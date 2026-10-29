#!/usr/bin/env python3
"""Matched e2 vs OpenVINO weight-only INT8 latency; see tools/BENCHMARKS.md."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parents[2]  # repository root: this script drives `cmake -S HERE`
CASES = ("query", "document", "query_batch8", "document_batch32", "image", "image_batch4",
         "audio", "audio_batch4")
# The image and audio towers have no batched entry point in either engine, so their "batch" cases
# repeat the single call inside one timed call: throughput, not a batched kernel. Text batches are
# genuine batched calls.
REPEATS = {"image_batch4": 4, "audio_batch4": 4}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


class Fixture:
    def __init__(self, path):
        self.data = path.read_bytes()
        self.offset = 0

    def take(self, dtype, count):
        dtype = np.dtype(dtype)
        end = self.offset + count * dtype.itemsize
        if count < 0 or end > len(self.data):
            raise ValueError("truncated/invalid fixture")
        out = np.frombuffer(self.data, dtype, count, self.offset).copy()
        self.offset = end
        return out

    def end(self):
        if self.offset != len(self.data):
            raise ValueError("unexpected fixture tail")


MIN_COSINE = 0.999


def cosine(output, reference):
    """Structure/unit-norm validation plus the worst row cosine; no accuracy gate."""
    a, b = np.asarray(output, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid output shape/values: {a.shape}, expected {b.shape}")
    a, b = a.reshape(-1, 768), b.reshape(-1, 768)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(abs(na - 1) > 0.001) or np.any(nb == 0):
        raise ValueError("non-unit or zero output")
    return float(np.min(np.sum(a * b, axis=1) / (na * nb)))


def check(output, reference, minimum=MIN_COSINE):
    value = cosine(output, reference)
    if value < minimum:
        raise ValueError(f"cosine {value:.9f} < {minimum}")
    return value


def summary(samples):
    ordered = sorted(samples)
    if not ordered or not all(math.isfinite(x) and x > 0 for x in ordered):
        raise ValueError("invalid timing samples")
    return {"median_ms": statistics.median(ordered), "min_ms": ordered[0],
            "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1]}


def save_case(out, name, kind, batch, n, payload, feeds, ref, xml, boundary, repeat=1):
    ref = np.asarray(ref, dtype=np.float32).reshape(batch, 768)
    check(ref, ref)
    with open(out / f"{name}.bin", "wb") as f:
        np.array([kind, batch, n, repeat], dtype="<i4").tofile(f)
        for data in payload:
            f.write(np.ascontiguousarray(data).tobytes())
        ref.astype("<f4").tofile(f)
    np.savez(out / f"{name}.npz", reference=ref, **feeds)
    return {"name": name, "batch": batch, "length": n, "xml": str(xml), "boundary": boundary,
            "repeat": repeat}


def prepare_case(name, fixture_dir, ir_dir, out):
    if name.startswith(("query", "document")):
        base = "query" if name.startswith("query") else "doc"
        fx = Fixture(fixture_dir / f"text_{base}.bin")
        n, dim = fx.take("<i4", 2)
        if not 0 < n <= 2048 or dim != 768:
            raise ValueError("bad text header")
        ids, ref = fx.take("<i4", int(n)), fx.take("<f4", int(dim))
        fx.end()
        batch = 8 if name.endswith("batch8") else 32 if name.endswith("batch32") else 1
        ids = np.tile(ids, (batch, 1))
        feeds = {"input_ids": ids.astype(np.int64), "attention_mask": np.ones(ids.shape, np.int64)}
        return save_case(out, name, 0, batch, int(n), [ids], feeds, np.tile(ref, (batch, 1)),
                         ir_dir / "text_int8.xml", "token IDs -> normalized 768d; no tokenization")
    if name.startswith("image"):
        repeat = REPEATS.get(name, 1)
        fx = Fixture(fixture_dir / "image.bin")
        n, pd, npos, dim = fx.take("<i4", 4)
        if not 0 < n <= 2520 or (pd, npos, dim) != (768, 2, 768):
            raise ValueError("bad image header")
        pixels = fx.take("<f4", int(n * pd)).reshape(n, pd)
        pos = fx.take("<i4", int(n * npos)).reshape(n, npos)
        ref = fx.take("<f4", int(dim))
        fx.end()
        return save_case(out, name, 1, 1, int(n), [pixels, pos],
                         {"pixel_values": pixels[None], "image_position_ids": pos[None].astype(np.int64)},
                         ref, ir_dir / "image_int8.xml",
                         "same padded image patches -> normalized 768d (vision + text); no image decoding/"
                         f"resize; one timed call = {repeat} sequential image{'s' if repeat > 1 else ''}",
                         repeat)
    raise ValueError(name)


def audio_fixture(path):
    fx = Fixture(path)
    samples, frames, soft, dim, ids, width, _, text = map(int, fx.take("<i4", 8))
    if not (0 < samples <= 480000 and 0 < frames <= 3000 and 0 < soft <= 750 and
            dim == 768 and width == 512 and 0 < ids <= 2048 and 0 <= text <= 65536):
        raise ValueError("bad audio header")
    fx.take("u1", text)
    pcm = fx.take("<f4", samples)
    fx.take("<i4", ids)
    mel = fx.take("<f4", frames * 128).reshape(1, frames, 128)
    mask = fx.take("u1", frames).reshape(1, frames)
    fx.take("<f4", soft * width + dim * 2)
    fx.end()
    return pcm, mel, mask, soft


def export_audio(model_dir, fixture_dir, out):
    # Reuse the existing export workaround, but KEEP the fixture's real tail mask.
    os.environ["E2_MODEL_DIR"] = str(model_dir)
    from bench_audio_ov import patch_clamps, load_model
    import torch
    import openvino as ov
    import nncf

    torch.set_num_threads(4)
    patch_clamps()
    pcm, mel, mask, nsoft = audio_fixture(fixture_dir / "audio.bin")
    model = load_model(("text", "audio"))[0].model.eval()

    class AudioEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = model.audio_tower
            self.project = model.embed_audio
            self.text = model.language_model
            self.register_buffer("mask", torch.from_numpy(mask).bool())
            embedding = model.get_input_embeddings()
            self.register_buffer("head", embedding(torch.tensor([[2, 256000]])).detach())
            self.register_buffer("tail", embedding(torch.tensor([[258883, 1]])).detach())

        def forward(self, input_features):
            soft = self.project(inputs_embeds=self.tower(input_features=input_features,
                                                       attention_mask=self.mask)[0])
            embeds = torch.cat([self.head, soft, self.tail], dim=1)
            hidden = self.text(inputs_embeds=embeds, attention_mask=torch.ones(
                (1, nsoft + 4), dtype=torch.long, device=embeds.device)).last_hidden_state
            return torch.nn.functional.normalize(hidden.mean(dim=1), p=2, dim=-1)

    wrapper = AudioEmbedding().eval()
    features = torch.from_numpy(mel)
    with torch.no_grad():
        ref = wrapper(features).float().numpy()
        torch.onnx.export(wrapper, (features,), str(out / "audio.onnx"),
                          input_names=["input_features"], output_names=["embedding"],
                          dynamo=True, opset_version=18, external_data=True)
    ir = ov.convert_model(str(out / "audio.onnx"))
    ir = nncf.compress_weights(ir, mode=nncf.CompressWeightsMode.INT8_SYM)
    xml = out / "audio_int8.xml"
    ov.save_model(ir, str(xml), compress_to_fp16=False)
    spec = save_case(out, "audio", 2, 1, len(pcm), [pcm], {"input_features": mel}, ref, xml,
                     "e2: 16kHz PCM -> 768d; OV: reference mel -> 768d, same tail mask; "
                     "OV excludes mel extraction (boundary advantage to OV)")
    spec.update(mel_frames=mel.shape[1], valid_frames=int(mask.sum()), soft_tokens=nsoft)
    (out / "audio.case.json").write_text(json.dumps(spec, indent=2))


def ov_worker(spec):
    import openvino as ov
    core = ov.Core()
    data = dict(np.load(spec["feeds"]))
    reference = data.pop("reference")
    model = core.read_model(spec["xml"])
    model.reshape({k: list(v.shape) for k, v in data.items()})
    storage = Counter(str(op.get_output_element_type(0)) for op in model.get_ops()
                      if op.get_type_name() == "Constant")
    if not any("int8" in k or "uint8" in k or "i8" in k for k in storage):
        raise ValueError("IR contains no 8-bit constants; not an INT8-weight baseline")
    # GPU plugin requires NUM_STREAMS as a string (CPU accepts either form).
    config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    if spec["device"] == "CPU":
        config["INFERENCE_NUM_THREADS"] = spec["threads"]
    t = time.perf_counter()
    compiled = core.compile_model(model, spec["device"], config)
    compile_s = time.perf_counter() - t
    request = compiled.create_infer_request()
    samples, minimum = [], 1.0
    for r in range(-spec["warmup"], spec["runs"]):
        t = time.perf_counter_ns()
        for _ in range(spec.get("repeat", 1)):
            request.infer(data)
        elapsed = (time.perf_counter_ns() - t) / 1e6
        output = request.get_output_tensor(0).data
        minimum = min(minimum, cosine(output, reference))
        if r >= 0:
            samples.append(elapsed)
    np.asarray(output, dtype=np.float32).tofile(spec["output"])
    precision = Counter()
    for op in compiled.get_runtime_model().get_ops():
        info = op.get_rt_info()
        if "runtimePrecision" in info:
            precision[str(info["runtimePrecision"].value)] += 1
    return {"samples_ms": samples, "minimum_cosine": minimum, "compile_s": compile_s,
            "config": config, "constant_types": dict(storage), "runtime_precisions": dict(precision),
            "device_name": core.get_property(spec["device"], "FULL_DEVICE_NAME")}


def run_logged(command, log, env=None, timeout=1800):
    with open(log, "w") as f:
        proc = subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT,
                              env=env, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"exit {proc.returncode}; see {log}")
    return log.read_text()


def render(report):
    lines = ["# e2 vs OpenVINO weight-only INT8", "", f"Started: {report['started']}", "",
             "Warm, synchronous batch-call latency; median of round medians. Speedup = OV/e2 (>1 favors e2).",
             f"Both engines are checked against the same FP32 reference on EVERY call; the gate is "
             f"cosine >= {MIN_COSINE}.", "",
             "| Case | Device | e2 ms | OV ms | e2 speedup | e2 cos | OV cos | Status |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        for device in report["devices"]:
            rows = [r for r in report["results"] if r["case"] == case["name"] and r["device"] == device]
            groups = {engine: [r for r in rows if r["engine"] == engine] for engine in ("e2", "ov")}
            values, worst, accurate = {}, {}, []
            for engine, trials in groups.items():
                if len(trials) == report["rounds"] and all(t["status"] == "ok" for t in trials):
                    values[engine] = statistics.median(t["median_ms"] for t in trials)
                    measured = [t.get("minimum_cosine") for t in trials]
                    if all(c is not None for c in measured):
                        worst[engine] = min(measured)
                    accurate.append(all(t.get("accuracy") == "pass" for t in trials))
            ratio = f"{values['ov']/values['e2']:.2f}x" if len(values) == 2 else "—"
            cells = [f"{values[e]:.3f}" if e in values else "—" for e in ("e2", "ov")]
            coss = [f"{worst[e]:.6f}" if e in worst else "—" for e in ("e2", "ov")]
            if len(values) != 2:
                status = "FAILED / incomplete; inspect results.json + logs"
            elif not all(accurate):
                status = "ACCURACY FAIL"
            else:
                status = "checked"
            lines.append(f"| {case['name']} | {device} | {cells[0]} | {cells[1]} | {ratio} | "
                         f"{coss[0]} | {coss[1]} | {status} |")
    if report.get("preparation_errors"):
        lines += ["", "## Preparation failures (NOT benchmark successes)", ""]
        lines += [f"- **{name}**: {error}" for name, error in report["preparation_errors"].items()]
    lines += ["", "## Boundaries", ""]
    lines += [f"- **{c['name']}**: {c['boundary']}" for c in report["cases"]]
    lines += ["", "Not calibrated full W8A8 OpenVINO: NNCF INT8_SYM weight compression. Different activation/attention precision.",
              "Small frozen/synthetic corpus, not retrieval-quality or stability certification. Compilation/load excluded.",
              "GPU may have known intermittent failures; a clean run does not resolve them. No frequency/power controls changed."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "Projects/embeddinggemma-2")
    ap.add_argument("--ir-dir", type=Path, help="default: MODEL/onnx/ov (NNCF INT8_SYM IRs)")
    ap.add_argument("--fixture-dir", type=Path, default=HERE / "tests/fixtures")
    ap.add_argument("--build-dir", type=Path, default=HERE / "build-compare")
    ap.add_argument("--output", type=Path, default=HERE / "benchmark-results" / time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--devices", default="CPU,GPU")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800, help="per-worker seconds")
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--audio-export", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        spec = json.loads(args.worker.read_text())
        result = ov_worker(spec)
        Path(spec["result"]).write_text(json.dumps(result, indent=2))
        return 0
    if args.audio_export:
        export_audio(args.model_dir.resolve(), args.fixture_dir.resolve(), args.output.resolve())
        return 0
    names, devices = args.cases.split(","), args.devices.split(",")
    if (not names or set(names) - set(CASES) or set(devices) - {"CPU", "GPU"} or
            len(set(names)) != len(names) or len(set(devices)) != len(devices) or
            not 1 <= args.runs <= 10000 or not 0 <= args.warmup <= 10000 or
            not 1 <= args.rounds <= 100 or not 1 <= args.threads <= len(os.sched_getaffinity(0)) or args.timeout < 1):
        ap.error("invalid case/device list, repetitions, thread budget or timeout")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite evidence from an earlier run
    model_dir, fixture_dir, build = args.model_dir.resolve(), args.fixture_dir.resolve(), args.build_dir.resolve()
    ir_dir = args.ir_dir.resolve() if args.ir_dir else model_dir / "onnx/ov"
    env = os.environ.copy()
    # Diagnostic/precision/cache overrides make the default-engine comparison ambiguous.
    overrides = {k: v for k, v in env.items() if k.startswith("E2_")}
    for k in overrides:
        del env[k]
    env["E2_THREADS"] = str(args.threads)
    run_logged(["cmake", "-S", HERE, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DE2_WITH_HIP=OFF",
                f"-DE2_WITH_GPU={'ON' if 'GPU' in devices else 'OFF'}", f"-DE2_MODEL_DIR={model_dir}"],
               out / "configure.log", env, args.timeout)
    run_logged(["cmake", "--build", build, "-j4", "--target", "e2_compare_bench"],
               out / "build.log", env, args.timeout)
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rounds": args.rounds,
              "runs": args.runs, "warmup": args.warmup, "threads": args.threads, "devices": devices,
              "command": sys.argv, "platform": platform.platform(), "python": sys.version,
              "affinity": sorted(os.sched_getaffinity(0)), "removed_e2_overrides": overrides,
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "openvino", "nncf", "torch")},
              "model": digest(model_dir / "model.safetensors"), "cases": [], "results": [],
              "source_hashes": {}, "input_hashes": {}, "preparation_errors": {}}
    for pattern in ("src/**/*.c", "src/**/*.h", "src/**/*.cl", "include/*.h", "tools/bench/*.c", "tools/bench/*.py",
                    "tests/test_metrics.h", "tests/test_bench_compare.py", "CMakeLists.txt", "cmake/*.cmake"):
        for p in HERE.glob(pattern):
            if p.is_file():
                report["source_hashes"][str(p.relative_to(HERE))] = digest(p)
    for command, filename in [(["lscpu"], "lscpu.txt"), (["uname", "-a"], "uname.txt")]:
        run_logged(command, out / filename)
    report["environment"] = {k: v for k, v in os.environ.items()
                             if k.startswith(("OMP_", "MKL_", "OPENVINO_", "ZE_", "SYCL_")) or
                             k in ("LD_LIBRARY_PATH", "ONEAPI_DEVICE_SELECTOR")}
    report["fixtures"] = {str(p): digest(p) for p in fixture_dir.glob("*.bin")}
    report["build_cache"] = digest(build / "CMakeCache.txt")
    report["reference_scripts"] = {str(p): digest(p) for p in
                                   (model_dir / "scripts").glob("*.py")}
    report["packages"] = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                      if d.metadata["Name"]))
    report["power_state"] = {str(p): p.read_text().strip() for pattern in
                             ("devices/system/cpu/cpufreq/policy*/scaling_governor",
                              "class/power_supply/*/online", "class/power_supply/*/status")
                             for p in Path("/sys").glob(pattern)}
    try:
        run_logged(["lspci", "-nnk"], out / "lspci.txt")
    except (OSError, RuntimeError) as exc:
        report["lspci_error"] = str(exc)
    audio_exported = None
    for name in names:
        try:
            if name.startswith("audio"):
                if audio_exported is None:
                    run_logged([sys.executable, __file__, "--audio-export", "--model-dir", model_dir,
                                "--fixture-dir", fixture_dir, "--output", out], out / "audio-export.log",
                               env, args.timeout)
                    audio_exported = json.loads((out / "audio.case.json").read_text())
                repeat = REPEATS.get(name, 1)
                case = dict(audio_exported, name=name, repeat=repeat)
                case["boundary"] += (f"; one timed call = {repeat} sequential "
                                     f"clip{'s' if repeat > 1 else ''}")
                if repeat != 1:
                    # Same payload, only the header's repeat field differs.
                    payload = (out / "audio.bin").read_bytes()[16:]
                    with open(out / f"{name}.bin", "wb") as f:
                        np.array([2, 1, case["length"], repeat], dtype="<i4").tofile(f)
                        f.write(payload)
                    shutil.copyfile(out / "audio.npz", out / f"{name}.npz")
            else:
                case = prepare_case(name, fixture_dir, ir_dir, out)
            for p in (out / f"{name}.bin", out / f"{name}.npz", Path(case["xml"]), Path(case["xml"]).with_suffix(".bin")):
                report["input_hashes"][str(p)] = digest(p)
            report["cases"].append(case)
        except Exception as exc:
            report["preparation_errors"][name] = str(exc)
            print(f"{name}: preparation FAILED: {exc}", flush=True)
    def persist():
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        (out / "REPORT.md").write_text(render(report))
    persist()
    for case in report["cases"]:
        name = case["name"]
        for device in devices:
            for round_index in range(args.rounds):
                order = ("e2", "ov") if round_index % 2 == 0 else ("ov", "e2")
                for engine in order:
                    stem = f"{name}-{device}-{round_index}-{engine}"
                    raw = out / f"{stem}.f32"
                    result = {"case": name, "device": device, "round": round_index, "engine": engine}
                    try:
                        if engine == "e2":
                            command = [build / "e2_compare_bench", model_dir, device.lower(), out / f"{name}.bin",
                                       args.runs, args.warmup, raw]
                        else:
                            spec = {"xml": case["xml"], "feeds": str(out / f"{name}.npz"), "device": device,
                                    "threads": args.threads, "runs": args.runs, "warmup": args.warmup,
                                    "repeat": case.get("repeat", 1), "output": str(raw),
                                    "result": str(out / f"{stem}.result.json")}
                            (out / f"{stem}.spec.json").write_text(json.dumps(spec, indent=2))
                            command = [sys.executable, __file__, "--worker", out / f"{stem}.spec.json"]
                        log = run_logged(command, out / f"{stem}.log", env, args.timeout)
                        measurements = (json.loads(next(line for line in reversed(log.splitlines()) if line.startswith('{')))
                                        if engine == "e2" else json.loads(Path(spec["result"]).read_text()))
                        if len(measurements["samples_ms"]) != args.runs:
                            raise ValueError("missing timing samples")
                        result.update(measurements, **summary(measurements["samples_ms"]), status="ok")
                        ref = np.load(out / f"{name}.npz")["reference"]
                        try:
                            check(np.fromfile(raw, dtype=np.float32).reshape(ref.shape), ref)
                            result["accuracy"] = "pass"
                        except ValueError as exc:
                            result["accuracy"] = f"FAIL: {exc}"
                        print(f"{stem}: {result['median_ms']:.3f} ms cos>={result['minimum_cosine']:.6f}"
                              f"{'' if result['accuracy'] == 'pass' else ' ACCURACY ' + result['accuracy']}",
                              flush=True)
                    except Exception as exc:
                        result.update(status="failed", error=str(exc))
                        print(f"{stem}: FAILED: {exc}", flush=True)
                    report["results"].append(result)
                    persist()
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    persist()
    print(out / "REPORT.md")
    return int(bool(report["preparation_errors"]) or any(r["status"] != "ok" for r in report["results"])
               or any(r.get("accuracy") != "pass" for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
