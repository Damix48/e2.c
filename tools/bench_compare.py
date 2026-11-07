#!/usr/bin/env python3
"""Repeatable e2 (int8) vs OpenVINO (f32 / int8) benchmark.

Inputs are generated deterministically, both engines get the same bytes, and
both report min/median over a warm-up + N timed runs.

    . .venv/bin/activate
    uv pip install openvino nncf
    make tests/e2_bench_full
    python tools/bench_compare.py                     # e2 + ov f32 + ov int8
    python tools/bench_compare.py --sources e2        # e2 only, no torch/OV
    python tools/bench_compare.py --repeat 3 --json /tmp/bench.json
    python tools/bench_compare.py --sources ov-int8,ov-int8-gpu \
        --text-tokens 64 --batch-sizes 0 --video-frames 0

OpenVINO sources can be retargeted with ``--ov-device CPU|GPU|NPU`` (the
``-gpu`` suffixes are a shortcut for ``--ov-device GPU``). The e2 side is
targeted with ``--e2-device cpu|gpu``, which sets ``E2_DEVICE`` for the
subprocess.

Rows: text at 8/64/128/512/2048 tokens, text batches, image, audio, video.
``--repeat`` pools the samples of the whole run, so a noisy host cannot fake
an improvement.

OpenVINO / e2 fairness notes
- The text tower is converted once per sequence length: torch.jit.trace bakes
  the sliding-window branch (512) and the attention masks, so a trace cannot
  be reshaped to another length (verified: cosine drops below 0.99).
- Every conversion runs in its own worker process. The OpenVINO PyTorch
  frontend keeps a traced graph alive after the model is dropped (~0.8 GB per
  conversion, ~6 GB at 2048 tokens), which otherwise pushes the run into swap.
- The audio tower's mask construction does not trace, so it is split into
  subsample + layer stack with the blocked mask precomputed. Times are summed.
- image / video = vision tower + the text encoder at the processor's sequence
  length. e2 numbers are end-to-end (preprocess + towers + text encoder).
- video runs the vision tower once per frame on both sides (frames are
  independent; their position ids index the frame).
- e2 audio excludes WAV decode (its API takes raw PCM); e2 image includes PNG
  decode.
- OpenVINO int8 is NNCF data-free weight compression (INT8_SYM): int8 weights,
  f32 activations.
"""

import argparse
import gc
import json
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ---------------------------------------------------------------- assets

def make_assets(assets_dir, frames, image_size, audio_seconds):
    """Deterministic PNG image, PNG video frames and a 16 kHz WAV."""
    from PIL import Image

    os.makedirs(assets_dir, exist_ok=True)
    w, h = image_size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1),
                     (xx + yy) % 255], axis=-1).astype(np.uint8)

    image = os.path.join(assets_dir, f"image_{w}x{h}.png")
    Image.fromarray(base).save(image)

    frame_paths = []
    for i in range(frames):
        frame = ((base.astype(np.int16) + 7 * i) % 256).astype(np.uint8)
        path = os.path.join(assets_dir, f"frame_{i:02d}_{w}x{h}.png")
        Image.fromarray(frame).save(path)
        frame_paths.append(path)

    audio = os.path.join(assets_dir, f"tone_{audio_seconds:g}s.wav")
    sr = 16000
    t = np.arange(int(audio_seconds * sr)) / sr
    pcm = 0.3 * np.sin(2 * np.pi * (220 + 110 * t / audio_seconds) * t)
    with wave.open(audio, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((pcm * 32767).astype("<i2").tobytes())

    return {"image": image, "frames": frame_paths, "audio": audio}


def read_wav(path):
    with wave.open(path) as fh:
        assert fh.getnchannels() == 1 and fh.getsampwidth() == 2, "mono s16 only"
        sr = fh.getframerate()
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def ntokens(tok, text):
    return len(tok(text)["input_ids"]) if tok is not None else 0


def make_text(tok, n):
    """Repeated sentence, trimmed to just over n tokens (exact if tok given)."""
    base = ("The northern lights are caused by charged particles from the sun "
            "colliding with gases in the upper atmosphere. ")
    text = base * (n // 15 + 2)
    if tok is None:
        return text
    while ntokens(tok, text) < n:
        text += base
    words = text.split()
    while len(words) > 1 and ntokens(tok, " ".join(words[:-1])) >= n:
        words.pop()
    return " ".join(words)


# ---------------------------------------------------------------- e2 side

def run_e2(binary, spec_path, device=None):
    if not os.path.exists(binary):
        raise SystemExit(f"{binary} missing: run `make tests/e2_bench_full`")
    env = None
    if device:
        env = os.environ.copy()
        env["E2_DEVICE"] = device
    proc = subprocess.run([binary, spec_path], capture_output=True, text=True,
                          env=env)
    if proc.returncode != 0:
        raise SystemExit(f"e2 bench failed ({proc.returncode}):\n{proc.stderr}")
    times, info = {}, {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            info[parts[0]] = parts[1]
        elif len(parts) == 3:
            try:
                times[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                print(f"  e2 {parts[0]}: {parts[2]}", file=sys.stderr)
    return times, info


def write_spec(path, args, texts, assets, batch_sizes):
    rows = [("model", args.e2_model), ("tokenizer", args.e2_tok),
            ("iters", str(args.iters * args.repeat))]
    if args.threads:
        rows.append(("threads", str(args.threads)))
    for requested in texts:
        rows.append(("text", f"text {requested}", texts[requested][0]))
    batch_text = texts[args.batch_tokens][0]
    for b in batch_sizes:
        rows.append(("batch", f"batch {b}x{args.batch_tokens}", *([batch_text] * b)))
    rows.append(("image", "image", assets["image"]))
    rows.append(("audio", "audio", assets["audio"]))
    rows.append(("video", f"video {len(assets['frames'])}", *assets["frames"]))
    with open(path, "w") as fh:
        for row in rows:
            fh.write("\t".join(row) + "\n")


# ---------------------------------------------------------------- OV helpers

def ov_modules():
    import warnings
    import nncf
    import openvino as ov
    warnings.filterwarnings("ignore")
    try:
        nncf.set_log_level("ERROR")
    except Exception:
        pass
    return ov, nncf


_core = None


def ov_core():
    global _core
    if _core is None:
        ov, _ = ov_modules()
        _core = ov.Core()
    return _core


def ov_device(source, override=None):
    """Device for an ov source name; ``override`` (--ov-device) wins."""
    if override:
        return override.upper()
    return "GPU" if source.endswith("-gpu") else "CPU"


def ov_int8(source):
    return source.startswith("ov-int8")


class Unit:
    """One compiled OpenVINO model plus its feeds."""

    def __init__(self, model, arrays, threads, device="CPU"):
        self.device = device
        config = self._config(threads, device)
        core = ov_core()
        try:
            self.cm = core.compile_model(model, device, config)
        except Exception:
            self.cm = core.compile_model(model, device)
        self.feeds = {self.cm.inputs[i]: np.asarray(a)
                      for i, a in enumerate(arrays)}

    @staticmethod
    def _config(threads, device):
        if device.upper() == "GPU":
            config = {"PERFORMANCE_HINT": "LATENCY"}
            try:
                config["INFERENCE_PRECISION_HINT"] = "f16"
            except Exception:
                pass
            return config
        if device.upper() == "CPU" and threads:
            return {"INFERENCE_NUM_THREADS": str(threads),
                    "PERFORMANCE_HINT": "LATENCY"}
        return {}

    def run(self):
        self.cm(self.feeds)


def measure(units, iters, repeats):
    """2 warm-up passes, then iters*repeats timed passes; returns best, median."""
    for _ in range(2):
        for u in units:
            u.run()
    ts = []
    for _ in range(max(iters, 1) * max(repeats, 1)):
        t0 = time.perf_counter()
        for u in units:
            u.run()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return min(ts), statistics.median(ts)


def compress(model):
    _, nncf = ov_modules()
    try:
        return nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)
    except Exception as e:                                  # pragma: no cover
        print(f"  nncf int8 failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def quiet_transformers():
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()


# ---------------------------------------------------------------- OV workers
# Each worker runs in its own process so the traced graph of one model is
# released by the kernel when it exits.

def ov_text_job(job):
    """Convert the text tower at one sequence length and time every row at it."""
    model_dir, T, sources, iters, repeats, threads, rows, want_enc, ov_dev = job
    import torch
    from transformers import AutoModel
    ov, _ = ov_modules()
    quiet_transformers()

    class TextWrap(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids):
            return self.lm(input_ids=input_ids).last_hidden_state

    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32,
                                      local_files_only=True,
                                      attn_implementation="eager").eval()
    dummy = np.full((1, T), 5, dtype=np.int64)
    with torch.no_grad():
        conv = ov.convert_model(TextWrap(model.language_model),
                                example_input=(torch.from_numpy(dummy),),
                                share_weights=False)
    out = {"T": T, "rows": {}, "enc": {}}
    for source in sources:
        m = compress(conv) if ov_int8(source) else conv
        if m is None:
            continue
        device = ov_device(source, ov_dev)
        for row, arr in rows:
            m.reshape({m.input(0): ov.PartialShape(list(arr.shape))})
            t = measure([Unit(m, [arr], threads, device)], iters, repeats)
            out["rows"].setdefault(row, {})[source] = t
            print(f"    T={T} {row:<14}{source:>8} {t[1]:8.1f} ms", flush=True)
        if want_enc:
            m.reshape({m.input(0): ov.PartialShape([1, T])})
            out["enc"][source] = measure([Unit(m, [dummy], threads, device)],
                                         iters, repeats)
    return out


def ov_av_job(job):
    """Time the vision tower (image + one video frame) and the audio tower."""
    (model_dir, sources, iters, repeats, threads, nframes,
     img, frame, aud, ov_dev) = job
    import torch
    from transformers import AutoModel
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.models.gemma4.modeling_gemma4 import sliding_window_mask_function
    ov, _ = ov_modules()
    quiet_transformers()

    class VisionWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values, position_ids):
            out = self.m.vision_tower(pixel_values=pixel_values,
                                      pixel_position_ids=position_ids)
            return self.m.embed_vision(out.last_hidden_state)

    class AudioSubWrap(torch.nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.t = tower

        def forward(self, input_features, input_features_mask):
            hidden, _ = self.t.subsample_conv_projection(input_features,
                                                         input_features_mask)
            return hidden

    class AudioCoreWrap(torch.nn.Module):
        def __init__(self, tower, embed):
            super().__init__()
            self.t = tower
            self.embed = embed

        def forward(self, hidden_states, position_embeddings, attention_mask):
            x = hidden_states
            for layer in self.t.layers:
                x = layer(x, attention_mask=attention_mask,
                          position_embeddings=position_embeddings)
            return self.embed(self.t.output_proj(x))

    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32,
                                      local_files_only=True,
                                      attn_implementation="eager").eval()
    img_px, img_pos = img
    feats, amask_in = aud
    has_video = frame is not None

    print("    tracing vision...", flush=True)
    with torch.no_grad():
        vis_img = ov.convert_model(
            VisionWrap(model), example_input=(torch.from_numpy(img_px),
                                              torch.from_numpy(img_pos)),
            share_weights=False)
        vis_frm = None
        if has_video:
            frm_px, frm_pos = frame
            vis_frm = ov.convert_model(
                VisionWrap(model), example_input=(torch.from_numpy(frm_px),
                                                  torch.from_numpy(frm_pos)),
                share_weights=False)
    print("    tracing audio...", flush=True)
    at = model.audio_tower
    t_feats = torch.from_numpy(feats)
    t_amask = torch.from_numpy(amask_in)
    with torch.no_grad():
        hs, msub = at.subsample_conv_projection(t_feats, t_amask)
        pos = at.rel_pos_enc(hs)
        am = create_bidirectional_mask(
            config=at.config, inputs_embeds=hs, attention_mask=msub,
            and_mask_function=sliding_window_mask_function(
                (at.config.attention_context_left - 1,
                 at.config.attention_context_right)))
        am = at._convert_4d_mask_to_blocked_5d(am)
        sub_ov = ov.convert_model(AudioSubWrap(at),
                                  example_input=(t_feats, t_amask),
                                  share_weights=False)
        core_ov = ov.convert_model(AudioCoreWrap(at, model.embed_audio),
                                   example_input=(hs, pos, am),
                                   share_weights=False)

    out = {}
    for source in sources:
        i8 = ov_int8(source)
        device = ov_device(source, ov_dev)
        mi = compress(vis_img) if i8 else vis_img
        mf = compress(vis_frm) if (i8 and has_video) else vis_frm
        ms = compress(sub_ov) if i8 else sub_ov
        mc = compress(core_ov) if i8 else core_ov
        if None in (mi, ms, mc) or (has_video and mf is None):
            continue
        img_u = Unit(mi, [img_px, img_pos], threads, device)
        aud_u = [Unit(ms, [feats, amask_in], threads, device),
                 Unit(mc, [hs.numpy(), pos.numpy(), am.numpy()], threads, device)]
        out.setdefault("image", {})[source] = measure([img_u], iters, repeats)
        out.setdefault("audio", {})[source] = measure(aud_u, iters, repeats)
        if has_video:
            frm_u = Unit(mf, [frm_px, frm_pos], threads, device)
            out.setdefault("video", {})[source] = measure([frm_u] * nframes,
                                                          iters, repeats)
            del frm_u
        del img_u, aud_u, mi, mf, ms, mc
        gc.collect()
        for row in out:
            print(f"    {row:<10}{source:>8} {out[row][source][1]:8.1f} ms",
                  flush=True)
    return out


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/dev-box/embeddinggemma-2"),
                    help="HuggingFace checkpoint directory")
    ap.add_argument("--sources", default="e2,ov-f32,ov-int8",
                    help="comma list of e2, ov-f32, ov-int8, ov-f32-gpu, "
                         "ov-int8-gpu")
    ap.add_argument("--e2-bin", default="./tests/e2_bench_full")
    ap.add_argument("--e2-model", default="models/e2-text+image+audio+video.bin")
    ap.add_argument("--e2-tok", default="models/e2-tokenizer.bin")
    ap.add_argument("--ov-device", default=None, type=str.upper,
                    choices=["CPU", "GPU", "NPU"],
                    help="retarget all OpenVINO sources to this device")
    ap.add_argument("--e2-device", default=None, choices=["cpu", "gpu"],
                    help="set E2_DEVICE for the e2 subprocess (default: unset)")
    ap.add_argument("--assets", default="/tmp/e2bench")
    ap.add_argument("--text-tokens", default="8,64,128,512,2048")
    ap.add_argument("--batch-sizes", default="8,32")
    ap.add_argument("--batch-tokens", type=int, default=64)
    ap.add_argument("--video-frames", type=int, default=8)
    ap.add_argument("--image-size", default="640x480")
    ap.add_argument("--audio-seconds", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1,
                    help="pool this many extra measurement passes")
    ap.add_argument("--threads", type=int, default=8,
                    help="inference threads for both engines (0 = engine default)")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    args = ap.parse_args()

    sources = [s for s in args.sources.split(",") if s]
    unknown = set(sources) - {"e2", "ov-f32", "ov-int8", "ov-f32-gpu",
                              "ov-int8-gpu"}
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")
    text_tokens = [int(t) for t in args.text_tokens.split(",") if t]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    batch_sizes = [b for b in batch_sizes if b > 0]      # 0 = no batch rows
    w, h = (int(x) for x in args.image_size.split("x"))
    ov_sources = [s for s in sources if s != "e2"]

    devices = {}
    if "e2" in sources:
        devices["e2"] = (args.e2_device or "cpu").upper()
    for s in ov_sources:
        devices[s] = ov_device(s, args.ov_device)

    assets = make_assets(args.assets, args.video_frames, (w, h), args.audio_seconds)
    print(f"assets: {assets['image']}, {assets['audio']}, "
          f"{len(assets['frames'])} frames", flush=True)

    tok = None
    if ov_sources:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        quiet_transformers()
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    texts = {}                       # requested length -> (text, real token count)
    for n in text_tokens + [args.batch_tokens]:
        text = make_text(tok, n)
        texts[n] = (text, ntokens(tok, text))
    if tok is not None:
        print("tokens: " + ", ".join(f"{n}->{texts[n][1]}" for n in texts), flush=True)

    results, meta = {}, {
        "args": vars(args), "sources": sources, "host": platform.node(),
        "cpu": platform.processor() or platform.machine(), "assets": assets,
        "tokens": {str(k): v[1] for k, v in texts.items()},
        "devices": devices,
    }

    if "e2" in sources:
        print("e2...", flush=True)
        spec = os.path.join(args.assets, "spec.tsv")
        write_spec(spec, args, texts, assets, batch_sizes)
        times, info = run_e2(args.e2_bin, spec, args.e2_device)
        meta["e2_threads"] = info.get("threads")
        for label, t in times.items():
            results.setdefault(label, {})["e2"] = t

    if ov_sources:
        run_ov(args, ov_sources, texts, text_tokens, batch_sizes, assets,
               tok, results, meta)

    report(results, sources, args, meta, devices)


def run_ov(args, sources, texts, text_tokens, batch_sizes, assets, tok,
           results, meta):
    from transformers import AutoProcessor
    from PIL import Image

    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    img_in = proc(images=Image.open(assets["image"]).convert("RGB"),
                  return_tensors="np")
    vid_in = None
    if assets["frames"]:
        frames = [[Image.open(p).convert("RGB") for p in assets["frames"]]]
        vid_in = proc(videos=frames, text=["<|video|>"], return_tensors="np")
    wav, sr = read_wav(assets["audio"])
    aud_in = proc(audio=wav, sampling_rate=sr, return_tensors="np")

    seq = {"image": int(img_in["input_ids"].shape[1]),
           "audio": int(aud_in["input_ids"].shape[1])}
    if vid_in is not None:
        seq["video"] = int(vid_in["input_ids"].shape[1])
    meta["seq"] = seq
    print(f"sequence lengths: image={seq['image']} "
          + (f"video={seq['video']} " if "video" in seq else "")
          + f"audio={seq['audio']}", flush=True)

    def ids_of(n):
        return np.array([tok(texts[n][0])["input_ids"]], dtype=np.int64)

    # rows to time per discovered text length
    jobs = {}
    for n in text_tokens:
        T = texts[n][1]
        jobs.setdefault(T, []).append((f"text {n}", ids_of(n)))
    for b in batch_sizes:
        T = texts[args.batch_tokens][1]
        ids = np.tile(ids_of(args.batch_tokens), (b, 1))
        jobs.setdefault(T, []).append((f"batch {b}x{args.batch_tokens}", ids))
    lengths = sorted(set(jobs) | set(seq.values()))
    enc = {}                       # source -> T -> (best, median)

    av_job = (args.model, sources, args.iters, args.repeat, args.threads,
              len(assets["frames"]),
              (img_in["pixel_values"], img_in["image_position_ids"]),
              ((vid_in["pixel_values_videos"][:1],
                vid_in["video_position_ids"][:1]) if vid_in is not None
               else None),
              (aud_in["input_features"], aud_in["input_features_mask"]),
              args.ov_device)

    print("openvino...", flush=True)
    pool_kw = {"max_workers": 1,
               "mp_context": multiprocessing.get_context("spawn")}
    if sys.version_info >= (3, 11):
        pool_kw["max_tasks_per_child"] = 1     # a fresh process per model
    with ProcessPoolExecutor(**pool_kw) as pool:
        futures = [pool.submit(ov_text_job, (args.model, T, sources, args.iters,
                                             args.repeat, args.threads,
                                             jobs.get(T, []), T in seq.values(),
                                             args.ov_device))
                   for T in lengths]
        for fut in futures:
            res = fut.result()
            for row, per_source in res["rows"].items():
                for source, t in per_source.items():
                    results.setdefault(row, {})[source] = t
            for source, t in res["enc"].items():
                enc.setdefault(source, {})[res["T"]] = t
        av = pool.submit(ov_av_job, av_job).result()

    # image / video / audio = tower + text encoder at that sequence length
    av_rows = [("image", seq["image"])]
    if "video" in seq:
        av_rows.append((f"video {len(assets['frames'])}", seq["video"]))
    av_rows.append(("audio", seq["audio"]))
    for key, length in av_rows:
        tower_key = "video" if key.startswith("video") else key
        for source, t in av.get(tower_key, {}).items():
            text_t = enc.get(source, {}).get(length)
            if text_t is None:
                continue
            results.setdefault(key, {})[source] = (t[0] + text_t[0],
                                                  t[1] + text_t[1])


def report(results, sources, args, meta, devices):
    order = [f"text {n}" for n in args.text_tokens.split(",")]
    order += [f"batch {b}x{args.batch_tokens}" for b in args.batch_sizes.split(",")]
    order += ["image", "audio", f"video {args.video_frames}"]
    order = [r for r in order if results.get(r)]
    cols = [s for s in sources if any(s in results[r] for r in order)]

    print(f"\nmedian ms  iters={args.iters}  repeat={args.repeat}  "
          f"threads={args.threads or 'engine default'}"
          + (f" (e2={meta.get('e2_threads')})" if meta.get("e2_threads") else ""))
    headers = [f"{c}@{devices.get(c)}" if devices.get(c) else c for c in cols]
    width = max(11, max(len(h) for h in headers)) + 1
    print(f"{'task':<18}{'length':>8}" + "".join(f"{h:>{width}}" for h in headers))
    for row in order:
        kind, _, rest = row.partition(" ")
        if row.startswith("text "):
            length = meta["tokens"].get(rest) or rest
        elif row.startswith("batch "):
            length = rest
        else:
            length = meta.get("seq", {}).get(kind, "")
        cells = "".join(f"{results[row][c][1]:{width}.1f}" if c in results[row]
                        else f"{'-':>{width}}" for c in cols)
        print(f"{row:<18}{str(length):>8}{cells}")

    print("\nnotes: ov image/video = tower(s) + text encoder at the processor's "
          "sequence length;")
    print("       e2 is end-to-end and includes the tokenizer.")
    print("       ov int8 = NNCF INT8_SYM weight-only compression.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"meta": meta,
                       "results": {row: {s: {"best_ms": v[0], "median_ms": v[1],
                                             "device": devices.get(s)}
                                         for s, v in r.items()}
                                   for row, r in results.items()}},
                      fh, indent=1)
        print(f"json: {args.json}")


if __name__ == "__main__":
    main()
