#!/usr/bin/env python
"""Same-session OpenVINO audio-tower benchmark for the numbers in README.

    E2_MODEL_DIR=~/Projects/embeddinggemma-2 .venv/bin/python tools/bench_audio_ov.py --frames 76

The graph is exported without an attention mask and with every frame valid. That is deliberate: the
masked graph reaches the GPU plugin as a uint8 GatherND, which OpenVINO 2026.4.1 cannot place
("[GPU] No layout format available for gathernd"). The full-valid graph has the same projections,
attention and convolutions, so its latency is the number the e2 GPU path is compared against; it is
not a fidelity comparison. For fidelity use the engine's own frozen fixture (`e2_test_gpu_audio`).

The 48 learned clamp buffers are exported as Python floats. Left as 0-d buffers, torch's ONNX
exporter emits Clip nodes whose min/max initializers are missing and the IR frontend rejects the
graph; the monkey patch below is what makes the comparison possible.
"""
import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__import__("os").environ.get("E2_MODEL_DIR", Path.home() / "Projects" / "embeddinggemma-2"))
sys.path.insert(0, str(ROOT / "scripts"))

import openvino as ov  # noqa: E402
import nncf  # noqa: E402
from onnx_common import load_model  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear  # noqa: E402


def patch_clamps():
    def forward(self, hidden_states):
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, min=float(self.input_min), max=float(self.input_max))
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, min=float(self.output_min), max=float(self.output_max))
        return hidden_states

    Gemma4ClippableLinear.forward = forward


def export(frames: int, path: Path):
    model = load_model(("text", "image", "audio"))[0].model
    tower, embed = model.audio_tower, model.embed_audio

    class W(torch.nn.Module):
        def forward(self, features):
            return embed(inputs_embeds=tower(input_features=features, attention_mask=None)[0])

    w = W().eval()
    with torch.no_grad():
        torch.onnx.export(w, torch.zeros(1, frames, 128), str(path), input_names=["input_features"],
                          output_names=["soft_tokens"], dynamo=True, opset_version=18,
                          external_data=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=76, help="mel frames (76 -> 19 soft tokens)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    patch_clamps()
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "audio.onnx"
        print("exporting", flush=True)
        export(args.frames, onnx_path)
        print("converting", flush=True)
        model = ov.convert_model(str(onnx_path))
        model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)

    core = ov.Core()
    feeds = {i.get_any_name(): np.zeros((1, args.frames, 128), np.float32) for i in model.inputs}
    out = {}
    for dev in ("CPU", "GPU"):
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(model, dev)
            compile_s = time.perf_counter() - t0
            for _ in range(args.warmup):
                compiled(feeds)
            wall = []
            for _ in range(args.runs):
                t = time.perf_counter()
                compiled(feeds)
                wall.append(time.perf_counter() - t)
            out[dev] = {"median_ms": round(statistics.median(wall) * 1e3, 2),
                        "min_ms": round(min(wall) * 1e3, 2), "compile_s": round(compile_s, 2)}
            print(f"openvino int8 {dev}: median {out[dev]['median_ms']:.2f} ms  "
                  f"min {out[dev]['min_ms']:.2f} ms  compile {compile_s:.1f} s", flush=True)
        except Exception as exc:
            print(f"openvino int8 {dev}: FAILED {str(exc).splitlines()[-1][:200]}", flush=True)
    if args.json:
        import json
        Path(args.json).write_text(json.dumps({"frames": args.frames, **out}, indent=2))


if __name__ == "__main__":
    main()
