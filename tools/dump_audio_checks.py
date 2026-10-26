#!/usr/bin/env python3
"""Freeze independent FP32 audio boundary references. Does not modify existing fixtures.

Development only: needs PyTorch, transformers and sentence-transformers plus the checkpoint.
  python tools/dump_audio_checks.py MODEL_DIR
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/audio_checks.bin")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = SentenceTransformer(str(args.model_dir), config_kwargs={"vision_config": None},
                               model_kwargs={"dtype": torch.float32, "local_files_only": True})
    base = model[0].model
    frontend = model[0].processor.feature_extractor
    # Same scalar double-precision formula as test_audio_reference.c, rounded to f32 once.
    pcm = np.asarray([.2 * math.sin(2 * math.pi * 220 * i / 16000) +
                      .1 * math.cos(2 * math.pi * 530 * i / 16000) for i in range(480000)], dtype=np.float32)
    records = []
    with torch.inference_mode():
        for n in (161, 640, 7680, 8320, 475520, 476160, 476800, 479360, 480000):
            feat = frontend([pcm[:n]], sampling_rate=16000, return_tensors="pt")
            output = base.audio_tower(input_features=feat["input_features"],
                                      attention_mask=feat["input_features_mask"])
            soft = base.embed_audio(inputs_embeds=output.last_hidden_state)
            # The reference frontend pads short clips; retain only real soft tokens.
            count = (int(feat["input_features_mask"].sum()) + 3) // 4
            soft = soft[:, :count]
            ids = torch.tensor([[2, 256000] + [258881] * count + [258883, 1]])
            embeds = base.get_input_embeddings()(ids).clone()
            embeds[:, 2:2 + count] = soft
            hidden = base.language_model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids)).last_hidden_state
            vector = torch.nn.functional.normalize(hidden.mean(dim=1), dim=-1)[0].float().numpy()
            assert vector.shape == (768,) and np.isfinite(vector).all()
            assert abs(float(np.linalg.norm(vector)) - 1) < 1e-4
            records.append(struct.pack("<II", n, count) + vector.astype("<f4").tobytes())
            print(f"{n} samples, {count} soft tokens", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(struct.pack("<4I", 0x41323245, 1, len(records), 768) + b"".join(records))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
