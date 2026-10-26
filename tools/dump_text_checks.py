#!/usr/bin/env python3
"""Freeze FP32 ONNX references for text kernel boundaries and a tiny retrieval smoke test.

Uses the reference project's existing numpy/onnxruntime/transformers environment:
  ../embeddinggemma-2/.venv/bin/python tools/dump_text_checks.py ../embeddinggemma-2
Also writes text_masked.bin beside the corpus for a non-compacted interior-mask oracle.
This is a regression corpus, not a retrieval benchmark or calibration dataset.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/text_checks.bin")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model_dir / "onnx/text.onnx"), options, providers=["CPUExecutionProvider"])
    pairs = [
        ("What causes the northern lights?", "The northern lights form when charged particles from the Sun collide with gases in Earth's atmosphere near the magnetic poles."),
        ("Why does bread dough rise?", "Yeast ferments sugars in bread dough and produces carbon dioxide. The trapped gas expands the dough as it rises."),
        ("How does TCP deliver data reliably?", "TCP uses sequence numbers, acknowledgments and retransmission to deliver an ordered stream of bytes reliably over a network."),
        ("How do plants turn sunlight into food?", "Photosynthesis lets plants use sunlight to convert water and carbon dioxide into sugars, releasing oxygen."),
        ("Why do volcanoes erupt?", "A volcanic eruption happens when pressure drives molten rock and gas from beneath Earth's crust onto the surface."),
        ("How does a hash table find a key?", "A hash table computes a hash from a key to choose a bucket. It handles collisions when different keys select the same bucket."),
        ("How does gzip compress files?", "Gzip compresses data with DEFLATE, which combines repeated-string references with Huffman coding, and includes a checksum."),
        ("How can an index speed up a SQL query?", "A database index maps column values to rows. A SQL query can look up matching rows in the index instead of scanning the entire table."),
    ]
    cases = []
    for i, (query, _) in enumerate(pairs):
        cases.append((f"query_{i}", tokenizer.encode("task: search result | query: " + query)))
    for i, (_, document) in enumerate(pairs):
        cases.append((f"document_{i}", tokenizer.encode(document)))
    prose = " ".join(document for _, document in pairs)
    body = tokenizer.encode((prose + " ") * 30, add_special_tokens=False)
    lengths = (1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
               196, 255, 256, 257, 511, 512, 513, 514, 1024, 2048)
    for length in lengths:
        ids = [2] if length == 1 else [2] + body[:length - 2] + [1]
        assert len(ids) == length
        cases.append((f"length_{length}", ids))
    references = []
    records = []
    for name, ids in cases:
        inputs = np.asarray([ids], dtype=np.int64)
        vector = np.asarray(session.run(None, {"input_ids": inputs, "attention_mask": np.ones_like(inputs)})[0], dtype="<f4").reshape(-1)
        assert vector.size == 768 and np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector)) - 1) < 1e-4
        references.append(vector)
        records.append(name.encode().ljust(64, b"\0") + struct.pack("<i", len(ids)) + np.asarray(ids, dtype="<i4").tobytes() + vector.tobytes())
        print(f"{name}: {len(ids)} tokens", flush=True)
    scores = np.stack(references[:len(pairs)]) @ np.stack(references[len(pairs):2 * len(pairs)]).T
    assert np.array_equal(scores.argmax(axis=1), np.arange(len(pairs))), "FP32 corpus must retrieve each paired document"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(struct.pack("<4I", 0x54583245, len(records), len(pairs), 768) + b"".join(records))
    print(f"wrote {args.output}: {len(records)} cases, {len(pairs)} retrieval pairs")
    ids = np.asarray([[2] + body[:1022] + [1]], dtype=np.int64)
    mask = np.ones_like(ids)
    mask[:, 128:896] = 0  # the surviving blocks must stay 768 positions apart
    vector = session.run(None, {"input_ids": ids, "attention_mask": mask})[0].astype("<f4").reshape(-1)
    compact = ids[mask != 0].reshape(1, -1)
    compact_vector = session.run(None, {"input_ids": compact, "attention_mask": np.ones_like(compact)})[0].reshape(-1)
    cosine = float(np.dot(vector, compact_vector))
    assert np.isfinite(vector).all() and abs(float(np.linalg.norm(vector)) - 1) < 1e-4
    assert cosine < 0.998, f"masked oracle must distinguish compaction, cosine={cosine}"
    path = args.output.with_name("text_masked.bin")
    path.write_bytes(struct.pack("<ii", ids.size, vector.size) + ids.astype("<i4").tobytes() +
                     mask.astype("u1").tobytes() + vector.tobytes())
    print(f"wrote {path}: masked-vs-compacted FP32 cosine {cosine:.9f}")


if __name__ == "__main__":
    main()
