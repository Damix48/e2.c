#!/usr/bin/env python
"""Dump engine test fixtures (token ids + fp32 reference embedding) from bench/fixtures/.

    uv run python engine/tools/dump_fixtures.py
Writes engine/tests/fixtures/<case>.bin for the cases the engine supports.
"""
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "bench" / "fixtures"
DST = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DST.mkdir(parents=True, exist_ok=True)

inputs = dict(np.load(SRC / "inputs.npz"))
ref = dict(np.load(SRC / "reference.npz"))

cases = {
    "text_query": (inputs["text_query_input_ids"], ref["text_query"]),
    "text_doc": (inputs["text_doc_input_ids"], ref["text_doc"]),
}
for name, (ids, emb) in cases.items():
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    with open(DST / f"{name}.bin", "wb") as f:
        f.write(struct.pack("<ii", ids.size, emb.size))
        f.write(ids.tobytes())
        f.write(emb.tobytes())
    print(f"{name}: {ids.size} ids, {emb.size} dims -> {DST / (name + '.bin')}")
