#!/usr/bin/env python
"""Dump the tokenizer fixture: a corpus of (utf-8 text -> token ids) pairs produced by the HF fast
tokenizer, plus the multimodal placeholder expansion the `EmbeddingGemma2Processor` lays out.

    uv run python engine/tools/dump_tokenizer_fixture.py

Fixture format (little endian):

    u32 n_cases
    per case: u32 n_bytes  bytes text  u32 n_ids  i32 ids[n_ids]
    u32 n_mm
    per mm case: u32 n_bytes bytes text  u32 n_images  i32 n_soft[n_images]  u32 n_ids i32 ids[n_ids]

`ids` include the `<bos>`/`<eos>` template, `image_position_ids`/`pixel_values` for the mm cases come
from the processor so the layout is the model's, not ours.
"""
import json
import random
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DST = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DST.mkdir(parents=True, exist_ok=True)

from transformers import AutoProcessor, AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(ROOT))
proc = AutoProcessor.from_pretrained(str(ROOT))

QUERY = "task: search result | query: What causes the northern lights?"
LONG = (
    "Retrieval augmented generation combines a retriever over a document collection with a "
    "generator that conditions on the retrieved passages. The retriever embeds the query and "
    "the documents into one vector space and returns the nearest neighbours, which are then "
    "inserted into the generator's context window. Quality depends on both the retriever and "
    "how the passages are chunked. "
) * 3
CAPTION = "task: sentence similarity | query: A photograph of a landscape. <|image|>"

cases = [
    "",  # empty string still gets the template
    "a",
    QUERY,
    LONG,
    CAPTION,
    "  leading and trailing  ",
    "Hello world! 123 \u00e9\u00e0 \u4f60\u597d",
    "\n\n\tmixed\twhitespace\n\n",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059",
    "emoji \U0001f389\U0001f680 combining e\u0301 vs \u00e9",
    "Z\u00fcrich, Stra\u00dfe, \ufb01nal \ufb03 \u00bd",
    "\u0391\u0392\u0393\u0394 \u03b1\u03b2\u03b3\u03b4 \u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439 \u043c\u0438\u0440",
    "\u05e2\u05d1\u05e8\u05d9\u05ea \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0939\u093f\u0928\u094d\u0926\u0940 \ud55c\uad6d\uc5b4",
    "code:\n  def f(x: int) -> int:\n      return x**2  # comment\n",
    "<bos>hello<eos>",
    "a <|image|> b",
    "<|image>X<image|>",
    "<|IMAGE|> <b|> <|audio|>",
    "\u2581 literal \u2581\u2581",
    "a" * 300,
    " " * 40,
    "0.5e-3 + 1/3 = 3.14%",
    "https://example.com/a?b=c&d=e#f",
]

random.seed(1234)
latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_'\"\n\t#@$%&*()[]{}\\/|<>+=~`^;:"
wide = "àéüßñçøå日本語中文한국어हिन्दीالعربيةᚠᚢᚦ🎉🚀ﬁ½\u0301\u2581"
for _ in range(2000):
    n = random.randint(1, 80)
    s = "".join(random.choice(latin if random.random() < 0.85 else wide) for _ in range(n))
    cases.append(s)

# a few long single-run documents: the heap BPE reallocates its scratch and the merge heap grows
random.seed(99)
for _ in range(6):
    block = "".join(random.choice(latin) for _ in range(random.randint(300, 900)))
    cases.append(block * random.randint(2, 8))

with open(DST / "tokenizer.bin", "wb") as f:
    f.write(struct.pack("<I", len(cases)))
    for s in cases:
        raw = s.encode("utf-8")
        ids = tok(s)["input_ids"]
        f.write(struct.pack("<I", len(raw)))
        f.write(raw)
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack("<%di" % len(ids), *ids))
print(f"tokenizer.bin: {len(cases)} cases, {len(wide)}/-")

# --- multimodal layout: the processor expands every <|image|> for the image it belongs to ---
from PIL import Image, ImageDraw


def make_image(w, h, i):
    img = Image.new("RGB", (w, h), (20 + 40 * i, 60, 120 - 20 * i))
    d = ImageDraw.Draw(img)
    for k in range(12):
        x0 = (k * 97 + 13 * i) % w
        y0 = (k * 53 + 7 * i) % h
        d.ellipse([x0, y0, x0 + w // 4, y0 + h // 4], fill=(255 - 15 * k, 100 + 12 * k, 30 + 20 * i))
    d.rectangle([w // 8, h // 8, w // 2, h // 2], outline=(255, 255, 255), width=3)
    return img


img = make_image(512, 512, 0)
batch = proc(text=[CAPTION], images=[[img]], return_tensors="pt")
valid = (batch["image_position_ids"] != -1).all(dim=-1)
n_soft = int(valid.sum().item()) // 9
ids = batch["input_ids"][0].tolist()
with open(DST / "tokenizer_mm.bin", "wb") as f:
    f.write(struct.pack("<I", 1))
    raw = CAPTION.encode("utf-8")
    f.write(struct.pack("<I", len(raw)))
    f.write(raw)
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<i", n_soft))
    f.write(struct.pack("<I", len(ids)))
    f.write(struct.pack("<%di" % len(ids), *ids))
print(f"tokenizer_mm.bin: 1 case, {n_soft} soft tokens, {len(ids)} ids")
