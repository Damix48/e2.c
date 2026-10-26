#!/usr/bin/env python
"""Dump the codec fixture: JPEG, WebP and PNG files in every EXIF orientation, plus the RGB the
reference path produces for them.

    uv run python engine/tools/dump_codec_fixture.py

The expectation for each case is `ImageOps.exif_transpose(Image.open(f)).convert("RGB")` - which is
exactly what transformers' `load_image` does before the processor resizes. So a case passing means
the engine agrees with the reference on both the decode and the orientation.

Fixture format (little endian):

    u32 n_cases
    per case: u32 format (1 png, 2 jpeg, 4 webp), u32 file_len, u8 file[],
              i32 exif_orientation, i32 width, i32 height, u32 rgb_len, u8 rgb[rgb_len]
    rgb_len == 0 marks a file that must be REJECTED (corrupt/truncated), never crash or hang.
"""
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent.parent
DST = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DST.mkdir(parents=True, exist_ok=True)

FMT = {"png": 1, "jpeg": 2, "webp": 4}
W, H = 34, 22

# a little structure so a wrong orientation is visible rather than a symmetric blur
rng = np.random.default_rng(4)
base = np.zeros((H, W, 3), np.uint8)
base[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
base[:, :, 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
base[:, :, 2] = (rng.random((H, W)) * 255).astype(np.uint8)
base[3 : H - 3, 5 : W - 5] = 200
img = Image.fromarray(base, "RGB")


def with_exif(image, orientation):
    e = Image.Exif()
    e[0x0112] = orientation
    return e.tobytes()


def encode(image, fmt, orientation, **kw):
    buf = io.BytesIO()
    image.save(buf, format=fmt.upper(), exif=with_exif(image, orientation), **kw)
    return buf.getvalue()


cases = []  # (fmt, bytes, orientation, expected_rgb_bytes) - expected None = must be rejected

for orientation in range(1, 9):
    for fmt in ("jpeg", "webp", "png"):
        kw = {"quality": 90} if fmt == "jpeg" else ({"quality": 90} if fmt == "webp" else {})
        data = encode(img, fmt, orientation, **kw)
        # the reference path: open -> transpose by EXIF -> RGB
        ref = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
        cases.append((FMT[fmt], data, orientation, ref.tobytes(), ref.shape))

# JPEG variants the decoder has to get right
variants = [
    ("baseline 4:2:0 q30", dict(quality=30, subsampling=0)),
    ("baseline 4:4:4 q95", dict(quality=95, subsampling=2)),
    ("progressive", dict(quality=90, progressive=True)),
    ("optimized", dict(quality=90, optimize=True)),
]
for name, kw in variants:
    data = encode(img, "jpeg", 1, **kw)
    ref = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    cases.append((FMT["jpeg"], data, 1, ref.tobytes(), ref.shape))
    print(f"jpeg {name}: {len(data)} B")

gray = img.convert("L")
data = encode(gray, "jpeg", 1, quality=90)
ref = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
cases.append((FMT["jpeg"], data, 1, ref.tobytes(), ref.shape))
print(f"jpeg grayscale: {len(data)} B")

for name, kw in [("webp lossless", dict(lossless=True)), ("webp q30", dict(quality=30))]:
    data = encode(img, "webp", 1, **kw)
    ref = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    cases.append((FMT["webp"], data, 1, ref.tobytes(), ref.shape))
    print(f"{name}: {len(data)} B")

# things that must be refused cleanly, never crash
good_jpeg = encode(img, "jpeg", 1, quality=90)
good_webp = encode(img, "webp", 1, quality=90)
good_png = encode(img, "png", 1)
bad = [
    (FMT["jpeg"], good_jpeg[: len(good_jpeg) // 2]),
    (FMT["jpeg"], good_jpeg[:2] + b"\xff\xd8" + good_jpeg[4:40]),
    (FMT["jpeg"], b"\xff\xd8" + b"\x00" * 40),
    (FMT["webp"], good_webp[: len(good_webp) // 2]),
    (FMT["webp"], b"RIFF\x10\x00\x00\x00WEBP" + b"XXXX" + b"\x00" * 8),
    (FMT["png"], good_png[: len(good_png) // 2]),
]
for fmt, data in bad:
    cases.append((fmt, data, 1, b"", None))
print(f"{len(bad)} corrupt/truncated cases")

with open(DST / "codec.bin", "wb") as f:
    f.write(struct.pack("<I", len(cases)))
    for fmt, data, orientation, rgb, shape in cases:
        f.write(struct.pack("<II", fmt, len(data)))
        f.write(data)
        f.write(struct.pack("<i", orientation))
        if rgb:
            h, w = shape[0], shape[1]
            f.write(struct.pack("<iii", w, h, len(rgb)))
            f.write(rgb)
        else:
            f.write(struct.pack("<iii", 0, 0, 0))
print(f"codec.bin: {len(cases)} cases, {sum(len(c[1]) for c in cases)} bytes of images")
