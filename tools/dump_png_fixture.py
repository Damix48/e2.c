#!/usr/bin/env python
"""Dump the PNG-decoder fixture: hand-built PNGs that force each scanline filter on each row,
for every colour type and bit depth the reader supports.

    uv run python engine/tools/dump_png_fixture.py

Why hand-built: the FILTER BYTES live inside the compressed stream, so neither a mutation fuzzer
(which can only corrupt compressed bytes) nor an ordinary library-written file reaches all of them.
Every image the processor writes happened to use Sub or None on its first row, which hid a NULL
previous-row dereference in the Up/Average/Paeth paths. This covers all five filters on row 0 and on
every later row, for 8- and 16-bit, gray / gray+alpha / RGB / RGBA / palette.

Fixture format (little endian):

    u32 n_cases
    per case: i32 width, i32 height, u32 png_len, u8 png[], u32 rgb_len, u8 rgb[rgb_len]
"""
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
DST = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DST.mkdir(parents=True, exist_ok=True)

CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
W, H = 23, 13


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def filter_row(ft, row, prev, bpp):
    out = bytearray(len(row))
    for x, v in enumerate(row):
        a = row[x - bpp] if x >= bpp else 0
        b = prev[x]
        c = prev[x - bpp] if x >= bpp else 0
        if ft == 0:
            f = 0
        elif ft == 1:
            f = a
        elif ft == 2:
            f = b
        elif ft == 3:
            f = (a + b) >> 1
        else:
            f = paeth(a, b, c)
        out[x] = (v - f) & 0xFF
    return bytes(out)


def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def build(w, h, ctype, depth, rows, filters, plte=None):
    """rows: list of bytearray (unfiltered), filters: list of filter types, one per row."""
    bpp = CHANNELS[ctype] * (depth // 8)
    raw = bytearray()
    prev = bytes(len(rows[0]))
    for y, row in enumerate(rows):
        raw += bytes([filters[y]]) + filter_row(filters[y], row, prev, bpp)
        prev = bytes(row)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
    if plte is not None:
        out += chunk(b"PLTE", plte)
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    return out


rng = np.random.default_rng(11)
cases = []

for ctype in (0, 2, 3, 4, 6):
    for depth in ((8,) if ctype == 3 else (8, 16)):  # palette PNGs are 8-bit only
        ch, step = CHANNELS[ctype], depth // 8
        pal = None
        if ctype == 3:
            pal = bytes(rng.integers(0, 256, size=3 * 16, dtype=np.uint8).tolist())
        for first in range(5):
            rows = []
            for y in range(H):
                n = ch * step
                if ctype == 3:
                    row = bytearray(rng.integers(0, 16, size=W, dtype=np.uint8).tobytes())
                else:
                    row = bytearray(rng.integers(0, 256, size=W * ch * step, dtype=np.uint8).tobytes())
                rows.append(row)
            # row 0 gets the filter under test, the rest cycle through all five
            filters = [first] + [(1 + first + y) % 5 for y in range(H - 1)]
            png = build(W, H, ctype, depth, rows, filters, pal)

            im = Image.open(__import__("io").BytesIO(png))
            if ctype == 0 and depth == 16:
                # Pillow clips 16-bit gray to 255 on convert("RGB"); the reader takes the high byte
                a = np.array(im)
                ref = np.repeat((a >> 8).astype(np.uint8)[:, :, None], 3, axis=2)
            else:
                ref = np.asarray(im.convert("RGB"))
            assert ref.shape == (H, W, 3), ref.shape
            cases.append((W, H, png, ref.tobytes()))
            print(f"ctype {ctype} depth {depth:2d} row0 filter {first}: {len(png):5d} B png")

with open(DST / "png.bin", "wb") as f:
    f.write(struct.pack("<I", len(cases)))
    for w, h, png, rgb in cases:
        f.write(struct.pack("<ii", w, h))
        f.write(struct.pack("<I", len(png)))
        f.write(png)
        f.write(struct.pack("<I", len(rgb)))
        f.write(rgb)
print(f"png.bin: {len(cases)} cases")
