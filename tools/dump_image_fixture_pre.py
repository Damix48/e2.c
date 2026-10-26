#!/usr/bin/env python
"""Dump the image-preprocessing fixture: PNG bytes + the exact tensors the HF Gemma4 image processor
produces for them.

    uv run python engine/tools/dump_image_fixture_pre.py

Fixture format (little endian):

    u32 n_images
    per image:
      i32 width, i32 height
      u32 png_len, u8 png[png_len]
      i32 target_width, i32 target_height                    (aspect-preserving resize result)
      u32 n_resized (= target_height*target_width*3), u8 resized[]   (the resized RGB image)
      i32 n_patches                                          (real patches, before padding)
      u32 full                                               (1 = pixel_values/pos follow)
      f32 pixel_values[2520*768]                             (0..1, patch-major)
      i32 position_ids[2520*2]                               (-1 marks padding)
"""
import io
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import functional as tvF

ROOT = Path(__file__).resolve().parent.parent.parent
DST = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DST.mkdir(parents=True, exist_ok=True)

from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(str(ROOT))
ip = proc.image_processor
PATCH, POOL, MAX_SOFT = ip.patch_size, ip.pooling_kernel_size, ip.max_soft_tokens
MAX_PATCHES = MAX_SOFT * POOL**2


def make_image(w, h, i):
    img = Image.new("RGB", (w, h), (20 + 40 * i, 60, 120 - 20 * i))
    d = ImageDraw.Draw(img)
    for k in range(12):
        x0 = (k * 97 + 13 * i) % w
        y0 = (k * 53 + 7 * i) % h
        d.ellipse([x0, y0, x0 + w // 4, y0 + h // 4], fill=(255 - 15 * k, 100 + 12 * k, 30 + 20 * i))
    d.rectangle([w // 8, h // 8, w // 2, h // 2], outline=(255, 255, 255), width=3)
    return img


def target_size(h, w):
    import math

    factor = math.sqrt(MAX_PATCHES * PATCH**2 / (h * w))
    side = POOL * PATCH
    return int(math.floor(factor * h / side)) * side, int(math.floor(factor * w / side)) * side


def patchify(resized_u8, th, tw):
    """(H,W,3) uint8 -> (ph*pw, 768) float32 in [0,1], the reference layout.

    The rescale is a multiply by the config's rescale_factor (not a divide by 255): the two differ
    in the last ulp and the fixture is bit-exact.
    """
    ph, pw = th // PATCH, tw // PATCH
    x = resized_u8.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(ph * pw, -1)
    return x.astype(np.float32) * np.float32(ip.rescale_factor)


images = [make_image(512, 512, 0), make_image(1024, 256, 1), make_image(256, 1024, 2), make_image(768, 480, 3)]
out = []
for i, img in enumerate(images):
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    th, tw = target_size(h, w)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
    resized = tvF.resize(t, size=[th, tw], interpolation=InterpolationMode.BICUBIC, antialias=True)
    resized_u8 = resized.permute(1, 2, 0).contiguous().numpy()
    assert resized_u8.dtype == np.uint8, resized_u8.dtype
    th, tw = resized_u8.shape[0], resized_u8.shape[1]

    # the processor's own layout, which is the thing we must reproduce
    batch = proc(images=[[img]], return_tensors="pt")
    pv = batch["pixel_values"][0].numpy().astype(np.float32)
    pos = batch["image_position_ids"][0].numpy().astype(np.int32)
    npv = int((pos != -1).all(axis=-1).sum())
    mine = patchify(resized_u8, th, tw)
    assert np.array_equal(mine, pv[: npv]), f"image {i}: patchify != processor pixel_values"
    assert (pos[npv:] == -1).all(), "padding should be -1"

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    out.append((w, h, buf.getvalue(), tw, th, resized_u8, npv, pv, pos))
    print(f"image {i}: {w}x{h} -> {tw}x{th}  patches={npv} soft={npv // POOL**2} png={len(buf.getvalue())}B")

with open(DST / "image_pre.bin", "wb") as f:
    f.write(struct.pack("<I", len(out)))
    for idx, (w, h, png, tw, th, resized_u8, npv, pv, pos) in enumerate(out):
        f.write(struct.pack("<ii", w, h))
        f.write(struct.pack("<I", len(png)))
        f.write(png)
        f.write(struct.pack("<ii", tw, th))
        f.write(struct.pack("<I", resized_u8.size))
        f.write(resized_u8.tobytes())
        f.write(struct.pack("<i", npv))
        full = 1 if idx == 0 else 0
        f.write(struct.pack("<I", full))
        if full:
            f.write(pv.astype("<f4").tobytes())
            f.write(pos.astype("<i4").tobytes())
print("image_pre.bin written")
