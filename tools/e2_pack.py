#!/usr/bin/env python3
"""
e2_pack.py - Quantize EmbeddingGemma-2 to the E2 int8 model format.

Produces a single self-contained binary (weights + tokenizer + mel filters)
for one of the six modality combinations:

    text | text+image | text+audio | text+video | text+image+audio | text+image+audio+video

Video uses the vision tower, so "text+video" and "text+image+video" contain
the same towers; the modality flag only changes the runtime mask.

Usage:
    python tools/e2_pack.py --model ~/dev-box/embeddinggemma-2 \
        --modalities text,image,audio --out e2-int8.bin
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MAGIC = b"E2MODEL1"
VERSION = 1

TEXT_LAYERS = 24
TEXT_HIDDEN = 512
TEXT_HEADS = 4
TEXT_KV = 2
TEXT_HEAD_DIM = 256          # sliding layers
TEXT_HEAD_DIM_FULL = 512     # full layers
TEXT_KV_FULL = 1
TEXT_INTER = 2048
TEXT_PLE = 512
TEXT_EMBED_DIM = 768
TEXT_VOCAB = 262144
TEXT_SLIDING = 512
VISION_LAYERS = 16
VISION_HIDDEN = 768
VISION_HEADS = 12
VISION_HEAD_DIM = 64
VISION_INTER = 3072
VISION_PATCH = 16
VISION_POS_TABLE = 10240
VISION_POOL = 3
AUDIO_LAYERS = 12
AUDIO_HIDDEN = 1024
AUDIO_HEADS = 8
AUDIO_HEAD_DIM = 128
AUDIO_INTER = 4096
AUDIO_OUT = 1536
AUDIO_CONV_K = 5
AUDIO_CHUNK = 12
AUDIO_CTX_LEFT = 13
AUDIO_CTX_RIGHT = 0

FULL_LAYER_MASK = sum(1 << i for i in (5, 11, 17, 23))


def log(*a):
    print(*a, file=sys.stderr)


# ---------------------------------------------------------------------------
# Binary writer
# ---------------------------------------------------------------------------

class Writer:
    def __init__(self):
        self.meta = []       # (name, dtype, shape, flags, offset, nbytes)
        self.blobs = []
        self.offset = 0

    def add(self, name, dtype, shape, flags, data: bytes):
        pad = (-self.offset) % 64
        if pad:
            self.blobs.append(b"\0" * pad)
            self.offset += pad
        self.meta.append({
            "name": name, "dtype": dtype, "shape": list(shape),
            "flags": flags, "offset": self.offset, "nbytes": len(data),
        })
        self.blobs.append(data)
        self.offset += len(data)

    def write(self, path):
        idx = b""
        for m in self.meta:
            nb = m["name"].encode()
            assert len(nb) < 96
            shape = list(m["shape"]) + [0] * (4 - len(m["shape"]))
            idx += struct.pack("<96sII4Q4Q", nb.ljust(96, b"\0"), m["dtype"],
                               len(m["shape"]), *shape, m["offset"],
                               m["nbytes"], m["flags"], 0)
        header = MAGIC + struct.pack("<III", VERSION, len(self.meta), 0)
        data = header + idx + b"".join(self.blobs)
        with open(path, "wb") as f:
            f.write(data)
        log(f"wrote {path}: {len(data)/1e6:.1f} MB, {len(self.meta)} tensors")


# dtype ids
DT_F32, DT_I8, DT_I32 = 0, 1, 2
FLAG_PACKED = 1


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quant_row(W: torch.Tensor):
    """Symmetric per-output-row int8 quantization. W: [N,K] float."""
    W = W.float()
    N, K = W.shape
    amax = W.abs().amax(dim=1)
    scale = (amax / 127.0).clamp_min(1e-12)
    assert K % 128 == 0, (N, K)
    Q = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)
    wsum = Q.to(torch.int32).reshape(N, K // 128, 128).sum(dim=2, dtype=torch.int32)
    return Q, scale.numpy().astype(np.float32), wsum.numpy().astype(np.int32)


def pack_int8(Q: torch.Tensor) -> bytes:
    """Pack [N,K] int8 into [K/4][N/16][16][4] for VNNI/NEON dot kernels."""
    N, K = Q.shape
    nb16 = (N + 15) // 16
    nk4 = (K + 3) // 4
    P = torch.zeros((nb16 * 16, nk4 * 4), dtype=torch.int8)
    P[:N, :K] = Q
    # [nblock][kblock][16 rows][4 k] so one output tile is contiguous over K
    P = P.reshape(nb16, 16, nk4, 4).permute(0, 2, 1, 3).contiguous()
    return P.numpy().tobytes()


def add_int8(w: Writer, name: str, W: torch.Tensor, packed=True, unquantized_row_major=False):
    if W.ndim != 2:
        raise ValueError(f"{name}: expected 2D")
    N, K = W.shape
    Q, scale, wsum = quant_row(W)
    if unquantized_row_major:
        w.add(name, DT_I8, (N, K), 0, Q.numpy().tobytes())
    elif packed:
        w.add(name, DT_I8, (N, K), FLAG_PACKED, pack_int8(Q))
    else:
        w.add(name, DT_I8, (N, K), 0, Q.numpy().tobytes())
    w.add(name + ".scale", DT_F32, (N,), 0, scale.tobytes())
    if packed:
        w.add(name + ".wsum", DT_I32, (N, K // 128), 0, wsum.tobytes())
    return Q, scale


def add_f32(w: Writer, name: str, T):
    if torch.is_tensor(T):
        a = T.float().cpu().numpy().astype(np.float32)
    else:
        a = np.asarray(T, dtype=np.float32)
    w.add(name, DT_F32, a.shape, 0, a.tobytes())


def add_i32(w: Writer, name: str, arr):
    a = np.asarray(arr, dtype=np.int32)
    w.add(name, DT_I32, a.shape, 0, a.tobytes())


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def pack_tokenizer(tok_json_path: str, out_path: str):
    t = json.load(open(tok_json_path))
    model = t["model"]
    vocab = model["vocab"]            # str -> id
    merges = model["merges"]          # list of [a,b] or "a b"
    added = t.get("added_tokens", [])

    items = [None] * len(vocab)
    for s, i in vocab.items():
        items[i] = s.encode("utf-8")
    assert all(x is not None for x in items)

    # merges -> (left_id, right_id) sorted by rank; ids looked up by string
    mlist = []
    for rank, m in enumerate(merges):
        if isinstance(m, str):
            a, b = m.split(" ")
        else:
            a, b = m
        ia = vocab.get(a)
        ib = vocab.get(b)
        ia = vocab.get(a); ib = vocab.get(b)
        if ia is None or ib is None:
            continue
        im = vocab.get(a + b)
        if im is None:
            continue
        mlist.append((ia, ib, rank, im))

    specials = []
    for a in added:
        s = a["content"]
        if s in vocab:
            specials.append((s.encode("utf-8"), vocab[s]))

    with open(out_path, "wb") as f:
        f.write(b"E2TOK1\0\0")
        f.write(struct.pack("<II", len(items), len(mlist)))
        for s in items:
            f.write(struct.pack("<I", len(s)))
            f.write(s)
        for a, b, r, m in mlist:
            f.write(struct.pack("<IIII", a, b, r, m))
        f.write(struct.pack("<I", len(specials)))
        for s, i in specials:
            f.write(struct.pack("<II", len(s), i))
            f.write(s)
    log(f"wrote {out_path}: {len(items)} tokens, {len(mlist)} merges, {len(specials)} specials")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dev-box/embeddinggemma-2"))
    ap.add_argument("--modalities", default="text,image,audio",
                    help="comma list of text,image,audio,video")
    ap.add_argument("--out", default="e2-int8.bin")
    ap.add_argument("--tokenizer-out", default="e2-tokenizer.bin")
    args = ap.parse_args()

    mods = set(args.modalities.split(","))
    unknown = mods - {"text", "image", "audio", "video"}
    if unknown:
        raise SystemExit(f"unknown modalities: {unknown}")
    if "video" in mods:
        mods.add("image")            # video shares the vision tower
    have_vision = "image" in mods
    have_audio = "audio" in mods
    flags = (1 if "text" in mods else 0) | (2 if have_vision else 0) | \
            (4 if have_audio else 0) | (8 if "video" in mods else 0)

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    tc = cfg["text_config"]

    st_path = os.path.join(args.model, "model.safetensors")
    f = safe_open(st_path, "pt")
    keys = set(f.keys())

    w = Writer()

    # --- metadata ---
    ints = [
        1, flags, TEXT_LAYERS, TEXT_HIDDEN, TEXT_HEADS, TEXT_KV,
        TEXT_HEAD_DIM, TEXT_HEAD_DIM_FULL, TEXT_KV_FULL, TEXT_INTER,
        TEXT_PLE, TEXT_EMBED_DIM, TEXT_VOCAB, TEXT_SLIDING,
        VISION_LAYERS, VISION_HIDDEN, VISION_HEADS, VISION_HEAD_DIM,
        VISION_INTER, VISION_PATCH, VISION_POS_TABLE, VISION_POOL,
        AUDIO_LAYERS, AUDIO_HIDDEN, AUDIO_HEADS, AUDIO_HEAD_DIM,
        AUDIO_INTER, AUDIO_OUT, AUDIO_CONV_K, AUDIO_CHUNK,
        AUDIO_CTX_LEFT, AUDIO_CTX_RIGHT, FULL_LAYER_MASK,
        int(cfg.get("vision_soft_tokens_per_image", 280)), 128,
    ]
    add_i32(w, "__meta__", ints)

    params = [
        tc["rms_norm_eps"],
        float(math.sqrt(TEXT_HIDDEN)),
        float(1.0 / math.sqrt(TEXT_HIDDEN)),        # PLE projection scale
        cfg["vision_config"]["rms_norm_eps"],
        cfg["vision_config"]["rope_parameters"]["rope_theta"],
        float(math.sqrt(VISION_HIDDEN)),            # pool scaling
        cfg["audio_config"]["rms_norm_eps"],
        cfg["audio_config"]["attention_logit_cap"],
        cfg["audio_config"]["residual_weight"],
        cfg["audio_config"]["gradient_clipping"],
        (AUDIO_HEAD_DIM ** -0.5) / math.log(2.0),   # audio q_scale
        math.log(1.0 + math.e) / math.log(2.0),     # audio k_scale
        10000.0,                                    # text sliding rope theta
        1000000.0,                                  # text full rope theta
        1e-3,                                       # mel floor
    ]
    add_f32(w, "__params__", params)

    # mel filterbank + hann window, computed here to avoid C-side formula drift
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mel import mel_filters, hann_window
    add_f32(w, "__mel_filters__", mel_filters(257, 128, 0.0, 8000.0, 16000))
    add_f32(w, "__hann_window__", hann_window(320))

    # --- text tower ---
    text_prefixes = ["language_model."]
    if have_vision:
        text_prefixes.append("embed_vision.")
    if have_audio:
        text_prefixes.append("embed_audio.")

    def wanted(k):
        if k.startswith("language_model."):
            return True
        if have_vision and k.startswith("embed_vision."):
            return True
        if have_audio and k.startswith("embed_audio."):
            return True
        if have_vision and k.startswith("vision_tower."):
            return True
        if have_audio and k.startswith("audio_tower."):
            return True
        return False

    n_int8 = 0
    n_f32 = 0
    for k in sorted(keys):
        if not wanted(k):
            continue
        t = f.get_tensor(k)
        # Matrices -> int8. Lookup table stays int8 row-major.
        if t.ndim == 2 and k.endswith(".weight"):
            if "position_embedding_table" in k:
                add_int8(w, k, t, packed=False, unquantized_row_major=True)
            elif "embed_tokens" in k:
                add_int8(w, k, t, packed=False, unquantized_row_major=True)
            else:
                add_int8(w, k, t, packed=True)
            n_int8 += 1
        elif t.ndim == 2 and k.endswith("position_embedding_table"):
            add_f32(w, k, t)
            n_f32 += 1
        else:
            add_f32(w, k, t)
            n_f32 += 1

    log(f"quantized {n_int8} matrices, kept {n_f32} tensors f32")
    w.write(args.out)
    pack_tokenizer(os.path.join(args.model, "tokenizer.json"), args.tokenizer_out)


if __name__ == "__main__":
    main()
