#!/usr/bin/env python3
"""e2_acc.py - cosine similarity of e2 against the float32 HuggingFace run.

    . .venv/bin/activate
    python tools/e2_acc.py text  [--text "..."] [--levels]
    python tools/e2_acc.py image [--image pic.png]
    python tools/e2_acc.py audio [--audio clip.wav]

Generates the reference with tools/e2_ref.py into a temp dir, runs e2cli with
E2_DUMP into another, and prints the cosine of every pair of tensors that
exists on both sides.  Exits non-zero if a final embedding drops below 0.999.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def load(p):
    return np.fromfile(p, dtype="<f4")


def cos(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["text", "image", "audio"])
    ap.add_argument("--model", default=os.path.expanduser("~/dev-box/embeddinggemma-2"))
    ap.add_argument("--e2-model", default="models/e2-text+image+audio+video.bin")
    ap.add_argument("--e2-tok", default="models/e2-tokenizer.bin")
    ap.add_argument("--e2-bin", default="./e2cli")
    ap.add_argument("--text", default="The northern lights are caused by charged particles "
                                       "from the sun colliding with gases in the upper atmosphere.")
    ap.add_argument("--image", default="/tmp/e2bench/image_640x480.png")
    ap.add_argument("--audio", default="/tmp/e2bench/tone_2s.wav")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep", default=None, help="keep the dump dirs here")
    ap.add_argument("--min", type=float, default=0.999, dest="floor")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="e2acc_")
    ref = os.path.join(tmp, "ref")
    cd = os.path.join(tmp, "c")
    os.makedirs(ref, exist_ok=True)
    os.makedirs(cd, exist_ok=True)

    refcmd = [sys.executable, "tools/e2_ref.py", args.what, "--model", args.model,
              "--out", ref]
    if args.what == "text":
        refcmd += ["--text", args.text]
        cli = [args.e2_bin, "text", args.e2_model, args.e2_tok, args.text]
    elif args.what == "image":
        refcmd += ["--image", args.image]
        cli = [args.e2_bin, "image", args.e2_model, args.e2_tok, args.image]
    else:
        refcmd += ["--audio", args.audio]
        cli = [args.e2_bin, "audio", args.e2_model, args.e2_tok, args.audio]
    if args.threads:
        cli.append(str(args.threads))

    print("reference...", flush=True)
    run(refcmd)
    print("e2...", flush=True)
    run(cli, {"E2_DUMP": cd})

    pairs = {
        "text": [("embeddings", "embeddings"), ("ple", "ple"),
                 ("last_hidden_state", "last_hidden_state"),
                 ("embedding", "embedding")],
        "image": [("vision_last", "vis_out"), ("image_soft", "image_soft"),
                  ("vis_pool", "vis_pool")],
        "audio": [("audio_last", "aud_out"), ("audio_soft", "aud_soft")],
    }.get(args.what, [])
    # per-layer text dumps, if any
    man = json.load(open(os.path.join(ref, "manifest.json")))
    if args.what == "text":
        for i in range(1, int(man.get("layers", 0))):
            pairs.append((f"layer{i}", f"layer{i}"))

    floor = 1.0
    for r, c in pairs:
        rp, cp = os.path.join(ref, r + ".bin"), os.path.join(cd, c + ".bin")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        a, b = load(rp), load(cp)
        v = cos(a, b)
        print(f"  {r:<22} {v:.5f}" + ("   <-- LOW" if v < args.floor else ""))
        if r in ("embedding", "image_soft", "audio_soft", "vis_pool"):
            floor = min(floor, v)
    if args.keep:
        print(f"dumps in {tmp}")
    else:
        subprocess.run(["rm", "-rf", tmp])
    if floor < args.floor:
        print(f"FAIL: worst final cosine {floor:.5f} < {args.floor}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
