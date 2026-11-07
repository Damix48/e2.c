#!/usr/bin/env python3
"""
e2_ref.py - Generate reference tensors from the HuggingFace implementation.

    python tools/e2_ref.py text --text "hello world" --out /tmp/ref
    python tools/e2_ref.py image --image pic.jpg --out /tmp/ref
    python tools/e2_ref.py audio --audio clip.wav --out /tmp/ref
    python tools/e2_ref.py embed --text "hello" --image pic.jpg --audio clip.wav --out /tmp/ref

Outputs raw little-endian f32 / i32 .bin files plus a JSON manifest.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def save(out, name, t, dtype="f32"):
    if torch.is_tensor(t):
        t = t.detach().cpu()
        if dtype == "i32":
            a = t.to(torch.int32).numpy()
        else:
            a = t.float().numpy()
    else:
        a = np.asarray(t, dtype=np.int32 if dtype == "i32" else np.float32)
    a.astype(a.dtype.newbyteorder("<") if hasattr(a.dtype, "newbyteorder") else a.dtype)
    path = os.path.join(out, name + ".bin")
    a.tofile(path)
    return {"file": name + ".bin", "shape": list(a.shape), "dtype": dtype}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["text", "image", "audio", "embed"])
    ap.add_argument("--model", default=os.path.expanduser("~/dev-box/embeddinggemma-2"))
    ap.add_argument("--text", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", default="/tmp/e2ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32,
                                      local_files_only=True, attn_implementation="eager")
    model.eval()
    manifest = {"cmd": args.cmd}

    if args.cmd == "text":
        enc = tok(args.text, return_tensors="pt")
        ids = enc["input_ids"]
        manifest["input_ids"] = save(args.out, "input_ids", ids, "i32")
        manifest["text"] = args.text
        with torch.no_grad():
            lm = model.language_model
            emb = lm.embed_tokens(ids)
            manifest["embeddings"] = save(args.out, "embeddings", emb[0])
            ple = lm.ple(emb)
            manifest["ple"] = save(args.out, "ple", ple[0])
            out = lm(input_ids=ids, attention_mask=enc["attention_mask"],
                     output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            for i, h in enumerate(hs):
                save(args.out, f"layer{i}", h[0])
            manifest["layers"] = len(hs)
            manifest["last_hidden_state"] = save(args.out, "last_hidden_state", out.last_hidden_state[0])
            pooled = out.last_hidden_state.mean(dim=1)
            embn = torch.nn.functional.normalize(pooled, dim=-1)
            manifest["embedding"] = save(args.out, "embedding", embn[0])

    if args.cmd in ("image", "embed"):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        enc = proc(images=img, return_tensors="pt")
        pv, pos = enc["pixel_values"], enc["image_position_ids"]
        manifest["pixel_values"] = save(args.out, "pixel_values", pv[0])
        manifest["image_position_ids"] = save(args.out, "image_position_ids", pos[0], "i32")
        with torch.no_grad():
            vout = model.vision_tower(pixel_values=pv, pixel_position_ids=pos)
            manifest["vision_last"] = save(args.out, "vision_last", vout.last_hidden_state)
            soft = model.embed_vision(vout.last_hidden_state)
            manifest["image_soft"] = save(args.out, "image_soft", soft)
            manifest["num_image_tokens"] = int(soft.shape[0])

    if args.cmd in ("audio", "embed"):
        import soundfile as sf
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        wav, sr = sf.read(args.audio, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        enc = proc(audio=wav, sampling_rate=16000, return_tensors="pt")
        feats, mask = enc["input_features"], enc["input_features_mask"]
        manifest["input_features"] = save(args.out, "input_features", feats[0])
        manifest["input_features_mask"] = save(args.out, "input_features_mask", mask[0], "i32")
        with torch.no_grad():
            aout = model.audio_tower(feats, mask)
            manifest["audio_last"] = save(args.out, "audio_last", aout.last_hidden_state[0])
            soft = model.embed_audio(aout.last_hidden_state)
            manifest["audio_soft"] = save(args.out, "audio_soft", soft[0])
            manifest["num_audio_tokens"] = int(soft.shape[1])

    if args.cmd == "embed":
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        content = []
        if args.text:
            content.append({"type": "text", "text": args.text})
        if args.image:
            content.append({"type": "image", "image": Image.open(args.image).convert("RGB")})
        if args.audio:
            import soundfile as sf
            wav, sr = sf.read(args.audio, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            content.append({"type": "audio", "audio": wav, "sampling_rate": sr})
        enc = proc.apply_chat_template([{"role": "user", "content": content}],
                                       tokenize=True, return_dict=True, return_tensors="pt")
        manifest["input_ids"] = save(args.out, "input_ids", enc["input_ids"], "i32")
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            h = out.last_hidden_state
            manifest["last_hidden_state"] = save(args.out, "last_hidden_state", h[0])
            emb = torch.nn.functional.normalize(h.mean(dim=1), dim=-1)
            manifest["embedding"] = save(args.out, "embedding", emb[0])

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
