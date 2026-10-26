#!/usr/bin/env python
"""Dump the audio fixture and the audio-tower reference activations for the engine.

    E2_MODEL_DIR=~/Projects/embeddinggemma-2 .venv/bin/python tools/dump_audio_fixture.py

Writes:
  tests/fixtures/audio.bin     pcm + token layout + reference soft tokens + the reference 768d
                               embeddings of text+audio and text+image+audio
  tests/ref_act/a_*.f32        audio-tower intermediates, raw float32, for bisection
                               (shapes are appended to ref_act/shapes_audio.txt)

The signal is synthetic (three swept tones under an amplitude envelope, 0.75 s at 16 kHz) so the
fixture is deterministic; 0.75 s is deliberately not a whole number of 4-frame groups, which is what
exercises the tail mask and the ceil() in the soft-token count (75 mel frames -> 19 soft tokens).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

# the checkpoint and its scripts/ + bench/ live in the model directory, which is not necessarily next
# to this engine copy
ROOT = Path(os.environ.get("E2_MODEL_DIR", Path.home() / "Projects" / "embeddinggemma-2"))
sys.path.insert(0, str(ROOT / "scripts"))

from onnx_common import (  # noqa: E402
    BOS_TOKEN_ID,
    BOI_TOKEN_ID,
    EOI_TOKEN_ID,
    EOS_TOKEN_ID,
    IMAGE_TOKEN_ID,
    load_model,
    mean_pool_normalize,
    num_soft_tokens_from_positions,
)

HERE = Path(__file__).resolve().parent.parent
FIX = HERE / "tests" / "fixtures"
ACT = HERE / "tests" / "ref_act"
FIX.mkdir(parents=True, exist_ok=True)
ACT.mkdir(parents=True, exist_ok=True)

SR = 16000
TEXT = "a person humming a quiet melody"
AOA, ATK, EOA = 256000, 258881, 258883  # <|audio>, <|audio|>, <audio|>

shapes = []


def dump(name, t):
    a = np.asarray(t.detach().float().cpu().numpy(), dtype=np.float32)
    a.tofile(ACT / f"a_{name}.f32")
    shapes.append(f"a_{name} {list(a.shape)}")
    print(f"  a_{name:18s} {list(a.shape)}")


def pcm_fixture():
    t = np.arange(int(SR * 0.75)) / SR
    env = np.exp(-1.7 * t) * (1.0 - np.exp(-90.0 * t))
    x = 0.30 * np.sin(2 * np.pi * (170 + 60 * t) * t)
    x += 0.18 * np.sin(2 * np.pi * (430 - 90 * t) * t + 0.7)
    x += 0.09 * np.sin(2 * np.pi * 1750 * t + 1.9)
    return (env * x).astype(np.float32)


m = load_model(("text", "image", "audio"))
model = m[0].model
at = model.audio_tower
emb_a = model.embed_audio
text_lm = model.language_model
tokenizer = m[0].tokenizer
fe = m[0].processor.feature_extractor

pcm = pcm_fixture()
feat = fe([pcm], sampling_rate=SR, return_tensors="pt")
features, fmask = feat["input_features"], feat["input_features_mask"]
print("mel", tuple(features.shape), "valid frames", int(fmask.sum()))

dump("filters", torch.from_numpy(fe.mel_filters))
dump("window", torch.from_numpy(fe.window))
dump("mel", features[0])

with torch.no_grad():
    # ---- stage by stage, exactly the body of Gemma4AudioModel.forward, so the C side can bisect
    sub = at.subsample_conv_projection
    h, msk = sub(features, fmask)
    dump("conv", h[0])
    pos = at.rel_pos_enc(h)
    dump("pos", pos[0])
    dump("relk0", at.layers[0].self_attn.relative_k_proj(pos)[0])

    from transformers.models.gemma4.modeling_gemma4 import (
        create_bidirectional_mask,
        sliding_window_mask_function,
    )

    attn_mask = create_bidirectional_mask(
        config=at.config,
        inputs_embeds=h,
        attention_mask=msk,
        and_mask_function=sliding_window_mask_function(
            (at.config.attention_context_left - 1, at.config.attention_context_right)
        ),
    )
    blocked = at._convert_4d_mask_to_blocked_5d(attn_mask) if attn_mask is not None else None

    gc = min(at.config.gradient_clipping, torch.finfo(torch.float32).max)
    if blocked is not None:
        dump("blocked_mask", blocked[0, 0].to(torch.float32))

    L0 = at.layers[0]
    h1 = L0.feed_forward1(h)
    dump("l0_ff1", h1[0])
    h2 = L0.norm_pre_attn(torch.clamp(h1, -gc, gc))
    dump("l0_attn_in", h2[0])
    a0, scores0 = L0.self_attn(hidden_states=h2, position_embeddings=pos, attention_mask=blocked)
    dump("l0_attn_out", a0[0])
    if scores0 is not None:
        dump("l0_scores", scores0[0, 0])
    h3 = L0.norm_post_attn(torch.clamp(a0, -gc, gc)) + h1
    dump("l0_lconv_in", h3[0])
    h4 = L0.lconv1d(h3)
    dump("l0_lconv_out", h4[0])
    h5 = L0.feed_forward2(h4)
    dump("l0_ff2", h5[0])
    h6 = L0.norm_out(torch.clamp(h5, -gc, gc))
    print("layer0 replay max|d| vs module:", float((h6 - L0(h, attention_mask=blocked,
                                                              position_embeddings=pos)).abs().max()))

    # ---- the whole tower, one layer at a time (output_hidden_states skips the last layer's output)
    hh = h
    for i, layer in enumerate(at.layers):
        hh = layer(hh, attention_mask=blocked, position_embeddings=pos)
        dump(f"l{i}", hh[0])
    out = at.output_proj(hh)
    dump("out_proj", out[0])
    soft = emb_a(inputs_embeds=out)
    dump("soft", soft[0])
    n_soft = soft.shape[1]
    print("soft tokens", tuple(soft.shape))

    # ---- the token layout the engine builds: <bos> text <|audio> N*<|audio|> <audio|> <eos>
    body = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([[BOS_TOKEN_ID] + body + [AOA] + [ATK] * n_soft + [EOA] + [EOS_TOKEN_ID]])

    def embed(ids_, pieces):
        e = model.get_input_embeddings()(ids_).clone()
        for tok, piece in pieces:
            e = e.masked_scatter((ids_ == tok).unsqueeze(-1), piece.reshape(-1, e.shape[-1]))
        lm = text_lm(inputs_embeds=e, attention_mask=torch.ones_like(ids_))
        return mean_pool_normalize(lm.last_hidden_state, torch.ones_like(ids_))[0]

    ref_ta = embed(ids, [(ATK, soft)])

    pv = torch.from_numpy(np.load(ROOT / "bench" / "fixtures" / "inputs.npz")["pixel_values"])
    pid = torch.from_numpy(np.load(ROOT / "bench" / "fixtures" / "inputs.npz")["image_position_ids"])
    n_img = int(num_soft_tokens_from_positions(pid)[0])
    vsoft = model.embed_vision(inputs_embeds=model.vision_tower(pixel_values=pv,
                                                               pixel_position_ids=pid).last_hidden_state)
    vsoft = vsoft[:n_img]  # already stripped of the padded patches by the vision tower
    ids2 = torch.tensor([[BOS_TOKEN_ID] + body + [BOI_TOKEN_ID] + [IMAGE_TOKEN_ID] * n_img +
                         [EOI_TOKEN_ID] + [AOA] + [ATK] * n_soft + [EOA] + [EOS_TOKEN_ID]])
    ref_tia = embed(ids2, [(IMAGE_TOKEN_ID, vsoft), (ATK, soft)])
    print("text+image+audio:", tuple(ref_tia.shape), "image soft tokens", n_img)

ids_arr = ids[0].numpy().astype(np.int32)
text_b = TEXT.encode()
head = np.array([pcm.size, features.shape[1], n_soft, 768, ids_arr.size, 512, n_img, len(text_b)],
                dtype=np.int32)
with open(FIX / "audio.bin", "wb") as f:
    head.tofile(f)
    f.write(text_b)
    pcm.astype(np.float32).tofile(f)
    ids_arr.tofile(f)
    np.asarray(features[0].cpu().numpy(), dtype=np.float32).tofile(f)   # [F, 128] reference mel
    np.asarray(fmask[0].cpu().numpy(), dtype=np.uint8).tofile(f)        # [F] reference mask
    np.asarray(soft[0].cpu().numpy(), dtype=np.float32).tofile(f)       # [N, 512] soft tokens
    np.asarray(ref_ta.cpu().numpy(), dtype=np.float32).tofile(f)        # 768 text+audio
    np.asarray(ref_tia.cpu().numpy(), dtype=np.float32).tofile(f)       # 768 text+image+audio
print("wrote", FIX / "audio.bin", (FIX / "audio.bin").stat().st_size, "bytes")
(ACT / "shapes_audio.txt").write_text("\n".join(shapes) + "\n")
print("wrote", ACT / "shapes_audio.txt")
