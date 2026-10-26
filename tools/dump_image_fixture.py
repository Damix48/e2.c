#!/usr/bin/env python
"""Dump the image fixture and the vision-tower reference activations for the engine.

    uv run python engine/tools/dump_image_fixture.py

Writes:
  engine/tests/fixtures/image.bin        padded pixel_values + position_ids + the reference 768d
                                         embedding, in the layout tests/test_image.c reads
  engine/tests/ref_act/v_*.f32           vision-tower intermediates, raw float32, for bisection
                                         (shapes are appended to ref_act/shapes.txt)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from onnx_common import image_only_ids, load_model  # noqa: E402

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ACT = Path(__file__).resolve().parent.parent / "tests" / "ref_act"
FIX.mkdir(parents=True, exist_ok=True)
ACT.mkdir(parents=True, exist_ok=True)

inputs = dict(np.load(ROOT / "bench" / "fixtures" / "inputs.npz"))
ref = dict(np.load(ROOT / "bench" / "fixtures" / "reference.npz"))
pv = torch.from_numpy(inputs["pixel_values"])
pid = torch.from_numpy(inputs["image_position_ids"])

m = load_model(("image",))
model = m[0].model
vt = model.vision_tower
emb_v = model.embed_vision
text = model.language_model
shapes = []


def dump(name, t):
    a = np.asarray(t.detach().float().cpu().numpy(), dtype=np.float32)
    a.tofile(ACT / f"v_{name}.f32")
    shapes.append(f"v_{name} {list(a.shape)}")
    print(f"  v_{name:18s} {list(a.shape)}")


with torch.no_grad():
    # --- reference path, mirroring EmbeddingGemma2Model.get_image_features + the image-only layout
    out = vt(pixel_values=pv, pixel_position_ids=pid)
    soft = emb_v(out.last_hidden_state)
    print("vision tokens", out.last_hidden_state.shape, "-> soft", soft.shape)

    ids = image_only_ids(soft.shape[0])
    embeds = model.get_input_embeddings()(ids).clone()
    image_mask = (ids == model.config.image_token_id).unsqueeze(-1)
    embeds = embeds.masked_scatter(image_mask, soft.reshape(-1, embeds.shape[-1]).to(embeds.dtype))
    lm = text(inputs_embeds=embeds, attention_mask=torch.ones_like(ids))
    # language_model.forward already ends in embedding_projection (512 -> 768), so pool and normalize
    final = torch.nn.functional.normalize(lm.last_hidden_state.mean(dim=1), p=2, dim=-1)
    got = np.asarray(final.float().cpu().numpy(), dtype=np.float32).reshape(-1)
    want = np.asarray(ref["image"], dtype=np.float32).reshape(-1)
    print("image-only cos vs reference.npz:", float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want))))

    # --- intermediates for bisection
    padding = (pid == -1).all(dim=-1)
    pe = vt.patch_embedder(pv, pid, padding)
    dump("patch_embed", pe)
    # the engine gathers the valid patches first and runs without a mask; the same intermediates
    # restricted to those rows are what it can be bisected against
    valid = ~padding[0]
    dump("patch_embed_s", pe[:, valid])
    # layer 0 internals, in the engine's (stripped) layout
    with torch.no_grad():
        L = vt.encoder.layers[0]
        sa = L.self_attn
        hd = sa.head_dim
        hh = sa.config.num_attention_heads
        xin = L.input_layernorm(pe)
        dump("l0_input_ln", xin[:, valid])
        qp, kp, vp = sa.q_proj(xin), sa.k_proj(xin), sa.v_proj(xin)
        dump("l0_q_proj", qp[:, valid])
        dump("l0_k_proj", kp[:, valid])
        dump("l0_v_proj", vp[:, valid])
        qn = sa.q_norm(qp.view(1, -1, hh, hd))
        kn = sa.k_norm(kp.view(1, -1, hh, hd))
        vn = sa.v_norm(vp.view(1, -1, hh, hd))
        dump("l0_q_norm", qn[:, valid])
        dump("l0_k_norm", kn[:, valid])
        dump("l0_v_norm", vn[:, valid])
        cos, sin = vt.encoder.rotary_emb(pe, pid)
        dump("l0_cos", cos[0][valid])
        from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
        qr = apply_multidimensional_rope(qn, cos, sin, pid)
        kr = apply_multidimensional_rope(kn, cos, sin, pid)
        dump("l0_q_rope", qr[:, valid])
        dump("l0_k_rope", kr[:, valid])
        qv = qr[0][valid][:, 0, :]  # head 0, [P, 64]
        kv = kr[0][valid][:, 0, :]
        vv = vn[0][valid][:, 0, :]
        sc = qv @ kv.transpose(0, 1)  # [P, P] logits for head 0
        dump("l0_scores", sc)
        pr = torch.softmax(sc.float(), dim=-1)
        dump("l0_probs", pr)
        ctx = pr @ vv
        dump("l0_ctx_h0", ctx)
        # full context needs all heads; recompute the whole attention the module way
        qa_ = qr.transpose(1, 2)
        ka_ = kr.transpose(1, 2)
        va_ = vn.transpose(1, 2)
        aw = torch.matmul(qa_, ka_.transpose(2, 3)) * 1.0
        aw = torch.softmax(aw.float(), dim=-1).to(qa_.dtype)
        co = torch.matmul(aw, va_).transpose(1, 2).reshape(1, -1, hh * hd)
        dump("l0_ctx", co[:, valid])
        op = sa.o_proj(co)
        dump("l0_o_proj", op[:, valid])
        patn = L.post_attention_layernorm(op)
        x1 = pe + patn
        dump("l0_x_attn", x1[:, valid])
        xff = L.pre_feedforward_layernorm(x1)
        g = L.mlp.gate_proj(xff)
        u = L.mlp.up_proj(xff)
        act = L.mlp.act_fn(g) * u
        dump("l0_mlp_gate", act[:, valid])
        mo = L.mlp.down_proj(act)
        dump("l0_mlp_out", mo[:, valid])

    h = pe
    for i in range(16):
        h = vt.encoder.layers[i](h, attention_mask=~padding, position_embeddings=vt.encoder.rotary_emb(h, pid),
                                 position_ids=pid)
        if i in (0, 1, 15):
            dump(f"layer{i}", h)
            dump(f"layer{i}_s", h[:, valid])
    pooled_v, mask = vt.pooler(h, pid, padding, output_length=pv.shape[-2] // 9)
    dump("pooler_out", pooled_v[mask])
    dump("soft_tokens", soft)

    # --- text + image (the caption case): <|image|> placeholders already expanded
    cap_ids = torch.from_numpy(np.asarray(inputs["image_caption_input_ids"], dtype=np.int64))
    cap_mask = torch.from_numpy(np.asarray(inputs["image_caption_attention_mask"], dtype=np.int64))
    cap_soft = model.embed_vision(inputs_embeds=out.last_hidden_state)
    cap_emb = model.get_input_embeddings()(
        torch.where(cap_ids == model.config.image_token_id, model.config.text_config.pad_token_id, cap_ids)
    ).clone()
    cap_mask3 = (cap_ids == model.config.image_token_id).unsqueeze(-1)
    cap_emb = cap_emb.masked_scatter(cap_mask3, cap_soft.reshape(-1, cap_emb.shape[-1]).to(cap_emb.dtype))
    cap_lm = text(inputs_embeds=cap_emb, attention_mask=cap_mask)
    cap_final = torch.nn.functional.normalize(cap_lm.last_hidden_state.mean(dim=1), p=2, dim=-1)
    cap_got = np.asarray(cap_final.float().cpu().numpy(), dtype=np.float32).reshape(-1)
    cap_want = np.asarray(ref["image_caption"], dtype=np.float32).reshape(-1)
    print("image_caption cos vs reference.npz:",
          float(cap_got @ cap_want / (np.linalg.norm(cap_got) * np.linalg.norm(cap_want))))
    import struct
    with open(FIX / "image_caption.bin", "wb") as f:
        n_ids = cap_ids.shape[1]
        f.write(struct.pack("<iiii", n_ids, pv.shape[-2], pv.shape[-1], cap_got.size))
        f.write(np.ascontiguousarray(cap_ids[0].numpy(), dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(pv[0].numpy(), dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(pid[0].numpy(), dtype=np.int32).tobytes())
        f.write(cap_got.tobytes())
    print(f"image_caption.bin: {n_ids} ids, {(FIX / 'image_caption.bin').stat().st_size / 1e6:.1f} MB")

    # --- the fixture the C test reads
    with open(FIX / "image.bin", "wb") as f:
        import struct
        n_patch, p_dim = pv.shape[-2], pv.shape[-1]
        f.write(struct.pack("<iiii", n_patch, p_dim, 2, got.size))
        f.write(np.ascontiguousarray(pv[0].numpy(), dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(pid[0].numpy(), dtype=np.int32).tobytes())
        f.write(got.tobytes())
    print(f"image.bin: {n_patch} patches x {p_dim}, {got.size} dims, "
          f"{(FIX / 'image.bin').stat().st_size / 1e6:.1f} MB")

with open(ACT / "shapes.txt", "w") as f:
    f.write("\n".join(shapes) + "\n")
