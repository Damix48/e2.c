#!/usr/bin/env python
"""Dump reference activations from the PyTorch model for one fixture, to bisect engine bugs.

    uv run python engine/tools/dump_ref_activations.py [case]

Writes engine/tests/ref_act/<name>.f32 (raw float32, shapes are printed and recorded in shapes.txt)
for: x0 (scaled token embeddings), ple, and for layer 0: input_ln, attn_out, x_after_attn,
pre_ffn_ln, mlp_out, x_after_mlp, ple_block_out, x_after_layer, plus final_norm, emb_proj.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "tests" / "ref_act"
OUT.mkdir(parents=True, exist_ok=True)

case = sys.argv[1] if len(sys.argv) > 1 else "text_query"
inputs = dict(np.load(ROOT / "bench" / "fixtures" / "inputs.npz"))
ids = torch.from_numpy(np.asarray(inputs[f"{case}_input_ids"], dtype=np.int64))
mask = torch.from_numpy(np.asarray(inputs[f"{case}_attention_mask"], dtype=np.int64))

from sentence_transformers import SentenceTransformer  # noqa: E402

m = SentenceTransformer(str(ROOT), config_kwargs={"vision_config": None, "audio_config": None},
                        model_kwargs={"dtype": torch.float32})
tm = m[0].model.language_model
tm.eval()
shapes = []


def dump(name, t):
    a = np.asarray(t.detach().float().cpu().numpy(), dtype=np.float32)
    a.tofile(OUT / f"{name}.f32")
    shapes.append(f"{name} {list(a.shape)}")
    print(f"  {name:16s} {list(a.shape)}")


with torch.no_grad():
    x0 = tm.embed_tokens(ids)
    dump("x0", x0)
    ple = tm.ple(x0)
    dump("ple", ple)

    cfg = tm.config
    mask_kwargs = {"config": cfg, "inputs_embeds": x0, "attention_mask": mask}
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2RotaryEmbedding,
        EmbeddingGemma2TextPLEBlock,
    )

    amap = {"full_attention": create_bidirectional_mask(**mask_kwargs)}
    from transformers.masking_utils import create_bidirectional_sliding_window_mask
    amap["sliding_attention"] = create_bidirectional_sliding_window_mask(**mask_kwargs)
    pos = torch.arange(x0.shape[1]).unsqueeze(0)
    pe = {t: tm.rotary_emb(x0, pos, t) for t in ("sliding_attention", "full_attention")}

    L = tm.layers[0]
    h = L.input_layernorm(x0)
    dump("l0_input_ln", h)
    sa = L.self_attn
    H, D = sa.config.num_attention_heads, sa.head_dim
    qp = sa.q_proj(h)
    kp = sa.k_proj(h)
    vp = sa.v_proj(h)
    dump("l0_q_proj", qp)
    dump("l0_k_proj", kp)
    qn = sa.q_norm(qp.view(1, -1, H, D))
    kn = sa.k_norm(kp.view(1, -1, sa.config.per_layer_config[cfg.layer_types[0]]["num_key_value_heads"], D) if False else kp.view(1, -1, kp.shape[-1] // D, D))
    vn = sa.v_norm(vp.view(1, -1, vp.shape[-1] // D, D))
    dump("l0_q_norm", qn)
    dump("l0_k_norm", kn)
    dump("l0_v_norm", vn)
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import apply_rotary_pos_emb
    cos, sin = pe[cfg.layer_types[0]]
    # mirror the module: (B,S,H,D) -> (B,H,S,D) -> rope -> scores -> softmax -> context
    qh = qn.transpose(1, 2)
    kh = kn.transpose(1, 2)
    vh = vn.transpose(1, 2)
    qr, kr = apply_rotary_pos_emb(qh, kh, cos, sin)
    dump("l0_q_rope", qr.transpose(1, 2))
    dump("l0_k_rope", kr.transpose(1, 2))
    kv_groups = H // kr.shape[1]
    kg = kr.repeat_interleave(kv_groups, dim=1)
    vg = vh.repeat_interleave(kv_groups, dim=1)
    scores = torch.matmul(qr, kg.transpose(2, 3)) * sa.scaling
    probs = torch.softmax(scores.float(), dim=-1)
    dump("l0_probs_h0", probs[0, 0])
    ctx = torch.matmul(probs, vg)
    dump("l0_ctx_pre_oproj", ctx.transpose(1, 2).reshape(1, -1, H * D))
    attn, _ = L.self_attn(hidden_states=h, attention_mask=amap[cfg.layer_types[0]],
                          position_embeddings=pe[cfg.layer_types[0]])
    dump("l0_attn_out", attn)
    h = L.post_attention_layernorm(attn)
    x = x0 + h
    dump("l0_x_after_attn", x)
    h = L.pre_feedforward_layernorm(x)
    dump("l0_pre_ffn_ln", h)
    mlp = L.mlp(h)
    dump("l0_mlp_out", mlp)
    h = L.post_feedforward_layernorm(mlp)
    x = x + h
    dump("l0_x_after_mlp", x)
    x = L.ple_block(x, ple[:, :, 0, :])
    dump("l0_ple_block_out", x)
    x = x * L.layer_scalar
    dump("l0_x_after_layer", x)

    # continue through the remaining layers for the true final state
    for l in range(1, len(tm.layers)):
        Ll = tm.layers[l]
        t = cfg.layer_types[l]
        r = x
        hh = Ll.input_layernorm(x)
        a, _ = Ll.self_attn(hidden_states=hh, attention_mask=amap[t], position_embeddings=pe[t])
        hh = Ll.post_attention_layernorm(a)
        x = r + hh
        r = x
        hh = Ll.pre_feedforward_layernorm(x)
        hh = Ll.mlp(hh)
        hh = Ll.post_feedforward_layernorm(hh)
        x = r + hh
        x = Ll.ple_block(x, ple[:, :, l, :])
        x = x * Ll.layer_scalar
    hn = tm.norm(x)
    dump("final_norm", hn)
    proj = hn @ tm.embedding_projection.weight.T
    dump("emb_proj", proj)

(OUT / f"shapes_{case}.txt").write_text("\n".join(shapes) + "\n")
print(f"wrote {OUT}/shapes_{case}.txt")
