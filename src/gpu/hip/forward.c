/* forward.c — the AMD/HIP forward pass: one upload, one recorded graph per shape, one launcher.
 *
 * The graph is a transcription of the CPU tower (src/text.c) with the same int8 recipe, in the same
 * order as the Intel backend's record() (src/gpu/gpu.c), so all three engines agree on an input.
 * Differences from the Intel recording are deliberate and small:
 *
 *  - Attention runs on the dot4 datapath (see e2k_attn_scores8 in kernels.hip): Q/K/V quantized
 *    once after norm+RoPE, scores and context through v_dot4, P quantized per 128-key block by the
 *    softmax. The fp32 scalar and fused-blocked kernels stay behind E2_HIP_ATTN=old|lds as the
 *    A/B references. Short shapes (seq_len < E2_HIP_ATTN_I8_MIN) stay on the fused fp32 kernel:
 *    the pack + softmax launches do not pay for themselves there.
 *  - The `valid` mask is only used where it is meaningful (the embedding gather, the attention
 *    keys, the mean pool). The norm/quantize rows always run: a padded row's norm is harmless and
 *    running it keeps the kernel's row index equal to the tensor's row index.
 *  - The weights are the CPU engine's quad-interleaved int8 matrix, copied as-is. Only the
 *    per-tensor descriptors and the norm/rope tables are added.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "hip.h"

static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }

/* ---- weight bookkeeping: w (quad-interleaved int8) + per-row scale. The per-block weight sums of
 * the CPU/Intel recipe are not needed because this path stores activations signed (see kernels.hip). */
static size_t q8w_bytes(const e2_q8w *q) {
    return align64((size_t)q->Npad * q->K + 4) + (size_t)q->Npad * 4;
}

static int upload_tensor(e2_hip *g, uint8_t *wbuf, size_t *woff, int32_t *desc, int di,
                         const e2_q8w *q, int y_region, int qa_region, int asc_region) {
    if (q->N <= 0 || q->K <= 0 || q->w == NULL) {
        e2h_set_error(g, "tensor %d has no int8 weights", di);
        return -1;
    }
    if (q->K % 128 || q->N % 8) {
        e2h_set_error(g, "tensor %d: N=%d K=%d is not a multiple of 8/128", di, q->N, q->K);
        return -1;
    }
    *woff = align64(*woff);
    int32_t *dd = desc + di * 12;
    dd[0] = (int32_t)*woff;
    memcpy(wbuf + *woff, q->w, (size_t)q->Npad * q->K);
    *woff += (size_t)q->Npad * q->K;
    *woff = align64(*woff);
    dd[1] = (int32_t)*woff;
    memcpy(wbuf + *woff, q->scale, sizeof(float) * (size_t)q->Npad);
    *woff += (size_t)q->Npad * 4;
    dd[2] = 0; /* per-block weight sums: unused on this backend */
    dd[3] = (int32_t)g->reg[y_region].off;
    dd[4] = q->N;
    dd[5] = q->K;
    dd[6] = q->nblocks;
    dd[7] = (int32_t)g->reg[qa_region].off;
    dd[8] = (int32_t)g->reg[asc_region].off;
    dd[9] = 0;
    dd[10] = q->Npad;
    dd[11] = 0;
    return 0;
}

/* One fixed layout for every pass, sized for E2_HIP_MAX_ROWS so all offsets are graph constants. */
static void layout(e2_hip *g) {
    const size_t R = (size_t)E2_HIP_MAX_ROWS;
    static const struct { int id; size_t bytes; } dim[] = {
        { HA_X, R * 512 * 4 },    { HA_RES, R * 512 * 4 },  { HA_Q, R * 2048 * 4 },
        { HA_V, R * 1024 * 4 },   { HA_CTX, R * 2048 * 4 }, { HA_GATE, R * 2048 * 4 },
        { HA_UP, R * 2048 * 4 },  { HA_TMP, R * 512 * 4 },  { HA_PLE, R * 12288 * 4 },
        { HA_EMB, R * 768 * 4 },  { HA_K2, R * 512 * 4 },   { HA_V2, R * 512 * 4 },
        { HA_QAX, R * 512 },      { HA_QAXS, R * 4 * 4 },
        { HA_QAC, R * 2048 },     { HA_QACS, R * 16 * 4 },  { HA_QAM, R * 2048 },
        { HA_QAMS, R * 16 * 4 },  { HA_QAP, R * 512 },      { HA_QAPS, R * 4 * 4 },
        /* int8 attention. K/V^T packs are per (sequence, kv-head): nseq*KV*sup*D peaks at
         * S = attn_i8_min (nseq = 2048/S, sup = 128) -> 4096 rows-total x 512 bytes = 2 MB.
         * Scores and P are per (sequence, head): one shared slice of S*sup <= 4M elements in
         * loop mode, and sliced mode is capped at the same total (E2_HIP_A8_SLICES). The pads
         * cover the products' unguarded activation reads on the last 8-row tile. */
        { HA_QQ, R * 2048 + (64 << 10) }, { HA_QQS, R * 16 * 4 + (4 << 10) },
        { HA_KPK, (size_t)3 << 20 },      { HA_KSC, (size_t)64 << 10 },
        { HA_VPK, (size_t)3 << 20 },      { HA_VSC, (size_t)128 << 10 },
        { HA_SCR, ((size_t)16 << 20) + ((size_t)1 << 20) },
        { HA_PB, ((size_t)4 << 20) + ((size_t)1 << 20) },
        { HA_PBS, (size_t)160 << 10 },
        /* vision tower, sized for E2_HIP_VP patches padded to 128 for the key axis. */
        { HA_VX, (size_t)E2_HIP_VP * 768 * 4 },    { HA_VRES, (size_t)E2_HIP_VP * 768 * 4 },
        { HA_VTMP, (size_t)E2_HIP_VP * 768 * 4 },  { HA_VQ, (size_t)E2_HIP_VP * 768 * 4 },
        { HA_VK, (size_t)E2_HIP_VP * 768 * 4 },    { HA_VV, (size_t)E2_HIP_VP * 768 * 4 },
        { HA_VCTX, (size_t)E2_HIP_VP * 768 * 4 },
        { HA_VGATE, (size_t)E2_HIP_VP * 3072 * 4 }, { HA_VUP, (size_t)E2_HIP_VP * 3072 * 4 },
        { HA_VQX, (size_t)E2_HIP_VP * 768 },       { HA_VQXS, (size_t)E2_HIP_VP * 6 * 4 },
        { HA_VQH, (size_t)E2_HIP_VP * 768 },       { HA_VQHS, (size_t)E2_HIP_VP * 6 * 4 },
        { HA_VQM, (size_t)E2_HIP_VP * 3072 },      { HA_VQMS, (size_t)E2_HIP_VP * 24 * 4 },
        { HA_VPP, (size_t)E2_HIP_VGROUPS * 768 },  { HA_VPPS, (size_t)E2_HIP_VGROUPS * 6 * 4 },
        { HA_VQPK, (size_t)E2_V_HEADS * E2_HIP_VP * 64 },
        { HA_VQPKS, (size_t)E2_V_HEADS * E2_HIP_VP * 4 },
        { HA_VKPK, (size_t)E2_V_HEADS * E2_HIP_VP * 64 },
        { HA_VKSC, (size_t)E2_V_HEADS * E2_HIP_VP * 4 },
        { HA_VVPK, (size_t)E2_V_HEADS * 64 * E2_HIP_VP },
        { HA_VVSC, (size_t)E2_V_HEADS * 64 * 4 },
        { HA_VSCR, (size_t)E2_HIP_VP * E2_HIP_VP * 4 },
        { HA_VPB, (size_t)E2_HIP_VP * E2_HIP_VP },
        { HA_VPBS, (size_t)E2_HIP_VP * (E2_HIP_VP / 128) * 4 },
        { HA_VPOOL, (size_t)E2_HIP_VGROUPS * 768 * 4 },
        { HA_VSOFT, (size_t)E2_MAX_SEQ * 512 * 4 }, { HA_VEMBIN, (size_t)E2_MAX_SEQ * 512 * 4 },
    };
    size_t off = 0;
    for (size_t i = 0; i < sizeof dim / sizeof dim[0]; i++) {
        g->reg[dim[i].id].off = off;
        g->reg[dim[i].id].bytes = dim[i].bytes;
        off = align64(off + dim[i].bytes);
    }
    /* tail pad for the GEMM's unguarded activation reads on the last tile */
    off += (size_t)64 * 2048 * 4;
    g->acts_bytes = off;
}

int e2h_forward_build(e2_hip *g) {
    e2_model *m = g->m;
    layout(g);

    const int ntext = m->n_layers * 9 + 2;
    const int nvis = m->vision ? E2_V_LAYERS * 7 + 2 : 0;
    const int ntensors = ntext + nvis;
    g->vdesc_base = ntext;
    g->desc_tensors = ntensors;
    size_t wbytes = 0;
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        wbytes += q8w_bytes(&L->q8_q) + q8w_bytes(&L->q8_k) + q8w_bytes(&L->q8_v) +
                  q8w_bytes(&L->q8_o) + q8w_bytes(&L->q8_gate) + q8w_bytes(&L->q8_up) +
                  q8w_bytes(&L->q8_down) + q8w_bytes(&L->q8_ple_gate) + q8w_bytes(&L->q8_ple_proj);
    }
    wbytes += q8w_bytes(&m->q8_ple_model) + q8w_bytes(&m->q8_embed_proj);
    wbytes += (size_t)(m->n_layers * 7 + 2) * 512 * 4;
    wbytes += (size_t)2 * (256 + 512) * E2_MAX_SEQ * 4;
    if (m->vision) {
        e2_vision *v = (e2_vision *)m->vision;
        for (int l = 0; l < E2_V_LAYERS; l++) {
            const e2_vlayer *L = &v->layers[l];
            wbytes += q8w_bytes(&L->q8_q) + q8w_bytes(&L->q8_k) + q8w_bytes(&L->q8_v) +
                      q8w_bytes(&L->q8_o) + q8w_bytes(&L->q8_gate) + q8w_bytes(&L->q8_up) +
                      q8w_bytes(&L->q8_down);
        }
        wbytes += q8w_bytes(&v->q8_patch) + q8w_bytes(&v->q8_embed_proj);
        wbytes += (size_t)(E2_V_LAYERS * 6) * 768 * 4;
        wbytes += (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2;
    }
    wbytes = align64(wbytes) + (1u << 20);
    g->wbuf_bytes = wbytes;

    uint8_t *wbuf = calloc(1, wbytes);
    int32_t *desc = calloc((size_t)ntensors * 12, 4);
    if (!wbuf || !desc) {
        free(wbuf);
        free(desc);
        e2h_set_error(g, "oom for %zu bytes of weights", wbytes);
        return -1;
    }

    size_t woff = 0;
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        const int b = l * 9;
        if (upload_tensor(g, wbuf, &woff, desc, b + 0, &L->q8_q, HA_Q, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 1, &L->q8_k, HA_K2, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 2, &L->q8_v, HA_V2, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 3, &L->q8_o, HA_TMP, HA_QAC, HA_QACS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 4, &L->q8_gate, HA_GATE, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 5, &L->q8_up, HA_UP, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 6, &L->q8_down, HA_TMP, HA_QAM, HA_QAMS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 7, &L->q8_ple_gate, HA_GATE, HA_QAX,
                          HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + 8, &L->q8_ple_proj, HA_TMP, HA_QAP, HA_QAPS))
            goto bad;
    }
    {
        const int gb = m->n_layers * 9;
        if (upload_tensor(g, wbuf, &woff, desc, gb, &m->q8_ple_model, HA_PLE, HA_QAX, HA_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, gb + 1, &m->q8_embed_proj, HA_EMB, HA_QAX,
                          HA_QAXS))
            goto bad;
    }
    /* norm gammas: q_norm/k_norm are head_dim wide, the rest hidden (512) wide */
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        const float *ps[7] = { L->q_norm, L->k_norm,      L->input_ln, L->post_attn_ln,
                               L->pre_ffn_ln, L->post_ffn_ln, L->ple_norm };
        const int ns[7] = { L->head_dim, L->head_dim, 512, 512, 512, 512, 512 };
        for (int i = 0; i < 7; i++) {
            woff = align64(woff);
            g->noff[l].n[i] = (int)woff;
            memcpy(wbuf + woff, ps[i], sizeof(float) * (size_t)ns[i]);
            woff += 512 * 4; /* one slot per gamma keeps the offsets uniform */
        }
    }
    woff = align64(woff);
    g->final_norm_off = (int)woff;
    memcpy(wbuf + woff, m->final_norm, sizeof(float) * 512);
    woff += 512 * 4;
    woff = align64(woff);
    g->ple_proj_norm_off = (int)woff;
    memcpy(wbuf + woff, m->ple_proj_norm, sizeof(float) * 512);
    woff += 512 * 4;
    for (int t = 0; t < 2; t++) {
        const int D = t == 0 ? 256 : 512;
        woff = align64(woff);
        g->cos_off[t] = (int)woff;
        memcpy(wbuf + woff, m->rope_cos[t], sizeof(float) * (size_t)E2_MAX_SEQ * D);
        woff += (size_t)E2_MAX_SEQ * D * 4;
        woff = align64(woff);
        g->sin_off[t] = (int)woff;
        memcpy(wbuf + woff, m->rope_sin[t], sizeof(float) * (size_t)E2_MAX_SEQ * D);
        woff += (size_t)E2_MAX_SEQ * D * 4;
    }
    /* ---- vision tower: weights, norm gammas, the bf16 position table (31 MB, kept resident) */
    if (m->vision) {
        e2_vision *v = (e2_vision *)m->vision;
        static const int vy[7] = { HA_VQ, HA_VK, HA_VV, HA_VTMP, HA_VGATE, HA_VUP, HA_VTMP };
        static const int vqa[7] = { HA_VQX, HA_VQX, HA_VQX, HA_VQH, HA_VQX, HA_VQX, HA_VQM };
        static const int vasc[7] = { HA_VQXS, HA_VQXS, HA_VQXS, HA_VQHS, HA_VQXS, HA_VQXS,
                                     HA_VQMS };
        for (int l = 0; l < E2_V_LAYERS; l++) {
            const e2_vlayer *L = &v->layers[l];
            const e2_q8w *q[7] = { &L->q8_q, &L->q8_k, &L->q8_v, &L->q8_o,
                                   &L->q8_gate, &L->q8_up, &L->q8_down };
            for (int s2 = 0; s2 < 7; s2++)
                if (upload_tensor(g, wbuf, &woff, desc, g->vdesc_base + l * 7 + s2, q[s2], vy[s2],
                                  vqa[s2], vasc[s2]))
                    goto bad;
            const float *ps[6] = { L->input_ln, L->post_attn_ln, L->pre_ffn_ln, L->post_ffn_ln,
                                   L->q_norm, L->k_norm };
            const int ns[6] = { 768, 768, 768, 768, 64, 64 };
            for (int i = 0; i < 6; i++) {
                woff = align64(woff);
                g->vnoff[l].n[i] = (int)woff;
                memcpy(wbuf + woff, ps[i], sizeof(float) * (size_t)ns[i]);
                woff += 768 * 4; /* one stride for all six: the head norms fit in a hidden */
            }
        }
        const int vb = g->vdesc_base + E2_V_LAYERS * 7;
        if (upload_tensor(g, wbuf, &woff, desc, vb, &v->q8_patch, HA_VRES, HA_VQX, HA_VQXS) ||
            upload_tensor(g, wbuf, &woff, desc, vb + 1, &v->q8_embed_proj, HA_VSOFT, HA_VPP,
                          HA_VPPS))
            goto bad;
        woff = align64(woff);
        g->vpostab_off = (int)woff;
        memcpy(wbuf + woff, v->pos_table, (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2);
        woff += (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2;
    }
    if (woff > wbytes) {
        e2h_set_error(g, "weight buffer overflow");
        goto bad;
    }

    /* ---- device allocations */
    const size_t part_bytes = (size_t)3 * E2_HIP_MAX_NSL * E2_HIP_GEMV_TM * 12288 * 4;
    if (hipMalloc(&g->wbuf, wbytes) || hipMalloc(&g->acts, g->acts_bytes) ||
        hipMalloc(&g->part, part_bytes) ||
        hipMalloc(&g->desc, (size_t)ntensors * 12 * 4) ||
        hipMalloc(&g->toktab, (size_t)E2_VOCAB * m->hidden * 2) ||
        hipMalloc((void **)&g->d_ids, sizeof(int32_t) * (size_t)E2_HIP_MAX_ROWS) ||
        hipMalloc((void **)&g->d_valid, E2_HIP_MAX_ROWS) ||
        hipMalloc((void **)&g->d_embout, sizeof(float) * 768 * (size_t)E2_HIP_MAX_ROWS) ||
        hipHostMalloc((void **)&g->ids, sizeof(int32_t) * (size_t)E2_HIP_MAX_ROWS, 0) ||
        hipHostMalloc((void **)&g->valid, E2_HIP_MAX_ROWS, 0) ||
        hipMalloc(&g->vpix, sizeof(float) * (size_t)E2_V_MAX_PATCHES * E2_V_PATCH) ||
        hipMalloc(&g->vpos, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES * 2) ||
        hipMalloc(&g->vorder, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES) ||
        hipMalloc(&g->vgpatch, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES) ||
        hipMalloc(&g->vgstart, sizeof(int32_t) * (E2_HIP_VGROUPS + 1)) ||
        hipMalloc(&g->vrope, sizeof(float) * 2 * (size_t)E2_HIP_VP * 64) ||
        hipMalloc(&g->vpool_ng, sizeof(int32_t))) {
        e2h_set_error(g, "device allocation failed: %s", hipGetErrorString(hipGetLastError()));
        goto bad;
    }
    if (hipMemcpy(g->wbuf, wbuf, wbytes, hipMemcpyHostToDevice) ||
        hipMemcpy(g->desc, desc, (size_t)ntensors * 12 * 4, hipMemcpyHostToDevice) ||
        hipMemcpy(g->toktab, m->embed_tokens, (size_t)E2_VOCAB * m->hidden * 2,
                  hipMemcpyHostToDevice)) {
        e2h_set_error(g, "upload failed: %s", hipGetErrorString(hipGetLastError()));
        goto bad;
    }
    memset(g->valid, 1, E2_HIP_MAX_ROWS);
    g->hdesc = desc; /* kept: the GEMM launcher sizes the column grid per tensor from it */
    free(wbuf);
    return 0;
bad:
    free(desc);
    free(wbuf);
    return -1;
}

/* ---------------------------------------------------------------- recording */

static int gemm(e2_hip *g, int rows, int first, int count, int nmax) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    if (rows <= 16) {
        /* k-split fills the machine when N is short; a wide projection already has thousands of
         * work items, and there a split is pure redundancy (each slice re-reads the activations and
         * the reduce pass re-reads the partials). nsl = 1 writes the projection directly. */
        int nsl = nmax >= 2048 ? 1 : nmax >= 1024 ? 2 : E2_HIP_MAX_NSL;
        { const char *ns = getenv("E2_HIP_NSL"); if (ns) nsl = atoi(ns); } /* A/B knob */
        e2h_gemv(g->stream, g->wbuf, g->acts, g->part, g->desc, nsl, rows, nmax, first, count);
        g->kernels_last += nsl > 1 ? 2 : 1;
    } else {
        /* Column blocks run back to back over the group's tensors (128 columns per LDS block,
         * 256 per direct block) instead of count grids each sized by the widest tensor: a
         * q|k|v launch no longer computes 3/4 of k's and v's blocks only to discard them. */
        const int ncb = g->gemm_lds ? 128 : 256;
        int ytotal = 0;
        for (int i = 0; i < count; i++) {
            const int Nt = g->hdesc[(first + i) * 12 + 4];
            ytotal += (Nt + ncb - 1) / ncb;
        }
        /* rows (not the 32-row pad) is what the kernel knows: the partial last tile skips the
         * warps that have no valid row. */
        if (g->gemm_lds)
            e2h_gemm_lds(g->stream, g->wbuf, g->acts, g->desc, rows, first, count, ytotal);
        else
            e2h_gemm(g->stream, g->wbuf, g->acts, g->desc, rows, first, count, ytotal);
        g->kernels_last++;
    }
    return 0;
}

static void rowops(e2_hip *g, int rows, int K, int region_x, int region_src, int region_res,
                   int region_qa, int region_asc, int gamma_off, int gamma2_off, int flags,
                   float prescale, float scale) {
    if (g->limit && g->kernels_last >= g->limit) return;
    void *x = (char *)g->acts + g->reg[region_x].off;
    void *src = region_src >= 0 ? (char *)g->acts + g->reg[region_src].off : NULL;
    void *res = region_res >= 0 ? (char *)g->acts + g->reg[region_res].off : NULL;
    void *qa = region_qa >= 0 ? (char *)g->acts + g->reg[region_qa].off : NULL;
    void *asc = region_asc >= 0 ? (char *)g->acts + g->reg[region_asc].off : NULL;
    void *gam = gamma_off >= 0 ? (char *)g->wbuf + gamma_off : NULL;
    void *gam2 = gamma2_off >= 0 ? (char *)g->wbuf + gamma2_off : NULL;
    e2h_rowops(g->stream, x, src, res, qa, asc, gam, gam2, g->m->rms_eps, prescale, scale, rows, K,
               K / 128, flags, NULL);
    g->kernels_last++;
}

int e2h_launch_all(e2_hip *g, int rows, int seq_len, int from_embeds) {
    e2_model *m = g->m;
    hipStream_t s = g->stream;
    g->kernels_last = 0;
    e2h_prof_rec = e2h_prof_on && g->calls == 0;
    if (hipMemcpyAsync(g->d_ids, g->ids, sizeof(int32_t) * (size_t)rows, hipMemcpyHostToDevice, s) ||
        hipMemcpyAsync(g->d_valid, g->valid, E2_HIP_MAX_ROWS, hipMemcpyHostToDevice, s))
        return e2h_ok(g, hipGetLastError(), "id upload");

    void *x = (char *)g->acts + g->reg[HA_X].off;
    void *emb = (char *)g->acts + g->reg[HA_EMB].off;
    const float embed_scale = m->embed_scale, ple_scale = m->ple_scale;

/* E2_HIP_LIMIT=n stops the graph after n kernels: with the fixed order this bisects the recording
 * against the CPU engine's E2_DUMP regions one kernel at a time. */
#define KERN(call) \
    do { \
        if (g->limit && g->kernels_last >= g->limit) goto done; \
        call; \
        g->kernels_last++; \
    } while (0)

    if (from_embeds) {
        const int hidden = 512;
        KERN(e2h_copy_rows(s, x, (const float *)((char *)g->acts + g->reg[HA_VEMBIN].off), rows,
                           hidden));
    } else {
        KERN(e2h_embed(s, x, (const uint16_t *)g->toktab, g->d_ids, g->d_valid, rows, m->hidden,
                       embed_scale));
    }
    rowops(g, rows, 512, HA_X, -1, -1, HA_QAX, HA_QAXS, -1, -1, 16 | 2, 1.f, 1.f);
    /* PLE: project to n_layers x 512 per token, scale, then RMSNorm every 512-slice */
    if (gemm(g, rows, m->n_layers * 9, 1, m->n_layers * 512)) return -1;
    rowops(g, rows * m->n_layers, 512, HA_PLE, -1, -1, -1, -1, g->ple_proj_norm_off, -1, 0,
           ple_scale, 1.f);

    const int nlayers = g->layer_limit > 0 && g->layer_limit < m->n_layers ? g->layer_limit
                                                                          : m->n_layers;
    for (int l = 0; l < nlayers; l++) {
        const e2_layer *L = &m->layers[l];
        const int base = l * 9;
        const int rope = L->sliding ? 0 : 1;
        const int D = L->head_dim, H = 4, KV = L->n_kv_heads;
        const int q_stride = H * D, kv_stride = KV * D;
        /* 1. residual copy + input norm + activation quantization */
        rowops(g, rows, 512, HA_X, -1, HA_RES, HA_QAX, HA_QAXS, g->noff[l].n[2], -1, 1 | 2, 1.f,
               1.f);
        /* 2. q | k | v from the normed input */
        if (gemm(g, rows, base + 0, 3, H * D)) return -1;
        /* 3. per-head RMSNorm + rotary embedding */
        KERN(e2h_norm_rope3(s, (float *)((char *)g->acts + g->reg[HA_Q].off),
                            (float *)((char *)g->wbuf + g->noff[l].n[0]), rows * H,
                            (float *)((char *)g->acts + g->reg[HA_K2].off),
                            (float *)((char *)g->wbuf + g->noff[l].n[1]), rows * KV,
                            (float *)((char *)g->acts + g->reg[HA_V2].off), NULL, rows * KV,
                            (float *)((char *)g->wbuf + g->cos_off[rope]),
                            (float *)((char *)g->wbuf + g->sin_off[rope]), m->rms_eps, H, KV, D, seq_len));
        /* 4. attention (context quantized on the way out) */
        {
            const int window = L->sliding ? L->window : 0;
            const int S = seq_len > 0 ? seq_len : rows;
            const int nseq = seq_len > 0 ? rows / seq_len : 1;
            const int sup = (S + 127) / 128 * 128;
            const int nb = D / 128;
            /* the int8 dot4 path: Q/K/V quantized once, scores and context through v_dot4.
             * Shapes that outgrow the pack regions (only possible with a lowered
             * E2_HIP_ATTN_I8_MIN) fall back to the fp32 kernel. */
            if (g->attn_i8 && S >= g->attn_i8_min &&
                (size_t)nseq * KV * sup * D <= g->reg[HA_KPK].bytes) {
                int8_t *qq = (int8_t *)((char *)g->acts + g->reg[HA_QQ].off);
                float *qqs = (float *)((char *)g->acts + g->reg[HA_QQS].off);
                int8_t *kpk = (int8_t *)((char *)g->acts + g->reg[HA_KPK].off);
                float *ksc = (float *)((char *)g->acts + g->reg[HA_KSC].off);
                int8_t *vpk = (int8_t *)((char *)g->acts + g->reg[HA_VPK].off);
                float *vsc = (float *)((char *)g->acts + g->reg[HA_VSC].off);
                float *scr = (float *)((char *)g->acts + g->reg[HA_SCR].off);
                int8_t *pb = (int8_t *)((char *)g->acts + g->reg[HA_PB].off);
                float *pbs = (float *)((char *)g->acts + g->reg[HA_PBS].off);
                const float *kf = (const float *)((char *)g->acts + g->reg[HA_K2].off);
                const float *vf = (const float *)((char *)g->acts + g->reg[HA_V2].off);
                float *ctxf = (float *)((char *)g->acts + g->reg[HA_CTX].off);
                /* Q quantized per (row, head) by the row kernel: rows*H rows of D elements is
                 * the same memory as rows rows of q_stride. */
                rowops(g, rows * H, D, HA_Q, -1, -1, HA_QQ, HA_QQS, -1, -1, 16 | 2, 1.f, 1.f);
                KERN(e2h_attn_packk(s, kpk, ksc, kf, S, sup, D, KV, kv_stride, nseq));
                KERN(e2h_attn_packv(s, vpk, vsc, vf, S, sup, D, KV, kv_stride, nseq));
                /* The LDS-staged products restage K/VT for every 16-row group, which is a win once
                 * the sequence is long enough for the direct kernels' per-warp working set to fall
                 * out of L1 (measured crossover ~512 keys on gfx1036); below it the direct-load
                 * kernels are faster. */
                const int a8_lds = g->attn_i8_lds && S >= E2_HIP_A8_LDS_MIN;
                if ((size_t)nseq * H * S * sup <= (size_t)E2_HIP_A8_SLICES) {
                    /* every (sequence, head) gets a slice: three launches per layer */
                    if (a8_lds) {
                        KERN(e2h_attn_scores8_lds(s, scr, qq, qqs, kpk, ksc, S, sup, D, H, KV,
                                                  q_stride, nseq * H));
                        KERN(e2h_attn_softmax8(s, scr, pb, pbs, g->d_valid, S, sup, H, window,
                                               nseq * H));
                        KERN(e2h_attn_ctx8_lds(s, ctxf, pb, pbs, vpk, vsc, S, sup, D, H, KV,
                                               q_stride, nseq * H));
                    } else {
                        KERN(e2h_attn_scores8(s, scr, qq, qqs, kpk, ksc, S, sup, D, H, KV,
                                              q_stride, nseq * H));
                        KERN(e2h_attn_softmax8(s, scr, pb, pbs, g->d_valid, S, sup, H, window,
                                               nseq * H));
                        KERN(e2h_attn_ctx8(s, ctxf, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride,
                                           nseq * H));
                    }
                } else {
                    /* long sequences: one shared scores/P slice, serialized per (s, h) */
                    for (int sq = 0; sq < nseq; sq++)
                        for (int h = 0; h < H; h++) {
                            const int kvh = (h * KV) / H;
                            int8_t *qp = qq + (size_t)sq * S * q_stride + (size_t)h * D;
                            float *qs = qqs + ((size_t)sq * S * H + h) * nb;
                            int8_t *kp = kpk + (size_t)(sq * KV + kvh) * sup * D;
                            float *ks = ksc + (size_t)(sq * KV + kvh) * sup;
                            int8_t *vp = vpk + (size_t)(sq * KV + kvh) * D * sup;
                            float *vs = vsc + (size_t)(sq * KV + kvh) * D;
                            float *cp = ctxf + (size_t)sq * S * q_stride + (size_t)h * D;
                            if (a8_lds) {
                                KERN(e2h_attn_scores8_lds(s, scr, qp, qs, kp, ks, S, sup, D, H,
                                                          KV, q_stride, 1));
                                KERN(e2h_attn_softmax8(s, scr, pb, pbs,
                                                       g->d_valid + (size_t)sq * S, S, sup, H,
                                                       window, 1));
                                KERN(e2h_attn_ctx8_lds(s, cp, pb, pbs, vp, vs, S, sup, D, H, KV,
                                                       q_stride, 1));
                            } else {
                                KERN(e2h_attn_scores8(s, scr, qp, qs, kp, ks, S, sup, D, H, KV,
                                                      q_stride, 1));
                                KERN(e2h_attn_softmax8(s, scr, pb, pbs,
                                                       g->d_valid + (size_t)sq * S, S, sup, H,
                                                       window, 1));
                                KERN(e2h_attn_ctx8(s, cp, pb, pbs, vp, vs, S, sup, D, H, KV,
                                                   q_stride, 1));
                            }
                        }
                }
                /* the context is the o projection's input: quantize it (the fused kernel used
                 * to do this on the way out) */
                rowops(g, rows, q_stride, HA_CTX, -1, -1, HA_QAC, HA_QACS, -1, -1, 16 | 2, 1.f,
                       1.f);
            } else if (g->attn_lds) {
            KERN(e2h_attn_lds(s, (float *)((char *)g->acts + g->reg[HA_CTX].off),
                              (float *)((char *)g->acts + g->reg[HA_Q].off),
                              (float *)((char *)g->acts + g->reg[HA_K2].off),
                              (float *)((char *)g->acts + g->reg[HA_V2].off),
                              (int8_t *)((char *)g->acts + g->reg[HA_QAC].off),
                              (float *)((char *)g->acts + g->reg[HA_QACS].off), g->d_valid, rows, H,
                              D, KV, q_stride, kv_stride, window, seq_len));
        } else {
            KERN(e2h_attn(s, (float *)((char *)g->acts + g->reg[HA_CTX].off),
                          (float *)((char *)g->acts + g->reg[HA_Q].off),
                          (float *)((char *)g->acts + g->reg[HA_K2].off),
                          (float *)((char *)g->acts + g->reg[HA_V2].off),
                          (int8_t *)((char *)g->acts + g->reg[HA_QAC].off),
                          (float *)((char *)g->acts + g->reg[HA_QACS].off), g->d_valid, rows, H, D,
                          KV, q_stride, kv_stride, window, seq_len));
        }
        } /* end of the int8/fp32 attention switch */
        /* 5. o projection from the quantized context */
        if (gemm(g, rows, base + 3, 1, 512)) return -1;
        /* 6. attention residual + pre-FFN norm/quantization */
        rowops(g, rows, 512, HA_X, HA_TMP, HA_RES, HA_QAX, HA_QAXS, g->noff[l].n[3],
               g->noff[l].n[4], 4 | 8 | 32 | 2, 1.f, 1.f);
        /* 7. gate | up */
        if (gemm(g, rows, base + 4, 2, m->intermediate)) return -1;
        /* 8. gelu(gate) * up, quantized */
        KERN(e2h_gelu_mul_quant(s, (float *)((char *)g->acts + g->reg[HA_GATE].off),
                                (const float *)((char *)g->acts + g->reg[HA_UP].off), NULL,
                                (int8_t *)((char *)g->acts + g->reg[HA_QAM].off),
                                (float *)((char *)g->acts + g->reg[HA_QAMS].off), rows,
                                m->intermediate, m->intermediate / 128, 0, 0));
        /* 9. down projection */
        if (gemm(g, rows, base + 6, 1, 512)) return -1;
        /* 10. MLP residual + quantization of x for the PLE gate */
        rowops(g, rows, 512, HA_X, HA_TMP, HA_RES, HA_QAX, HA_QAXS, g->noff[l].n[5], -1, 4 | 8 | 2,
               1.f, 1.f);
        /* 11. PLE gate */
        if (gemm(g, rows, base + 7, 1, 512)) return -1;
        /* 12. gate = gelu(gate) * this layer's PLE slice, quantized */
        KERN(e2h_gelu_mul_quant(s, (float *)((char *)g->acts + g->reg[HA_GATE].off), NULL,
                                (const float *)((char *)g->acts + g->reg[HA_PLE].off),
                                (int8_t *)((char *)g->acts + g->reg[HA_QAP].off),
                                (float *)((char *)g->acts + g->reg[HA_QAPS].off), rows, 512, 4,
                                m->n_layers * 512, l * 512));
        /* 13. PLE projection */
        if (gemm(g, rows, base + 8, 1, 512)) return -1;
        /* 14. PLE residual + the layer scalar */
        rowops(g, rows, 512, HA_X, HA_TMP, HA_X, -1, -1, g->noff[l].n[6], -1, 4 | 8, 1.f,
               L->layer_scalar);
    }
    if (g->layer_limit == 0) goto done;
    /* final norm (quantized for the embedder projection), the projection, mean pool + L2 */
    rowops(g, rows, 512, HA_X, -1, -1, HA_QAX, HA_QAXS, g->final_norm_off, -1, 2, 1.f, 1.f);
    if (gemm(g, rows, m->n_layers * 9 + 1, 1, 768)) return -1;
    {
        const int dim = 768, nseq = seq_len > 0 ? rows / seq_len : 1;
        (void)nseq;
        KERN(e2h_pool_l2(s, g->d_embout, (const float *)emb, g->d_valid, rows, dim, seq_len));
    }
#undef KERN
done:;
    return 0;
}

int e2h_record(e2_hip *g, int rows, int seq_len, int from_embeds, e2_hip_slot *slot) {
    if (!g->nograph &&
        e2h_ok(g, hipStreamBeginCapture(g->stream, hipStreamCaptureModeThreadLocal), "begin capture"))
        return -1;
    if (e2h_launch_all(g, rows, seq_len, from_embeds)) {
        if (!g->nograph) {
            hipGraph_t gr = NULL;
            hipStreamEndCapture(g->stream, &gr);
            if (gr) hipGraphDestroy(gr);
        }
        return -1;
    }
    slot->kernels = (int)g->kernels_last;
    if (!g->nograph) {
        hipGraph_t gr = NULL;
        if (e2h_ok(g, hipStreamEndCapture(g->stream, &gr), "end capture")) return -1;
        if (e2h_ok(g, hipGraphInstantiate(&slot->exec, gr, NULL, NULL, 0), "graph instantiate")) {
            hipGraphDestroy(gr);
            return -1;
        }
        /* Pre-stage the graph so the first (and every) launch does not pay the upload. Not fatal:
         * older runtimes may not implement it. */
        hipGraphUpload(slot->exec, g->stream);
        slot->graph = gr;
    }
    return 0;
}

/* ---------------------------------------------------------------- vision tower recording
 * The Gemma4 encoder as one recorded graph, in the same order as the Intel backend's record_vision
 * and src/vision.c. The host does the gather/CSR/rope-table work (vision_soft_tokens in hip.c); here
 * only the kernel sequence is laid down. The projections go through gemm()/rowops() unchanged: the
 * weights are the same quad-interleaved int8 matrix and every projection's K is a multiple of 128. */
int e2h_launch_vision(e2_hip *g, int P, int ng) {
    const int D = E2_V_HIDDEN, H = E2_V_HEADS;
    const int Ppad = (P + 127) / 128 * 128;
    const int vl = g->vlayer_limit;
    g->kernels_last = 0;
    e2h_prof_rec = e2h_prof_on && g->calls == 0;

    int8_t *vqx = (int8_t *)((char *)g->acts + g->reg[HA_VQX].off);
    float *vqxs = (float *)((char *)g->acts + g->reg[HA_VQXS].off);
    float *vx = (float *)((char *)g->acts + g->reg[HA_VX].off);
    float *vres = (float *)((char *)g->acts + g->reg[HA_VRES].off);
    float *vq = (float *)((char *)g->acts + g->reg[HA_VQ].off);
    float *vk = (float *)((char *)g->acts + g->reg[HA_VK].off);
    float *vv = (float *)((char *)g->acts + g->reg[HA_VV].off);
    float *vctx = (float *)((char *)g->acts + g->reg[HA_VCTX].off);
    float *vgate = (float *)((char *)g->acts + g->reg[HA_VGATE].off);
    float *vup = (float *)((char *)g->acts + g->reg[HA_VUP].off);
    int8_t *vqm = (int8_t *)((char *)g->acts + g->reg[HA_VQM].off);
    float *vqms = (float *)((char *)g->acts + g->reg[HA_VQMS].off);
    int8_t *vqpk = (int8_t *)((char *)g->acts + g->reg[HA_VQPK].off);
    float *vqpks = (float *)((char *)g->acts + g->reg[HA_VQPKS].off);
    int8_t *vkpk = (int8_t *)((char *)g->acts + g->reg[HA_VKPK].off);
    float *vksc = (float *)((char *)g->acts + g->reg[HA_VKSC].off);
    int8_t *vvpk = (int8_t *)((char *)g->acts + g->reg[HA_VVPK].off);
    float *vvsc = (float *)((char *)g->acts + g->reg[HA_VVSC].off);
    float *vscr = (float *)((char *)g->acts + g->reg[HA_VSCR].off);
    int8_t *vpb = (int8_t *)((char *)g->acts + g->reg[HA_VPB].off);
    float *vpbs = (float *)((char *)g->acts + g->reg[HA_VPBS].off);
    float *vpool = (float *)((char *)g->acts + g->reg[HA_VPOOL].off);
    const float *vrope = (const float *)((char *)g->vrope);

#define KERNV(call) \
    do { \
        if (g->limit && g->kernels_last >= g->limit) goto done; \
        call; \
        g->kernels_last++; \
    } while (0)

    /* 1. gather the valid patches, rescale and quantize (K=768 -> six 128-blocks) */
    KERNV(e2h_vpatch_q(g->stream, vqx, vqxs, (const float *)g->vpix, (const int32_t *)g->vorder,
                       P, D));
    /* 2. patch projection 768->768, then 3. the 2D position term */
    if (gemm(g, P, g->vdesc_base + E2_V_LAYERS * 7, 1, D)) return -1;
    KERNV(e2h_vpos_add(g->stream, vx, vres, (const uint16_t *)((char *)g->wbuf + g->vpostab_off),
                       (const int32_t *)g->vorder, (const int32_t *)g->vpos, P, D, E2_V_POS));
    if (vl == 0) goto done; /* diagnostic: patch embed only */

    for (int l = 0; l < E2_V_LAYERS; l++) {
        const int b = g->vdesc_base + l * 7;
        /* 1. res = x; x = norm(x, input_ln); quantize */
        rowops(g, P, D, HA_VX, -1, HA_VRES, HA_VQX, HA_VQXS, g->vnoff[l].n[0], -1, 1 | 2, 1.f,
               1.f);
        /* 2. q | k | v */
        if (gemm(g, P, b, 3, D)) return -1;
        /* 3. per-head RMSNorm + axial rope (v without rope or gamma) */
        KERNV(e2h_vnorm_rope3(g->stream, vq, (const float *)((char *)g->wbuf + g->vnoff[l].n[4]),
                              vk, (const float *)((char *)g->wbuf + g->vnoff[l].n[5]), vv, NULL,
                              vrope, vrope + (size_t)E2_HIP_VP * 64, g->m->rms_eps, P * H, H));
        /* 4. Q, K and V into the attention's int8 layouts */
        KERNV(e2h_vattn_packq(g->stream, vqpk, vqpks, vq, P * H, Ppad, H));
        KERNV(e2h_vattn_packk(g->stream, vkpk, vksc, vk, P, Ppad, H));
        KERNV(e2h_vattn_packv(g->stream, vvpk, vvsc, vv, P, Ppad, H));
        /* 5. attention, one head at a time: the scores/P scratch is shared, and the host loop
         * serialises the three kernels of a head. */
        for (int hh = 0; hh < H; hh++) {
            KERNV(e2h_vattn_scores8(g->stream, vscr, vqpk, vqpks, vkpk, vksc, P, Ppad, H, hh));
            KERNV(e2h_vattn_softmax8(g->stream, vscr, vpb, vpbs, P, Ppad));
            KERNV(e2h_vattn_ctx8(g->stream, vctx, vpb, vpbs, vvpk, vvsc, P, Ppad, H, hh));
        }
        /* 6. quantize the context for the o projection */
        rowops(g, P, D, HA_VCTX, -1, -1, HA_VQH, HA_VQHS, -1, -1, 16 | 2, 1.f, 1.f);
        /* 7. o_proj, then x = norm(tmp, post_attn_ln) + res */
        if (gemm(g, P, b + 3, 1, D)) return -1;
        rowops(g, P, D, HA_VX, HA_VTMP, HA_VRES, -1, -1, g->vnoff[l].n[1], -1, 4 | 8, 1.f, 1.f);
        /* 8. res = x; x = norm(x, pre_ffn_ln); quantize */
        rowops(g, P, D, HA_VX, -1, HA_VRES, HA_VQX, HA_VQXS, g->vnoff[l].n[2], -1, 1 | 2, 1.f,
               1.f);
        /* 9. gate | up, then gelu(gate) * up, quantized for the down projection */
        if (gemm(g, P, b + 4, 2, E2_V_INTER)) return -1;
        KERNV(e2h_gelu_mul_quant(g->stream, vgate, vup, NULL, vqm, vqms, P, E2_V_INTER,
                                 E2_V_INTER / 128, 0, 0));
        /* 10. down_proj, then x = norm(tmp, post_ffn_ln) + res */
        if (gemm(g, P, b + 6, 1, D)) return -1;
        rowops(g, P, D, HA_VX, HA_VTMP, HA_VRES, -1, -1, g->vnoff[l].n[3], -1, 4 | 8, 1.f, 1.f);
        if (vl >= 0 && l + 1 >= vl) goto done; /* diagnostic: stop after layer l */
    }

    /* pooler, then the scale-free RMSNorm and the 768->512 soft-token projection. The grid is the
     * fixed group bound and the count comes from the device, so the graph is valid for any image
     * with this patch count whose group count is the one it was recorded for. */
    {
        const float scale = sqrtf((float)D) / (float)(E2_V_POOL * E2_V_POOL);
        KERNV(e2h_vpool(g->stream, vpool, vx, (const int32_t *)g->vgstart,
                        (const int32_t *)g->vgpatch, (const int32_t *)g->vpool_ng, D, scale));
    }
    rowops(g, ng, D, HA_VPOOL, -1, -1, HA_VPP, HA_VPPS, -1, -1, 2, 1.f, 1.f);
    if (gemm(g, ng, g->vdesc_base + E2_V_LAYERS * 7 + 1, 1, 512)) return -1;
#undef KERNV
    return 0;
done:;
    return 0;
}

int e2h_record_vision(e2_hip *g, int P, int ng, e2_hip_slot *slot) {
    if (!g->nograph &&
        e2h_ok(g, hipStreamBeginCapture(g->stream, hipStreamCaptureModeThreadLocal), "begin capture"))
        return -1;
    if (e2h_launch_vision(g, P, ng)) {
        if (!g->nograph) {
            hipGraph_t gr = NULL;
            hipStreamEndCapture(g->stream, &gr);
            if (gr) hipGraphDestroy(gr);
        }
        return -1;
    }
    slot->kernels = (int)g->kernels_last;
    if (!g->nograph) {
        hipGraph_t gr = NULL;
        if (e2h_ok(g, hipStreamEndCapture(g->stream, &gr), "end capture")) return -1;
        if (e2h_ok(g, hipGraphInstantiate(&slot->exec, gr, NULL, NULL, 0), "graph instantiate")) {
            hipGraphDestroy(gr);
            return -1;
        }
        hipGraphUpload(slot->exec, g->stream);
        slot->graph = gr;
    }
    return 0;
}

/* ---------------------------------------------------------------- diagnostics */

static const struct { const char *name; int region; } g_dump[] = {
    { "x", HA_X },     { "res", HA_RES }, { "q", HA_Q },     { "k", HA_K2 },   { "v", HA_V2 },
    { "ctx", HA_CTX }, { "gate", HA_GATE }, { "up", HA_UP }, { "tmp", HA_TMP }, { "ple", HA_PLE },
    { "emb", HA_EMB }, { "qax", HA_QAX }, { "qaxs", HA_QAXS }, { "qac", HA_QAC },
    { "qacs", HA_QACS }, { "qam", HA_QAM }, { "qams", HA_QAMS }, { "qap", HA_QAP },
    { "qaps", HA_QAPS },
};

void e2h_dump_regions(e2_hip *g, int rows) {
    for (size_t i = 0; i < sizeof g_dump / sizeof g_dump[0]; i++) {
        size_t bytes = g->reg[g_dump[i].region].bytes;
        if (g_dump[i].region == HA_PLE) bytes = (size_t)rows * g->m->n_layers * 512 * 4;
        void *host = malloc(bytes);
        if (!host) return;
        if (hipMemcpy(host, (char *)g->acts + g->reg[g_dump[i].region].off, bytes,
                      hipMemcpyDeviceToHost)) {
            free(host);
            continue;
        }
        char path[512];
        snprintf(path, sizeof path, "%s/hip_%s.f32", g->dump_dir, g_dump[i].name);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(host, 1, bytes, f);
            fclose(f);
        }
        free(host);
    }
    {
        char path[512];
        snprintf(path, sizeof path, "%s/hip_out.f32", g->dump_dir);
        float *tmp = malloc(4 * 768 * (size_t)rows);
        if (tmp && hipMemcpy(tmp, g->d_embout, 4 * 768 * (size_t)rows, hipMemcpyDeviceToHost) ==
                       hipSuccess) {
            FILE *f = fopen(path, "wb");
            if (f) {
                fwrite(tmp, 4, 768 * (size_t)rows, f);
                fclose(f);
            }
        }
        free(tmp);
    }
}
