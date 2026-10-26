/* gpu.c — the GPU backend: model upload, graph recording, forward pass.
 *
 * Two ideas carry this file:
 *
 *  - Everything the forward pass needs is uploaded once and every offset in it is static, so the whole
 *    sequence of kernels for a given shape is recorded into one Level Zero command list and replayed.
 *    A call then costs a host-side copy of the token ids, one submission and a read of the output
 *    vectors. Graphs are keyed by (rows, sequence length).
 *
 *  - The graph is a transcription of the CPU tower in src/text.c, in the same order and with the same
 *    int8 recipe, so the two engines agree on every input. Where the CPU pass runs a serial loop over
 *    S x hidden this runs one sub-group per row; where it materializes an (S,S) score block this keeps
 *    an online softmax in registers. The fusions (norm+quantize in one pass, gelu+product+quantize in
 *    one pass, the context quantized inside the attention) exist to keep dependent kernel boundaries
 *    down: each one costs a command-list barrier, measured at 2 us on this iGPU against 0.26 us for a
 *    kernel with no dependent successor.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../model.h"
#include "../validation.h"
#include "gpu.h"

static char g_err[1024];

const char *e2_gpu_last_error(void) { return g_err; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int fail(e2_gpu *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    if (g) snprintf(g->err, sizeof g->err, "%s", g_err);
    return -1;
}

/* ---------------------------------------------------------------- weights */

static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }

/* The output clamp belongs to the projection, not to a separate pass: a clipped linear's learned
 * bounds are applied at the GEMM store. Descriptor fields 10..12 are the flag and the two bounds
 * (the int array is also the place the kernel reads them from). */
static void set_clamp(int32_t *dd, const e2_q8w *q) {
    const float lo = q->out_min, hi = q->out_max;
    dd[E2_DS_CLAMP] = lo > -1e30f ? 1 : 0;
    memcpy(&dd[E2_DS_LO], &lo, 4);
    memcpy(&dd[E2_DS_HI], &hi, 4);
}

static size_t q8w_bytes(const e2_q8w *q) {
    /* k-quad layout + scales + block sums, then the DPAS A layout (see q8w_dpas) */
    const size_t base = (size_t)(q->K / 4) * q->N * 4 + (size_t)q->N * 4 + (size_t)q->nblocks * q->N * 4;
    return align64(base) + (size_t)q->N * q->K;
}

/* The XMX GEMM wants the weights as its A operand: lane l of the sub-group holds dword r =
 * W[n0+r][4l..4l+3], so lane l's eight dwords have to be one 32-byte run. The runs of one k-step
 * (8 quads) are stored lanes-contiguous (E2_DPAS_OFF in e2_tile.h) so that the eight addresses of one
 * operand load are 32 bytes apart instead of N*4, which is what lets that load coalesce. Same numbers
 * as the quad layout. */
static void q8w_dpas(uint32_t *dst, const e2_q8w *q) {
    const int N = q->N, KQ = q->K / 4, NQ = N / 8;
    for (int n = 0; n < N; n++)
        for (int kq = 0; kq < KQ; kq++) {
            const uint8_t *src = (const uint8_t *)(q->w + (size_t)(n >> 3) * KQ * 32 + (n & 7) * 4) +
                                 (size_t)kq * 32;
            memcpy((uint8_t *)dst + E2_DPAS_OFF(kq, n >> 3, n & 7, 0, NQ), src, 4);
        }
}

/* Repacks the CPU engine's quad-interleaved int8 weights into the GPU's k-quad-contiguous layout. Same
 * numbers: same quantized values, same per-row scales, same per-block sums. */
static void q8w_pack(uint8_t *dst, const e2_q8w *q, int32_t *desc) {
    const int N = q->N, K = q->K, KQ = K / 4, nb = q->nblocks;
    int8_t *w = (int8_t *)dst;
    for (int n = 0; n < N; n++) {
        const int8_t *row = q->w + (size_t)(n >> 3) * KQ * 32 + (n & 7) * 4;
        for (int kq = 0; kq < KQ; kq++) memcpy(w + ((size_t)kq * N + n) * 4, row + (size_t)kq * 32, 4);
    }
    float *wsc = (float *)(dst + (size_t)KQ * N * 4);
    memcpy(wsc, q->scale, sizeof(float) * (size_t)N);
    int32_t *bsq = (int32_t *)(dst + (size_t)KQ * N * 4 + (size_t)N * 4);
    for (int b = 0; b < nb; b++)
        for (int n = 0; n < N; n++) bsq[(size_t)b * N + n] = q->bsq[(size_t)b * q->Npad + n];
    desc[4] = N;
    desc[5] = K;
    desc[6] = nb;
}

static int upload_tensor(e2_gpu *g, uint8_t *wbuf, size_t *woff, int32_t *desc, int di,
                         const e2_q8w *q, int y_region, int qa_region, int asc_region) {
    if (q->N <= 0 || q->K <= 0 || q->w == NULL) return fail(g, "descriptor %d has no int8 weights", di);
    if (q->K % 128) return fail(g, "descriptor %d: K=%d is not a multiple of 128", di, q->K);
    if (q->N % 8) return fail(g, "descriptor %d: N=%d is not a multiple of 8", di, q->N);
    *woff = align64(*woff);
    q8w_pack(wbuf + *woff, q, desc + di * E2_DS_STRIDE);
    const size_t kqw = (size_t)(q->K / 4) * q->N * 4;
    int32_t *dd = desc + di * E2_DS_STRIDE;
    dd[0] = (int32_t)*woff;
    dd[1] = (int32_t)(dd[0] + kqw);
    dd[2] = (int32_t)(dd[0] + kqw + (size_t)q->N * 4);
    dd[3] = (int32_t)g->reg[y_region].off;
    dd[7] = (int32_t)g->reg[qa_region].off;
    dd[8] = (int32_t)g->reg[asc_region].off;
    const size_t base = kqw + (size_t)q->N * 4 + (size_t)q->nblocks * q->N * 4;
    dd[9] = (int32_t)align64(*woff + base);
    q8w_dpas((uint32_t *)(wbuf + dd[9]), q);
    set_clamp(dd, q);
    *woff = (size_t)dd[9] + (size_t)q->N * q->K;
    return 0;
}

/* ---- vision tower weights. Only k_gemm_dpas runs on this half of the model, and it reads the DPAS
 * A layout rather than the quad-major one the dp4a kernels use, so a vision tensor skips the quad
 * copy: the same numbers for half the bytes and half the upload. */
static size_t vq8w_bytes(const e2_q8w *q) {
    return align64((size_t)q->N * 4 + (size_t)q->nblocks * q->N * 4 + (size_t)q->N * q->K);
}

static int vupload_tensor(e2_gpu *g, uint8_t *wbuf, size_t *woff, int32_t *desc, int di,
                          const e2_q8w *q, int y_region, int qa_region, int asc_region) {
    if (q->N <= 0 || q->K <= 0 || q->w == NULL) return fail(g, "vision descriptor %d has no weights", di);
    if (q->K % 128 || q->N % 8) return fail(g, "vision descriptor %d: N=%d K=%d", di, q->N, q->K);
    *woff = align64(*woff);
    int32_t *dd = desc + di * E2_DS_STRIDE;
    const size_t base = *woff;
    memcpy(wbuf + base, q->scale, sizeof(float) * (size_t)q->N);
    int32_t *bsq = (int32_t *)(wbuf + base + (size_t)q->N * 4);
    for (int b = 0; b < q->nblocks; b++)
        for (int n = 0; n < q->N; n++) bsq[(size_t)b * q->N + n] = q->bsq[(size_t)b * q->Npad + n];
    dd[0] = 0; /* no quad-major copy: nothing here runs a dp4a kernel */
    dd[1] = (int32_t)base;
    dd[2] = (int32_t)(base + (size_t)q->N * 4);
    dd[3] = (int32_t)g->reg[y_region].off;
    dd[4] = q->N;
    dd[5] = q->K;
    dd[6] = q->nblocks;
    dd[7] = (int32_t)g->reg[qa_region].off;
    dd[8] = (int32_t)g->reg[asc_region].off;
    dd[9] = (int32_t)align64(base + (size_t)q->N * 4 + (size_t)q->nblocks * q->N * 4);
    q8w_dpas((uint32_t *)(wbuf + dd[9]), q);
    set_clamp(dd, q);
    *woff = (size_t)dd[9] + (size_t)q->N * q->K;
    return 0;
}

/* One fixed layout for every pass: regions are sized for E2_GPU_MAX_ROWS rows, so the offsets are
 * constants and a graph recorded for one shape stays valid for any shape that fits. */
static void layout(e2_gpu *g) {
    const size_t R = (size_t)E2_GPU_MAX_ROWS;
    static const struct { int id; size_t bytes; } dim[] = {
        { A_X, R * 512 * 4 },    { A_RES, R * 512 * 4 },  { A_Q, R * 2048 * 4 },
        { A_V, R * 1024 * 4 },   { A_CTX, R * 2048 * 4 }, { A_GATE, R * 2048 * 4 },
        { A_UP, R * 2048 * 4 },  { A_TMP, R * 512 * 4 },  { A_PLE, R * 12288 * 4 },
        { A_EMB, R * 768 * 4 },  { A_K2, R * 512 * 4 },   { A_V2, R * 512 * 4 },
        { A_QAX, R * 512 },      { A_QAXS, R * 4 * 4 },
        { A_QAC, R * 2048 },     { A_QACS, R * 16 * 4 },  { A_QAM, R * 2048 },
        { A_QAMS, R * 16 * 4 },  { A_QAP, R * 512 },      { A_QAPS, R * 4 * 4 },
        /* attention on the XMX engine (see k_attn_gemm in kernels/e2.cl). Q is packed per (head,row);
         * K and V^T are packed per (sequence, kv-head) into the DPAS A layout, whose size is
         * sup*D bytes per (sequence, kv-head) with sup the sequence length padded to 128 rows. */
        { A_QPK, (size_t)E2_NUM_HEADS * R * 512 + (64 << 10) },
        { A_QPS, (size_t)E2_NUM_HEADS * R * 4 * 4 },
        { A_KPK, (size_t)4 << 20 }, { A_KSC, (size_t)64 << 10 }, { A_KBSQ, (size_t)512 << 10 },
        { A_VPK, (size_t)4 << 20 }, { A_VSC, (size_t)128 << 10 }, { A_VBSQ, (size_t)1 << 20 },
        { A_SCR, (size_t)17 << 20 }, { A_PB, (size_t)5 << 20 }, { A_PBS, R * 16 * 4 },
        /* vision tower. The activations are sized for E2_V_MAX_PATCHES patches, padded to E2_DM
         * (the GEMM reads its activation tile unguarded) and to 128 for the key axis. */
        { A_VX, (size_t)E2_GPU_VP * 768 * 4 },   { A_VRES, (size_t)E2_GPU_VP * 768 * 4 },
        { A_VTMP, (size_t)E2_GPU_VP * 768 * 4 }, { A_VQ, (size_t)E2_GPU_VP * 768 * 4 },
        { A_VK, (size_t)E2_GPU_VP * 768 * 4 },   { A_VV, (size_t)E2_GPU_VP * 768 * 4 },
        { A_VCTX, (size_t)E2_GPU_VP * 768 * 4 },
        { A_VGATE, (size_t)E2_GPU_VP * 3072 * 4 }, { A_VUP, (size_t)E2_GPU_VP * 3072 * 4 },
        { A_VQX, (size_t)E2_GPU_VP * 768 },      { A_VQXS, (size_t)E2_GPU_VP * 6 * 4 },
        { A_VQH, (size_t)E2_GPU_VP * 768 },      { A_VQHS, (size_t)E2_GPU_VP * 6 * 4 },
        { A_VQM, (size_t)E2_GPU_VP * 3072 },     { A_VQMS, (size_t)E2_GPU_VP * 24 * 4 },
        { A_VPP, (size_t)E2_GPU_VGROUPS * 768 }, { A_VPPS, (size_t)E2_GPU_VGROUPS * 6 * 4 },
        { A_VQPK, (size_t)E2_GPU_VP * 768 },     { A_VQPKS, (size_t)E2_GPU_VP * 12 * 4 },
        { A_VKPK, (size_t)12 * E2_GPU_VK * 64 }, { A_VKSC, (size_t)12 * E2_GPU_VK * 4 },
        { A_VKBSQ, (size_t)12 * E2_GPU_VK * 4 },
        { A_VVPK, (size_t)12 * 64 * E2_GPU_VK }, { A_VVSC, (size_t)12 * 64 * 4 },
        { A_VVBSQ, (size_t)12 * (E2_GPU_VK / 128) * 64 * 4 },
        /* one head's scores at a time: the tower processes heads serially and reuses this */
        { A_VSCR, (size_t)E2_GPU_VK * E2_GPU_VK * 4 }, { A_VPB, (size_t)E2_GPU_VK * E2_GPU_VK },
        { A_VPBS, (size_t)E2_GPU_VP * (E2_GPU_VK / 32) * 4 },
        { A_VSBM, (size_t)E2_GPU_VP * (E2_GPU_VK / 32) * 4 },
        { A_VSBS, (size_t)E2_GPU_VP * (E2_GPU_VK / 32) * 4 },
        { A_VPOOL, (size_t)E2_GPU_VGROUPS * 768 * 4 },
        { A_VSOFT, (size_t)E2_MAX_SEQ * 512 * 4 }, { A_VEMBIN, (size_t)E2_MAX_SEQ * 512 * 4 },
        /* audio tower: 750 tokens is the processor's 30 s truncation point. The 4096-wide FF
         * intermediate and its int8 copy dominate (12.3 MB + 3 MB); the whole set is under 50 MB. */
        { A_AX, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_ARES, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_ATMP, (size_t)E2_A_MAX_TOK * 1536 * 4 },
        { A_AFF, (size_t)E2_A_MAX_TOK * 4096 * 4 },
        { A_ALC, (size_t)E2_A_MAX_TOK * 2048 * 4 },
        { A_AGLU, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_AQ, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_AK, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_AV, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_AATT, (size_t)E2_A_MAX_TOK * 1024 * 4 },
        { A_AQA, (size_t)E2_A_MAX_TOK * 4096 },
        { A_AQAS, (size_t)E2_A_MAX_TOK * 32 * 4 },
        { A_ASOFT, (size_t)E2_A_MAX_TOK * 512 * 4 },
        { A_ARELK, (size_t)E2_A_LAYERS * E2_A_RELPOS * 1024 * 4 },
    };
    size_t off = 0;
    for (size_t i = 0; i < sizeof dim / sizeof dim[0]; i++) {
        g->reg[dim[i].id].off = off;
        g->reg[dim[i].id].bytes = dim[i].bytes;
        off = align64(off + dim[i].bytes);
    }
    /* Belt and braces for k_gemm_tiled's padded M: rows is capped at E2_GPU_MAX_ROWS and is a multiple
     * of E2_BM for the last tile (2048 = 64 x 32), so every read and write already lands inside a
     * region - but that is a property of two constants in two files, and the failure mode of getting it
     * wrong is a device-page fault in a kernel rather than something the tests would catch. */
    off += (size_t)E2_BM * 12288 * 4;
    g->acts_bytes = off;
}

e2_gpu *e2_gpu_create(e2_model *m) {
    if (!m) { fail(NULL, "model is NULL"); return NULL; }
    e2_gpu *g = calloc(1, sizeof *g);
    if (!g) { fail(NULL, "oom"); return NULL; }
    g->max_rows = E2_GPU_MAX_ROWS;
    g->m = m;
    const char *lim = getenv("E2_GPU_LAYERS");
    g->layer_limit = lim ? atoi(lim) : -1;
    g->dump_dir = getenv("E2_GPU_DUMP");
    { const char *dc = getenv("E2_GPU_DUMP_CALL"); g->dump_call = dc ? atoi(dc) : 1; }
    g->gemm_dp4a = getenv("E2_GPU_DP4A") != NULL;
    g->attn_dpas = getenv("E2_GPU_ATTN_OLD") == NULL;
    g->vattn_old = getenv("E2_GPU_VATTN_OLD") != NULL;
    g->only = getenv("E2_GPU_ONLY") ? atoi(getenv("E2_GPU_ONLY")) : 0;
    g->limit = getenv("E2_GPU_LIMIT") ? atoi(getenv("E2_GPU_LIMIT")) : 0;
    { const char *v = getenv("E2_GPU_VLAYERS"); g->vlayer_limit = v ? atoi(v) : -1; }
    char err[2048];
    g->l0 = e2_l0_create(err, sizeof err);
    if (!g->l0) { fail(g, "Level Zero: %s", err); free(g); return NULL; }
    if (!m->use_int8) { fail(g, "the GPU backend needs the int8 weights (E2_INT8=0 disables them)"); goto bad; }
    layout(g);

    e2_audio *au = (e2_audio *)m->audio;
    if (au && e2_audio_prepare(m)) goto bad;
    const int ntensors = m->n_layers * E2_DS_PER_LAYER + 2 + E2_V_LAYERS * 7 + 2 +
                         (au ? E2_A_LAYERS * E2_ADS_PER_LAYER + 3 : 0);
    g->vdesc_base = m->n_layers * E2_DS_PER_LAYER + 2;
    g->adesc_base = g->vdesc_base + E2_V_LAYERS * 7 + 2;
    size_t wbytes = 0;
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        wbytes += q8w_bytes(&L->q8_q) + q8w_bytes(&L->q8_k) + q8w_bytes(&L->q8_v) +
                  q8w_bytes(&L->q8_o) + q8w_bytes(&L->q8_gate) + q8w_bytes(&L->q8_up) +
                  q8w_bytes(&L->q8_down) + q8w_bytes(&L->q8_ple_gate) + q8w_bytes(&L->q8_ple_proj);
    }
    wbytes += q8w_bytes(&m->q8_ple_model) + q8w_bytes(&m->q8_embed_proj);
    /* the norm gammas (7 per layer + 2 global), the rope tables (2 x 2048 x 512 worst case) */
    wbytes += (size_t)(m->n_layers * 7 + 2) * 512 * 4;
    wbytes += (size_t)2 * (256 + 512) * E2_MAX_SEQ * 4;
    /* vision tower: the seven projections per layer, the two ends, the norm gammas, the axial rope
     * frequencies, and the 31 MB bf16 position table (2 x 10240 x 768), which is small enough to
     * keep resident rather than gather rows out of every call. */
    {
        e2_vision *v = (e2_vision *)m->vision;
        if (!v) { fail(g, "the model has no vision tower"); goto bad; }
        for (int l = 0; l < E2_V_LAYERS; l++) {
            const e2_vlayer *L = &v->layers[l];
            wbytes += vq8w_bytes(&L->q8_q) + vq8w_bytes(&L->q8_k) + vq8w_bytes(&L->q8_v) +
                      vq8w_bytes(&L->q8_o) + vq8w_bytes(&L->q8_gate) + vq8w_bytes(&L->q8_up) +
                      vq8w_bytes(&L->q8_down);
        }
        wbytes += vq8w_bytes(&v->q8_patch) + vq8w_bytes(&v->q8_embed_proj);
        wbytes += (size_t)(E2_V_LAYERS * 6 + 2) * 768 * 4;
        wbytes += (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2;
        wbytes += 16 * 4;
    }
    /* audio tower: 135 int8 projections (the DPAS layout only), nine norm gammas, the per-dim query
     * scale and the transposed depthwise kernel per layer, plus the twelve relative-key blocks
     * computed on the host once and the output-projection bias. */
    if (au) {
        for (int l = 0; l < E2_A_LAYERS; l++) {
            const e2_alayer *L = &au->layers[l];
            const e2_q8w *q[11] = { &L->q.q8, &L->k.q8, &L->v.q8, &L->post.q8,
                                    &L->ff1a.q8, &L->ff1b.q8, &L->ff2a.q8, &L->ff2b.q8,
                                    &L->lc_start.q8, &L->lc_end.q8, &L->q8_rel_k };
            for (int i = 0; i < 11; i++) wbytes += vq8w_bytes(q[i]);
            wbytes += (size_t)(9 * 1024 + 128 + E2_A_LCONV_K * 1024 + E2_A_RELPOS * 1024) * 4;
        }
        const e2_q8w *gq[3] = { &au->in_proj.q8, &au->out_proj.q8, &au->embed.q8 };
        for (int i = 0; i < 3; i++) wbytes += vq8w_bytes(gq[i]);
        wbytes += (size_t)E2_A_OUT * 4;
    }
    wbytes = align64(wbytes) + (1u << 20);
    g->wbuf_bytes = wbytes;

    uint8_t *wbuf = calloc(1, wbytes);
    int32_t *desc = calloc((size_t)ntensors * E2_DS_STRIDE, 4);
    if (!wbuf || !desc) { free(wbuf); free(desc); fail(g, "oom for %zu bytes of weights", wbytes); goto bad; }

    size_t woff = 0;
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        const int b = l * E2_DS_PER_LAYER;
        if (upload_tensor(g, wbuf, &woff, desc, b + E2_DS_Q, &L->q8_q, A_Q, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_K, &L->q8_k, A_K2, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_V, &L->q8_v, A_V2, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_O, &L->q8_o, A_TMP, A_QAC, A_QACS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_GATE, &L->q8_gate, A_GATE, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_UP, &L->q8_up, A_UP, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_DOWN, &L->q8_down, A_TMP, A_QAM, A_QAMS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_PLE_GATE, &L->q8_ple_gate, A_GATE, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, b + E2_DS_PLE_PROJ, &L->q8_ple_proj, A_TMP, A_QAP, A_QAPS))
            goto bad;
    }
    {
        const int gb = m->n_layers * E2_DS_PER_LAYER;
        if (upload_tensor(g, wbuf, &woff, desc, gb, &m->q8_ple_model, A_PLE, A_QAX, A_QAXS) ||
            upload_tensor(g, wbuf, &woff, desc, gb + 1, &m->q8_embed_proj, A_EMB, A_QAX, A_QAXS))
            goto bad;
    }
    for (int l = 0; l < m->n_layers; l++) {
        const e2_layer *L = &m->layers[l];
        const float *ps[7] = { L->q_norm, L->k_norm,      L->input_ln, L->post_attn_ln,
                               L->pre_ffn_ln, L->post_ffn_ln, L->ple_norm };
        const int ns[7] = { L->head_dim, L->head_dim, 512, 512, 512, 512, 512 };
        for (int i = 0; i < 7; i++) {
            woff = align64(woff);
            g->noff[l].n[i] = (int)woff;
            memcpy(wbuf + woff, ps[i], sizeof(float) * (size_t)ns[i]);
            woff += (size_t)ns[i] * 4;
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
    /* ---- vision tower: weights, norm gammas, axial rope frequencies, position table */
    {
        e2_vision *v = (e2_vision *)m->vision;
        static const int vy[7] = { A_VQ, A_VK, A_VV, A_VTMP, A_VGATE, A_VUP, A_VTMP };
        static const int vqa[7] = { A_VQX, A_VQX, A_VQX, A_VQH, A_VQX, A_VQX, A_VQM };
        static const int vasc[7] = { A_VQXS, A_VQXS, A_VQXS, A_VQHS, A_VQXS, A_VQXS, A_VQMS };
        for (int l = 0; l < E2_V_LAYERS; l++) {
            const e2_vlayer *L = &v->layers[l];
            const e2_q8w *q[7] = { &L->q8_q, &L->q8_k, &L->q8_v, &L->q8_o,
                                   &L->q8_gate, &L->q8_up, &L->q8_down };
            for (int s = 0; s < 7; s++)
                if (vupload_tensor(g, wbuf, &woff, desc, g->vdesc_base + l * 7 + s, q[s], vy[s],
                                   vqa[s], vasc[s]))
                    goto bad;
            const float *ps[6] = { L->input_ln, L->post_attn_ln, L->pre_ffn_ln, L->post_ffn_ln,
                                   L->q_norm, L->k_norm };
            const int ns[6] = { 768, 768, 768, 768, 64, 64 };
            for (int i = 0; i < 6; i++) {
                woff = align64(woff);
                g->vnoff[l].n[i] = (int)woff;
                memcpy(wbuf + woff, ps[i], sizeof(float) * (size_t)ns[i]);
                woff += (size_t)768 * 4; /* one stride for all six: the head norms fit in a hidden */
            }
        }
        const int vb = g->vdesc_base + E2_V_LAYERS * 7;
        if (vupload_tensor(g, wbuf, &woff, desc, vb, &v->q8_patch, A_VRES, A_VQX, A_VQXS) ||
            vupload_tensor(g, wbuf, &woff, desc, vb + 1, &v->q8_embed_proj, A_VSOFT, A_VPP, A_VPPS))
            goto bad;
        woff = align64(woff);
        g->vropeinv_off = (int)woff;
        memcpy(wbuf + woff, v->rope_inv, sizeof(float) * 16);
        woff += 16 * 4;
        woff = align64(woff);
        g->vpostab_off = (int)woff;
        memcpy(wbuf + woff, v->pos_table, (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2);
        woff += (size_t)2 * E2_V_POS * E2_V_HIDDEN * 2;
    }
    /* ---- audio tower. The relative-position projection is the one weight that is cheaper to run
     * once on the host than to replay per layer on the device: it is 13 rows against a 1024x1024
     * matrix, twelve times, and its input (the sinusoidal position table) never changes. */
    if (au) {
        static const int ay[11] = { A_AQ, A_AK, A_AV, A_ATMP, A_AFF, A_ATMP, A_AFF, A_ATMP,
                                    A_ALC, A_ATMP, A_ARELK };
        for (int l = 0; l < E2_A_LAYERS; l++) {
            const e2_alayer *L = &au->layers[l];
            const e2_q8w *q[11] = { &L->q.q8, &L->k.q8, &L->v.q8, &L->post.q8,
                                    &L->ff1a.q8, &L->ff1b.q8, &L->ff2a.q8, &L->ff2b.q8,
                                    &L->lc_start.q8, &L->lc_end.q8, &L->q8_rel_k };
            for (int i = 0; i < 11; i++)
                if (vupload_tensor(g, wbuf, &woff, desc, g->adesc_base + l * E2_ADS_PER_LAYER + i,
                                   q[i], ay[i], A_AQA, A_AQAS))
                    goto bad;
            const float *ps[9] = { L->n_pre_attn, L->n_post_attn, L->n_out,
                                   L->n_ff1_pre, L->n_ff1_post, L->n_ff2_pre, L->n_ff2_post,
                                   L->n_lc_pre, L->n_lc_conv };
            for (int i = 0; i < 9; i++) {
                woff = align64(woff);
                g->anoff[l].n[i] = (int)woff;
                memcpy(wbuf + woff, ps[i], sizeof(float) * 1024);
                woff += 1024 * 4;
            }
            woff = align64(woff);
            g->aqmult_off[l] = (int)woff;
            memcpy(wbuf + woff, L->qmult, sizeof(float) * E2_A_HEAD_DIM);
            woff += (size_t)E2_A_HEAD_DIM * 4;
            woff = align64(woff);
            g->alc_conv_off[l] = (int)woff;
            memcpy(wbuf + woff, L->lc_conv_t, sizeof(float) * E2_A_LCONV_K * 1024);
            woff += (size_t)E2_A_LCONV_K * 1024 * 4;
        }
        const int ab = g->adesc_base + E2_A_LAYERS * E2_ADS_PER_LAYER;
        if (vupload_tensor(g, wbuf, &woff, desc, ab, &au->in_proj.q8, A_AX, A_AQA, A_AQAS) ||
            vupload_tensor(g, wbuf, &woff, desc, ab + 1, &au->out_proj.q8, A_ATMP, A_AQA, A_AQAS) ||
            vupload_tensor(g, wbuf, &woff, desc, ab + 2, &au->embed.q8, A_ASOFT, A_AQA, A_AQAS))
            goto bad;
        woff = align64(woff);
        g->aout_bias_off = (int)woff;
        for (int i = 0; i < E2_A_OUT; i++) {
            const float b = e2_bf16f(au->out_proj_b[i]);
            memcpy(wbuf + woff + (size_t)i * 4, &b, 4);
        }
        woff += (size_t)E2_A_OUT * 4;

        int8_t *pq = malloc((size_t)E2_A_RELPOS * 1024);
        float *psc = malloc(sizeof(float) * (size_t)E2_A_RELPOS * 8);
        float *relk = malloc(sizeof(float) * (size_t)E2_A_RELPOS * 1024);
        if (!pq || !psc || !relk) { free(pq); free(psc); free(relk); fail(g, "oom: audio relk"); goto bad; }
        e2_i8_quantize(pq, psc, au->pos, E2_A_RELPOS, 1024, -INFINITY, INFINITY);
        for (int l = 0; l < E2_A_LAYERS; l++) {
            e2_i8_gemm(relk, pq, psc, &au->layers[l].q8_rel_k, E2_A_RELPOS, 1024, 0, 1024);
            woff = align64(woff);
            g->arelk_off[l] = (int)woff;
            memcpy(wbuf + woff, relk, sizeof(float) * (size_t)E2_A_RELPOS * 1024);
            woff += (size_t)E2_A_RELPOS * 1024 * 4;
        }
        free(pq); free(psc); free(relk);
    }
    if (woff > wbytes) { free(desc); free(wbuf); fail(g, "weight buffer overflow"); goto bad; }
    if (getenv("E2_GPU_AUDIO_DEBUG") && au) {
        const float *g0 = (const float *)(wbuf + g->anoff[0].n[2]);
        printf("[audio] anoff0 %d %d %d %d %d %d %d %d %d n_out=%g %g %g %g\n",
               g->anoff[0].n[0], g->anoff[0].n[1], g->anoff[0].n[2], g->anoff[0].n[3],
               g->anoff[0].n[4], g->anoff[0].n[5], g->anoff[0].n[6], g->anoff[0].n[7],
               g->anoff[0].n[8], g0[0], g0[1], g0[2], g0[3]);
        const float *qm = (const float *)(wbuf + g->aqmult_off[0]);
        printf("[audio] qmult %g %g  arelk0 %g\n", qm[0], qm[1], ((const float *)(wbuf + g->arelk_off[0]))[0]);
    }

    g->desc_tensors = ntensors;
    /* 3 tensors per launch at most, 4 k-slices, 8 tokens, and the widest projection (the PLE one,
     * 12288 wide). Undersizing this overflowed the buffer next to it and quietly corrupted the
     * embeddings - worth the few megabytes. */
    const size_t part_bytes = (size_t)3 * E2_GPU_MAX_NSL * E2_NM * 12288 * 4;
    g->wbuf = e2_l0_alloc(g->l0, wbytes);
    g->acts = e2_l0_alloc(g->l0, g->acts_bytes);
    g->part = e2_l0_alloc(g->l0, part_bytes);
    g->part_bytes = part_bytes;
    g->desc = e2_l0_alloc(g->l0, (size_t)ntensors * E2_DS_STRIDE * 4);
    g->toktab = e2_l0_alloc(g->l0, (size_t)E2_VOCAB * m->hidden * 2);
    g->ids = e2_l0_alloc_shared(g->l0, sizeof(int32_t) * (size_t)E2_GPU_MAX_ROWS);
    g->valid = e2_l0_alloc_shared(g->l0, E2_GPU_VP);
    g->embout = e2_l0_alloc_shared(g->l0, sizeof(float) * 768 * (size_t)E2_GPU_MAX_ROWS);
    /* per-call vision uploads: the raw patches, their position ids, the valid-patch gather and the
     * axial rope tables, plus the pooler's CSR lists (a patch list and a group offset array). */
    g->vpix = e2_l0_alloc(g->l0, sizeof(float) * (size_t)E2_V_MAX_PATCHES * E2_V_PATCH);
    g->vpos = e2_l0_alloc(g->l0, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES * 2);
    g->vorder = e2_l0_alloc(g->l0, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES);
    g->vgpatch = e2_l0_alloc(g->l0, sizeof(int32_t) * (size_t)E2_V_MAX_PATCHES);
    g->vgstart = e2_l0_alloc(g->l0, sizeof(int32_t) * (E2_GPU_VGROUPS + 1));
    g->vpool_ng = e2_l0_alloc_shared(g->l0, sizeof(int32_t));
    g->vrope = e2_l0_alloc(g->l0, sizeof(float) * 2 * (size_t)E2_GPU_VP * E2_V_HEAD_DIM);
    g->axin = e2_l0_alloc_shared(g->l0, sizeof(float) * (size_t)E2_A_MAX_TOK * 1024);
    g->asoft = e2_l0_alloc_shared(g->l0, sizeof(float) * (size_t)E2_A_MAX_TOK * 512);
    if (!g->wbuf || !g->acts || !g->part || !g->desc || !g->toktab || !g->ids || !g->valid ||
        !g->embout || !g->vpix || !g->vpos || !g->vorder || !g->vgpatch || !g->vgstart ||
        !g->vpool_ng || !g->vrope || !g->axin || !g->asoft) {
        free(desc); free(wbuf); fail(g, "device allocation failed"); goto bad;
    }
    if (e2_l0_write(g->l0, g->wbuf, wbytes, wbuf) ||
        e2_l0_write(g->l0, g->desc, (size_t)ntensors * E2_DS_STRIDE * 4, desc) ||
        e2_l0_write(g->l0, g->toktab, (size_t)E2_VOCAB * m->hidden * 2, m->embed_tokens)) {
        free(desc); free(wbuf); fail(g, "upload failed: %s", e2_l0_error(g->l0)); goto bad;
    }
    if (getenv("E2_GPU_DESC")) {
        static const char *const rn[A_NREGION] = {
            [A_X] = "X",     [A_RES] = "RES",   [A_Q] = "Q",     [A_V] = "V",
            [A_CTX] = "CTX", [A_GATE] = "GATE", [A_UP] = "UP",   [A_TMP] = "TMP",
            [A_PLE] = "PLE", [A_EMB] = "EMB",   [A_K2] = "K2",   [A_V2] = "V2",
            [A_QAX] = "QAX", [A_QAXS] = "QAXS", [A_QAC] = "QAC", [A_QACS] = "QACS",
            [A_QAM] = "QAM", [A_QAMS] = "QAMS", [A_QAP] = "QAP", [A_QAPS] = "QAPS",
            [A_VX] = "VX",   [A_VRES] = "VRES", [A_VTMP] = "VTMP", [A_VQ] = "VQ",
            [A_VK] = "VK",   [A_VV] = "VV",   [A_VCTX] = "VCTX", [A_VQX] = "VQX",
            [A_VQM] = "VQM", [A_VPP] = "VPP", [A_VQPK] = "VQPK", [A_VKPK] = "VKPK",
            [A_VVPK] = "VVPK", [A_VSCR] = "VSCR", [A_VPB] = "VPB",
            [A_VPOOL] = "VPOOL", [A_VSOFT] = "VSOFT", [A_VEMBIN] = "VEMBIN",
            [A_AX] = "AX", [A_ARES] = "ARES", [A_ATMP] = "ATMP", [A_AFF] = "AFF",
            [A_ALC] = "ALC", [A_AGLU] = "AGLU", [A_AQ] = "AQ", [A_AK] = "AK", [A_AV] = "AV",
            [A_AATT] = "AATT", [A_AQA] = "AQA", [A_AQAS] = "AQAS", [A_ASOFT] = "ASOFT",
            [A_ARELK] = "ARELK",
        };
        printf("[gpu] regions:");
        for (int i = 0; i < A_NREGION; i++) printf(" %s@%zu", rn[i] ? rn[i] : "?", g->reg[i].off);
        printf("\n[gpu] weight bytes %zu, acts %zu, used %zu\n", wbytes, g->acts_bytes, woff);
        for (int i = 0; i < 9; i++) {
            const int32_t *dd = desc + i * E2_DS_STRIDE;
            printf("[gpu] desc[%d] w=%d wsc=%d bsq=%d y=%d N=%d K=%d nb=%d qa=%d asc=%d\n", i, dd[0],
                   dd[1], dd[2], dd[3], dd[4], dd[5], dd[6], dd[7], dd[8]);
        }
        }
    free(desc);
    free(wbuf);
    return g;
bad:
    e2_gpu_destroy(g);
    return NULL;
}

void e2_gpu_destroy(e2_gpu *g) {
    if (!g) return;
    for (int i = 0; i < g->nslots; i++) e2_l0_graph_free(g->l0, g->slots[i].gr);
    if (g->l0) {
        e2_l0_free(g->l0, g->wbuf);
        e2_l0_free(g->l0, g->acts);
        e2_l0_free(g->l0, g->part);
        e2_l0_free(g->l0, g->vpix);
        e2_l0_free(g->l0, g->vpos);
        e2_l0_free(g->l0, g->vorder);
        e2_l0_free(g->l0, g->vgpatch);
        e2_l0_free(g->l0, g->vgstart);
        e2_l0_free(g->l0, g->vpool_ng);
        e2_l0_free(g->l0, g->vrope);
        e2_l0_free(g->l0, g->axin);
        e2_l0_free(g->l0, g->asoft);
        e2_l0_free(g->l0, g->desc);
        e2_l0_free(g->l0, g->toktab);
        e2_l0_free(g->l0, g->ids);
        e2_l0_free(g->l0, g->valid);
        e2_l0_free(g->l0, g->embout);
    }
    e2_l0_destroy(g->l0);
    free(g);
}

const char *e2_gpu_device(e2_gpu *g) { return e2_l0_device_name(g->l0); }

int e2_gpu_stats(e2_gpu *g, long *calls, long *kernels_last, double *last_ms) {
    if (calls) *calls = g->calls;
    if (kernels_last) *kernels_last = g->kernels_last; /* kernels of the most recent shape */
    if (last_ms) *last_ms = g->last_ms;
    return 0;
}

/* ---------------------------------------------------------------- graph */

static void *woff_ptr(e2_gpu *g, int off) { return (char *)g->wbuf + off; }
static void *aoff_ptr(e2_gpu *g, int region) { return (char *)g->acts + g->reg[region].off; }

/* Sub-groups per DPAS work group. The small-M single-sequence path and the large-M batch/image path
 * want different values (see E2_GEMM_GY in e2_tile.h); E2_GPU_GY forces one for the A/B. */
static int gemm_gy(int rows) {
    const char *e = getenv("E2_GPU_GY");
    if (e) return atoi(e);
    /* A one-query call is 16 tokens of work in an E2_DM-tall tile: at 32 sub-groups every work group
     * is mostly epilogue, and 8 measured 6.37 -> 6.08 ms end to end at seq 17. */
    if (rows <= 64) return 8;
    return rows <= 512 ? E2_GEMM_GY_SMALL : E2_GEMM_GY;
}

/* One or more int8 projections that share their activation input, in one launch (plus the k-slice
 * reduction for the short-sequence path). */
static int gemm(e2_gpu *g, e2_graph *gr, int rows, int first, int count, int nmax, int tiled) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    void *wbuf = g->wbuf, *acts = g->acts, *part = g->part;
    void *d = (char *)g->desc + (size_t)first * E2_DS_STRIDE * 4;
    /* The split+reduce pair only pays when the tile would be almost entirely padding: it costs two
     * launches and 16 activation loads per 4 weight loads, where k_gemm_dpas2 needs 3 per k-step.
     * Measured seq 16 (pair) 7.20 ms vs seq 17 (DPAS) 6.37 ms, and 146 fewer launches. */
    if (rows <= 4 && !tiled) {
        const int nsl = 4; /* four k-slices: enough work items to fill 128 EUs at N=512 */
        const uint32_t gx = (uint32_t)((nmax + 127) / 128);
        if (e2_l0_append(g->l0, gr, "k_gemv_split", gx, (uint32_t)nsl, (uint32_t)count, 128, 1, 1, 7,
                         &wbuf, sizeof wbuf, &acts, sizeof acts, &part, sizeof part, &d, sizeof d,
                         &nsl, sizeof nsl, &rows, sizeof rows, &nmax, sizeof nmax))
            return -1;
        if (e2_l0_append(g->l0, gr, "k_gemv_reduce", gx, (uint32_t)count, 1, 128, 1, 1, 6, &acts,
                         sizeof acts, &part, sizeof part, &d, sizeof d, &nsl, sizeof nsl, &rows,
                         sizeof rows, &nmax, sizeof nmax))
            return -1;
        g->kernels_last += 2;
        return 0;
    }
    /* Padded row count: k_gemm_dpas loads its activations unguarded over the E2_DM-tall tile, so the
     * grid rounds M up to E2_DM. The extra rows compute but are never stored (the kernel guards on the
     * real `rows`), and the activation buffer is sized for E2_GPU_MAX_ROWS = 2048, a multiple of E2_DM,
     * so the reads stay inside the region. */
    if (g->gemm_dp4a) {
        const int mpad = (rows + E2_BM - 1) / E2_BM * E2_BM;
        if (e2_l0_append(g->l0, gr, "k_gemm_tiled", (uint32_t)(mpad / E2_BM),
                         (uint32_t)((nmax + E2_BN - 1) / E2_BN), (uint32_t)count, E2_GEMM_NG,
                         E2_GEMM_MG, 1, 4, &wbuf, sizeof wbuf, &acts, sizeof acts, &d, sizeof d, &mpad,
                         sizeof mpad))
            return -1;
        g->kernels_last++;
        return 0;
    }
    const int mpad = (rows + E2_DM - 1) / E2_DM * E2_DM;
    const int gy = gemm_gy(rows); /* sub-groups per work group, each E2_DN features */
    /* k_gemm_dpas2 is the same tile with a vectorised epilogue (vload8 scales + float8 accumulate);
     * E2_GPU_GEMM_OLD=1 selects the first version, which is the A/B switch. */
    const char *kname = getenv("E2_GPU_GEMM_OLD") ? "k_gemm_dpas" : "k_gemm_dpas2";
    if (e2_l0_append(g->l0, gr, kname, (uint32_t)(mpad / E2_DM),
                     (uint32_t)((nmax + E2_DN * gy - 1) / (E2_DN * gy)), (uint32_t)count, 8,
                     (uint32_t)gy, 1, 4, &wbuf, sizeof wbuf, &acts, sizeof acts, &d, sizeof d, &rows,
                     sizeof rows))
        return -1;
    g->kernels_last++;
    return 0;
}

static int rowops(e2_gpu *g, e2_graph *gr, int rows, int K, int region_x, int region_src,
                  int region_res, int region_qa, int region_asc, int gamma_off, int gamma2_off,
                  int flags, float prescale, float scale) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    const uint32_t gy = 4; /* four sub-groups per work group */
    void *x = aoff_ptr(g, region_x);
    void *src = region_src >= 0 ? aoff_ptr(g, region_src) : NULL;
    void *res = region_res >= 0 ? aoff_ptr(g, region_res) : NULL;
    void *qa = region_qa >= 0 ? aoff_ptr(g, region_qa) : NULL;
    void *asc = region_asc >= 0 ? aoff_ptr(g, region_asc) : NULL;
    void *gam = gamma_off >= 0 ? woff_ptr(g, gamma_off) : NULL;
    void *gam2 = gamma2_off >= 0 ? woff_ptr(g, gamma2_off) : NULL;
    const int nb = K / 128;
    const float clo = 0.f, chi = 0.f;
    void *valid = g->valid;
    if (e2_l0_append(g->l0, gr, "k_rowops", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 17, &x,
                     sizeof x, &src, sizeof src, &res, sizeof res, &qa, sizeof qa, &asc, sizeof asc,
                     &gam, sizeof gam, &gam2, sizeof gam2, &g->m->rms_eps, sizeof g->m->rms_eps,
                     &prescale, sizeof prescale, &scale, sizeof scale, &rows, sizeof rows, &K, sizeof K,
                     &nb, sizeof nb, &flags, sizeof flags, &clo, sizeof clo, &chi, sizeof chi,
                     &valid, sizeof valid))
        return -1;
    g->kernels_last++;
    return 0;
}

/* Audio rows are never masked (the front end takes only complete mel windows), and every projection
 * in the tower gets an explicit input clamp. This is k_rowops without the text/vision arguments that
 * do not apply, with the clamp carried into the same pass as the norm and the int8 quantization. */
static int arowops(e2_gpu *g, e2_graph *gr, int rows, int K, int region_x, int region_src,
                   int region_res, int region_qa, int gamma_off, int flags, float clo, float chi,
                   float scale) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    const uint32_t gy = 4;
    void *x = aoff_ptr(g, region_x);
    void *src = region_src >= 0 ? aoff_ptr(g, region_src) : NULL;
    void *res = region_res >= 0 ? aoff_ptr(g, region_res) : NULL;
    void *qa = region_qa >= 0 ? aoff_ptr(g, region_qa) : NULL;
    void *asc = region_qa >= 0 ? aoff_ptr(g, A_AQAS) : NULL;
    void *gam = gamma_off >= 0 ? woff_ptr(g, gamma_off) : NULL;
    void *gam2 = NULL;
    const int nb = K / 128;
    const float prescale = 1.f;
    void *valid = NULL;
    if (e2_l0_append(g->l0, gr, "k_rowops", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 17, &x,
                     sizeof x, &src, sizeof src, &res, sizeof res, &qa, sizeof qa, &asc, sizeof asc,
                     &gam, sizeof gam, &gam2, sizeof gam2, &g->m->rms_eps, sizeof g->m->rms_eps,
                     &prescale, sizeof prescale, &scale, sizeof scale, &rows, sizeof rows, &K, sizeof K,
                     &nb, sizeof nb, &flags, sizeof flags, &clo, sizeof clo, &chi, sizeof chi,
                     &valid, sizeof valid))
        return -1;
    g->kernels_last++;
    return 0;
}

/* The audio GEMM always uses the XMX tile, including for a 19-token clip: k_gemv_split's ceiling is
 * 16 rows, and a 19-row audio clip is common enough that the two-row overflow would otherwise select
 * a kernel with the wrong token count. */
static int ageam(e2_gpu *g, e2_graph *gr, int rows, int first, int count, int nmax) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    void *wbuf = g->wbuf, *acts = g->acts, *part = g->part;
    void *d = (char *)g->desc + (size_t)first * E2_DS_STRIDE * 4;
    const char *ge = getenv("E2_AUDIO_GY");
    const int lgy = ge ? atoi(ge) : (rows <= 128 ? 8 : 32);
    if (getenv("E2_AUDIO_DPAS_OLD")) return gemm(g, gr, rows, first, count, nmax, 1);
    const int ntile = (lgy / E2_AMG) * 16;
    const int mtile = E2_DM * E2_AMG; /* rows one work group covers */
    int ksplit = rows <= 128 ? 2 : 1;
    const char *ke = getenv("E2_AUDIO_KSPLIT");
    if (ke) ksplit = atoi(ke);
    if (ksplit < 1) ksplit = 1;
    int pstride = rows * nmax;
    while (ksplit > 1 && (size_t)count * ksplit * pstride * 4 > g->part_bytes) ksplit >>= 1;
    if (ksplit <= 1) {
        if (e2_l0_append(g->l0, gr, "k_audio_dpas4", (uint32_t)((rows + mtile - 1) / mtile),
                         (uint32_t)((nmax + ntile - 1) / ntile), (uint32_t)count, 8, (uint32_t)lgy,
                         1, 7, &wbuf, sizeof wbuf, &acts, sizeof acts, &part, sizeof part, &d,
                         sizeof d, &rows, sizeof rows, &ksplit, sizeof ksplit, &pstride,
                         sizeof pstride))
            return -1;
        g->kernels_last++;
        return 0;
    }
    if (e2_l0_append(g->l0, gr, "k_audio_dpas4", (uint32_t)((rows + mtile - 1) / mtile),
                     (uint32_t)((nmax + ntile - 1) / ntile), (uint32_t)(count * ksplit), 8,
                     (uint32_t)lgy, 1, 7, &wbuf, sizeof wbuf, &acts, sizeof acts, &part,
                     sizeof part, &d, sizeof d, &rows, sizeof rows, &ksplit, sizeof ksplit,
                     &pstride, sizeof pstride))
        return -1;
    if (e2_l0_append(g->l0, gr, "k_audio_reduce_k", (uint32_t)((nmax + 127) / 128),
                     (uint32_t)count, 1, 128, 1, 1, 8, &part, sizeof part, &acts, sizeof acts,
                     &d, sizeof d, &count, sizeof count, &ksplit, sizeof ksplit, &rows,
                     sizeof rows, &nmax, sizeof nmax, &pstride, sizeof pstride))
        return -1;
    g->kernels_last += 2;
    return 0;
}

static int record(e2_gpu *g, int rows, int seq_len, int from_embeds, e2_graph **out) {
    e2_model *m = g->m;
    e2_graph *gr = e2_l0_graph_begin(g->l0, 1);
    if (!gr) return fail(g, "command list: %s", e2_l0_error(g->l0));
    g->kernels_last = 0;
    void *toktab = g->toktab, *ids = g->ids, *x = aoff_ptr(g, A_X), *emb = aoff_ptr(g, A_EMB);
    void *valid = g->valid;
    const float embed_scale = m->embed_scale, ple_scale = m->ple_scale;
    const float eps = m->rms_eps;

/* g->limit (E2_GPU_LIMIT=n) stops the recording after n kernels: with the fixed order this bisects
 * the graph against the CPU engine's E2_DUMP activations one kernel at a time. */
#define KERN(name, ...) \
    do { \
        if (g->limit && g->kernels_last >= g->limit) goto done; \
        if (e2_l0_append(g->l0, gr, name, __VA_ARGS__)) goto fail2; \
        g->kernels_last++; \
    } while (0)

    /* input embeddings: token ids -> x (bf16 table, scaled), or a copy of the block the caller
     * built in A_VEMBIN (the image path: soft tokens have no token id) */
    if (from_embeds) {
        void *src = aoff_ptr(g, A_VEMBIN);
        const int hidden = 512;
        KERN("k_copy_rows", (uint32_t)((rows + 3) / 4), 2, 1, 64, 4, 1, 4, &x, sizeof x, &src,
             sizeof src, &rows, sizeof rows, &hidden, sizeof hidden);
    } else {
        KERN("k_embed", (uint32_t)((rows + 3) / 4), (uint32_t)(512 / 4 / 64), 1, 64, 4, 1, 7, &x,
             sizeof x, &toktab, sizeof toktab, &ids, sizeof ids, &valid, sizeof valid, &rows,
             sizeof rows, &m->hidden, sizeof m->hidden, &embed_scale, sizeof embed_scale);
    }
    if (g->only == 1) goto done; /* diagnostic: only the embedding gather */
    /* quantize the input for the PLE projection (and the first layer re-quantizes after its norm) */
    if (rowops(g, gr, rows, 512, A_X, -1, -1, A_QAX, A_QAXS, -1, -1, 16 | 2, 1.f, 1.f)) goto fail2;
    if (g->only == 2) goto done;
    /* PLE: project to 24 x 512 per token, then normalize slices (not token-mask rows). */
    if (gemm(g, gr, rows, m->n_layers * E2_DS_PER_LAYER, 1, m->n_layers * 512, 0)) goto fail2;
    if (g->only == 3) goto done;
    if (rowops(g, gr, rows * m->n_layers, 512, A_PLE, -1, -1, -1, -1, g->ple_proj_norm_off, -1, 64,
               ple_scale, 1.f))
        goto fail2;

    /* only the region dump reads the fp32 gelu results; the projections read the int8 copies */
    const int wr = g->dump_dir != NULL;
    const int nlayers = g->layer_limit > 0 && g->layer_limit < m->n_layers ? g->layer_limit
                                                                              : m->n_layers;
    for (int l = 0; l < nlayers; l++) {
        const e2_layer *L = &m->layers[l];
        const int base = l * E2_DS_PER_LAYER;
        const int rope = L->sliding ? 0 : 1;
        const int D = L->head_dim, H = E2_NUM_HEADS, KV = L->n_kv_heads;
        /* 1. residual copy + input norm + activation quantization */
        if (rowops(g, gr, rows, 512, A_X, -1, A_RES, A_QAX, A_QAXS, g->noff[l].n[2], -1, 1 | 2, 1.f, 1.f))
            goto fail2;
        /* 2. q | k | v from the normed input */
        if (gemm(g, gr, rows, base + E2_DS_Q, 3, H * D, 0)) goto fail2;
        /* 3. per-head RMSNorm + rotary embedding, q and k with rope, v without */
        {
            void *xq = aoff_ptr(g, A_Q), *xk = aoff_ptr(g, A_K2), *xv = aoff_ptr(g, A_V2);
            void *gq = woff_ptr(g, g->noff[l].n[0]), *gk = woff_ptr(g, g->noff[l].n[1]);
            void *gv = NULL; /* v_norm has no learned scale */
            void *cs = woff_ptr(g, g->cos_off[rope]), *sn = woff_ptr(g, g->sin_off[rope]);
            const int rq = rows * H, rk = rows * KV, rv = rows * KV;
            KERN("k_norm_rope3", (uint32_t)((rq + 1) / 2), 1, 1, 32, 2, 1, 16, &xq, sizeof xq, &gq,
                 sizeof gq, &rq, sizeof rq, &xk, sizeof xk, &gk, sizeof gk, &rk, sizeof rk, &xv,
                 sizeof xv, &gv, sizeof gv, &rv, sizeof rv, &cs, sizeof cs, &sn, sizeof sn, &eps,
                 sizeof eps, &H, sizeof H, &KV, sizeof KV, &D, sizeof D, &seq_len, sizeof seq_len);
        }
        /* 4. attention: context, quantized on the way out for the o projection */
        {
            const int window = L->sliding ? L->window : 0;
            const int kv_stride = KV * D, q_stride = H * D;
            /* Pack all sequence/KV-heads together, then run each query tile across all
             * sequence/query-heads. The scratch cap bounds concurrent scores/P, not input shapes. */
            if (g->attn_dpas && seq_len >= 64) {
                const int nseq_a = rows / seq_len, nz = nseq_a * H;
                const int sup = (seq_len + 127) / 128 * 128;
                const int nbv = sup / 128, gy = 4, one = 1;
                if (rowops(g, gr, rows * H, D, A_Q, -1, -1, A_QPK, A_QPS, -1, -1, 16 | 2 | 64, 1.f,
                           1.f))
                    goto fail2;
                void *krow = aoff_ptr(g, A_K2), *vrow = aoff_ptr(g, A_V2);
                void *kd = aoff_ptr(g, A_KPK), *vd = aoff_ptr(g, A_VPK);
                void *ksc = aoff_ptr(g, A_KSC), *vsc = aoff_ptr(g, A_VSC);
                void *kbq = aoff_ptr(g, A_KBSQ), *vbq = aoff_ptr(g, A_VBSQ);
                KERN("k_attn_packw", (uint32_t)((sup + gy - 1) / gy), 1, (uint32_t)(nseq_a * KV),
                     32, gy, 1, 13, &krow, sizeof krow, &kd, sizeof kd, &ksc, sizeof ksc,
                     &kbq, sizeof kbq, &sup, sizeof sup, &D, sizeof D, &kv_stride, sizeof kv_stride,
                     &one, sizeof one, &seq_len, sizeof seq_len, &D, sizeof D,
                     &seq_len, sizeof seq_len, &KV, sizeof KV, &valid, sizeof valid);
                KERN("k_attn_packw", (uint32_t)((D + gy - 1) / gy), 1, (uint32_t)(nseq_a * KV),
                     32, gy, 1, 13, &vrow, sizeof vrow, &vd, sizeof vd, &vsc, sizeof vsc,
                     &vbq, sizeof vbq, &D, sizeof D, &sup, sizeof sup, &one, sizeof one,
                     &kv_stride, sizeof kv_stride, &D, sizeof D, &seq_len, sizeof seq_len,
                     &seq_len, sizeof seq_len, &KV, sizeof KV, &valid, sizeof valid);
                size_t cap = g->reg[A_SCR].bytes / (sizeof(float) * (size_t)nz * sup);
                const size_t pcap = g->reg[A_PB].bytes / ((size_t)nz * sup);
                const size_t scap = g->reg[A_PBS].bytes / (sizeof(float) * (size_t)nz * nbv);
                if (cap > pcap) cap = pcap;
                if (cap > scap) cap = scap;
                const int mpad = (seq_len + E2_DM - 1) / E2_DM * E2_DM;
                const int tile = cap < (size_t)mpad ? (int)(cap / E2_DM) * E2_DM : mpad;
                if (tile < E2_DM) { fail(g, "attention scratch too small"); goto fail2; }
                void *qp = aoff_ptr(g, A_QPK), *qps = aoff_ptr(g, A_QPS);
                void *scr = aoff_ptr(g, A_SCR), *pb = aoff_ptr(g, A_PB);
                void *pasc = aoff_ptr(g, A_PBS), *ctxp = aoff_ptr(g, A_CTX);
                for (int q0 = 0; q0 < seq_len; q0 += tile) {
                    const int live = seq_len - q0 < tile ? seq_len - q0 : tile;
                    const int scores = 0, context = 1;
                    KERN("k_attn_batch_gemm", (uint32_t)(tile / E2_DM),
                         (uint32_t)((sup + E2_DN * gy - 1) / (E2_DN * gy)), (uint32_t)nz,
                         8, gy, 1, 14, &kd, sizeof kd, &ksc, sizeof ksc, &kbq, sizeof kbq,
                         &qp, sizeof qp, &qps, sizeof qps, &scr, sizeof scr,
                         &seq_len, sizeof seq_len, &sup, sizeof sup, &D, sizeof D, &H, sizeof H,
                         &KV, sizeof KV, &q0, sizeof q0, &tile, sizeof tile, &scores, sizeof scores);
                    KERN("k_attn_softmax", (uint32_t)((live + gy - 1) / gy), 1, (uint32_t)nz,
                         32, gy, 1, 12, &scr, sizeof scr, &pb, sizeof pb, &pasc, sizeof pasc,
                         &valid, sizeof valid, &live, sizeof live, &sup, sizeof sup,
                         &seq_len, sizeof seq_len, &window, sizeof window, &H, sizeof H,
                         &nbv, sizeof nbv, &q0, sizeof q0, &tile, sizeof tile);
                    KERN("k_attn_batch_gemm", (uint32_t)(tile / E2_DM),
                         (uint32_t)((D + E2_DN * gy - 1) / (E2_DN * gy)), (uint32_t)nz,
                         8, gy, 1, 14, &vd, sizeof vd, &vsc, sizeof vsc, &vbq, sizeof vbq,
                         &pb, sizeof pb, &pasc, sizeof pasc, &ctxp, sizeof ctxp,
                         &seq_len, sizeof seq_len, &sup, sizeof sup, &D, sizeof D, &H, sizeof H,
                         &KV, sizeof KV, &q0, sizeof q0, &tile, sizeof tile, &context, sizeof context);
                }
                /* the context is the o projection's input: quantize it (k_attn used to do this) */
                if (rowops(g, gr, rows, q_stride, A_CTX, -1, -1, A_QAC, A_QACS, -1, -1, 16 | 2, 1.f,
                           1.f))
                    goto fail2;
                goto attn_done;
            }
            {
                void *ctx = aoff_ptr(g, A_CTX), *q = aoff_ptr(g, A_Q), *k = aoff_ptr(g, A_K2);
                void *v = aoff_ptr(g, A_V2);
                void *qa = aoff_ptr(g, A_QAC), *asc = aoff_ptr(g, A_QACS);
                const char *attn_kernel = "k_attn";
                KERN(attn_kernel, (uint32_t)((rows + 3) / 4), (uint32_t)H, 1, 32, 4, 1, 15, &ctx,
                     sizeof ctx, &q, sizeof q, &k, sizeof k, &v, sizeof v, &qa, sizeof qa, &asc,
                     sizeof asc, &valid, sizeof valid, &rows, sizeof rows, &H, sizeof H, &D,
                     sizeof D, &KV, sizeof KV, &q_stride, sizeof q_stride, &kv_stride,
                     sizeof kv_stride, &window, sizeof window, &seq_len, sizeof seq_len);
            }
        attn_done:;
        }
        /* 5. o projection from the quantized context */
        if (gemm(g, gr, rows, base + E2_DS_O, 1, 512, 0)) goto fail2;
        /* 6. attention residual, then the pre-FFN norm and its quantization (one pass) */
        if (rowops(g, gr, rows, 512, A_X, A_TMP, A_RES, A_QAX, A_QAXS, g->noff[l].n[3],
                   g->noff[l].n[4], 4 | 8 | 32 | 2, 1.f, 1.f))
            goto fail2;
        /* 7. gate | up */
        if (gemm(g, gr, rows, base + E2_DS_GATE, 2, 2048, 0)) goto fail2;
        /* 8. gelu(gate) * up, quantized for the down projection */
        {
            void *gate = aoff_ptr(g, A_GATE), *up = aoff_ptr(g, A_UP), *nople = NULL;
            void *qa = aoff_ptr(g, A_QAM), *asc = aoff_ptr(g, A_QAMS);
            const int K = m->intermediate, nb = K / 128, stride = 0, poff = 0;
            KERN("k_gelu_mul_quant", (uint32_t)(((size_t)rows * nb + 3) / 4), 1, 1, 32, 4, 1, 11,
                 &gate, sizeof gate, &up, sizeof up, &nople, sizeof nople, &qa, sizeof qa, &asc,
                 sizeof asc, &rows, sizeof rows, &K, sizeof K, &nb, sizeof nb, &stride, sizeof stride,
                 &poff, sizeof poff, &wr, sizeof wr);
        }
        /* 9. down projection */
        if (gemm(g, gr, rows, base + E2_DS_DOWN, 1, 512, 0)) goto fail2;
        /* 10. MLP residual + quantization of x for the PLE gate */
        if (rowops(g, gr, rows, 512, A_X, A_TMP, A_RES, A_QAX, A_QAXS, g->noff[l].n[5], -1, 4 | 8 | 2,
                   1.f, 1.f))
            goto fail2;
        /* 11. PLE gate */
        if (gemm(g, gr, rows, base + E2_DS_PLE_GATE, 1, 512, 0)) goto fail2;
        /* 12. gate = gelu(gate) * this layer's PLE slice, quantized */
        {
            void *gate = aoff_ptr(g, A_GATE), *ple = aoff_ptr(g, A_PLE), *noup = NULL;
            void *qa = aoff_ptr(g, A_QAP), *asc = aoff_ptr(g, A_QAPS);
            const int K = 512, nb = 4, stride = m->n_layers * 512, poff = l * 512;
            KERN("k_gelu_mul_quant", (uint32_t)(((size_t)rows * nb + 3) / 4), 1, 1, 32, 4, 1, 11,
                 &gate, sizeof gate, &noup, sizeof noup, &ple, sizeof ple, &qa, sizeof qa, &asc,
                 sizeof asc, &rows, sizeof rows, &K, sizeof K, &nb, sizeof nb, &stride, sizeof stride,
                 &poff, sizeof poff, &wr, sizeof wr);
        }
        /* 13. PLE projection */
        if (gemm(g, gr, rows, base + E2_DS_PLE_PROJ, 1, 512, 0)) goto fail2;
        /* 14. PLE residual (note the aliasing: the residual is x itself) + the layer scalar */
        if (rowops(g, gr, rows, 512, A_X, A_TMP, A_X, -1, -1, g->noff[l].n[6], -1, 4 | 8, 1.f,
                   L->layer_scalar))
            goto fail2;
    }
    if (g->layer_limit == 0) goto done; /* diagnostic: stop after the PLE prologue */
    /* final norm (quantized for the embedder projection), the projection, mean pool + L2 */
    if (rowops(g, gr, rows, 512, A_X, -1, -1, A_QAX, A_QAXS, g->final_norm_off, -1, 2, 1.f, 1.f))
        goto fail2;
    if (gemm(g, gr, rows, m->n_layers * E2_DS_PER_LAYER + 1, 1, 768, 0)) goto fail2;
    {
        void *out = g->embout;
        const int dim = 768, nseq = seq_len > 0 ? rows / seq_len : 1;
        KERN("k_pool_l2", (uint32_t)nseq, 1, 1, 256, 1, 1, 6, &out, sizeof out, &emb, sizeof emb,
             &valid, sizeof valid, &rows, sizeof rows, &dim, sizeof dim, &seq_len, sizeof seq_len);
    }
#undef KERN
done:
    if (e2_l0_graph_close(g->l0, gr)) goto fail2;
    *out = gr;
    return 0;
fail2:
    e2_l0_graph_free(g->l0, gr);
    return fail(g, "recording failed: %s", e2_l0_error(g->l0));
}

/* Calls complete synchronously, so the oldest graph has no in-flight users at eviction. */
static e2_slot *new_slot(e2_gpu *g) {
    if (g->nslots == (int)(sizeof g->slots / sizeof g->slots[0])) {
        e2_l0_graph_free(g->l0, g->slots[0].gr);
        g->nslots--;
        memmove(g->slots, g->slots + 1, (size_t)g->nslots * sizeof g->slots[0]);
    }
    return &g->slots[g->nslots];
}

static e2_graph *graph_for(e2_gpu *g, int rows, int seq_len, int from_embeds) {
    for (int i = 0; i < g->nslots; i++)
        if (g->slots[i].rows == rows && g->slots[i].seq_len == seq_len &&
            g->slots[i].from_embeds == from_embeds) {
            g->kernels_last = g->slots[i].kernels;
            return g->slots[i].gr;
        }
    e2_slot *sl = new_slot(g);
    e2_graph *gr = NULL;
    if (record(g, rows, seq_len, from_embeds, &gr)) return NULL;
    *sl = (e2_slot){rows, seq_len, from_embeds, gr, (int)g->kernels_last};
    g->nslots++;
    return gr;
}

/* ---------------------------------------------------------------- diagnostics
 * E2_GPU_DUMP=<dir> writes the activation regions back as raw f32 after a call, and E2_GPU_LAYERS=n
 * stops the graph after n layers (0 = after the PLE prologue). That is what the CPU engine's
 * E2_DUMP=dir does for the CPU tower, so the two can be compared stage by stage. */
static const struct { const char *name; int region; } g_dump[] = {
    { "x", A_X },   { "res", A_RES }, { "q", A_Q },     { "k", A_K2 },   { "v", A_V2 },
    { "ctx", A_CTX },
    { "gate", A_GATE }, { "up", A_UP }, { "tmp", A_TMP }, { "ple", A_PLE }, { "emb", A_EMB },
    { "scr", A_SCR }, { "pb", A_PB }, { "pbs", A_PBS }, { "qpk", A_QPK }, { "kpk", A_KPK },
    { "qps", A_QPS }, { "ksc", A_KSC }, { "kbq", A_KBSQ }, { "vpk", A_VPK },
    { "vsc", A_VSC }, { "vbq", A_VBSQ },
};

static void dump_regions(e2_gpu *g, int rows, int seq_len) {
    for (size_t i = 0; i < sizeof g_dump / sizeof g_dump[0]; i++) {
        size_t bytes = g->reg[g_dump[i].region].bytes;
        /* the PLE region is only rows*12288 deep, not the whole allocation */
        if (g_dump[i].region == A_PLE) bytes = (size_t)rows * g->m->n_layers * 512 * 4;
        void *host = malloc(bytes);
        if (!host) return;
        if (e2_l0_read(g->l0, aoff_ptr(g, g_dump[i].region), bytes, host)) { free(host); continue; }
        char path[512];
        snprintf(path, sizeof path, "%s/gpu_%s.f32", g->dump_dir, g_dump[i].name);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(host, 1, bytes, f); fclose(f); }
        free(host);
    }
    (void)seq_len;
}

/* ---------------------------------------------------------------- forward */

static int run_chunk(e2_gpu *g, const int32_t *ids, const uint8_t *valid, int rows, int seq_len,
                     int from_embeds, float *out) {
    e2_graph *gr = graph_for(g, rows, seq_len, from_embeds);
    if (!gr) return -1;
    if (ids) memcpy(g->ids, ids, sizeof(int32_t) * (size_t)rows);
    /* Reset the full mask because cached graphs also serve differently padded batches. */
    memset(g->valid, 1, (size_t)E2_GPU_MAX_ROWS);
    if (valid) memcpy(g->valid, valid, (size_t)rows);
    double t0 = now_ms();
    if (e2_l0_graph_run(g->l0, gr)) return fail(g, "run: %s", e2_l0_error(g->l0));
    g->last_ms = now_ms() - t0;
    const int nseq = seq_len > 0 ? rows / seq_len : 1;
    memcpy(out, g->embout, sizeof(float) * 768 * (size_t)nseq);
    g->calls++;
    if (g->dump_dir && g->calls == g->dump_call) dump_regions(g, rows, seq_len);
    return 0;
}

int e2_gpu_text_embed(e2_gpu *g, const int32_t *ids, int n_ids, float *out768) {
    char verr[256];
    if (!g) return fail(NULL, "GPU context is NULL");
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) return fail(g, "sequence length %d out of range", n_ids);
    if (!out768) return fail(g, "output pointer is NULL");
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) return fail(g, "%s", verr);
    return run_chunk(g, ids, NULL, n_ids, n_ids, 0, out768);
}

int e2_gpu_text_embed_batch(e2_gpu *g, const int32_t *ids, const uint8_t *valid, int batch,
                            int seq_len, float *out) {
    char verr[256];
    if (!g) return fail(NULL, "GPU context is NULL");
    if (batch <= 0 || seq_len <= 0 || seq_len > E2_MAX_SEQ ||
        (size_t)batch > SIZE_MAX / (size_t)seq_len / sizeof(*ids) ||
        (size_t)batch > SIZE_MAX / E2_EMBED_DIM / sizeof(*out))
        return fail(g, "bad batch geometry");
    if (!out) return fail(g, "output pointer is NULL");
    /* The pad rows are gathered too, so "pad with any valid id" means every id in the block must be
     * a vocabulary entry - validated once, up front, not per chunk. */
    if (e2_valid_ids(ids, (size_t)batch * (size_t)seq_len, verr, sizeof verr))
        return fail(g, "%s", verr);
    /* Sequences beyond what one pass holds are processed in chunks of whole sequences. */
    int per = g->max_rows / seq_len;
    if (per < 1) per = 1;
    for (int b0 = 0; b0 < batch;) {
        const int n = batch - b0 < per ? batch - b0 : per;
        const int rows = n * seq_len;
        if (run_chunk(g, ids + (size_t)b0 * seq_len, valid ? valid + (size_t)b0 * seq_len : NULL, rows,
                      seq_len, 0, out + (size_t)b0 * 768))
            return -1;
        b0 += n; /* by n, not per: b0 + per can leave int range on the last chunk */
    }
    return 0;
}

int e2_gpu_audio_embed(e2_gpu *g, const float *pcm, int n_samples, float *out768) {
    if (!g) return fail(NULL, "GPU context is NULL");
    if (!pcm || n_samples <= 0) return fail(g, "empty audio input");
    if (!out768) return fail(g, "output pointer is NULL");
    float *soft = malloc(sizeof(float) * (size_t)E2_A_MAX_TOK * 512);
    if (!soft) return fail(g, "oom: audio soft tokens");
    int n = 0;
    if (e2_gpu_audio_soft(g, pcm, n_samples, soft, &n)) { free(soft); return -1; }
    const int rows = n + 4;
    float *emb = malloc(sizeof(float) * (size_t)rows * 512);
    if (!emb) { free(soft); return fail(g, "oom: audio text embeddings"); }
    e2_embed_token(g->m, 2, emb);              /* <bos> */
    e2_embed_token(g->m, 256000, emb + 512);   /* <|audio> */
    memcpy(emb + (size_t)2 * 512, soft, sizeof(float) * (size_t)n * 512);
    e2_embed_token(g->m, 258883, emb + (size_t)(2 + n) * 512); /* <audio|> */
    e2_embed_token(g->m, 1, emb + (size_t)(3 + n) * 512);      /* <eos> */
    free(soft);
    int rc = 0;
    if (e2_l0_write(g->l0, aoff_ptr(g, A_VEMBIN), sizeof(float) * (size_t)rows * 512, emb))
        rc = fail(g, "audio text input: %s", e2_l0_error(g->l0));
    else
        rc = run_chunk(g, NULL, NULL, rows, rows, 1, out768);
    free(emb);
    return rc;
}

/* ---------------------------------------------------------------- vision tower
 * The same model as src/vision.c, with the same structure and the same int8 recipe, recorded as one
 * command list per (patch count, pool-group count). Three things are different from the text half:
 *
 *  - The padding mask is a gather. The host walks the position ids once, builds the index list of the
 *    real patches (and the pooler's group lists), and every kernel downstream runs on a dense tensor.
 *    That is also what removes the reference's (1,1,2520,2520) attention bias.
 *  - head_dim is 64, so the attention does not use the text tower's kernels: k_vattn_packk/packq and
 *    k_vattn_scores quantize per row with a single 64-wide block (two DPAS k-steps, no block loop)
 *    while the context product reuses k_attn_gemm and k_vattn_packv, whose k dimension is the key
 *    count and stays on 128-element blocks.
 *  - The axial RoPE table is built on the host, per distinct coordinate: a patch grid of 2520 points
 *    has ~100 distinct x and y values, so the ~160k transcendental calls a naive build would make
 *    collapse to a few thousand. The table itself is 1.3 MB per call, which is 13 us of upload.
 */

static void vrope_build(e2_gpu *g, const int32_t *pos, const int32_t *order, int P, float *cs,
                        float *sn) {
    e2_vision *v = (e2_vision *)g->m->vision;
    static uint8_t seen[2][E2_V_POS];
    static float cx[E2_V_POS][16], sx[E2_V_POS][16], cy[E2_V_POS][16], sy[E2_V_POS][16];
    memset(seen, 0, sizeof seen);
    for (int p = 0; p < P; p++) {
        const int x = pos[2 * order[p]], y = pos[2 * order[p] + 1];
        if (!seen[0][x]) {
            seen[0][x] = 1;
            for (int i = 0; i < 16; i++) {
                const float a = (float)x * v->rope_inv[i];
                cx[x][i] = cosf(a);
                sx[x][i] = sinf(a);
            }
        }
        if (!seen[1][y]) {
            seen[1][y] = 1;
            for (int i = 0; i < 16; i++) {
                const float a = (float)y * v->rope_inv[i];
                cy[y][i] = cosf(a);
                sy[y][i] = sinf(a);
            }
        }
        float *c = cs + (size_t)p * 64, *s = sn + (size_t)p * 64;
        for (int i = 0; i < 16; i++) {
            c[i] = c[i + 16] = cx[x][i];
            s[i] = s[i + 16] = sx[x][i];
            c[32 + i] = c[48 + i] = cy[y][i];
            s[32 + i] = s[48 + i] = sy[y][i];
        }
    }
}

/* y[M,N] = x @ W^T for the vision tower: always the DPAS kernel (its rows are never a short
 * sequence), with the padded M the kernel's unguarded activation reads need. */
static int vgemm(e2_gpu *g, e2_graph *gr, int rows, int first, int count, int nmax) {
    if (g->limit && g->kernels_last >= g->limit) return 0;
    void *wbuf = g->wbuf, *acts = g->acts;
    void *d = (char *)g->desc + (size_t)first * E2_DS_STRIDE * 4;
    const int mpad = (rows + E2_DM - 1) / E2_DM * E2_DM;
    const int gy = gemm_gy(rows);
    const char *kname = getenv("E2_GPU_GEMM_OLD") ? "k_gemm_dpas" : "k_gemm_dpas2";
    if (e2_l0_append(g->l0, gr, kname, (uint32_t)(mpad / E2_DM),
                     (uint32_t)((nmax + E2_DN * gy - 1) / (E2_DN * gy)), (uint32_t)count, 8, gy, 1, 4,
                     &wbuf, sizeof wbuf, &acts, sizeof acts, &d, sizeof d, &rows, sizeof rows))
        return -1;
    g->kernels_last++;
    return 0;
}

static int record_vision(e2_gpu *g, int P, int ng, e2_graph **out) {
    e2_model *m = g->m;
    const int D = E2_V_HIDDEN, H = E2_V_HEADS;
    const int Ppad = (P + 127) / 128 * 128;
    const int mpad = (P + E2_DM - 1) / E2_DM * E2_DM;
    const int mps = (P + E2_SDM - 1) / E2_SDM * E2_SDM;
    const int gy = 4;
    const int vl = g->vlayer_limit;
    e2_graph *gr = e2_l0_graph_begin(g->l0, 1);
    if (!gr) return fail(g, "command list: %s", e2_l0_error(g->l0));
    g->kernels_last = 0;

#define KERN(name, ...) \
    do { \
        if (g->limit && g->kernels_last >= g->limit) goto done; \
        if (e2_l0_append(g->l0, gr, name, __VA_ARGS__)) goto fail2; \
        g->kernels_last++; \
    } while (0)

    /* 1. gather the valid patches, rescale and quantize them (K=768 -> six 128-blocks) */
    {
        void *qa = aoff_ptr(g, A_VQX), *asc = aoff_ptr(g, A_VQXS);
        void *pix = g->vpix, *order = g->vorder;
        KERN("k_vpatch_q", (uint32_t)((P + gy - 1) / gy), 1, 1, 32, gy, 1, 6, &qa, sizeof qa, &asc,
             sizeof asc, &pix, sizeof pix, &order, sizeof order, &P, sizeof P, &D, sizeof D);
    }
    /* 2. patch projection 768 -> 768, then 3. the 2D position term */
    if (vgemm(g, gr, P, g->vdesc_base + E2_V_LAYERS * 7, 1, D)) goto fail2;
    {
        void *x = aoff_ptr(g, A_VX), *proj = aoff_ptr(g, A_VRES);
        void *ptab = woff_ptr(g, g->vpostab_off), *order = g->vorder, *pids = g->vpos;
        const int npos = E2_V_POS;
        KERN("k_vpos_add", (uint32_t)((P + 3) / 4), (uint32_t)((D + 255) / 256), 1, 64, 4, 1, 8, &x,
             sizeof x, &proj, sizeof proj, &ptab, sizeof ptab, &order, sizeof order, &pids,
             sizeof pids, &P, sizeof P, &D, sizeof D, &npos, sizeof npos);
    }
    if (vl == 0) goto done; /* diagnostic: patch embed only */

    for (int l = 0; l < E2_V_LAYERS; l++) {
        const int b = g->vdesc_base + l * 7;
        /* 1. res = x; x = norm(x, input_ln); quantize */
        if (rowops(g, gr, P, D, A_VX, -1, A_VRES, A_VQX, A_VQXS, g->vnoff[l].n[0], -1, 1 | 2, 1.f,
                   1.f))
            goto fail2;
        /* 2. q | k | v */
        if (vgemm(g, gr, P, b, 3, D)) goto fail2;
        /* 3. per-head RMSNorm + axial rope (v without rope or gamma) */
        {
            void *xq = aoff_ptr(g, A_VQ), *xk = aoff_ptr(g, A_VK), *xv = aoff_ptr(g, A_VV);
            void *gq = woff_ptr(g, g->vnoff[l].n[4]), *gk = woff_ptr(g, g->vnoff[l].n[5]);
            void *gv = NULL;
            void *cs = g->vrope, *sn = (char *)g->vrope + (size_t)E2_GPU_VP * 64 * 4;
            const int rows = P * H;
            KERN("k_vnorm_rope3", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 11, &xq,
                 sizeof xq, &gq, sizeof gq, &xk, sizeof xk, &gk, sizeof gk, &xv, sizeof xv, &gv,
                 sizeof gv, &cs, sizeof cs, &sn, sizeof sn, &m->rms_eps, sizeof m->rms_eps, &rows,
                 sizeof rows, &H, sizeof H);
        }
        /* 4. Q, K and V into the attention's operand layouts */
        {
            void *q = aoff_ptr(g, A_VQ), *qa = aoff_ptr(g, A_VQPK), *asc = aoff_ptr(g, A_VQPKS);
            const int rows = P * H;
            KERN("k_vattn_packq", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 6, &q, sizeof q,
                 &qa, sizeof qa, &asc, sizeof asc, &rows, sizeof rows, &P, sizeof P, &H, sizeof H);
        }
        {
            void *k = aoff_ptr(g, A_VK), *wd = aoff_ptr(g, A_VKPK), *wsc = aoff_ptr(g, A_VKSC);
            void *bsq = aoff_ptr(g, A_VKBSQ);
            const int rows = H * Ppad;
            KERN("k_vattn_packk", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 7, &k, sizeof k,
                 &wd, sizeof wd, &wsc, sizeof wsc, &bsq, sizeof bsq, &P, sizeof P, &Ppad,
                 sizeof Ppad, &D, sizeof D);
        }
        {
            void *vv = aoff_ptr(g, A_VV), *wd = aoff_ptr(g, A_VVPK), *wsc = aoff_ptr(g, A_VVSC);
            void *bsq = aoff_ptr(g, A_VVBSQ);
            const int rows = H * 64, nb = Ppad / 128;
            KERN("k_vattn_packv", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 8, &vv,
                 sizeof vv, &wd, sizeof wd, &wsc, sizeof wsc, &bsq, sizeof bsq, &P, sizeof P, &Ppad,
                 sizeof Ppad, &D, sizeof D, &nb, sizeof nb);
        }
        /* 5. attention, one head at a time: the three kernels of one head share one score/P scratch
         * (the command list serialises them), and there is no window and no mask, so the products
         * are plain GEMMs over the padded key range. The fused path writes the int8 P instead of a
         * fp16 score matrix; E2_GPU_VATTN_OLD=1 is the pre-fusion A/B. */
        {
            const int nbs = Ppad / 128;                       /* context k-blocks (old path) */
            const int nb32 = Ppad / 32;                       /* 32-key blocks (fused path) */
            const int nts = (Ppad + E2_SDN * gy - 1) / (E2_SDN * gy);
            const int ntc = (64 + E2_DN * gy - 1) / (E2_DN * gy);
            const int xq = 64, aq = 1, hd = 64;   /* head-major packed Q: row stride 64, scale stride 1 */
            for (int h = 0; h < H; h++) {
                void *kpk = (char *)aoff_ptr(g, A_VKPK) + (size_t)h * Ppad * 64;
                void *ksc = (char *)aoff_ptr(g, A_VKSC) + (size_t)h * Ppad * 4;
                void *kbsq = (char *)aoff_ptr(g, A_VKBSQ) + (size_t)h * Ppad * 4;
                void *qp8 = (char *)aoff_ptr(g, A_VQPK) + (size_t)h * P * 64;
                void *qsc = (char *)aoff_ptr(g, A_VQPKS) + (size_t)h * P * 4;
                void *vpk = (char *)aoff_ptr(g, A_VVPK) + (size_t)h * Ppad * 64;
                void *vsc = (char *)aoff_ptr(g, A_VVSC) + (size_t)h * 64 * 4;
                void *scr = aoff_ptr(g, A_VSCR), *pb = aoff_ptr(g, A_VPB);
                void *pbs = aoff_ptr(g, A_VPBS);
                void *sbm = aoff_ptr(g, A_VSBM);
                void *sbs = aoff_ptr(g, A_VSBS);
                void *ctx = (char *)aoff_ptr(g, A_VCTX) + (size_t)h * 64 * 4;
                if (!g->vattn_old) {
                    KERN("k_vattn_psoft", (uint32_t)(mps / E2_SDM), (uint32_t)nts, 1, 8, gy, 1, 14,
                         &kpk, sizeof kpk, &ksc, sizeof ksc, &kbsq, sizeof kbsq, &qp8, sizeof qp8,
                         &qsc, sizeof qsc, &pb, sizeof pb, &sbm, sizeof sbm, &sbs, sizeof sbs,
                         &Ppad, sizeof Ppad, &P, sizeof P, &mps, sizeof mps, &xq, sizeof xq,
                         &aq, sizeof aq, &P, sizeof P);
                    KERN("k_vattn_pasc", (uint32_t)P, 1, 1, 32, 1, 1, 5, &sbm, sizeof sbm, &sbs,
                         sizeof sbs, &pbs, sizeof pbs, &P, sizeof P, &nb32, sizeof nb32);
                    KERN("k_vattn_ctx", (uint32_t)(mpad / E2_DM), (uint32_t)ntc, 1, 8, gy, 1, 12,
                         &vpk, sizeof vpk, &vsc, sizeof vsc, &pb, sizeof pb, &pbs, sizeof pbs,
                         &ctx, sizeof ctx, &hd, sizeof hd, &Ppad, sizeof Ppad, &nb32, sizeof nb32,
                         &P, sizeof P, &D, sizeof D, &mps, sizeof mps, &nb32, sizeof nb32);
                } else {
                    void *vbsq = (char *)aoff_ptr(g, A_VVBSQ) + (size_t)h * nbs * 64 * 4;
                    KERN("k_vattn_scores", (uint32_t)(mps / E2_SDM), (uint32_t)nts, 1, 8, gy, 1, 13,
                         &kpk, sizeof kpk, &ksc, sizeof ksc, &kbsq, sizeof kbsq, &qp8, sizeof qp8, &qsc,
                         sizeof qsc, &scr, sizeof scr, &sbm, sizeof sbm, &Ppad, sizeof Ppad, &P, sizeof P,
                         &mps, sizeof mps, &xq, sizeof xq, &aq, sizeof aq, &P, sizeof P);
                    KERN("k_vattn_softmax", (uint32_t)((P + gy - 1) / gy), 1, 1, 32, gy, 1, 9, &scr,
                         sizeof scr, &sbm, sizeof sbm, &pb, sizeof pb, &pbs, sizeof pbs, &P, sizeof P,
                         &Ppad, sizeof Ppad, &mps, sizeof mps, &P, sizeof P, &nbs, sizeof nbs);
                    KERN("k_attn_gemm", (uint32_t)(mpad / E2_DM), (uint32_t)ntc, 1, 8, gy, 1, 13, &vpk,
                         sizeof vpk, &vsc, sizeof vsc, &vbsq, sizeof vbsq, &pb, sizeof pb, &pbs,
                         sizeof pbs, &ctx, sizeof ctx, &hd, sizeof hd, &Ppad, sizeof Ppad, &nbs,
                         sizeof nbs, &P, sizeof P, &D, sizeof D, &Ppad, sizeof Ppad, &nbs, sizeof nbs);
                }
            }
        }
        /* 6. quantize the context for the o projection */
        if (rowops(g, gr, P, D, A_VCTX, -1, -1, A_VQH, A_VQHS, -1, -1, 16 | 2, 1.f, 1.f)) goto fail2;
        /* 7. o_proj, then x = norm(tmp, post_attn_ln) + res (one pass: the residual is added by the
         * normalizing kernel, so the CPU path's separate copy is not needed) */
        if (vgemm(g, gr, P, b + 3, 1, D)) goto fail2;
        if (rowops(g, gr, P, D, A_VX, A_VTMP, A_VRES, -1, -1, g->vnoff[l].n[1], -1, 4 | 8, 1.f, 1.f))
            goto fail2;
        /* 8. res = x; x = norm(x, pre_ffn_ln); quantize */
        if (rowops(g, gr, P, D, A_VX, -1, A_VRES, A_VQX, A_VQXS, g->vnoff[l].n[2], -1, 1 | 2, 1.f,
                   1.f))
            goto fail2;
        /* 9. gate | up, then gelu(gate) * up, quantized for the down projection */
        if (vgemm(g, gr, P, b + 4, 2, E2_V_INTER)) goto fail2;
        const int wr = g->dump_dir != NULL;
        {
            void *gate = aoff_ptr(g, A_VGATE), *up = aoff_ptr(g, A_VUP), *nople = NULL;
            void *qa = aoff_ptr(g, A_VQM), *asc = aoff_ptr(g, A_VQMS);
            const int K = E2_V_INTER, nb = K / 128, stride = 0, poff = 0;
            KERN("k_gelu_mul_quant", (uint32_t)(((size_t)P * nb + 3) / 4), 1, 1, 32, 4, 1, 11,
                 &gate, sizeof gate, &up, sizeof up, &nople, sizeof nople, &qa, sizeof qa, &asc,
                 sizeof asc, &P, sizeof P, &K, sizeof K, &nb, sizeof nb, &stride, sizeof stride,
                 &poff, sizeof poff, &wr, sizeof wr);
        }
        /* 10. down_proj, then x = norm(tmp, post_ffn_ln) + res */
        if (vgemm(g, gr, P, b + 6, 1, D)) goto fail2;
        if (rowops(g, gr, P, D, A_VX, A_VTMP, A_VRES, -1, -1, g->vnoff[l].n[3], -1, 4 | 8, 1.f, 1.f))
            goto fail2;
        if (vl >= 0 && l + 1 >= vl) goto done; /* diagnostic: stop after layer l */
    }

    /* pooler, then the scale-free RMSNorm and the 768 -> 512 soft-token projection. The grid is the
     * fixed group bound and the count comes from the device, so the command list is valid for any
     * image with this patch count whose group count is the one it was recorded for. */
    {
        void *pool = aoff_ptr(g, A_VPOOL), *x = aoff_ptr(g, A_VX);
        void *gs = g->vgstart, *gp = g->vgpatch, *ngp = g->vpool_ng;
        const float scale = sqrtf((float)D) / (float)(E2_V_POOL * E2_V_POOL);
        KERN("k_vpool", (uint32_t)E2_GPU_VGROUPS, 1, 1, 128, 1, 1, 7, &pool, sizeof pool, &x,
             sizeof x, &gs, sizeof gs, &gp, sizeof gp, &ngp, sizeof ngp, &D, sizeof D, &scale,
             sizeof scale);
    }
    if (rowops(g, gr, ng, D, A_VPOOL, -1, -1, A_VPP, A_VPPS, -1, -1, 2, 1.f, 1.f)) goto fail2;
    if (vgemm(g, gr, ng, g->vdesc_base + E2_V_LAYERS * 7 + 1, 1, 512)) goto fail2;
#undef KERN
done:
    if (e2_l0_graph_close(g->l0, gr)) goto fail2;
    *out = gr;
    return 0;
fail2:
    e2_l0_graph_free(g->l0, gr);
    return fail(g, "vision recording failed: %s", e2_l0_error(g->l0));
}

static e2_graph *vgraph_for(e2_gpu *g, int P) {
    const int key = g->vgroups;
    for (int i = 0; i < g->nslots; i++)
        if (g->slots[i].from_embeds == 2 && g->slots[i].rows == P && g->slots[i].seq_len == key) {
            g->kernels_last = g->slots[i].kernels;
            return g->slots[i].gr;
        }
    e2_slot *sl = new_slot(g);
    e2_graph *gr = NULL;
    if (record_vision(g, P, key, &gr)) return NULL;
    *sl = (e2_slot){P, key, 2, gr, (int)g->kernels_last};
    g->nslots++;
    return gr;
}

/* ---------------------------------------------------------------- audio tower
 * The same twelve layers as src/audio.c, recorded once per soft-token count. The differences from a
 * graph-runtime transcription are deliberate:
 *
 *  - every clipped linear's input and output bound is part of the projection's descriptor, so the
 *    output clamp is the GEMM store and the input clamp rides the activation quantizer;
 *  - attention is one fused kernel (relative term, tanh cap, softmax, value accumulation, context
 *    clamp and int8 quantization), because at twelve keys a Q/K pack plus two GEMMs is all overhead;
 *  - the light conv is three kernels: GLU, depthwise+RMSNorm+SiLU+quantize, and clamp+residual.
 *
 * The relative-key projection is precomputed at upload time (see e2_gpu_create), so the per-call
 * graph never pays for a 13x1024 GEMM that only depends on constants.
 */
static int audio_silu_quant(e2_gpu *g, e2_graph *gr, int rows, int K, int region_x, float clo,
                            float chi) {
    const uint32_t gy = 4;
    void *x = aoff_ptr(g, region_x), *qa = aoff_ptr(g, A_AQA), *asc = aoff_ptr(g, A_AQAS);
    const int nb = K / 128;
    if (e2_l0_append(g->l0, gr, "k_audio_silu_quant", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32,
                     gy, 1, 8, &x, sizeof x, &qa, sizeof qa, &asc, sizeof asc, &rows, sizeof rows,
                     &K, sizeof K, &nb, sizeof nb, &clo, sizeof clo, &chi, sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_norm2(e2_gpu *g, e2_graph *gr, int rows, int region_src, int region_res,
                      int region_resout, int region_x, int region_qa, int g1_off, int g2_off,
                      int flags, float scale, float clo, float chi) {
    const uint32_t gy = 4;
    void *src = aoff_ptr(g, region_src), *res = aoff_ptr(g, region_res);
    void *res_out = region_resout >= 0 ? aoff_ptr(g, region_resout) : NULL;
    void *x = aoff_ptr(g, region_x);
    void *qa = region_qa >= 0 ? aoff_ptr(g, region_qa) : NULL;
    void *asc = region_qa >= 0 ? aoff_ptr(g, A_AQAS) : NULL;
    void *g1 = woff_ptr(g, g1_off), *g2 = woff_ptr(g, g2_off);
    const int K = 1024, nb = 8;
    if (e2_l0_append(g->l0, gr, "k_audio_norm2", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1,
                     15, &src, sizeof src, &res, sizeof res, &res_out, sizeof res_out, &x,
                     sizeof x, &qa, sizeof qa, &asc, sizeof asc, &g1, sizeof g1, &g2, sizeof g2,
                     &rows, sizeof rows, &K, sizeof K, &nb, sizeof nb, &flags, sizeof flags, &scale,
                     sizeof scale, &clo, sizeof clo, &chi, sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_glu(e2_gpu *g, e2_graph *gr, int rows, float clo, float chi) {
    const uint32_t gy = 4;
    void *lc = aoff_ptr(g, A_ALC), *out = aoff_ptr(g, A_AGLU);
    if (e2_l0_append(g->l0, gr, "k_audio_glu", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1, 5,
                     &lc, sizeof lc, &out, sizeof out, &rows, sizeof rows, &clo, sizeof clo, &chi,
                     sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_dwconv(e2_gpu *g, e2_graph *gr, int rows, const e2_alayer *L, int l, float clo,
                        float chi) {
    const uint32_t gy = 4;
    void *x = aoff_ptr(g, A_AGLU), *w = woff_ptr(g, g->alc_conv_off[l]);
    void *gam = woff_ptr(g, g->anoff[l].n[8]);
    void *qa = aoff_ptr(g, A_AQA), *asc = aoff_ptr(g, A_AQAS);
    const float eps = 1e-6f;
    (void)L;
    if (e2_l0_append(g->l0, gr, "k_audio_dwconv", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy, 1,
                     9, &x, sizeof x, &w, sizeof w, &gam, sizeof gam, &qa, sizeof qa, &asc,
                     sizeof asc, &rows, sizeof rows, &eps, sizeof eps, &clo, sizeof clo, &chi,
                     sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_add_clamp(e2_gpu *g, e2_graph *gr, int rows, float clo, float chi) {
    const uint32_t gy = 4;
    void *src = aoff_ptr(g, A_ATMP), *res = aoff_ptr(g, A_ARES), *out = aoff_ptr(g, A_AX);
    if (e2_l0_append(g->l0, gr, "k_audio_add_clamp", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy,
                     1, 6, &src, sizeof src, &res, sizeof res, &out, sizeof out, &rows, sizeof rows,
                     &clo, sizeof clo, &chi, sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_bias_norm(e2_gpu *g, e2_graph *gr, int rows) {
    const uint32_t gy = 4;
    void *x = aoff_ptr(g, A_ATMP), *bias = woff_ptr(g, g->aout_bias_off);
    void *qa = aoff_ptr(g, A_AQA), *asc = aoff_ptr(g, A_AQAS);
    const float eps = 1e-6f;
    if (e2_l0_append(g->l0, gr, "k_audio_bias_norm", (uint32_t)((rows + gy - 1) / gy), 1, 1, 32, gy,
                     1, 6, &x, sizeof x, &bias, sizeof bias, &qa, sizeof qa, &asc, sizeof asc,
                     &rows, sizeof rows, &eps, sizeof eps))
        return -1;
    g->kernels_last++;
    return 0;
}

static int audio_attn(e2_gpu *g, e2_graph *gr, int rows, int l, float clo, float chi) {
    void *q = aoff_ptr(g, A_AQ), *k = aoff_ptr(g, A_AK), *v = aoff_ptr(g, A_AV);
    void *relk = woff_ptr(g, g->arelk_off[l]), *qm = woff_ptr(g, g->aqmult_off[l]);
    void *ctx = aoff_ptr(g, A_AATT), *qa = aoff_ptr(g, A_AQA), *asc = aoff_ptr(g, A_AQAS);
    const float cap = 50.f, kscale = 1.4426950408889634f; /* log2(e), the reference's k_scale */
    if (e2_l0_append(g->l0, gr, "k_audio_attn", (uint32_t)rows, 1, 1, 32, 8, 1, 13, &q, sizeof q,
                     &k, sizeof k, &v, sizeof v, &relk, sizeof relk, &qm, sizeof qm, &ctx,
                     sizeof ctx, &qa, sizeof qa, &asc, sizeof asc, &rows, sizeof rows, &cap,
                     sizeof cap, &kscale, sizeof kscale, &clo, sizeof clo, &chi, sizeof chi))
        return -1;
    g->kernels_last++;
    return 0;
}

static int record_audio(e2_gpu *g, int S, e2_graph **out) {
    e2_model *m = g->m;
    e2_audio *a = (e2_audio *)m->audio;
    if (!a) return fail(g, "the model has no audio tower");
    e2_graph *gr = e2_l0_graph_begin(g->l0, 1);
    if (!gr) return fail(g, "command list: %s", e2_l0_error(g->l0));
    g->kernels_last = 0;
    void *dst = g->asoft, *src = aoff_ptr(g, A_ASOFT);
    void *axin = g->axin, *ax = aoff_ptr(g, A_AX);
    const int hidden = 512, H = E2_A_HIDDEN, hid_in = 1024;
    if (e2_l0_append(g->l0, gr, "k_copy_rows", (uint32_t)((S + 3) / 4),
                     (uint32_t)(1024 / 4 / 64), 1, 64, 4, 1, 4, &ax, sizeof ax, &axin, sizeof axin,
                     &S, sizeof S, &hid_in, sizeof hid_in))
        goto fail2;
    g->kernels_last++;
    const int nlayers = getenv("E2_GPU_ALAYERS") ? atoi(getenv("E2_GPU_ALAYERS")) : E2_A_LAYERS;
    const int ab = g->adesc_base + E2_A_LAYERS * E2_ADS_PER_LAYER;
#define AK(expr) do { if (expr) goto fail2; } while (0)
    for (int l = 0; l < nlayers && l < E2_A_LAYERS; l++) {
        const e2_alayer *L = &a->layers[l];
        const int base = g->adesc_base + l * E2_ADS_PER_LAYER;
        /* feed-forward 1: residual + pre-norm + clipped 1024->4096, SiLU+clamp+quantize,
         * clipped 4096->1024, post-norm * 0.5 + residual */
        AK(arowops(g, gr, S, H, A_AX, -1, A_ARES, A_AQA, g->anoff[l].n[3], 1 | 2 | 128,
                   L->ff1a.in_min, L->ff1a.in_max, 1.f));
        AK(ageam(g, gr, S, base + E2_ADS_FF1A, 1, E2_A_INTER));
        AK(audio_silu_quant(g, gr, S, E2_A_INTER, A_AFF, L->ff1b.in_min, L->ff1b.in_max));
        AK(ageam(g, gr, S, base + E2_ADS_FF1B, 1, H));
        AK(audio_norm2(g, gr, S, A_ATMP, A_ARES, A_ARES, A_ATMP, A_AQA, g->anoff[l].n[4],
                       g->anoff[l].n[0], 2 | 128, 0.5f, L->q.in_min, L->q.in_max));

        /* attention: the pre-norm was fused into the FF1 close; q|k|v then the relative/window kernel */
        AK(ageam(g, gr, S, base + E2_ADS_Q, 3, H));
        AK(audio_attn(g, gr, S, l, L->post.in_min, L->post.in_max));
        AK(ageam(g, gr, S, base + E2_ADS_POST, 1, H));
        AK(audio_norm2(g, gr, S, A_ATMP, A_ARES, A_ARES, A_ATMP, A_AQA, g->anoff[l].n[1],
                       g->anoff[l].n[7], 2 | 128, 1.f, L->lc_start.in_min, L->lc_start.in_max));

        /* light conv: the pre-norm was fused into the attention close; GLU, depthwise k=5 + norm +
         * SiLU + quantize, clipped 1024->1024, clamp + residual */
        AK(ageam(g, gr, S, base + E2_ADS_LC_START, 1, 2 * H));
        AK(audio_glu(g, gr, S, L->lc_start.out_min, L->lc_start.out_max));
        AK(audio_dwconv(g, gr, S, L, l, L->lc_end.in_min, L->lc_end.in_max));
        AK(ageam(g, gr, S, base + E2_ADS_LC_END, 1, H));
        AK(audio_add_clamp(g, gr, S, L->lc_end.out_min, L->lc_end.out_max));

        /* feed-forward 2: same shape as FF1, then the layer's final norm into A_AX */
        AK(arowops(g, gr, S, H, A_AX, -1, A_ARES, A_AQA, g->anoff[l].n[5], 1 | 2 | 128,
                   L->ff2a.in_min, L->ff2a.in_max, 1.f));
        AK(ageam(g, gr, S, base + E2_ADS_FF2A, 1, E2_A_INTER));
        AK(audio_silu_quant(g, gr, S, E2_A_INTER, A_AFF, L->ff2b.in_min, L->ff2b.in_max));
        AK(ageam(g, gr, S, base + E2_ADS_FF2B, 1, H));
        AK(audio_norm2(g, gr, S, A_ATMP, A_ARES, -1, A_AX, -1, g->anoff[l].n[6],
                       g->anoff[l].n[2], 0, 0.5f, 0.f, 0.f));
    }
    /* out_proj (1024->1536 + bias), scale-free RMSNorm, embed_audio (1536->512) */
    AK(arowops(g, gr, S, H, A_AX, -1, -1, A_AQA, -1, 16 | 2, 0.f, 0.f, 1.f));
    AK(ageam(g, gr, S, ab + 1, 1, E2_A_OUT));
    if (getenv("E2_GPU_ASKIP_HEAD")) goto done;
    AK(audio_bias_norm(g, gr, S));
    AK(ageam(g, gr, S, ab + 2, 1, 512));
    AK(e2_l0_append(g->l0, gr, "k_copy_rows", (uint32_t)((S + 3) / 4),
                    (uint32_t)(512 / 4 / 64), 1, 64, 4, 1, 4, &dst, sizeof dst, &src, sizeof src,
                    &S, sizeof S, &hidden, sizeof hidden));
    g->kernels_last++;
#undef AK
done:
    if (e2_l0_graph_close(g->l0, gr)) goto fail2;
    *out = gr;
    return 0;
fail2:
    e2_l0_graph_free(g->l0, gr);
    return fail(g, "audio recording failed: %s", e2_l0_error(g->l0));
}

static e2_graph *agraph_for(e2_gpu *g, int S) {
    for (int i = 0; i < g->nslots; i++)
        if (g->slots[i].from_embeds == 3 && g->slots[i].rows == S) {
            g->kernels_last = g->slots[i].kernels;
            return g->slots[i].gr;
        }
    e2_slot *sl = new_slot(g);
    e2_graph *gr = NULL;
    if (record_audio(g, S, &gr)) return NULL;
    *sl = (e2_slot){S, 0, 3, gr, (int)g->kernels_last};
    g->nslots++;
    return gr;
}

int e2_gpu_audio_soft(e2_gpu *g, const float *pcm, int n_samples, float *soft, int *n_soft) {
    if (!g) return fail(NULL, "GPU context is NULL");
    e2_audio *a = (e2_audio *)g->m->audio;
    if (!a) return fail(g, "the model has no audio tower");
    if (!pcm || n_samples <= 0) return fail(g, "empty audio input");
    if (!soft) return fail(g, "soft-token output pointer is NULL");
    const float *fx = NULL;
    int S = 0;
    if (e2_audio_frontend(g->m, pcm, n_samples, &fx, &S) || !fx || S <= 0)
        return fail(g, "audio front end: %s", e2_last_error());
    if (S > E2_A_MAX_TOK) return fail(g, "audio tower produced %d tokens (max %d)", S, E2_A_MAX_TOK);
    memcpy(g->axin, fx, sizeof(float) * (size_t)S * 1024);
    e2_graph *gr = agraph_for(g, S);
    if (!gr) return -1;
    const double t0 = now_ms();
    if (e2_l0_graph_run(g->l0, gr)) return fail(g, "audio run: %s", e2_l0_error(g->l0));
    g->last_ms = now_ms() - t0;
    g->calls++;
    memcpy(soft, g->asoft, sizeof(float) * (size_t)S * 512);
    if (n_soft) *n_soft = S;
    {
        const char *ad = getenv("E2_GPU_AUDIO_DUMP");
        if (ad) {
            const struct { const char *name; int region; size_t n; } d[8] = {
                { "ax", A_AX, (size_t)S * 1024 },
                { "att", A_AATT, (size_t)S * 1024 },
                { "asoft", A_ASOFT, (size_t)S * 512 },
                { "aq", A_AQ, (size_t)S * 1024 },
                { "ak", A_AK, (size_t)S * 1024 },
                { "av", A_AV, (size_t)S * 1024 },
                { "aqa", A_AQA, (size_t)S * 1024 },
                { "atmp", A_ATMP, (size_t)S * 1536 },
            };
            for (int i = 0; i < 8; i++) {
                float *host = malloc(sizeof(float) * d[i].n);
                if (!host) continue;
                if (!e2_l0_read(g->l0, aoff_ptr(g, d[i].region), sizeof(float) * d[i].n, host)) {
                    char path[512];
                    snprintf(path, sizeof path, "%s_%s.f32", ad, d[i].name);
                    FILE *fo = fopen(path, "wb");
                    if (fo) { fwrite(host, 4, d[i].n, fo); fclose(fo); }
                }
                free(host);
            }
        }
    }
    return 0;
}

int e2_gpu_audio_stats(e2_gpu *g, long *kernels, double *last_ms) {
    if (kernels) *kernels = g->kernels_last;
    if (last_ms) *last_ms = g->last_ms;
    return 0;
}

/* ---------------------------------------------------------------- vision entry points
 * Runs the vision tower and leaves the soft tokens in `soft` (n_soft x 512 floats, host memory). */
static int vision_soft_tokens(e2_gpu *g, const float *pixels, const int32_t *position_ids,
                              int n_patches, float *soft, int *n_soft_out) {
    e2_model *m = g->m;
    char verr[256];
    if (n_patches <= 0 || n_patches > E2_V_MAX_PATCHES)
        return fail(g, "patch count %d out of range (max %d)", n_patches, E2_V_MAX_PATCHES);
    if (!pixels) return fail(g, "pixels pointer is NULL");
    /* The positions bound the axial table the patch embedder reads and the pool grid the kernels
     * write to (A_VPOOL holds E2_GPU_VGROUPS rows); both are settled before the gather. */
    int cols = 0, grows = 0;
    if (e2_valid_positions(position_ids, n_patches, E2_GPU_VGROUPS, &cols, &grows, verr, sizeof verr))
        return fail(g, "%s", verr);
    const int groups = cols * grows;
    const size_t nscr = 2 * (size_t)E2_V_MAX_PATCHES + 4 * (size_t)(E2_GPU_VGROUPS + 1);
    int32_t *scr = malloc(sizeof(int32_t) * nscr);
    if (!scr) return fail(g, "oom for the image scratch");
    int32_t *order = scr, *gpatch = scr + E2_V_MAX_PATCHES;
    int32_t *cnt = gpatch + E2_V_MAX_PATCHES;
    int32_t *gmap = cnt + E2_GPU_VGROUPS + 1;
    int32_t *gstart = gmap + E2_GPU_VGROUPS + 1;
    int32_t *wpos = gstart + E2_GPU_VGROUPS + 1;

    int P = 0;
    for (int i = 0; i < n_patches; i++)
        if (position_ids[2 * i] >= 0) order[P++] = i; /* the only negative pair is (-1,-1) padding */

    const int pool = E2_V_POOL;
    memset(cnt, 0, sizeof(int32_t) * (size_t)(groups + 1));
    for (int p = 0; p < P; p++) {
        const int i = order[p];
        cnt[position_ids[2 * i] / pool + cols * (position_ids[2 * i + 1] / pool)]++;
    }
    int ng = 0, acc = 0;
    for (int gg = 0; gg < groups; gg++) {
        if (cnt[gg]) {
            gmap[gg] = ng;
            gstart[ng++] = acc;
            acc += cnt[gg];
        } else {
            gmap[gg] = -1;
        }
    }
    if (ng > E2_GPU_VGROUPS) { free(scr); return fail(g, "too many pool groups"); }
    gstart[ng] = acc;
    memcpy(wpos, gstart, sizeof(int32_t) * (size_t)(ng + 1));
    for (int p = 0; p < P; p++) {
        const int i = order[p];
        const int gg = position_ids[2 * i] / pool + cols * (position_ids[2 * i + 1] / pool);
        gpatch[wpos[gmap[gg]]++] = p;
    }

    /* rope tables, then the per-call uploads */
    float *rope = malloc(sizeof(float) * 2 * (size_t)E2_GPU_VP * 64);
    if (!rope) { free(scr); return fail(g, "oom for the rope tables"); }
    vrope_build(g, position_ids, order, P, rope, rope + (size_t)E2_GPU_VP * 64);

    g->vgroups = ng;
    /* the vision kernels never read the validity mask (the gather already removed the padding), but
     * k_rowops does, and this path does not go through run_chunk's initialization. The mask is sized
     * for the vision's row count, not the text tower's. */
    memset(g->valid, 1, E2_GPU_VP);
    e2_graph *gr = vgraph_for(g, P);
    if (!gr) { free(scr); free(rope); return -1; }
    if (e2_l0_write(g->l0, g->vpix, sizeof(float) * (size_t)n_patches * E2_V_PATCH, pixels) ||
        e2_l0_write(g->l0, g->vpos, sizeof(int32_t) * (size_t)n_patches * 2, position_ids) ||
        e2_l0_write(g->l0, g->vorder, sizeof(int32_t) * (size_t)P, order) ||
        e2_l0_write(g->l0, g->vgstart, sizeof(int32_t) * (size_t)(ng + 1), gstart) ||
        e2_l0_write(g->l0, g->vgpatch, sizeof(int32_t) * (size_t)P, gpatch) ||
        e2_l0_write(g->l0, g->vpool_ng, sizeof(int32_t), &ng) ||
        e2_l0_write(g->l0, g->vrope, sizeof(float) * 2 * (size_t)E2_GPU_VP * 64, rope)) {
        free(scr); free(rope);
        return fail(g, "vision upload failed: %s", e2_l0_error(g->l0));
    }
    free(scr);
    free(rope);

    double t0 = now_ms();
    if (e2_l0_graph_run(g->l0, gr)) return fail(g, "vision run: %s", e2_l0_error(g->l0));
    g->vlast_ms = now_ms() - t0;
    g->vkernels_last = g->kernels_last;
    if (g->vlayer_limit >= 0 || g->limit) {
        /* diagnostic graph: the tail never ran, so there are no soft tokens to hand back. Dump the
         * regions the CPU engine's E2_DUMP writes under the same names, so the port bisects stage by
         * stage (v_l0_input_ln, v_l0_q_proj, v_l0_q_rope, v_l0_ctx, v_l0_o_proj, v_l0_x_attn,
         * v_l0_mlp_gate, v_l0_mlp_out, v_layer0). */
        if (g->dump_dir) {
            static const struct { const char *name; int region; int bytes_per_row; } vr[] = {
                { "vx", A_VX, 768 * 4 },      { "vres", A_VRES, 768 * 4 },
                { "vtmp", A_VTMP, 768 * 4 },  { "vq", A_VQ, 768 * 4 },
                { "vk", A_VK, 768 * 4 },      { "vv", A_VV, 768 * 4 },
                { "vctx", A_VCTX, 768 * 4 },  { "vgate", A_VGATE, 3072 * 4 },
                { "vup", A_VUP, 3072 * 4 },   { "vpool", A_VPOOL, 768 * 4 },
                { "vscr", A_VSCR, 0 },        { "vpb", A_VPB, 0 },
                { "vkpk", A_VKPK, 0 },        { "vqpk", A_VQPK, 0 },
                { "vqps", A_VQPKS, 0 },       { "vksc", A_VKSC, 0 },
                { "vkbsq", A_VKBSQ, 0 },      { "vpbs", A_VPBS, 0 },
            };
            for (size_t i = 0; i < sizeof vr / sizeof vr[0]; i++) {
                size_t bytes = (size_t)vr[i].bytes_per_row * (size_t)E2_V_MAX_PATCHES;
                if (vr[i].region == A_VPOOL) bytes = (size_t)E2_GPU_VGROUPS * 768 * 4;
                if (vr[i].region == A_VSCR) bytes = (size_t)E2_GPU_VK * E2_GPU_VK * 4;
                if (vr[i].region == A_VPB) bytes = (size_t)E2_GPU_VK * E2_GPU_VK;
                if (vr[i].region == A_VKPK || vr[i].region == A_VQPK)
                    bytes = (size_t)12 * E2_GPU_VK * 64;
                if (vr[i].region == A_VQPKS) bytes = (size_t)E2_V_MAX_PATCHES * 12 * 4;
                if (vr[i].region == A_VKSC || vr[i].region == A_VKBSQ)
                    bytes = (size_t)12 * E2_GPU_VK * 4;
                if (vr[i].region == A_VPBS) bytes = (size_t)E2_GPU_VK * (E2_GPU_VK / 128) * 4;
                void *host = malloc(bytes);
                if (!host) continue;
                if (!e2_l0_read(g->l0, aoff_ptr(g, vr[i].region), bytes, host)) {
                    char path[512];
                    snprintf(path, sizeof path, "%s/gpu_%s.f32", g->dump_dir, vr[i].name);
                    FILE *f = fopen(path, "wb");
                    if (f) { fwrite(host, 1, bytes, f); fclose(f); }
                }
                free(host);
            }
            size_t bytes = sizeof(float) * (size_t)E2_GPU_VGROUPS * 512;
            void *host = malloc(bytes);
            if (host && !e2_l0_read(g->l0, aoff_ptr(g, A_VSOFT), bytes, host)) {
                char path[512];
                snprintf(path, sizeof path, "%s/gpu_vsoft.f32", g->dump_dir);
                FILE *f = fopen(path, "wb");
                if (f) { fwrite(host, 1, bytes, f); fclose(f); }
            }
            free(host);
        }
        *n_soft_out = 0;
        return 0;
    }
    if (e2_l0_read(g->l0, aoff_ptr(g, A_VSOFT), sizeof(float) * (size_t)ng * 512, soft)) {
        return fail(g, "soft token read: %s", e2_l0_error(g->l0));
    }
    *n_soft_out = ng;
    (void)m;
    return 0;
}

int e2_gpu_image_embed(e2_gpu *g, const float *pixels, const int32_t *position_ids, int n_patches,
                       float *out768) {
    if (!g) return fail(NULL, "GPU context is NULL");
    if (!out768) return fail(g, "output pointer is NULL");
    float *soft = malloc(sizeof(float) * (size_t)E2_GPU_VGROUPS * 512);
    if (!soft) return fail(g, "oom");
    int n_soft = 0;
    if (vision_soft_tokens(g, pixels, position_ids, n_patches, soft, &n_soft)) { free(soft); return -1; }
    if (n_soft <= 0) { free(soft); return 0; } /* E2_GPU_VLAYERS: diagnostic run */
    const int S = n_soft + 4;
    float *e = calloc((size_t)S * 512, sizeof(float));
    if (!e) { free(soft); return fail(g, "oom"); }
    /* <bos> <|image> <|image|> x n_soft <image|> <eos>, the layout src/vision.c builds */
    e2_embed_token(g->m, 2, e);
    e2_embed_token(g->m, 255999, e + 512);
    memcpy(e + 2 * 512, soft, sizeof(float) * (size_t)n_soft * 512);
    e2_embed_token(g->m, 258882, e + (size_t)(2 + n_soft) * 512);
    e2_embed_token(g->m, 1, e + (size_t)(3 + n_soft) * 512);
    free(soft);
    if (e2_l0_write(g->l0, aoff_ptr(g, A_VEMBIN), sizeof(float) * (size_t)S * 512, e)) {
        free(e);
        return fail(g, "embedding upload: %s", e2_l0_error(g->l0));
    }
    free(e);
    return run_chunk(g, NULL, NULL, S, S, 1, out768);
}

int e2_gpu_text_image_embed(e2_gpu *g, const int32_t *ids, int n_ids, const float *pixels,
                            const int32_t *position_ids, int n_patches, float *out768) {
    char verr[256];
    if (!g) return fail(NULL, "GPU context is NULL");
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) return fail(g, "sequence length %d out of range", n_ids);
    if (!out768) return fail(g, "output pointer is NULL");
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) return fail(g, "%s", verr);
    float *soft = malloc(sizeof(float) * (size_t)E2_GPU_VGROUPS * 512);
    float *e = calloc((size_t)n_ids * 512, sizeof(float));
    if (!soft || !e) { free(soft); free(e); return fail(g, "oom"); }
    int n_soft = 0;
    if (vision_soft_tokens(g, pixels, position_ids, n_patches, soft, &n_soft)) {
        free(soft); free(e);
        return -1;
    }
    if (n_soft <= 0) { free(soft); free(e); return 0; }
    int slot = 0;
    for (int s = 0; s < n_ids; s++) {
        if (ids[s] == 258880) { /* <|image|> */
            if (slot >= n_soft) { free(soft); free(e); return fail(g, "more <|image|> slots than soft tokens"); }
            memcpy(e + (size_t)s * 512, soft + (size_t)slot++ * 512, sizeof(float) * 512);
        } else {
            e2_embed_token(g->m, ids[s], e + (size_t)s * 512);
        }
    }
    free(soft);
    if (slot != n_soft) {
        free(e);
        return fail(g, "%d soft tokens for %d slots", n_soft, slot);
    }
    if (e2_l0_write(g->l0, aoff_ptr(g, A_VEMBIN), sizeof(float) * (size_t)n_ids * 512, e)) {
        free(e);
        return fail(g, "embedding upload: %s", e2_l0_error(g->l0));
    }
    free(e);
    return run_chunk(g, NULL, NULL, n_ids, n_ids, 1, out768);
}

int e2_gpu_vision_stats(e2_gpu *g, long *kernels, double *last_ms) {
    if (kernels) *kernels = g->vkernels_last;
    if (last_ms) *last_ms = g->vlast_ms;
    return 0;
}
