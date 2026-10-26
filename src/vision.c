/* vision.c — Gemma4 vision tower (16 layers, 768 wide, 12 heads, axial 2D RoPE) plus the
 * image-only and text+image entry points.
 *
 * The reference (transformers Gemma4VisionModel) runs the encoder over every padded patch with a
 * bidirectional mask and strips the padding at the pooler. That mask is pure overhead here: a padded
 * patch has zero embeddings, a zero position embedding and an attention mask that makes its keys
 * invisible to every real query, so the real tokens' outputs do not depend on it at all (checked
 * against the reference: cosine 1.000002, max|delta| 0.08 on activations of magnitude ~30). The
 * engine therefore gathers the valid patches once and never builds a mask, which also removes the
 * (1,1,2520,2520) bias the ONNX export materialises inside every layer.
 *
 * Attention is a real blocked f32 GEMM pair (logits = Q K^T, context = P V) rather than a per-row
 * dot/softmax/axpy loop, because at 2520 patches the naive form re-reads V once per query row and
 * head: ~19 GB per layer against ~40 MB for the blocked form.
 */
#include <math.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "threads.h"
#include "validation.h"

/* E2_PROFILE=1 prints a wall-clock breakdown of one image call. */
#define VPROF_N 8
static const char *g_vprof_name[VPROF_N] = { "patch", "qkv+o", "attention", "mlp", "norm", "elem", "text", "attn_pack" };
static double g_vprof[VPROF_N];
static int g_vprof_on;
static int g_vdump; /* E2_DUMP set: dev path, keeps the materialised attention matrices */

static double vnow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void vprof_flush(int P) {
    if (!g_vprof_on) return;
    double tot = 0;
    for (int i = 0; i < VPROF_N; i++) tot += g_vprof[i];
    printf("[vprofile P=%d] total %.1f ms\n", P, tot);
    for (int i = 0; i < VPROF_N; i++) {
        printf("   %-10s %8.1f ms  %5.1f%%\n", g_vprof_name[i], g_vprof[i], 100.0 * g_vprof[i] / tot);
        g_vprof[i] = 0;
    }
}

/* E2_DUMP=<dir> writes raw f32 vision intermediates for bisecting against dump_image_fixture.py */
static void vdump(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof path, "%s/v_%s.f32", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(p, 4, n, f); fclose(f); }
}

int e2_vision_init(e2_model *m);
void e2_embed_token(e2_model *m, int id, float *dst);
int e2_text_embed_embeds(e2_model *m, const float *embeds, int S, float *out768);

/* ---------------------------------------------------------------- workspace
 * One function defines size and layout (same rule as the text workspace, for the same reason). */
static size_t vws_layout(e2_vision *v, float *base, int P) {
    float *p = base;
    size_t n = 0;
#define VSLOT(field, cnt)                \
    do {                                 \
        size_t c_ = (size_t)(cnt);       \
        if (v) v->field = p;             \
        p += c_;                         \
        n += c_;                         \
    } while (0)
    const size_t D = E2_V_HIDDEN, I = E2_V_INTER;
    const size_t S = (size_t)P;
    VSLOT(x, S * D);
    VSLOT(res, S * D);
    VSLOT(q, S * D);
    VSLOT(k, S * D);
    VSLOT(v, S * D);
    VSLOT(att, S * D);
    VSLOT(tmp, S * D);
    VSLOT(gate, S * I);
    VSLOT(up, S * I);
    VSLOT(kt, E2_V_HEAD_DIM * S);
    VSLOT(scores, (size_t)e2_nthreads() * E2_V_ATT_BQ * S);
    /* int8 attention scratch. Sizes are in floats; the byte views are taken in vws_alloc. Pp is P
     * padded to 16 so the kernel's 16-key operand loads never cross the packed panels. */
    {
        const size_t Pp = (S + 15) / 16 * 16;
        const size_t nb = (S + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
        const size_t nt = (size_t)e2_nthreads();
        VSLOT(att_q8_words, (S * 64 + 3) / 4);
        VSLOT(att_qs, S);
        VSLOT(att_kp_words, Pp * 16);
        VSLOT(att_ks, Pp);
        VSLOT(att_kz_words, Pp);
        VSLOT(att_vp_words, (size_t)E2_V_HEADS * Pp * 16);
        VSLOT(att_vs, (size_t)E2_V_HEADS * 64);
        VSLOT(att_vz_words, (size_t)E2_V_HEADS * nb * 64);
        VSLOT(att_p8_words, (nt * E2_V_ATT_BQ * S + 3) / 4 + 4);
        VSLOT(att_ps, nt * E2_V_ATT_BQ * nb);
    }
    VSLOT(cos, S * E2_V_HEAD_DIM);
    VSLOT(sin, S * E2_V_HEAD_DIM);
    VSLOT(pool, S * D);
    VSLOT(emb, (size_t)E2_MAX_SEQ * 512);
    VSLOT(qa_words, (S * I + 3) / 4);
    VSLOT(qa_scale, S * (I / E2_I8_BLOCK + 1));
    VSLOT(order_words, S);        /* S ints: the gathered valid patch indices */
    VSLOT(seen_words, (S + 3) / 4); /* S bytes: pooler group occupancy */
#undef VSLOT
    return n;
}

static int vws_alloc(e2_vision *v, int max_patches) {
    if (v->x) return 0;
    const size_t floats = vws_layout(NULL, NULL, max_patches);
    v->bytes = floats * sizeof(float) + 64;
    void *blk = malloc(v->bytes);
    if (!blk) { e2_set_error("oom: %zu bytes of vision workspace", v->bytes); return -1; }
    const size_t used = vws_layout(v, (float *)blk, max_patches);
    if (used != floats) { e2_set_error("vision workspace layout mismatch"); free(blk); memset(v, 0, sizeof *v); return -1; }
    v->qa = (uint8_t *)v->qa_words;
    v->order = (int *)v->order_words;
    v->seen = (uint8_t *)v->seen_words;
    v->att_q8 = (uint8_t *)v->att_q8_words;
    v->att_kp = (uint8_t *)v->att_kp_words;
    v->att_vp = (uint8_t *)v->att_vp_words;
    v->att_p8 = (uint8_t *)v->att_p8_words;
    v->att_kz = (int32_t *)v->att_kz_words;
    v->att_vz = (int32_t *)v->att_vz_words;
    return 0;
}

/* ---------------------------------------------------------------- int8 projections
 * Same scheme as the text tower: per-row weight scales, per-block activation scales, work claimed
 * off a shared counter so the P/E-core speed difference does not gate every region.
 *
 * The projection is segmented so the two groups that read the same normalised activation (q|k|v and
 * gate|up) share *one* quantisation and *one* region: the standalone form quantised the same [P,768]
 * block three times per layer and paid a barrier for each. */
#define E2_VGEMM_MAXSEG 3

struct vgemm_seg {
    float *y;
    int N;
    const e2_q8w *q;
};

struct vgemm_task {
    e2_vision *v;
    const float *x; /* non-NULL: phase 1 quantises x into v->qa */
    int M, Mp, K, nb, total, tile, nseg;
    struct vgemm_seg seg[E2_VGEMM_MAXSEG];
    _Atomic int qnext, qdone, next;
};

static void vgemm_claim(struct vgemm_task *t, int a, int b) {
    int off = 0;
    for (int s = 0; s < t->nseg && off < b; s++) {
        const int n = t->seg[s].N;
        const int n0 = a > off ? a - off : 0;
        const int n1 = b - off < n ? b - off : n;
        if (n1 > n0)
            e2_i8_gemm(t->seg[s].y, (const int8_t *)t->v->qa, t->v->qa_scale, t->seg[s].q, t->Mp, n,
                       n0, n1);
        off += n;
    }
}

static void vgemm_worker(void *arg, int tid, int nthreads) {
    struct vgemm_task *t = arg;
    (void)tid;
    if (t->x) {
        int8_t *qa = (int8_t *)t->v->qa;
        float *asc = t->v->qa_scale;
        for (;;) {
            const int m0 = atomic_fetch_add_explicit(&t->qnext, 32, memory_order_relaxed);
            if (m0 >= t->Mp) break;
            int n = t->Mp - m0;
            if (n > 32) n = 32;
            int nr = n;
            if (m0 + nr > t->M) nr = t->M - m0;
            if (nr > 0)
                e2_i8_quantize(qa + (size_t)m0 * t->K, asc + (size_t)m0 * t->nb,
                               t->x + (size_t)m0 * t->K, nr, t->K, -INFINITY, INFINITY);
            if (nr < n) { /* padded tokens: kept out of the GEMM's way, never read back */
                memset(qa + (size_t)(m0 + nr) * t->K, 0, (size_t)(n - nr) * t->K);
                memset(asc + (size_t)(m0 + nr) * t->nb, 0, sizeof(float) * (size_t)(n - nr) * t->nb);
            }
        }
        atomic_fetch_add_explicit(&t->qdone, 1, memory_order_release);
        while (atomic_load_explicit(&t->qdone, memory_order_acquire) < nthreads) e2_pause();
    }
    for (;;) {
        const int a = atomic_fetch_add_explicit(&t->next, t->tile, memory_order_relaxed);
        if (a >= t->total) break;
        int b = a + t->tile;
        if (b > t->total) b = t->total;
        vgemm_claim(t, a, b);
    }
}

/* x == NULL when the activation block is already quantised in v->qa (the fused gelu below leaves
 * gelu(gate)*up there, so the down projection does not re-read the 28 MB fp32 matrix). */
static void vgemm_segs(e2_vision *v, const float *x, int M, int K, int nseg,
                       const struct vgemm_seg *segs) {
    struct vgemm_task t;
    memset(&t, 0, sizeof t);
    t.v = v;
    t.x = x;
    t.M = M;
    t.Mp = x ? (M + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR : M;
    t.K = K;
    t.nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    if (x) { atomic_init(&t.qnext, 0); atomic_init(&t.qdone, 0); }
    for (int s = 0; s < nseg && s < E2_VGEMM_MAXSEG; s++) {
        t.seg[t.nseg++] = segs[s];
        t.total += segs[s].N;
    }
    t.tile = t.total / (E2_TILE_PER_THREAD * e2_nthreads());
    t.tile = (t.tile + 31) & ~31; /* 32-aligned: the AVX-512 kernel advances 32 rows per load pair */
    if (t.tile < 32) t.tile = 32;
    atomic_init(&t.next, 0);
    e2_parallel(vgemm_worker, &t);
}

static void vgemm_run(e2_vision *v, float *y, const e2_q8w *q, int Mp, int N, int K) {
    const struct vgemm_seg seg = { y, N, q };
    vgemm_segs(v, NULL, Mp, K, 1, &seg);
}

/* The GEMM from an activation block the caller already quantized into v->qa. */
static void vgemm_ready(e2_vision *v, float *y, const e2_q8w *q, int M, int N) {
    const int Mp = (M + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR;
    const int nb = (q->K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    if (Mp > M) {
        memset((int8_t *)v->qa + (size_t)M * q->K, 0, (size_t)(Mp - M) * q->K);
        memset(v->qa_scale + (size_t)M * nb, 0, sizeof(float) * (size_t)(Mp - M) * nb);
    }
    vgemm_run(v, y, q, Mp, N, q->K);
}

static void vgemm(e2_model *m, e2_vision *v, float *y, const float *x, const e2_q8w *q,
                  const uint16_t *w, int M, int N, int K) {
    if (!m->use_int8) { e2_k()->gemm_nt(y, x, w, M, N, K, E2_DT_BF16); return; }
    const struct vgemm_seg seg = { y, N, q };
    vgemm_segs(v, x, M, K, 1, &seg);
}

/* Two/three projections off one activation block: one quantisation, one region. The bf16 path (not
 * the default) keeps the per-projection GEMM because it has no shared quantisation to amortise. */
static void vgemm2(e2_model *m, e2_vision *v, float *y0, float *y1, const float *x, const e2_q8w *q0,
                   const e2_q8w *q1, const uint16_t *w0, const uint16_t *w1, int M, int N, int K) {
    if (!m->use_int8) {
        e2_k()->gemm_nt(y0, x, w0, M, N, K, E2_DT_BF16);
        e2_k()->gemm_nt(y1, x, w1, M, N, K, E2_DT_BF16);
        return;
    }
    const struct vgemm_seg seg[2] = { { y0, N, q0 }, { y1, N, q1 } };
    vgemm_segs(v, x, M, K, 2, seg);
}

static void vgemm3(e2_model *m, e2_vision *v, float *y0, float *y1, float *y2, const float *x,
                   const e2_q8w *q0, const e2_q8w *q1, const e2_q8w *q2, const uint16_t *w0,
                   const uint16_t *w1, const uint16_t *w2, int M, int N, int K) {
    if (!m->use_int8) {
        e2_k()->gemm_nt(y0, x, w0, M, N, K, E2_DT_BF16);
        e2_k()->gemm_nt(y1, x, w1, M, N, K, E2_DT_BF16);
        e2_k()->gemm_nt(y2, x, w2, M, N, K, E2_DT_BF16);
        return;
    }
    const struct vgemm_seg seg[3] = { { y0, N, q0 }, { y1, N, q1 }, { y2, N, q2 } };
    vgemm_segs(v, x, M, K, 3, seg);
}

/* ---------------------------------------------------------------- blocked f32 GEMM
 * The two attention products (logits = Q K^T with K=64, context = P V with N=64) go through the
 * blocked kernel in kernels_avx2.c; here they are split into 64-row M blocks claimed off a counter,
 * which is where the parallelism comes from. The alternative - a dot/softmax/axpy loop per query row
 * and head - re-reads V once per row: ~19 GB per layer against ~40 MB for the blocked form.
 */
struct sgemm_task {
    float *C;
    const float *A, *B;
    int ldc, lda, ldb, M, N, K;
    _Atomic int next;
};

static void sgemm_worker(void *arg, int tid, int nthreads) {
    struct sgemm_task *t = arg;
    (void)tid;
    for (;;) {
        const int m0 = atomic_fetch_add_explicit(&t->next, 64, memory_order_relaxed);
        if (m0 >= t->M) break;
        int m1 = m0 + 64;
        if (m1 > t->M) m1 = t->M;
        e2_sgemm_nn_range(t->C, t->ldc, t->A, t->lda, t->B, t->ldb, t->M, t->N, t->K, m0, m1);
    }
}

static void sgemm_nn(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    struct sgemm_task t = { C, A, B, ldc, lda, ldb, M, N, K, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(sgemm_worker, &t);
}

/* ---------------------------------------------------------------- elementwise helpers */
struct vmap_task {
    float *a;
    const float *b;
    int n, kind; /* 0 gelu, 1 mul, 2 add, 3 copy, 4 scale */
    float s;
    _Atomic int next;
};

static void vmap_worker(void *arg, int tid, int nthreads) {
    struct vmap_task *t = arg;
    (void)tid;
    const int grain = 4096;
    for (;;) {
        const int i0 = atomic_fetch_add_explicit(&t->next, grain, memory_order_relaxed);
        if (i0 >= t->n) break;
        int i1 = i0 + grain;
        if (i1 > t->n) i1 = t->n;
        switch (t->kind) {
        case 0:
            e2_k()->gelu_tanh(t->a + i0, (size_t)(i1 - i0));
            break;
        case 1:
            e2_k()->mul(t->a + i0, t->b + i0, (size_t)(i1 - i0));
            break;
        case 2:
            e2_k()->add(t->a + i0, t->b + i0, (size_t)(i1 - i0));
            break;
        case 3:
            memcpy(t->a + i0, t->b + i0, sizeof(float) * (size_t)(i1 - i0));
            break;
        default:
            e2_k()->scale(t->a + i0, t->s, (size_t)(i1 - i0));
            break;
        }
    }
}

static void vmap(int kind, float *a, const float *b, int n, float s) {
    if (n <= 0) return;
    if (n <= 16384) {
        struct vmap_task t = { a, b, n, kind, s, 0 };
        vmap_worker(&t, 0, 1);
        return;
    }
    struct vmap_task t = { a, b, n, kind, s, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vmap_worker, &t);
}

/* gelu(gate) * up, quantized per E2_I8_BLOCK-element block, one chunk of rows at a time. The
 * standalone form streamed the 28 MB gate matrix three times (gelu, multiply, then the down
 * projection's own quantization); a chunk of 8 rows (~192 KB) is L2-resident, so the tanh result
 * is quantized before it leaves cache and the fp32 matrix is never re-read. */
struct vgu_task {
    float *gate;
    const float *up;
    e2_vision *v;
    int rows, I, nb;
    _Atomic int next;
};
static void vgu_worker(void *arg, int tid, int nthreads) {
    struct vgu_task *t = arg;
    const e2_kernels *K = e2_k();
    (void)tid;
    (void)nthreads;
    const int chunk = 8;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, chunk, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int rn = t->rows - r0;
        if (rn > chunk) rn = chunk;
        float *g = t->gate + (size_t)r0 * t->I;
        const float *u = t->up + (size_t)r0 * t->I;
        K->gelu_tanh(g, (size_t)rn * t->I);
        K->mul(g, u, (size_t)rn * t->I);
        e2_i8_quantize((int8_t *)t->v->qa + (size_t)r0 * t->I,
                       t->v->qa_scale + (size_t)r0 * t->nb, g, rn, t->I, -INFINITY, INFINITY);
    }
}
static void vgelu_up_quant(e2_vision *v, float *gate, const float *up, int rows, int I) {
    struct vgu_task t = { gate, up, v, rows, I, (I + E2_I8_BLOCK - 1) / E2_I8_BLOCK, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vgu_worker, &t);
}

static void vnorm(float *x, int rows, int D, const float *gamma, float eps) {
    for (int r = 0; r < rows; r++) e2_k()->rmsnorm(x + (size_t)r * D, 1, D, gamma, eps);
}

struct vnorm_task {
    float *x;
    int rows, D;
    const float *gamma;
    float eps;
    _Atomic int next;
};

static void vnorm_worker(void *arg, int tid, int nthreads) {
    struct vnorm_task *t = arg;
    (void)tid;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, 32, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int n = t->rows - r0;
        if (n > 32) n = 32;
        e2_k()->rmsnorm(t->x + (size_t)r0 * t->D, n, t->D, t->gamma, t->eps);
    }
}

static void vnorm_par(float *x, int rows, int D, const float *gamma, float eps) {
    if (rows <= 32) { vnorm(x, rows, D, gamma, eps); return; }
    struct vnorm_task t = { x, rows, D, gamma, eps, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vnorm_worker, &t);
}

/* Per-row softmax for the E2_DUMP (materialised attention) path. */
struct vsoft_task {
    float *s;
    int rows, n;
    _Atomic int next;
};

static void vsoft_worker(void *arg, int tid, int nthreads) {
    struct vsoft_task *t = arg;
    (void)tid;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int n = t->rows - r0;
        if (n > 8) n = 8;
        for (int r = r0; r < r0 + n; r++) e2_k()->softmax(t->s + (size_t)r * t->n, t->n);
    }
}

static void vsoftmax(float *s, int rows, int n) {
    struct vsoft_task t = { s, rows, n, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vsoft_worker, &t);
}

/* Fused residual copy + RMSNorm, and RMSNorm + residual add (same fusions as the text tower's
 * map_norm_split/map_norm_add). The vision tower used to run these as three separate map calls
 * each, which is what made the residual bookkeeping ~20% of an image. */
struct vfuse_task {
    float *out, *src, *res;
    int rows, D;
    const float *gamma;
    float eps;
    int mode; /* 0: norm_split (src unused), 1: norm_add */
    _Atomic int next;
};

static void vfuse_worker(void *arg, int tid, int nthreads) {
    struct vfuse_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int n = t->rows - r0;
        if (n > 8) n = 8;
        if (t->mode)
            e2_k()->norm_add(t->out + (size_t)r0 * t->D, t->src + (size_t)r0 * t->D,
                             t->res + (size_t)r0 * t->D, n, t->D, t->gamma, t->eps);
        else
            e2_k()->norm_split(t->out + (size_t)r0 * t->D, t->res + (size_t)r0 * t->D, n, t->D,
                               t->gamma, t->eps);
    }
}

static void vfuse_run(struct vfuse_task *t) {
    if (t->rows <= 8 || e2_nthreads() <= 1) {
        if (t->mode)
            e2_k()->norm_add(t->out, t->src, t->res, t->rows, t->D, t->gamma, t->eps);
        else
            e2_k()->norm_split(t->out, t->res, t->rows, t->D, t->gamma, t->eps);
        return;
    }
    atomic_init(&t->next, 0);
    e2_parallel(vfuse_worker, t);
}

static void vnorm_split_par(float *x, float *res, int rows, int D, const float *gamma, float eps) {
    struct vfuse_task t = { x, NULL, res, rows, D, gamma, eps, 0, 0 };
    vfuse_run(&t);
}

static void vnorm_add_par(float *out, float *src, const float *res, int rows, int D,
                          const float *gamma, float eps) {
    struct vfuse_task t = { out, src, (float *)res, rows, D, gamma, eps, 1, 0 };
    vfuse_run(&t);
}

/* kt[k][p] = k[p][h*64 + k]: the transposed key panel the logits GEMM wants (B is [K,N] row-major). */
struct vtr_task {
    float *kt;
    const float *k;
    int P, h;
    _Atomic int next;
};

static void vtr_worker(void *arg, int tid, int nthreads) {
    struct vtr_task *t = arg;
    (void)tid;
    for (;;) {
        const int k0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (k0 >= E2_V_HEAD_DIM) break;
        int kn = E2_V_HEAD_DIM - k0;
        if (kn > 8) kn = 8;
        for (int k = k0; k < k0 + kn; k++) {
            float *dst = t->kt + (size_t)k * t->P;
            const float *src = t->k + (size_t)t->h * E2_V_HEAD_DIM + k;
            for (int p = 0; p < t->P; p++) dst[p] = src[(size_t)p * E2_V_HIDDEN];
        }
    }
}

static void vtranspose_kt(float *kt, const float *k, int P, int h) {
    struct vtr_task t = { kt, k, P, h, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vtr_worker, &t);
}

/* Blocked attention, one head at a time: a claim owns E2_V_ATT_BQ query rows, computes their logits
 * against every key, softmaxes them and multiplies by V before the next claim. The (P,P) matrix
 * never exists, so it never crosses the memory hierarchy. Claims, not a static split, so the P/E core
 * speed difference does not gate the region. */
struct vatt_task {
    const float *q_h, *v_h, *kt;
    float *att_h, *scores;
    int P;
    _Atomic int next;
};

static void vatt_worker(void *arg, int tid, int nthreads) {
    struct vatt_task *t = arg;
    (void)nthreads;
    const int HD = E2_V_HEAD_DIM, D = E2_V_HIDDEN;
    float *sc = t->scores + (size_t)tid * E2_V_ATT_BQ * t->P;
    for (;;) {
        const int m0 = atomic_fetch_add_explicit(&t->next, E2_V_ATT_BQ, memory_order_relaxed);
        if (m0 >= t->P) break;
        int rows = t->P - m0;
        if (rows > E2_V_ATT_BQ) rows = E2_V_ATT_BQ;
        e2_sgemm_nn_range(sc, t->P, t->q_h + (size_t)m0 * D, D, t->kt, t->P, rows, t->P, HD, 0, rows);
        for (int r = 0; r < rows; r++) e2_k()->softmax(sc + (size_t)r * t->P, t->P);
        e2_sgemm_nn_range(t->att_h + (size_t)m0 * D, D, sc, t->P, t->v_h, D, rows, HD, t->P, 0, rows);
    }
}

static void vattention(e2_vision *v, int h, int P) {
    struct vatt_task t = { v->q + (size_t)h * E2_V_HEAD_DIM, v->v + (size_t)h * E2_V_HEAD_DIM,
                           v->kt, v->att + (size_t)h * E2_V_HEAD_DIM, v->scores, P, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vatt_worker, &t);
}

/* ---------------------------------------------------------------- int8 attention
 * The fp32 products above are bandwidth-bound at 8 threads: the K and V panels are 590 KB per head
 * and every 32-row claim re-reads them. Quantizing Q/K/P/V^T the way the text tower quantizes its
 * activations cuts every panel 4x (and the products become VNNI), which is what actually makes the
 * region scale. Same decomposition as the GPU backends: per-row Q/K scales, V^T per dim, P per
 * 128-key block; the int8 dot is unsigned(q+128) x signed, so each product subtracts 128*sum(B).
 *
 * One kernel serves both products - C[m][n] = bsc[n] * sum_b asc[m][b] * (A[m] . B[n] - 128*bz).
 * A is unsigned [M][K] with row stride lda, B is packed k-quad-major [K/4][Np][4] so 16 output
 * columns are one 64-byte load, and `ldc` lets the context product write straight into the strided
 * head columns of `att`. The kernel itself lives in kernels_i8_avx512_vnni.c and is reached through
 * the e2_i8_attn_gemm dispatch, so the text tower's attention shares it. */
static int vattn_have_simd(void) { return e2_i8_attn_gemm != NULL; }

/* Scalar form of the same operation, kept as the reference while bringing the SIMD one up
 * (E2_VATTN_REF=1 selects it). */
static void vattn_gemm_ref(float *C, int ldc, const uint8_t *A, const float *asc,
                           const uint8_t *Bp, const float *bsc, const int32_t *bz, int M, int Np,
                           int N, int K) {
    const int nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                int s = 0;
                int k1 = b * E2_I8_BLOCK + E2_I8_BLOCK;
                if (k1 > K) k1 = K;
                for (int k = b * E2_I8_BLOCK; k < k1; k++) {
                    const int av = (int)(uint8_t)A[(size_t)m * K + k];
                    const int bv = (int)(int8_t)Bp[((size_t)(k >> 2) * Np + n) * 4 + (k & 3)];
                    s += av * bv;
                }
                s -= 128 * bz[(size_t)b * Np + n];
                acc += asc[(size_t)m * nb + b] * (float)s;
            }
            C[(size_t)m * ldc + n] = acc * bsc[n];
        }
    }
}

/* Q per (patch, head) row -> unsigned int8 + one scale; the source is strided by the hidden. */
struct vq_task {
    const float *q;
    int stride, h, P;
    uint8_t *out;
    float *sc;
    _Atomic int next;
};
static void vq_worker(void *arg, int tid, int nthreads) {
    struct vq_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int r = atomic_fetch_add_explicit(&t->next, 16, memory_order_relaxed);
        if (r >= t->P) break;
        int rn = t->P - r;
        if (rn > 16) rn = 16;
        for (int i = 0; i < rn; i++) {
            const float *s = t->q + (size_t)(r + i) * t->stride + (size_t)t->h * 64;
            float amax = 0.f;
            for (int d = 0; d < 64; d++) {
                const float a = fabsf(s[d]);
                if (a > amax) amax = a;
            }
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            const float ri = amax > 0.f ? 127.f / amax : 0.f;
            uint8_t *o = t->out + (size_t)(r + i) * 64;
            for (int d = 0; d < 64; d++) {
                int v = (int)lrintf(s[d] * ri);
                if (v > 127) v = 127; else if (v < -127) v = -127;
                o[d] = (uint8_t)(v + 128);
            }
            t->sc[r + i] = sc;
        }
    }
}

/* K -> signed int8 packed k-quad-major [16][Pp][4], one scale per key row, and the row sum the
 * +128 activation bias needs. Keys past P are zero. */
struct vk_task {
    const float *k;
    int stride, h, P, Pp;
    uint8_t *out;
    float *sc;
    int32_t *z;
    _Atomic int next;
};
static void vk_worker(void *arg, int tid, int nthreads) {
    struct vk_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int n = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (n >= t->Pp) break;
        int nn = t->Pp - n;
        if (nn > 8) nn = 8;
        for (int i = 0; i < nn; i++) {
            const int row = n + i;
            int32_t zz = 0;
            if (row < t->P) {
                const float *s = t->k + (size_t)row * t->stride + (size_t)t->h * 64;
                float amax = 0.f;
                for (int d = 0; d < 64; d++) {
                    const float a = fabsf(s[d]);
                    if (a > amax) amax = a;
                }
                const float sc = amax > 0.f ? amax / 127.f : 1.f;
                const float ri = amax > 0.f ? 127.f / amax : 0.f; /* a multiply, not 64 divides */
                for (int d = 0; d < 64; d++) {
                    int v = (int)lrintf(s[d] * ri);
                    if (v > 127) v = 127; else if (v < -127) v = -127;
                    t->out[((size_t)(d >> 2) * t->Pp + row) * 4 + (d & 3)] = (uint8_t)(int8_t)v;
                    zz += v;
                }
                t->sc[row] = sc;
            } else {
                for (int d = 0; d < 64; d++)
                    t->out[((size_t)(d >> 2) * t->Pp + row) * 4 + (d & 3)] = 0;
                t->sc[row] = 1.f;
            }
            t->z[row] = zz;
        }
    }
}

/* V^T -> signed int8 packed [H][Pp/4][64][4] (key-major), one scale per dim and the per-128-key
 * block sums the context product's +128 bias needs. One work item is a whole head and reads the
 * source by the patch row (64 contiguous floats), so the reads stream instead of striding by the
 * hidden - the per-dim version was 15x this one. */
struct vv_task {
    const float *v;
    int P, Pp, nb, H;
    uint8_t *out;
    float *sc;
    int32_t *z;
    _Atomic int next;
};
static void vv_worker(void *arg, int tid, int nthreads) {
    struct vv_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int h = atomic_fetch_add_explicit(&t->next, 1, memory_order_relaxed);
        if (h >= t->H) break;
        uint8_t *out = t->out + (size_t)h * (size_t)(t->Pp / 4) * 64 * 4;
        float *sc = t->sc + (size_t)h * 64;
        int32_t *zz = t->z + (size_t)h * t->nb * 64;
        float amax[64], ri[64];
        for (int d = 0; d < 64; d++) amax[d] = 0.f;
        for (int j = 0; j < t->P; j++) {
            const float *s = t->v + (size_t)j * E2_V_HIDDEN + (size_t)h * 64;
            for (int d = 0; d < 64; d++) {
                const float a = fabsf(s[d]);
                if (a > amax[d]) amax[d] = a;
            }
        }
        for (int d = 0; d < 64; d++) {
            sc[d] = amax[d] > 0.f ? amax[d] / 127.f : 1.f;
            ri[d] = amax[d] > 0.f ? 127.f / amax[d] : 0.f;
        }
        for (int b = 0; b < t->nb; b++)
            for (int d = 0; d < 64; d++) zz[(size_t)b * 64 + d] = 0;
        for (int j = 0; j < t->Pp; j++) {
            int q[64];
            if (j < t->P) {
                const float *s = t->v + (size_t)j * E2_V_HIDDEN + (size_t)h * 64;
                for (int d = 0; d < 64; d++) {
                    int v = (int)lrintf(s[d] * ri[d]);
                    if (v > 127) v = 127; else if (v < -127) v = -127;
                    q[d] = v;
                    zz[(size_t)(j >> 7) * 64 + d] += v;
                }
            } else {
                for (int d = 0; d < 64; d++) q[d] = 0;
            }
            for (int d = 0; d < 64; d++)
                out[((size_t)(j >> 2) * 64 + d) * 4 + (j & 3)] = (uint8_t)(int8_t)q[d];
        }
    }
}

static int vattn_ref_on(void) {
    static int r = -1;
    if (r < 0) r = getenv("E2_VATTN_REF") != NULL;
    return r;
}

static void vattn_gemm(float *C, int ldc, const uint8_t *A, const float *asc, const uint8_t *Bp,
                       const float *bsc, const int32_t *bz, int M, int Np, int N, int K) {
    if (vattn_ref_on()) vattn_gemm_ref(C, ldc, A, asc, Bp, bsc, bz, M, Np, N, K);
    else e2_i8_attn_gemm(C, ldc, A, K, asc, Bp, bsc, bz, M, Np, N, K);
}

struct vatt8_task {
    float *att, *scores;
    uint8_t *q8, *kp, *vp, *p8;
    float *qs, *ks, *vs, *ps;
    int32_t *kz, *vz;
    int P, Pp, nb, h;
    _Atomic int next;
};
static void vatt8_worker(void *arg, int tid, int nthreads) {
    struct vatt8_task *t = arg;
    const e2_kernels *K = e2_k();
    (void)nthreads;
    const int BQ = E2_V_ATT_BQ;
    float *sc = t->scores + (size_t)tid * BQ * t->P;
    uint8_t *p8 = t->p8 + (size_t)tid * BQ * t->P;
    float *ps = t->ps + (size_t)tid * BQ * t->nb;
    for (;;) {
        const int m0 = atomic_fetch_add_explicit(&t->next, BQ, memory_order_relaxed);
        if (m0 >= t->P) break;
        int rows = t->P - m0;
        if (rows > BQ) rows = BQ;
        const uint8_t *vp = t->vp + (size_t)t->h * (size_t)(t->Pp / 4) * 64 * 4;
        const float *vs = t->vs + (size_t)t->h * 64;
        const int32_t *vz = t->vz + (size_t)t->h * t->nb * 64;
        vattn_gemm(sc, t->P, t->q8 + (size_t)m0 * 64, t->qs + m0, t->kp, t->ks, t->kz, rows,
                   t->Pp, t->P, 64);
        if (e2_softmax_quant_i8) {
            e2_softmax_quant_i8(sc, (int8_t *)p8, ps, rows, t->P, t->nb);
        } else {
            for (int i = 0; i < rows; i++) K->softmax(sc + (size_t)i * t->P, t->P);
            e2_i8_quantize((int8_t *)p8, ps, sc, rows, t->P, -INFINITY, INFINITY);
        }
        vattn_gemm(t->att + (size_t)m0 * 768 + (size_t)t->h * 64, 768, p8, ps, vp, vs, vz, rows, 64,
                   64, t->P);
    }
}

static int vattn_int8_ok(e2_model *m) {
    if (getenv("E2_VATTN_OFF")) return 0; /* A/B: force the fp32 attention */
    return m->use_int8 && e2_i8_quantize != NULL && vattn_have_simd();
}

static void vattn_pack_qk(e2_vision *v, int h, int P) {
    const int Pp = (P + 15) / 16 * 16;
    { /* Q */
        struct vq_task t = { v->q, E2_V_HIDDEN, h, P, v->att_q8, v->att_qs, 0 };
        atomic_init(&t.next, 0);
        e2_parallel(vq_worker, &t);
    }
    { /* K */
        struct vk_task t = { v->k, E2_V_HIDDEN, h, P, Pp, v->att_kp, v->att_ks, v->att_kz, 0 };
        atomic_init(&t.next, 0);
        e2_parallel(vk_worker, &t);
    }
}

static void vattn_pack_vt_all(e2_vision *v, int P) {
    const int Pp = (P + 15) / 16 * 16;
    struct vv_task t = { v->v, P, Pp, (P + E2_I8_BLOCK - 1) / E2_I8_BLOCK, E2_V_HEADS, v->att_vp,
                         v->att_vs, v->att_vz, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(vv_worker, &t);
}

static void vattention_i8(e2_vision *v, int h, int P) {
    struct vatt8_task t = { v->att,     v->scores, v->att_q8, v->att_kp, v->att_vp, v->att_p8,
                            v->att_qs,  v->att_ks, v->att_vs, v->att_ps, v->att_kz, v->att_vz,
                            P,          (P + 15) / 16 * 16, (P + E2_I8_BLOCK - 1) / E2_I8_BLOCK, h,
                            0 };
    atomic_init(&t.next, 0);
    e2_parallel(vatt8_worker, &t);
}

/* ---------------------------------------------------------------- axial RoPE tables
 * cos/sin for one patch: channels 0..31 rotate by the x coordinate, 32..63 by y, each with 16
 * frequencies repeated twice (the reference's H-H-W-W recomposition). */
static void rope_tables(e2_vision *v, const int32_t *pos, const int *order, int P) {
    for (int p = 0; p < P; p++) {
        const int x = pos[2 * order[p]], y = pos[2 * order[p] + 1];
        float *c = v->cos + (size_t)p * E2_V_HEAD_DIM;
        float *s = v->sin + (size_t)p * E2_V_HEAD_DIM;
        for (int i = 0; i < E2_V_HEAD_DIM / 4; i++) {
            const float fh = (float)x * v->rope_inv[i];
            const float fw = (float)y * v->rope_inv[i];
            c[i] = c[i + 16] = cosf(fh);
            s[i] = s[i + 16] = sinf(fh);
            c[32 + i] = c[48 + i] = cosf(fw);
            s[32 + i] = s[48 + i] = sinf(fw);
        }
    }
}

/* ---------------------------------------------------------------- load */
static float *vnorm_weight(e2_model *m, const char *name, int64_t n) {
    float *dst = malloc(sizeof(float) * (size_t)n);
    if (!dst) return NULL;
    if (e2_copy_bf16_f32(m, name, n, dst)) { free(dst); return NULL; }
    return dst;
}

int e2_vision_init(e2_model *m) {
    e2_vision *v = calloc(1, sizeof *v);
    if (!v) { e2_set_error("oom"); return -1; }
    m->vision = v;

    const int64_t dp[2] = { E2_V_HIDDEN, E2_V_PATCH };
    const int64_t dpos[3] = { 2, E2_V_POS, E2_V_HIDDEN }; /* checked exactly: positions are bounded by it */
    const int64_t dep[2] = { 512, E2_V_HIDDEN };
    v->patch_proj = e2_tensor_bf16(m, "vision_tower.patch_embedder.input_proj.weight", 2, dp);
    v->pos_table = e2_tensor_bf16(m, "vision_tower.patch_embedder.position_embedding_table", 3, dpos);
    v->embed_proj = e2_tensor_bf16(m, "embed_vision.embedding_projection.weight", 2, dep);
    if (!e2_weight_ok(v->patch_proj) || !e2_weight_ok(v->pos_table) || !e2_weight_ok(v->embed_proj))
        return -1;

    char n_[13][112];
    const char *suffix[13] = {
            "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight",
            "self_attn.v_proj.linear.weight", "self_attn.o_proj.linear.weight",
            "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight", "mlp.down_proj.linear.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    };
    for (int l = 0; l < E2_V_LAYERS; l++) {
        e2_vlayer *L = &v->layers[l];
        for (int s = 0; s < 13; s++)
            snprintf(n_[s], sizeof n_[s], "vision_tower.encoder.layers.%d.%s", l, suffix[s]);
        const int64_t dq[2] = { E2_V_HEADS * E2_V_HEAD_DIM, E2_V_HIDDEN };
        const int64_t di[2] = { E2_V_INTER, E2_V_HIDDEN };
        const int64_t dd[2] = { E2_V_HIDDEN, E2_V_INTER };
        const int64_t dh[2] = { E2_V_HIDDEN, E2_V_HIDDEN };
        L->q_proj = e2_tensor_bf16(m, n_[0], 2, dq);
        L->k_proj = e2_tensor_bf16(m, n_[1], 2, dq);
        L->v_proj = e2_tensor_bf16(m, n_[2], 2, dq);
        L->o_proj = e2_tensor_bf16(m, n_[3], 2, dh);
        L->gate_proj = e2_tensor_bf16(m, n_[4], 2, di);
        L->up_proj = e2_tensor_bf16(m, n_[5], 2, di);
        L->down_proj = e2_tensor_bf16(m, n_[6], 2, dd);
        if (!e2_weight_ok(L->q_proj) || !e2_weight_ok(L->k_proj) || !e2_weight_ok(L->v_proj) ||
            !e2_weight_ok(L->o_proj) || !e2_weight_ok(L->gate_proj) || !e2_weight_ok(L->up_proj) ||
            !e2_weight_ok(L->down_proj))
            return -1;
        L->input_ln = vnorm_weight(m, n_[7], E2_V_HIDDEN);
        L->post_attn_ln = vnorm_weight(m, n_[8], E2_V_HIDDEN);
        L->pre_ffn_ln = vnorm_weight(m, n_[9], E2_V_HIDDEN);
        L->post_ffn_ln = vnorm_weight(m, n_[10], E2_V_HIDDEN);
        L->q_norm = vnorm_weight(m, n_[11], E2_V_HEAD_DIM);
        L->k_norm = vnorm_weight(m, n_[12], E2_V_HEAD_DIM);
        if (!L->input_ln || !L->post_attn_ln || !L->pre_ffn_ln || !L->post_ffn_ln || !L->q_norm ||
            !L->k_norm)
            return -1;
    }

    /* rope_theta 100, spatial_dim = head_dim/2 = 32, arange(0,32,2)/32 */
    for (int i = 0; i < E2_V_HEAD_DIM / 4; i++)
        v->rope_inv[i] = 1.0f / powf(100.0f, (float)(2 * i) / (float)(E2_V_HEAD_DIM / 2));

    if (m->use_int8) {
        if (e2_q8w_load(m, &v->q8_patch, "vision_tower.patch_embedder.input_proj.weight",
                        v->patch_proj, E2_V_HIDDEN, E2_V_PATCH) ||
            e2_q8w_load(m, &v->q8_embed_proj, "embed_vision.embedding_projection.weight",
                        v->embed_proj, 512, E2_V_HIDDEN))
            return -1;
        for (int l = 0; l < E2_V_LAYERS; l++) {
            e2_vlayer *L = &v->layers[l];
            for (int s = 0; s < 7; s++)
                snprintf(n_[s], sizeof n_[s], "vision_tower.encoder.layers.%d.%s", l, suffix[s]);
            if (e2_q8w_load(m, &L->q8_q, n_[0], L->q_proj, E2_V_HEADS * E2_V_HEAD_DIM, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_k, n_[1], L->k_proj, E2_V_HEADS * E2_V_HEAD_DIM, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_v, n_[2], L->v_proj, E2_V_HEADS * E2_V_HEAD_DIM, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_o, n_[3], L->o_proj, E2_V_HIDDEN, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_gate, n_[4], L->gate_proj, E2_V_INTER, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_up, n_[5], L->up_proj, E2_V_INTER, E2_V_HIDDEN) ||
                e2_q8w_load(m, &L->q8_down, n_[6], L->down_proj, E2_V_HIDDEN, E2_V_INTER))
                return -1;
        }
    }
    if (vws_alloc(v, E2_V_MAX_PATCHES)) return -1;
    return 0;
}

void e2_vision_free(e2_vision *v) {
    if (!v) return;
    for (int l = 0; l < E2_V_LAYERS; l++) {
        e2_vlayer *L = &v->layers[l];
        free(L->input_ln);
        free(L->post_attn_ln);
        free(L->pre_ffn_ln);
        free(L->post_ffn_ln);
        free(L->q_norm);
        free(L->k_norm);
        e2_q8w_free(&L->q8_q);
        e2_q8w_free(&L->q8_k);
        e2_q8w_free(&L->q8_v);
        e2_q8w_free(&L->q8_o);
        e2_q8w_free(&L->q8_gate);
        e2_q8w_free(&L->q8_up);
        e2_q8w_free(&L->q8_down);
    }
    e2_q8w_free(&v->q8_patch);
    e2_q8w_free(&v->q8_embed_proj);
    free(v->x);
    free(v);
}

/* ---------------------------------------------------------------- the tower
 * Returns the number of soft tokens (one per 3x3 patch group, padding stripped) in v->pool. */
static int vision_tower(e2_model *m, const float *pixels, const int32_t *position_ids, int n_patches) {
    e2_vision *v = m->vision;
    const int D = E2_V_HIDDEN, I = E2_V_INTER, H = E2_V_HEADS, HD = E2_V_HEAD_DIM;
    g_vprof_on = getenv("E2_PROFILE") != NULL;
    g_vdump = getenv("E2_DUMP") != NULL;
    double tp0 = vnow();

    /* gather the valid patches: everything downstream then runs without a mask */
    int P = 0;
    for (int i = 0; i < n_patches; i++)
        if (position_ids[2 * i] >= 0 && position_ids[2 * i + 1] >= 0) v->order[P++] = i;
    if (P == 0) { e2_set_error("image has no valid patches"); return -1; }

    /* patch embedder: 2*(x-0.5) -> input_proj, plus the 2D position embedding sum */
    for (int p = 0; p < P; p++) {
        const float *src = pixels + (size_t)v->order[p] * E2_V_PATCH;
        float *dst = v->x + (size_t)p * E2_V_PATCH;
        for (int i = 0; i < E2_V_PATCH; i++) dst[i] = 2.0f * (src[i] - 0.5f);
    }
    {
        /* the projection output is [P,768]; run it into a scratch of its own width by using the
         * padded M of the GEMM helper, so `res` doubles as the destination here */
        vgemm(m, v, v->res, v->x, &v->q8_patch, v->patch_proj, P, D, E2_V_PATCH);
        for (int p = 0; p < P; p++) {
            const int x = position_ids[2 * v->order[p]];
            const int y = position_ids[2 * v->order[p] + 1];
            const uint16_t *ex = v->pos_table + ((size_t)x * D);
            const uint16_t *ey = v->pos_table + ((size_t)E2_V_POS + (size_t)y) * D;
            float *dst = v->res + (size_t)p * D;
            for (int d = 0; d < D; d++) dst[d] += e2_bf16f(ex[d]) + e2_bf16f(ey[d]);
        }
    }
    memcpy(v->x, v->res, sizeof(float) * (size_t)P * D);
    if (g_vprof_on) g_vprof[0] += vnow() - tp0;
    vdump("patch_embed", v->x, (size_t)P * D);
    rope_tables(v, position_ids, v->order, P);
    const int i8attn = vattn_int8_ok(m);

    for (int l = 0; l < E2_V_LAYERS; l++) {
        e2_vlayer *L = &v->layers[l];
        vnorm_split_par(v->x, v->res, P, D, L->input_ln, m->rms_eps);
        if (l == 0) vdump("l0_input_ln", v->x, (size_t)P * D);

        double t0 = g_vprof_on ? vnow() : 0;
        vgemm3(m, v, v->q, v->k, v->v, v->x, &L->q8_q, &L->q8_k, &L->q8_v, L->q_proj, L->k_proj,
               L->v_proj, P, H * HD, D);
        if (g_vprof_on) { g_vprof[1] += vnow() - t0; t0 = vnow(); }
        if (l == 0) {
            vdump("l0_q_proj", v->q, (size_t)P * D);
            vdump("l0_k_proj", v->k, (size_t)P * D);
            vdump("l0_v_proj", v->v, (size_t)P * D);
        }
        vnorm_par(v->q, P * H, HD, L->q_norm, m->rms_eps);
        vnorm_par(v->k, P * H, HD, L->k_norm, m->rms_eps);
        vnorm_par(v->v, P * H, HD, NULL, m->rms_eps); /* v_norm has no learned scale */
        if (l == 0) {
            vdump("l0_q_norm", v->q, (size_t)P * D);
            vdump("l0_k_norm", v->k, (size_t)P * D);
            vdump("l0_v_norm", v->v, (size_t)P * D);
        }
        /* block = head_dim/2: the reference splits each head into an x half and a y half and
         * rotate_half's inside each, so channels pair at distance head_dim/4, not head_dim/2 */
        if (g_vprof_on) { g_vprof[4] += vnow() - t0; t0 = vnow(); }
        e2_k()->rope_rows(v->q, 0, P * H, H, HD, HD / 2, v->cos, v->sin);
        e2_k()->rope_rows(v->k, 0, P * H, H, HD, HD / 2, v->cos, v->sin);
        if (g_vprof_on) { g_vprof[5] += vnow() - t0; t0 = vnow(); }
        if (l == 0) {
            vdump("l0_q_rope", v->q, (size_t)P * D);
            vdump("l0_k_rope", v->k, (size_t)P * D);
            vdump("l0_cos", v->cos, (size_t)P * HD);
        }

        double tpack = 0.0;
        if (i8attn) {
            const double tp = g_vprof_on ? vnow() : 0.0;
            vattn_pack_vt_all(v, P);
            if (g_vprof_on) tpack += vnow() - tp;
        }
        for (int h = 0; h < H; h++) {
            if (i8attn) {
                const double tp = g_vprof_on ? vnow() : 0.0;
                vattn_pack_qk(v, h, P);
                if (g_vprof_on) tpack += vnow() - tp;
                vattention_i8(v, h, P);
                continue;
            }
            vtranspose_kt(v->kt, v->k, P, h);
            if (g_vdump) {
                /* E2_DUMP: keep the materialised (P,P) form, which is what the bisection fixtures
                 * (tools/dump_image_fixture.py) compare against. Dev path only. */
                float *full = malloc(sizeof(float) * (size_t)P * P);
                sgemm_nn(full, P, v->q + (size_t)h * HD, D, v->kt, P, P, P, HD);
                if (l == 0 && h == 0) vdump("l0_scores", full, (size_t)P * P);
                vsoftmax(full, P, P);
                if (l == 0 && h == 0) vdump("l0_probs", full, (size_t)P * P);
                sgemm_nn(v->att + (size_t)h * HD, D, full, P, v->v + (size_t)h * HD, D, P, HD, P);
                free(full);
            } else {
                vattention(v, h, P);
            }
        }
        if (g_vprof_on) { g_vprof[2] += vnow() - t0 - tpack; g_vprof[7] += tpack; t0 = vnow(); }
        if (l == 0) vdump("l0_ctx", v->att, (size_t)P * D);
        vgemm(m, v, v->tmp, v->att, &L->q8_o, L->o_proj, P, D, D);
        if (g_vprof_on) { g_vprof[1] += vnow() - t0; t0 = vnow(); }
        if (l == 0) vdump("l0_o_proj", v->tmp, (size_t)P * D);
        vnorm_add_par(v->x, v->tmp, v->res, P, D, L->post_attn_ln, m->rms_eps);
        if (l == 0) vdump("l0_x_attn", v->x, (size_t)P * D);

        if (g_vprof_on) { g_vprof[4] += vnow() - t0; t0 = vnow(); }
        vnorm_split_par(v->x, v->res, P, D, L->pre_ffn_ln, m->rms_eps);
        if (g_vprof_on) { g_vprof[4] += vnow() - t0; t0 = vnow(); }
        vgemm2(m, v, v->gate, v->up, v->x, &L->q8_gate, &L->q8_up, L->gate_proj, L->up_proj, P, I, D);
        if (g_vprof_on) { g_vprof[3] += vnow() - t0; t0 = vnow(); }
        if (m->use_int8) {
            /* gelu(gate)*up and its quantization in one L2-resident pass, then the down projection */
            vgelu_up_quant(v, v->gate, v->up, P, I);
            if (l == 0) vdump("l0_mlp_gate", v->gate, (size_t)P * I);
            vgemm_ready(v, v->tmp, &L->q8_down, P, D);
            vnorm_add_par(v->x, v->tmp, v->res, P, D, L->post_ffn_ln, m->rms_eps);
        } else {
            vmap(0, v->gate, NULL, P * I, 0);
            vmap(1, v->gate, v->up, P * I, 0);
            if (l == 0) vdump("l0_mlp_gate", v->gate, (size_t)P * I);
            vgemm(m, v, v->tmp, v->gate, &L->q8_down, L->down_proj, P, D, I);
        }
        if (l == 0) vdump("l0_mlp_out", v->tmp, (size_t)P * D);
        if (!m->use_int8)
            vnorm_add_par(v->x, v->tmp, v->res, P, D, L->post_ffn_ln, m->rms_eps);
        if (g_vprof_on) { g_vprof[5] += vnow() - t0; }
        if (l == 0 || l == 1 || l == E2_V_LAYERS - 1) {
            char nm[32];
            snprintf(nm, sizeof nm, "layer%d", l);
            vdump(nm, v->x, (size_t)P * D);
        }
    }

    /* pooler: 3x3 average by patch position, /k^2, then scale by sqrt(hidden) */
    {
        int max_x = 0, max_y = 0;
        for (int p = 0; p < P; p++) {
            const int x = position_ids[2 * v->order[p]];
            const int y = position_ids[2 * v->order[p] + 1];
            if (x > max_x) max_x = x;
            if (y > max_y) max_y = y;
        }
        const int cols = max_x / E2_V_POOL + 1;
        const int groups = cols * (max_y / E2_V_POOL + 1);
        memset(v->pool, 0, sizeof(float) * (size_t)groups * D);
        uint8_t *seen = v->seen;
        memset(seen, 0, (size_t)groups);
        for (int p = 0; p < P; p++) {
            const int x = position_ids[2 * v->order[p]] / E2_V_POOL;
            const int y = position_ids[2 * v->order[p] + 1] / E2_V_POOL;
            const int g = x + cols * y;
            seen[g] = 1;
            float *dst = v->pool + (size_t)g * D;
            const float *src = v->x + (size_t)p * D;
            for (int d = 0; d < D; d++) dst[d] += src[d];
        }
        const float inv = (float)E2_V_HIDDEN, scale = 1.0f / (float)(E2_V_POOL * E2_V_POOL);
        (void)0;
        int out = 0;
        for (int g = 0; g < groups; g++) {
            if (!seen[g]) continue;
            float *dst = v->pool + (size_t)out * D;
            const float *src = v->pool + (size_t)g * D;
            for (int d = 0; d < D; d++) dst[d] = src[d] * scale * sqrtf(inv);
            out++;
        }
        return out;
    }
}

/* ---------------------------------------------------------------- entry points */
/* The public image arguments, checked before the tower touches a weight: it reads `pixels` and
 * `position_ids` and writes `out768`, so none of them may be NULL. */
static int vision_args_ok(e2_model *m, const float *pixels, const int32_t *position_ids,
                          float *out768) {
    if (!m) { e2_set_error("model is NULL"); return -1; }
    if (!pixels) { e2_set_error("pixels pointer is NULL"); return -1; }
    if (!position_ids) { e2_set_error("position ids pointer is NULL"); return -1; }
    if (!out768) { e2_set_error("output pointer is NULL"); return -1; }
    return 0;
}

static int image_soft_tokens(e2_model *m, const float *pixels, const int32_t *position_ids, int n_patches) {
    e2_vision *v = m->vision;
    char verr[256];
    if (!v) { e2_set_error("vision tower not loaded"); return -1; }
    if (!pixels) { e2_set_error("pixels pointer is NULL"); return -1; }
    if (n_patches <= 0 || n_patches > E2_V_MAX_PATCHES) {
        e2_set_error("patch count %d out of range (max %d)", n_patches, E2_V_MAX_PATCHES);
        return -1;
    }
    /* Two capacity bounds apply before the tower starts: the pool/seen buffers hold
     * E2_V_MAX_PATCHES rows (vws_alloc), the axial table is [2, E2_V_POS, 768], and the pooler's
     * output becomes the image's soft tokens, which the <bos> <|image|> <image|> <eos> layout wraps
     * in 4 more, so the grid also has to fit the context. */
    if (e2_valid_positions(position_ids, n_patches, E2_MAX_SEQ - 4, NULL, NULL, verr, sizeof verr)) {
        e2_set_error("%s", verr);
        return -1;
    }
    const int n = vision_tower(m, pixels, position_ids, n_patches);
    if (n < 0) return -1;
    const double t0 = g_vprof_on ? vnow() : 0;
    /* embed_vision: scale-free RMSNorm over 768, then 768 -> 512 */
    vdump("pooler_out", v->pool, (size_t)n * E2_V_HIDDEN);
    vnorm_par(v->pool, n, E2_V_HIDDEN, NULL, m->rms_eps);
    vgemm(m, v, v->emb, v->pool, &v->q8_embed_proj, v->embed_proj, n, 512, E2_V_HIDDEN);
    if (g_vprof_on) g_vprof[0] += vnow() - t0;
    vdump("soft_tokens", v->emb, (size_t)n * 512);
    return n;
}

/* The vision soft tokens live in v->emb; the audio-side splice reads them from there. */
int e2_vision_soft_tokens(e2_model *m, const float *pixels, const int32_t *position_ids, int n_patches,
                          const float **soft) {
    e2_vision *v = m ? m->vision : NULL;
    if (!soft) { e2_set_error("soft-token output pointer is NULL"); return -1; }
    if (!v) { e2_set_error("vision tower not loaded"); return -1; }
    const int n = image_soft_tokens(m, pixels, position_ids, n_patches);
    if (n < 0) return -1;
    *soft = v->emb;
    return n;
}

int e2_image_embed(e2_model *m, const float *pixels, const int32_t *position_ids, int n_patches, float *out768) {
    if (vision_args_ok(m, pixels, position_ids, out768)) return -1;
    e2_vision *v = m->vision;
    if (!v) { e2_set_error("vision tower not loaded"); return -1; }
    const double t_v = g_vprof_on ? vnow() : 0;
    const int n_soft = image_soft_tokens(m, pixels, position_ids, n_patches);
    if (n_soft < 0) return -1;
    if (g_vprof_on) { printf("[vprofile] vision tower+proj %.1f ms\n", vnow() - t_v); vprof_flush(n_soft * 9); }
    const int S = n_soft + 4;
    if (S > E2_MAX_SEQ) { e2_set_error("image layout of %d tokens exceeds the context", S); return -1; }

    /* v->emb holds n_soft rows of 512d soft tokens; build the text input embedding block in `pool`
     * (free now) and hand it to the text tower */
    float *e = v->pool;
    e2_embed_token(m, 2, e);
    e2_embed_token(m, 255999, e + 512);
    for (int i = 0; i < n_soft; i++)
        memcpy(e + (size_t)(2 + i) * 512, v->emb + (size_t)i * 512, sizeof(float) * 512);
    e2_embed_token(m, 258882, e + (size_t)(2 + n_soft) * 512);
    e2_embed_token(m, 1, e + (size_t)(3 + n_soft) * 512);
    const double t_t = g_vprof_on ? vnow() : 0;
    int rc = e2_text_embed_embeds(m, e, S, out768);
    if (g_vprof_on) printf("[vprofile] text tower %.1f ms\n", vnow() - t_t);
    return rc;
}

int e2_text_image_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pixels,
                        const int32_t *position_ids, int n_patches, float *out768) {
    char verr[256];
    if (vision_args_ok(m, pixels, position_ids, out768)) return -1;
    e2_vision *v = m->vision;
    if (!v) { e2_set_error("vision tower not loaded"); return -1; }
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) { e2_set_error("sequence length %d out of range", n_ids); return -1; }
    /* The ids are gathered after the tower, but a bad one is bad input either way: reject it before
     * the image costs ~700 ms of work. */
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) { e2_set_error("%s", verr); return -1; }
    const int n_soft = image_soft_tokens(m, pixels, position_ids, n_patches);
    if (n_soft < 0) return -1;

    float *e = v->pool;
    int slot = 0;
    for (int s = 0; s < n_ids; s++) {
        if (ids[s] == 258880) { /* <|image|> placeholder */
            if (slot >= n_soft) { e2_set_error("more <|image|> slots than soft tokens"); return -1; }
            memcpy(e + (size_t)s * 512, v->emb + (size_t)slot++ * 512, sizeof(float) * 512);
        } else {
            e2_embed_token(m, ids[s], e + (size_t)s * 512);
        }
    }
    if (slot != n_soft) { e2_set_error("%d soft tokens for %d slots", n_soft, slot); return -1; }
    return e2_text_embed_embeds(m, e, n_ids, out768);
}
