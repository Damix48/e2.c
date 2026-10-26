/* text.c — EmbeddingGemma 2 text tower, implemented from the reference (transformers) module.
 *
 * Per layer: RMSNorm -> bidirectional attention (q/k/v projections, q/k/v RMSNorm, RoPE, 512-token
 * sliding window on 20 of 24 layers, o_proj) -> residual -> RMSNorm -> gated MLP (gelu-tanh) ->
 * residual -> PLE block -> layer_scalar.
 *
 * Three details that are easy to get wrong and are matched exactly here:
 *   - attention scaling is 1.0 (not 1/sqrt(head_dim)), and head_dim differs per layer type:
 *     256 with 2 KV heads on sliding layers, 512 with 1 KV head on the 4 full-attention layers;
 *   - the sliding window keeps |i-j| <= 512 (inclusive), and the mask is never materialised: the
 *     key loop is simply bounded, which is what a specialized engine should do;
 *   - PLE is projection-only: project the *input* embeddings to 24x512, RMSNorm each 512 slice with
 *     a learned scale, and gate a mid-block bottleneck with it.
 */
#include <math.h>
#include <time.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "threads.h"
#include "validation.h"

/* E2_DUMP=<dir> writes raw f32 activations for bisecting against the PyTorch reference. */
static void e2_dump(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s.f32", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(p, 4, n, f); fclose(f); }
}



/* E2_PROFILE=1 prints a wall-clock breakdown of one embedding (the caller-side time of every
 * parallel region plus the serial work between them). */
#define PROF_N 8
static const char *g_prof_name[PROF_N] = { "gemm", "quantize", "attention", "norm", "gelu", "elem", "copy", "rope" };
static double g_prof_ms[PROF_N];
static _Atomic long g_prof_busy_us;   /* sum of per-thread GEMM compute time */
static _Atomic long g_prof_quant_us;  /* sum of per-thread activation-quantization time */
static _Atomic long g_prof_claims;
static _Atomic int  g_prof_threads[64];
static int g_prof_on;
static int g_dump_on;

static double e2_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void prof_flush(int S) {
    if (!g_prof_on) return;
    printf("[profile] gemm: %ld claims, %.1f thread-ms over %.1f ms wall = %.2f x parallelism; claims/thread:",
           (long)g_prof_claims, (double)g_prof_busy_us / 1000.0, g_prof_ms[0],
           (double)g_prof_busy_us / 1000.0 / (g_prof_ms[0] > 0 ? g_prof_ms[0] : 1));
    for (int i = 0; i < 16; i++) printf(" %d", g_prof_threads[i]);
    printf("\n");
    atomic_store(&g_prof_busy_us, 0);
    atomic_store(&g_prof_claims, 0);
    for (int i = 0; i < 16; i++) atomic_store(&g_prof_threads[i], 0);
    double tot = 0;
    for (int i = 0; i < PROF_N; i++) tot += g_prof_ms[i];
    printf("[profile S=%d] total %.2f ms\n", S, tot);
    for (int i = 0; i < PROF_N; i++) {
        printf("   %-10s %8.2f ms  %5.1f%%\n", g_prof_name[i], g_prof_ms[i], 100.0 * g_prof_ms[i] / tot);
        g_prof_ms[i] = 0;
    }
}

/* ---------------------------------------------------------------- parallel map
 * The work between the GEMMs (tanh-GELU, the gated-MLP product, residual adds, RMSNorm, RoPE,
 * residual copies) used to run single-threaded. Measured on a 196-token document that was ~90 ms of
 * a 133 ms embedding: the GEMMs scaled, this did not. One worker and one work counter covers all of
 * it, which keeps the number of parallel regions (and therefore barriers) per embedding bounded.
 */
enum {
    E2_OP_GELU,
    E2_OP_MUL,
    E2_OP_ADD,
    E2_OP_SCALE,
    E2_OP_COPY,
    E2_OP_RMSNORM,
    E2_OP_ROPE,
    E2_OP_MUL_ROWS, /* a[r*D + j] *= b[r*stride + j]: the PLE per-layer slice is strided */
    E2_OP_NORM_SPLIT, /* b := a (pre-norm), a := a * inv * gamma */
    E2_OP_NORM_ADD    /* a := norm(c) * gamma + b */
};

struct map_task {
    int op, D, ngroups, stride;
    float *a;
    const float *b, *c, *cs, *sn, *gamma;
    float s, eps;
    size_t n, grain;
    _Atomic size_t next;
};

static void map_worker(void *arg, int tid, int nthreads) {
    struct map_task *t = arg;
    (void)tid;
    for (;;) {
        const size_t i0 = atomic_fetch_add_explicit(&t->next, t->grain, memory_order_relaxed);
        if (i0 >= t->n) break;
        size_t i1 = i0 + t->grain;
        if (i1 > t->n) i1 = t->n;
        switch (t->op) {
        case E2_OP_GELU:
            e2_k()->gelu_tanh(t->a + i0, i1 - i0);
            break;
        case E2_OP_MUL:
            e2_k()->mul(t->a + i0, t->b + i0, i1 - i0);
            break;
        case E2_OP_ADD:
            e2_k()->add(t->a + i0, t->b + i0, i1 - i0);
            break;
        case E2_OP_SCALE:
            e2_k()->scale(t->a + i0, t->s, i1 - i0);
            break;
        case E2_OP_COPY:
            memcpy(t->a + i0, t->b + i0, sizeof(float) * (i1 - i0));
            break;
        case E2_OP_RMSNORM:
            e2_k()->rmsnorm(t->a + i0 * (size_t)t->D, (int)(i1 - i0), t->D, t->gamma, t->eps);
            break;
        case E2_OP_MUL_ROWS:
            e2_k()->mul_rows(t->a + i0 * (size_t)t->D, t->b + i0 * (size_t)t->stride,
                             (int)(i1 - i0), t->D, t->stride);
            break;
        case E2_OP_NORM_SPLIT:
            e2_k()->norm_split(t->a + i0 * (size_t)t->D, (float *)t->b + i0 * (size_t)t->D,
                               (int)(i1 - i0), t->D, t->gamma, t->eps);
            break;
        case E2_OP_NORM_ADD:
            e2_k()->norm_add(t->a + i0 * (size_t)t->D, t->c + i0 * (size_t)t->D,
                             t->b + i0 * (size_t)t->D, (int)(i1 - i0), t->D, t->gamma, t->eps);
            break;
        default:
            e2_k()->rope_rows(t->a, (int)i0, (int)i1, t->ngroups, t->D, t->D, t->cs, t->sn);
            break;
        }
    }
}

/* which E2_PROFILE bucket each op reports into (indexed by E2_OP_*) */
static const int g_prof_slot[] = { 4, 5, 5, 5, 6, 3, 7, 5, 3, 5 };

/* `n` is elements, or rows for the row-structured ops; `grain` is in the same unit. Serial when there
 * is not enough work to pay for a region (~170 of them per embedding already). */
static void run_map(struct map_task *t) {
    if (t->n == 0) return;
    double tp = e2_now_ms();
    if (t->n <= t->grain) { map_worker(t, 0, 1); }
    else { atomic_init(&t->next, 0); e2_parallel(map_worker, t); }
    if (g_prof_on) g_prof_ms[g_prof_slot[t->op]] += e2_now_ms() - tp;
}

/* ~8 KiB of elements per chunk: large enough that the per-chunk overhead (a switch and an atomic) is
 * noise, small enough that the work still divides across the pool. Row-structured ops take their
 * grain in rows, derived from the row length. */
#define E2_GRAIN 2048
#define ROW_GRAIN(n) ((E2_GRAIN / (n)) > 0 ? (E2_GRAIN / (n)) : 1)

static void map_flat(int op, float *a, const float *b, size_t n) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = op;
    t.a = a;
    t.b = b;
    t.n = n;
    t.grain = E2_GRAIN;
    run_map(&t);
}

static void map_norm(float *x, int rows, int D, const float *gamma, float eps) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_RMSNORM;
    t.a = x;
    t.D = D;
    t.gamma = gamma;
    t.eps = eps;
    t.n = (size_t)rows;
    t.grain = ROW_GRAIN(D);
    run_map(&t);
}

static void map_rope(float *x, int rows, int ngroups, int D, const float *cs, const float *sn) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_ROPE;
    t.a = x;
    t.D = D;
    t.ngroups = ngroups;
    t.cs = cs;
    t.sn = sn;
    t.n = (size_t)rows;
    t.grain = ROW_GRAIN(D);
    run_map(&t);
}

static void map_scale(float *a, float s, size_t n) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_SCALE;
    t.a = a;
    t.s = s;
    t.n = n;
    t.grain = E2_GRAIN;
    run_map(&t);
}

static void map_mul_rows(float *a, const float *b, int rows, int D, int stride) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_MUL_ROWS;
    t.a = a;
    t.b = b;
    t.D = D;
    t.stride = stride;
    t.n = (size_t)rows;
    t.grain = ROW_GRAIN(D);
    run_map(&t);
}

/* The residual copy and the normalisation it precedes read the same row, and the residual add
 * follows the normalisation of the same row: fusing them removes two of the ~9 flat-op regions per
 * layer and two passes over S x D. */
static void map_norm_split(float *x, float *res, int rows, int D, const float *gamma, float eps) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_NORM_SPLIT;
    t.a = x;
    t.b = res;
    t.D = D;
    t.gamma = gamma;
    t.eps = eps;
    t.n = (size_t)rows;
    t.grain = ROW_GRAIN(D);
    run_map(&t);
}

static void map_norm_add(float *out, const float *src, const float *res, int rows, int D,
                         const float *gamma, float eps) {
    struct map_task t;
    memset(&t, 0, sizeof t);
    t.op = E2_OP_NORM_ADD;
    t.a = out;
    t.c = src;
    t.b = res;
    t.D = D;
    t.gamma = gamma;
    t.eps = eps;
    t.n = (size_t)rows;
    t.grain = ROW_GRAIN(D);
    run_map(&t);
}

/* ---------------------------------------------------------------- layer types */
static int is_full_layer(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }

/* ---------------------------------------------------------------- GEMM dispatch
 * Every projection goes through here: int8 weights + per-row int8 activations when available, bf16
 * weights otherwise. The int8 path runs on the thread pool, split along N so no weight byte is ever
 * read by two threads, and the activation quantization is shared by every GEMM that reads the same
 * input (q/k/v read one tensor, gate/up read one tensor) so it is done once instead of three times.
 */
#define E2_GEMM_MAXSEG 4

struct gemm_seg {
    float *y;
    int N;
    const e2_q8w *q;   /* int8 weights */
    const uint16_t *w; /* bf16 weights, only used when m->use_int8 == 0 */
};

struct gemm_task {
    e2_model *m;
    int M, total, nseg, tile;
    /* Phase 1 (skipped when `x` is NULL, i.e. the activations are already quantized): quantize x
     * into ws.qa. Folding it into this region instead of running it as its own is worth a region per
     * GEMM group - six per layer - and those regions are where a 15-token query spends its time. */
    const float *x;
    int Mp, K, nb;
    _Atomic int qnext, qdone;
    struct gemm_seg seg[E2_GEMM_MAXSEG];
    _Atomic int next;
};

/* Work is claimed one tile at a time through a shared counter instead of being split statically:
 * this CPU has fast P-cores and slower E-cores, and with an equal static split every region ends up
 * gated by its slowest thread, which measured as only ~3.5x speedup on 16 threads (the cores
 * themselves sustain ~55 GMAC/s each, barrier-free, so the ceiling is much higher). */
static void gemm_claim(struct gemm_task *t, int a, int b) {
    int off = 0;
    for (int s = 0; s < t->nseg && off < b; s++) {
        const int n = t->seg[s].N;
        int n0 = a > off ? a - off : 0;
        int n1 = b - off < n ? b - off : n;
        if (n1 > n0) {
            const double t0 = g_prof_on ? e2_now_ms() : 0.0;
            e2_i8_gemm(t->seg[s].y, (const int8_t *)t->m->ws.qa, t->m->ws.qa_scale, t->seg[s].q,
                       t->Mp, n, n0, n1);
            if (g_prof_on) {
                atomic_fetch_add(&g_prof_busy_us, (long)((e2_now_ms() - t0) * 1000.0));
                atomic_fetch_add(&g_prof_claims, 1);
            }
        }
        off += n;
    }
}

/* Quantization phase: claim 16-row chunks, including the zeroed tail padded up to E2_I8_MR. */
static void gemm_quant_phase(struct gemm_task *t) {
    int8_t *qa = (int8_t *)t->m->ws.qa;
    float *asc = t->m->ws.qa_scale;
    for (;;) {
        const int m0 = atomic_fetch_add_explicit(&t->qnext, 16, memory_order_relaxed);
        if (m0 >= t->Mp) break;
        int n = t->Mp - m0;
        if (n > 16) n = 16;
        int nr = n;
        if (m0 + nr > t->M) nr = t->M - m0;
        if (nr > 0)
            e2_i8_quantize(qa + (size_t)m0 * t->K, asc + (size_t)m0 * t->nb, t->x + (size_t)m0 * t->K,
                           nr, t->K, -INFINITY, INFINITY);
        if (nr < n) { /* the padded tokens: never read back, only kept out of the GEMM's way */
            memset(qa + (size_t)(m0 + nr) * t->K, 0, (size_t)(n - nr) * t->K);
            memset(asc + (size_t)(m0 + nr) * t->nb, 0, sizeof(float) * (size_t)(n - nr) * t->nb);
        }
    }
}

static void gemm_worker(void *arg, int tid, int nthreads) {
    struct gemm_task *t = arg;
    (void)tid;
    if (t->x) {
        const double tq = g_prof_on ? e2_now_ms() : 0.0;
        gemm_quant_phase(t);
        if (g_prof_on) atomic_fetch_add(&g_prof_quant_us, (long)((e2_now_ms() - tq) * 1000.0));
        atomic_fetch_add_explicit(&t->qdone, 1, memory_order_release);
        while (atomic_load_explicit(&t->qdone, memory_order_acquire) < nthreads) e2_pause();
    }
    for (;;) {
        const int a = atomic_fetch_add_explicit(&t->next, t->tile, memory_order_relaxed);
        if (a >= t->total) break;
        int b = a + t->tile;
        if (b > t->total) b = t->total;
        gemm_claim(t, a, b);
    }
}

/* Tokens are padded up to E2_I8_MR: the tail of the tile loop in the kernel is one token per pass,
 * so a 15-token query would otherwise re-read every weight row 6 times instead of 2. The padding
 * rows are zeroed and their outputs are never read (the workspace is sized for E2_MAX_SEQ, a
 * multiple of the tile, so both the activations and y have room for them). */
static int pad_tokens(int M) { return (M + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR; }

/* One parallel region for one or more GEMMs that share their activation input (q|k|v, gate|up). */
static void gemm_segs(e2_model *m, const float *x, int M, int K, int reuse, int nseg,
                      const struct gemm_seg *segs) {
    double t0 = e2_now_ms();
    if (!m->use_int8) {
        for (int s = 0; s < nseg; s++)
            e2_k()->gemm_nt(segs[s].y, x, segs[s].w, M, segs[s].N, K, m->wdtype);
        return;
    }
    struct gemm_task t;
    memset(&t, 0, sizeof t);
    t.m = m;
    if (!reuse) {
        t.x = x;
        t.K = K;
        t.nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
        atomic_init(&t.qnext, 0);
        atomic_init(&t.qdone, 0);
    }
    t.M = M;
    t.Mp = pad_tokens(M);
    for (int s = 0; s < nseg && s < E2_GEMM_MAXSEG; s++) {
        t.seg[t.nseg++] = segs[s];
        t.total += segs[s].N;
    }
    /* Several tiles per thread: enough to even out the P/E-core speed differences that claiming
     * exists to fix, few enough that the shared counter stays off the critical path. Measured flat
     * over 4..32, so the exact value is not delicate. */
    t.tile = t.total / (E2_TILE_PER_THREAD * e2_nthreads());
    if (t.tile < E2_I8_MR) t.tile = E2_I8_MR;
    if (t.tile > 64) t.tile = 64;
    /* every segment width is a multiple of 16 and the AVX-512 kernel advances 16 output rows per
     * weight load: an 8-aligned claim would drop to the 256-bit path whenever n1-n0 < 16, which is
     * every claim of the N=512 projections once the pool is large. 16-aligned claims also mean the
     * 256-bit tail never runs on the dispatch path. */
    t.tile = (t.tile + 15) & ~15;
    if (t.tile < 16) t.tile = 16;
    atomic_init(&t.next, 0);
    e2_parallel(gemm_worker, &t);
    if (g_prof_on) {
        /* the quantize phase overlaps across workers, so its share of the region is the mean
         * per-thread time, not the sum */
        g_prof_ms[1] += (double)atomic_exchange(&g_prof_quant_us, 0) / 1000.0 / e2_nthreads();
        g_prof_ms[0] += e2_now_ms() - t0;
    }
}

static void gemm(e2_model *m, float *y, const float *x, const e2_q8w *q, const uint16_t *w,
                 int M, int N, int K, int reuse) {
    const struct gemm_seg seg = { y, N, q, w };
    if (!m->use_int8) {
        e2_k()->gemm_nt(y, x, w, M, N, K, m->wdtype);
        return;
    }
    gemm_segs(m, x, M, K, reuse, 1, &seg);
}

/* Keep the passes cache-resident without starving the pool on short sequences. */
struct gate_quant_task {
    e2_model *m;
    float *gate;
    const float *up;
    int rows, D, stride, nb, chunk;
    _Atomic int next;
};

static void gate_quant_worker(void *arg, int tid, int nthreads) {
    struct gate_quant_task *t = arg;
    const e2_kernels *K = e2_k();
    (void)tid;
    (void)nthreads;
    for (;;) {
        int r0 = atomic_fetch_add_explicit(&t->next, t->chunk, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int nr = t->rows - r0;
        if (nr > t->chunk) nr = t->chunk;
        float *g = t->gate + (size_t)r0 * t->D;
        K->gelu_tanh(g, (size_t)nr * t->D);
        K->mul_rows(g, t->up + (size_t)r0 * t->stride, nr, t->D, t->stride);
        e2_i8_quantize((int8_t *)t->m->ws.qa + (size_t)r0 * t->D,
                       t->m->ws.qa_scale + (size_t)r0 * t->nb, g, nr, t->D, -INFINITY, INFINITY);
    }
}

static void gate_quant(e2_model *m, float *gate, const float *up, int rows, int D, int stride) {
    if (!m->use_int8) {
        map_flat(E2_OP_GELU, gate, NULL, (size_t)rows * D);
        map_mul_rows(gate, up, rows, D, stride);
        return;
    }
    double t0 = e2_now_ms();
    int chunk = rows / (2 * e2_nthreads());
    if (chunk < 1) chunk = 1;
    if (chunk > 8) chunk = 8;
    struct gate_quant_task t = { m, gate, up, rows, D, stride,
                                 (D + E2_I8_BLOCK - 1) / E2_I8_BLOCK, chunk, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(gate_quant_worker, &t);
    int tail = pad_tokens(rows) - rows;
    memset(m->ws.qa + (size_t)rows * D, 0, (size_t)tail * D);
    memset(m->ws.qa_scale + (size_t)rows * t.nb, 0, sizeof(float) * (size_t)tail * t.nb);
    if (g_prof_on) g_prof_ms[1] += e2_now_ms() - t0;
}

static void layer_types_init(e2_model *m) {
    for (int l = 0; l < m->n_layers; l++) {
        int full = is_full_layer(l);
        e2_layer *L = &m->layers[l];
        L->sliding = !full;
        L->window = 512; /* config.sliding_window; the mask keeps |i-j| <= window */
        L->head_dim = full ? 512 : 256;
        L->n_kv_heads = full ? 1 : 2;
        L->q_dim = E2_NUM_HEADS * L->head_dim;
        L->kv_dim = L->n_kv_heads * L->head_dim;
    }
}

static int rope_init(e2_model *m) {
    for (int t = 0; t < 2; t++) {
        int head_dim = (t == 0) ? 256 : 512;
        float theta = (t == 0) ? m->rope_theta_sliding : m->rope_theta_full;
        size_t n = (size_t)E2_MAX_SEQ * head_dim;
        m->rope_cos[t] = malloc(sizeof(float) * n);
        m->rope_sin[t] = malloc(sizeof(float) * n);
        if (!m->rope_cos[t] || !m->rope_sin[t]) {
            e2_set_error("oom: text RoPE tables");
            return -1;
        }
        for (int p = 0; p < E2_MAX_SEQ; p++) {
            float *c = m->rope_cos[t] + (size_t)p * head_dim;
            float *s = m->rope_sin[t] + (size_t)p * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float inv = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float f = (float)p * inv;
                c[i] = c[i + head_dim / 2] = cosf(f);
                s[i] = s[i + head_dim / 2] = sinf(f);
            }
        }
    }
    return 0;
}

#define ATTN_ROWS 4 /* rows claimed per work item (see gemm_worker for why claiming beats a split) */

struct attn_task {
    e2_model *m;
    e2_layer *L;
    int S;
    const uint8_t *valid;
    float *q, *k, *v, *ctx, *attn;
    int dbg;
    _Atomic int next;
};

static void attn_rows(struct attn_task *t, int i0, int i1, int tid);

static void attn_worker(void *arg, int tid, int nthreads) {
    struct attn_task *t = arg;
    (void)nthreads;
    for (;;) {
        const int i0 = atomic_fetch_add_explicit(&t->next, ATTN_ROWS, memory_order_relaxed);
        if (i0 >= t->S) break;
        int i1 = i0 + ATTN_ROWS;
        if (i1 > t->S) i1 = t->S;
        attn_rows(t, i0, i1, tid);
    }
}

static void attn_rows(struct attn_task *t, int i0, int i1, int tid) {
    const e2_layer *L = t->L;
    const e2_kernels *K = e2_k();
    const int S = t->S, GD = L->q_dim, H = E2_NUM_HEADS, D = L->head_dim, KV = L->n_kv_heads;
    const int groups = H / KV;
    float *scores = t->attn + (size_t)tid * E2_MAX_SEQ;
    for (int i = i0; i < i1; i++) {
        if (!t->valid[i]) { memset(t->ctx + (size_t)i * GD, 0, sizeof(float) * GD); continue; }
        int lo = 0, hi = S;
        if (L->sliding) {
            lo = i - L->window > 0 ? i - L->window : 0;
            hi = i + L->window + 1 < S ? i + L->window + 1 : S;
        }
        for (int h = 0; h < H; h++) {
            const float *qh = t->q + (size_t)i * GD + (size_t)h * D;
            const int kv_head = h / groups;
            int n = 0;
            for (int j = lo; j < hi; j++) {
                if (!t->valid[j]) continue;
                scores[n++] = K->dot_f32(qh, t->k + (size_t)j * L->kv_dim + (size_t)kv_head * D, (size_t)D);
            }
            float *oh = t->ctx + (size_t)i * GD + (size_t)h * D;
            memset(oh, 0, sizeof(float) * D);
            if (n == 0) continue;
            K->softmax(scores, n);
            if (t->dbg && i0 == 0 && i == 0 && h == 0) {
                e2_dump("l0_p_used", scores, (size_t)n);
                e2_dump("l0_v_used", t->v, (size_t)S * L->kv_dim);
            }
            n = 0;
            for (int j = lo; j < hi; j++) {
                if (!t->valid[j]) continue;
                const float *vh = t->v + (size_t)j * L->kv_dim + (size_t)kv_head * D;
                K->axpy(oh, vh, scores[n++], D);
            }
            if (t->dbg && i == 0 && h == 0) e2_dump("l0_ctx_h0_row0", oh, (size_t)D);
        }
    }
}

/* forward declarations for the dump helper (defined below) */
static void e2_dump(const char *name, const float *p, size_t n);

/* ---------------------------------------------------------------- blocked attention
 * The per-row loop below re-reads every key and value vector once per query row: for a 196-token
 * document that is 196 passes over 400 KB of K and V per layer, which is what actually dominates it
 * (the MACs are a third of the GEMM's, the traffic is 100x). The blocked form is the same two
 * matrix products the vision tower uses - logits = Q K^T, context = P V - with K transposed once per
 * layer into w->kt so both can go through the blocked f32 kernel. A work item is (head, 32 query
 * rows): it computes the logits for those rows, softmaxes them, and immediately multiplies by V, so
 * the (S,S) scores never leave the worker's L1/L2 block. Query rows are claimed, not split, for the
 * same reason the GEMMs claim.
 *
 * Only the union of each query block's inclusive windows is multiplied; per-row masks trim
 * its edges. Every row must be valid. E2_DUMP retains the row-wise diagnostic reference.
 */
struct kt_task {
    const float *k;
    float *kt;
    int S, D, rows, kv_dim, groups;
    _Atomic int next;
};

static void kt_worker(void *arg, int tid, int nthreads) {
    struct kt_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int r1 = r0 + 8;
        if (r1 > t->rows) r1 = t->rows;
        for (int r = r0; r < r1; r++) {
            const int h = r / t->D, d = r - h * t->D;
            const float *src = t->k + (size_t)(h / t->groups) * t->D + d;
            float *dst = t->kt + (size_t)r * t->S;
            for (int n = 0; n < t->S; n++) dst[n] = src[(size_t)n * t->kv_dim];
        }
    }
}

struct attb_task {
    float *q, *v, *ctx, *kt, *scores;
    int S, D, GD, kv_dim, H, kv_groups, nblk, window;
    _Atomic int next;
};

static void attb_worker(void *arg, int tid, int nthreads) {
    struct attb_task *t = arg;
    const e2_kernels *K = e2_k();
    (void)nthreads;
    float *sc = t->scores + (size_t)tid * E2_ATT_BQ * t->S;
    for (;;) {
        const int it = atomic_fetch_add_explicit(&t->next, 1, memory_order_relaxed);
        if (it >= t->H * t->nblk) break;
        const int h = it / t->nblk, blk = it - h * t->nblk;
        const int m0 = blk * E2_ATT_BQ;
        int rows = t->S - m0;
        if (rows > E2_ATT_BQ) rows = E2_ATT_BQ;
        int lo = 0, hi = t->S;
        if (t->window >= 0) {
            lo = m0 > t->window ? m0 - t->window : 0;
            hi = m0 + rows + t->window;
            if (hi > t->S) hi = t->S;
        }
        const int keys = hi - lo;
        e2_sgemm_nn_range(sc, keys, t->q + (size_t)m0 * t->GD + (size_t)h * t->D, t->GD,
                          t->kt + (size_t)h * t->D * t->S + lo, t->S, rows, keys, t->D, 0, rows);
        for (int i = 0; i < rows; i++) {
            float *row = sc + (size_t)i * keys;
            int first = 0, end = keys;
            if (t->window >= 0) {
                first = m0 + i - t->window - lo;
                if (first < 0) first = 0;
                end = m0 + i + t->window + 1 - lo;
                if (end > keys) end = keys;
            }
            K->softmax(row + first, end - first);
            memset(row, 0, sizeof(float) * (size_t)first);
            memset(row + end, 0, sizeof(float) * (size_t)(keys - end));
        }
        e2_sgemm_nn_range(t->ctx + (size_t)m0 * t->GD + (size_t)h * t->D, t->GD, sc, keys,
                          t->v + (size_t)lo * t->kv_dim + (size_t)(h / t->kv_groups) * t->D,
                          t->kv_dim, rows, t->D, keys, 0, rows);
    }
}

static void attention_blocked(e2_layer *L, float *q, float *k, float *v, float *ctx, float *kt,
                              float *scores, int S) {
    const int D = L->head_dim;
    struct kt_task tk = { k, kt, S, D, E2_NUM_HEADS * D, L->kv_dim, E2_NUM_HEADS / L->n_kv_heads, 0 };
    atomic_init(&tk.next, 0);
    e2_parallel(kt_worker, &tk);
    struct attb_task at = { q, v, ctx, kt, scores, S, D, L->q_dim, L->kv_dim, E2_NUM_HEADS,
                            E2_NUM_HEADS / L->n_kv_heads, (S + E2_ATT_BQ - 1) / E2_ATT_BQ,
                            L->sliding ? L->window : -1, 0 };
    atomic_init(&at.next, 0);
    e2_parallel(attb_worker, &at);
}


static void attention(e2_model *m, e2_layer *L, float *h, int B, int seq_len, e2_ws *w,
                      int dbg) {    float *q = w->q, *k = w->k, *v = w->v, *ctx = w->att, *attn = w->attn, *tmp = w->tmp;
    const e2_kernels *K = e2_k();
    const int H = E2_NUM_HEADS, D = L->head_dim, KV = L->n_kv_heads, GD = L->q_dim;
    const int rope_type = L->sliding ? 0 : 1;

    const int M = B * seq_len;

    /* q/k/v share one activation quantization and one parallel region */
    {
        const struct gemm_seg seg[3] = {
            { q, GD, &L->q8_q, L->q_proj },
            { k, L->kv_dim, &L->q8_k, L->k_proj },
            { v, L->kv_dim, &L->q8_v, L->v_proj },
        };
        gemm_segs(m, h, M, m->hidden, 0, 3, seg);
    }

    if (dbg) { e2_dump("l0_q_proj", q, (size_t)M * GD); e2_dump("l0_k_proj", k, (size_t)M * L->kv_dim); }
    map_norm(q, M * H, D, L->q_norm, m->rms_eps);
    map_norm(k, M * KV, D, L->k_norm, m->rms_eps);
    map_norm(v, M * KV, D, NULL, m->rms_eps); /* v_norm has no learned scale */
    if (dbg) { e2_dump("l0_q_norm", q, (size_t)M * GD); e2_dump("l0_k_norm", k, (size_t)M * L->kv_dim);
               e2_dump("l0_v_norm", v, (size_t)M * L->kv_dim); }

    for (int b = 0; b < B; b++) {
        float *q = w->q + (size_t)b * seq_len * GD;
        float *k = w->k + (size_t)b * seq_len * L->kv_dim;
        float *v = w->v + (size_t)b * seq_len * L->kv_dim;
        float *ctx = w->att + (size_t)b * seq_len * GD;
        const uint8_t *valid = w->valid + (size_t)b * seq_len;
        /* Match the unpadded single's attention schedule and skip trailing padding. */
        int S = seq_len;
        while (S && !valid[S - 1]) S--;
        memset(ctx + (size_t)S * GD, 0, sizeof(float) * (size_t)(seq_len - S) * GD);
        if (!S) continue;
        map_rope(q, S * H, H, D, m->rope_cos[rope_type], m->rope_sin[rope_type]);
        map_rope(k, S * KV, KV, D, m->rope_cos[rope_type], m->rope_sin[rope_type]);

        if (dbg) { e2_dump("l0_q_rope", q, (size_t)S * GD); e2_dump("l0_k_rope", k, (size_t)S * L->kv_dim); }
        if (dbg && g_dump_on) {
            /* probabilities for one head, all query rows: (S, S); only the bisection tooling wants
             * this, and it costs a serial (S,S) dot loop - see e2_dump */
            float *pr = malloc(sizeof(float) * (size_t)S * S);
            if (pr) {
                for (int i = 0; i < S; i++) {
                    for (int j = 0; j < S; j++)
                        pr[(size_t)i * S + j] = K->dot_f32(q + (size_t)i * GD, k + (size_t)j * L->kv_dim, (size_t)D);
                    K->softmax(pr + (size_t)i * S, S);
                }
                e2_dump("l0_probs_h0", pr, (size_t)S * S);
                free(pr);
            }
        }
        {
            double ta = e2_now_ms();
            int blocked = !g_dump_on && S >= E2_ATT_BQ;
            for (int s = 0; blocked && s < S; s++)
                if (!valid[s]) blocked = 0;
            if (blocked) {
                attention_blocked(L, q, k, v, ctx, w->kt, w->asblk, S);
            } else {
                struct attn_task t = { m, L, S, valid, q, k, v, ctx, attn, dbg, 0 };
                atomic_init(&t.next, 0);
                if (S <= ATTN_ROWS) attn_worker(&t, 0, 1);
                else e2_parallel(attn_worker, &t);
            }
            if (g_prof_on) g_prof_ms[2] += e2_now_ms() - ta;
        }
    }
    if (dbg) e2_dump("l0_ctx_pre_oproj", ctx, (size_t)M * GD);
    /* o_proj output lands in tmp; the caller normalises tmp directly (the old ctx = tmp copy
     * existed only because ctx is q_dim-wide and the result is hidden-wide). */
    gemm(m, tmp, ctx, &L->q8_o, L->o_proj, M, m->hidden, GD, 0);
}

static void mlp(e2_model *m, e2_layer *L, float *x, int S, float *gate, float *up, float *tmp) {
    const int I = m->intermediate;
    /* gate/up share one activation quantization and one parallel region */
    {
        const struct gemm_seg seg[2] = { { gate, I, &L->q8_gate, L->gate_proj },
                                         { up, I, &L->q8_up, L->up_proj } };
        gemm_segs(m, x, S, m->hidden, 0, 2, seg);
    }
    gate_quant(m, gate, up, S, I, I);
    gemm(m, tmp, gate, &L->q8_down, L->down_proj, S, m->hidden, I, m->use_int8);
}

/* `ple` points at this layer's slice (gate[i] *= ple[s*stride + d]): the per-layer embeddings are
 * stored layer-fastest, so a layer slice is strided across tokens rather than contiguous. */
static void ple_block(e2_model *m, e2_layer *L, float *x, int S, const float *ple, int stride,
                      float *gate, float *tmp) {
    const int P = m->ple_dim;
    const int D = m->hidden;
    gemm(m, gate, x, &L->q8_ple_gate, L->ple_gate, S, P, D, 0);
    gate_quant(m, gate, ple, S, P, stride);
    gemm(m, tmp, gate, &L->q8_ple_proj, L->ple_proj, S, D, P, m->use_int8);
    map_norm_add(x, tmp, x, S, D, L->ple_norm, m->rms_eps);
}

void e2_embed_token(e2_model *m, int id, float *dst) {
    const uint16_t *row = m->embed_tokens + (size_t)id * m->hidden;
    e2_bf16_to_f32(dst, row, (size_t)m->hidden);
    for (int d = 0; d < m->hidden; d++) dst[d] *= m->embed_scale;
}

/* Projection regions span all sequences; attention, positions and pooling never do. */
static int text_forward(e2_model *m, int B, int seq_len, float *out) {
    const int S = B * seq_len, D = m->hidden;
    e2_ws *w = &m->ws;
    g_prof_on = getenv("E2_PROFILE") != NULL;
    g_dump_on = getenv("E2_DUMP") != NULL;

    e2_dump("x0", w->x, (size_t)S * D);

    /* PLE: project the input embeddings to 24 x 512 and norm each slice */
    gemm(m, w->ple, w->x, &m->q8_ple_model, m->ple_model_proj, S, m->n_layers * m->ple_dim, D, 0);
    map_scale(w->ple, m->ple_scale, (size_t)S * m->n_layers * m->ple_dim);
    map_norm(w->ple, S * m->n_layers, m->ple_dim, m->ple_proj_norm, m->rms_eps);

    if (m->n_layers) e2_dump("ple", w->ple, (size_t)S * m->n_layers * m->ple_dim);

    for (int l = 0; l < m->n_layers; l++) {
        e2_layer *L = &m->layers[l];
        const char *tag = (l == 0) ? "l0_" : NULL;

        map_norm_split(w->x, w->res, S, D, L->input_ln, m->rms_eps);
        if (tag) e2_dump("l0_input_ln", w->x, (size_t)S * D);
        attention(m, L, w->x, B, seq_len, w, l == 0);
        if (tag) e2_dump("l0_attn_out", w->tmp, (size_t)S * D);
        map_norm_add(w->x, w->tmp, w->res, S, D, L->post_attn_ln, m->rms_eps);
        if (tag) e2_dump("l0_x_after_attn", w->x, (size_t)S * D);

        map_norm_split(w->x, w->res, S, D, L->pre_ffn_ln, m->rms_eps);
        if (tag) e2_dump("l0_pre_ffn_ln", w->x, (size_t)S * D);
        mlp(m, L, w->x, S, w->gate, w->up, w->tmp);
        if (tag) e2_dump("l0_mlp_out", w->tmp, (size_t)S * D);
        map_norm_add(w->x, w->tmp, w->res, S, D, L->post_ffn_ln, m->rms_eps);
        if (tag) e2_dump("l0_x_after_mlp", w->x, (size_t)S * D);

        ple_block(m, L, w->x, S, w->ple + (size_t)l * m->ple_dim, m->n_layers * m->ple_dim, w->ple_g, w->tmp);
        if (tag) e2_dump("l0_ple_block_out", w->x, (size_t)S * D);

        if (L->layer_scalar != 1.0f) map_scale(w->x, L->layer_scalar, (size_t)S * D);
    }

    map_norm(w->x, S, D, m->final_norm, m->rms_eps);
    e2_dump("final_norm", w->x, (size_t)S * D);
    gemm(m, w->emb, w->x, &m->q8_embed_proj, m->embed_proj, S, E2_EMBED_DIM, D, 0);
    e2_dump("emb_proj", w->emb, (size_t)S * E2_EMBED_DIM);
    for (int b = 0; b < B; b++) {
        float dst[E2_EMBED_DIM];
        e2_meanpool_f32(dst, w->emb + (size_t)b * seq_len * E2_EMBED_DIM,
                        seq_len, E2_EMBED_DIM, w->valid + (size_t)b * seq_len);
        int any = 0;
        for (int i = 0; i < seq_len; i++) any |= w->valid[b * seq_len + i];
        if (any) {
            e2_l2norm_f32(dst, E2_EMBED_DIM);
            if (!e2_valid_embedding(dst)) {
                e2_set_error("text inference produced an invalid embedding");
                return -1;
            }
        }
        memcpy(out + (size_t)b * E2_EMBED_DIM, dst, sizeof dst);
    }
    prof_flush(S);
    return 0;
}

/* Image/audio callers supply already-spliced, all-valid embeddings. */
int e2_text_embed_embeds(e2_model *m, const float *embeds, int S, float *out768) {
    e2_set_error("");
    if (!m || !embeds || !out768) { e2_set_error("null text input"); return -1; }
    if (S <= 0 || S > E2_MAX_SEQ) { e2_set_error("sequence length %d out of range", S); return -1; }
    if (!m->ws.x) { e2_set_error("workspace not allocated"); return -1; }
    if (embeds != m->ws.x) memcpy(m->ws.x, embeds, sizeof(float) * (size_t)S * m->hidden);
    memset(m->ws.valid, 1, (size_t)S);
    return text_forward(m, 1, S, out768);
}

int e2_text_embed_batch(e2_model *m, const int32_t *ids, const uint8_t *valid,
                        int batch, int seq_len, float *out) {
    e2_set_error("");
    if (!m || !ids || !out) { e2_set_error("null text input"); return -1; }
    if (batch <= 0 || seq_len <= 0 || seq_len > E2_MAX_SEQ ||
        (size_t)batch > SIZE_MAX / (size_t)seq_len / sizeof(*ids) ||
        (size_t)batch > SIZE_MAX / E2_EMBED_DIM / sizeof(*out)) {
        e2_set_error("bad batch geometry"); return -1;
    }
    if (!m->ws.x) { e2_set_error("workspace not allocated"); return -1; }
    const size_t total = (size_t)batch * seq_len;
    /* Validate the entire request before touching output, including masked token IDs. */
    for (size_t i = 0; i < total; i++) {
        if (ids[i] < 0 || ids[i] >= E2_VOCAB) {
            e2_set_error("token id %d out of range", ids[i]); return -1;
        }
    }
    const int per = E2_MAX_SEQ / seq_len;
    for (int b0 = 0; b0 < batch;) {
        const int n = batch - b0 < per ? batch - b0 : per;
        const int rows = n * seq_len;
        const size_t off = (size_t)b0 * seq_len;
        for (int i = 0; i < rows; i++) {
            e2_embed_token(m, ids[off + i], m->ws.x + (size_t)i * m->hidden);
            m->ws.valid[i] = valid ? valid[off + i] : 1;
        }
        if (text_forward(m, n, seq_len, out + (size_t)b0 * E2_EMBED_DIM)) return -1;
        b0 += n;
    }
    return 0;
}

int e2_text_embed(e2_model *m, const int32_t *ids, int S, float *out768) {
    return e2_text_embed_batch(m, ids, NULL, 1, S, out768);
}

int e2_text_init(e2_model *m) {
    layer_types_init(m);
    return rope_init(m);
}
