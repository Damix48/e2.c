/* audio.c — Gemma4 audio tower (USM/Conformer: 12 layers, 1024 wide, 8 heads, chunked local
 * attention with a relative-position term) plus the audio and multimodal entry points.
 *
 * Reference: transformers Gemma4AudioModel + Gemma4AudioFeatureExtractor + the EmbeddingGemma2
 * glue (0.5-scaled residual feed-forwards, clipped linears, output_proj 1024->1536, then
 * embed_audio's scale-free RMSNorm + 1536->512). Everything below follows that code; the
 * intermediates in tests/ref_act/a_*.f32 are dumped stage by stage by tools/dump_audio_fixture.py,
 * so a mismatch can be localised in one run (E2_DUMP=<dir> writes the engine's own stages).
 *
 * Three shapes of the reference are easy to get subtly wrong, so they are named here:
 *   - the attention window is `sliding_window_mask_function((left - 1, right))`, whose left test is
 *     `dist >= 0 && dist < left - 1`: with context_left 13 that is 12 keys per query (the query
 *     itself plus 11 frames back), not 13;
 *   - the relative term covers those same distances: `_rel_shift` maps context offset i,j to
 *     relative index j - i, and the position table is built from position ids 12..0, so distance d
 *     uses relative row (RELPOS - 1 - d), i.e. position id d;
 *   - the block context is 24 wide (12 chunk + 12 past + 0 future), so a key at context offset j
 *     has distance 12 + i - j from query i of the same block.
 *
 * Every attention window is causal and every convolution is causal, so a valid token cannot depend
 * on anything after it. The engine therefore runs the frontend on the valid mel frames only, never
 * pads, and never builds the reference's blocked attention mask (which exists only to hide padding).
 */
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "threads.h"
#include "validation.h"

/* M_PI/M_E come from the POSIX/gnu headers, which a strict -std=c11 build does not expose */
#define A_PI 3.14159265358979323846
#define A_E 2.71828182845904523536

int e2_audio_init(e2_model *m);
void e2_audio_free(e2_audio *a);
int e2_audio_prepare(e2_model *m);
void e2_embed_token(e2_model *m, int id, float *dst);
int e2_text_embed_embeds(e2_model *m, const float *embeds, int S, float *out768);
int e2_vision_soft_tokens(e2_model *m, const float *pixels, const int32_t *pos, int n_patches,
                          const float **soft);

/* <|audio> <|audio|> <audio|> (the layout the processor builds around the soft tokens) */
#define A_BOA 256000
#define A_TOKEN 258881
#define A_EOA 258883
#define I_BOI 255999
#define I_TOKEN 258880
#define I_EOI 258882

static double g_gemm_ms;
static long g_gemm_calls;
struct aprof_slot { int M, N, K; double ms; int calls; };
static struct aprof_slot g_slots[24];
static int g_nslots;

static double anow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void adump(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof path, "%s/a_%s.f32", dir, name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(p, 4, n, f); fclose(f); }
}

/* ---------------------------------------------------------------- workspace */
static size_t aws_layout(e2_audio *a, float *base) {
    float *p = base;
    size_t n = 0;
#define ASLOT(field, cnt)                \
    do {                                 \
        size_t c_ = (size_t)(cnt);       \
        if (a) a->field = p;             \
        p += c_;                         \
        n += c_;                         \
    } while (0)
    /* Every projection writes full int8 row tiles, including the final padded rows. */
    const size_t S = (E2_A_MAX_TOK + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR;
    const size_t H = E2_A_HIDDEN, I = E2_A_INTER;
    ASLOT(x, S * H);
    ASLOT(res, S * H);
    ASLOT(tmp, S * E2_A_OUT);
    ASLOT(q, S * H);
    ASLOT(k, S * H);
    ASLOT(v, S * H);
    ASLOT(att, S * H);
    ASLOT(ff, S * I);
    ASLOT(lc, S * 2 * H); /* the light-conv GLU projection is 2048 wide */
    ASLOT(relk, (size_t)((E2_A_RELPOS + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR) * H);
    ASLOT(clamp8, (size_t)8 * I); /* bf16 fallback: 8 clamped rows at a time */
    ASLOT(soft, S * 512);
    ASLOT(emb, (size_t)E2_MAX_SEQ * 512);
    ASLOT(qa_words, (S * I + 3) / 4);
    ASLOT(qa_scale, S * (I / E2_I8_BLOCK + 1));
#undef ASLOT
    return n;
}

static int aws_alloc(e2_audio *a) {
    if (a->x) return 0;
    const size_t floats = aws_layout(NULL, NULL);
    a->bytes = floats * sizeof(float) + 64;
    void *blk = malloc(a->bytes);
    if (!blk) { e2_set_error("oom: %zu bytes of audio workspace", a->bytes); return -1; }
    const size_t used = aws_layout(a, (float *)blk);
    if (used != floats) {
        e2_set_error("audio workspace layout mismatch");
        free(blk);
        memset(a, 0, sizeof *a);
        return -1;
    }
    a->qa = (uint8_t *)a->qa_words;
    return 0;
}

/* Frontend and subsample scratch, sized by this call's mel frame count. */
static int asub_alloc(e2_audio *a, int n_frames) {
    const size_t T1 = (size_t)(n_frames + 1) / 2, T2 = (T1 + 1) / 2;
    const size_t want_mel = (size_t)n_frames * E2_A_MEL, want_c0 = T1 * 64 * E2_A_C0,
                 want_c1 = T2 * 32 * E2_A_C1;
    if (a->mel_cap < want_mel) {
        free(a->mel);
        a->mel = malloc(sizeof(float) * want_mel);
        a->mel_cap = a->mel ? want_mel : 0;
    }
    if (a->c0_cap < want_c0) {
        free(a->c0_out);
        a->c0_out = malloc(sizeof(float) * want_c0);
        a->c0_cap = a->c0_out ? want_c0 : 0;
    }
    if (a->c1_cap < want_c1) {
        free(a->c1_out);
        a->c1_out = malloc(sizeof(float) * want_c1);
        a->c1_cap = a->c1_out ? want_c1 : 0;
    }
    if (!a->mel || !a->c0_out || !a->c1_out) { e2_set_error("oom: audio frontend scratch"); return -1; }
    return 0;
}

/* ---------------------------------------------------------------- clipped-linear GEMM
 * y[M,N] = clamp_out(clamp_in(x) @ w[N,K]^T). The input clamp runs before the quantiser (it bounds
 * the activation range the reference was trained for, and it is what keeps int8 activations honest).
 * x == NULL means "the same clamped rows are already quantised in a->qa" - one region, no requant. */
struct agemm_task {
    e2_audio *a;
    const e2_alin *lin;
    const float *x;
    float *y;
    int M, Mp, N, K, tile;
    _Atomic int qnext, next, qdone;
};

static void agemm_worker(void *arg, int tid, int nthreads) {
    struct agemm_task *t = arg;
    e2_audio *a = t->a;
    (void)tid;
    if (t->x) {
        const int K = t->K, nb = (K + E2_I8_BLOCK - 1) / E2_I8_BLOCK;
        /* the clamp rides inside the quantiser: one pass over the block, not clamp-into-a-buffer
         * then quantise (which was 20 ms of an 11 s clip across the tower's 135 projections) */
        const float lo = t->lin->in_min, hi = t->lin->in_max;
        for (;;) {
            const int m0 = atomic_fetch_add_explicit(&t->qnext, 16, memory_order_relaxed);
            if (m0 >= t->Mp) break;
            int nr = t->Mp - m0;
            if (nr > 16) nr = 16;
            if (m0 + nr > t->M) nr = t->M - m0;
            if (nr > 0)
                e2_i8_quantize((int8_t *)a->qa + (size_t)m0 * K, a->qa_scale + (size_t)m0 * nb,
                               t->x + (size_t)m0 * K, nr, K, lo, hi);
            /* padded rows only have to be finite; the kernel's results for them are dropped */
            for (int i = nr; i < 16 && m0 + i < t->Mp; i++) {
                memset(a->qa + (size_t)(m0 + i) * K, 0, (size_t)K);
                memset(a->qa_scale + (size_t)(m0 + i) * nb, 0, sizeof(float) * (size_t)nb);
            }
        }
        atomic_fetch_add_explicit(&t->qdone, 1, memory_order_release);
        while (atomic_load_explicit(&t->qdone, memory_order_acquire) < nthreads) e2_pause();
    }
    /* claim width: ~8 claims per thread, 8-aligned because the kernel moves 8 output rows per
     * weight load (a fixed 32 left the last round of a 1024-wide projection 4 claims wide) */
    for (;;) {
        const int n0 = atomic_fetch_add_explicit(&t->next, t->tile, memory_order_relaxed);
        if (n0 >= t->N) break;
        int n1 = n0 + t->tile;
        if (n1 > t->N) n1 = t->N;
        e2_i8_gemm(t->y, (const int8_t *)a->qa, a->qa_scale, &t->lin->q8, t->Mp, t->N, n0, n1);
    }
}

static void agemm_run(e2_model *m, e2_audio *a, float *y, const float *x, const e2_alin *lin, int M,
                      int N, int K) {
    const double t0_ = anow();
    if (!m->use_int8) {
        /* the bf16 fallback still has to clamp: the reference's clipped linears are part of the
         * numerics, not a quantisation detail */
        const int clampin = lin->in_min > -1e30f, CH = 8;
        for (int m0 = 0; m0 < M; m0 += CH) {
            int mr = M - m0;
            if (mr > CH) mr = CH;
            const float *src = x + (size_t)m0 * K;
            if (clampin) {
                float *dst = a->clamp8;
                for (size_t i = 0; i < (size_t)mr * K; i++) {
                    float v = src[i];
                    dst[i] = v < lin->in_min ? lin->in_min : (v > lin->in_max ? lin->in_max : v);
                }
                src = dst;
            }
            e2_k()->gemm_nt(y + (size_t)m0 * N, src, lin->w, mr, N, K, E2_DT_BF16);
            if (lin->out_min > -1e30f)
                e2_k()->clamp(y + (size_t)m0 * N, lin->out_min, lin->out_max, (size_t)mr * N);
        }
    } else {
        struct agemm_task t;
        memset(&t, 0, sizeof t);
        t.a = a;
        t.lin = lin;
        t.x = x;
        t.y = y;
        t.M = M;
        t.Mp = x ? (M + E2_I8_MR - 1) / E2_I8_MR * E2_I8_MR : M;
        t.N = N;
        t.K = K;
        int tile = N / (E2_TILE_PER_THREAD * e2_nthreads());
        tile = (tile + 7) & ~7;
        if (tile < 8) tile = 8;
        if (tile > 64) tile = 64;
        t.tile = tile;
        atomic_init(&t.next, 0);
        if (x) { atomic_init(&t.qnext, 0); atomic_init(&t.qdone, 0); }
        e2_parallel(agemm_worker, &t);
    }
    /* accounting: an embedding is ~150 of these regions, so the split against attention/conv decides
     * whether a shape change is worth anything */
    const double t1 = anow();
    g_gemm_ms += t1 - t0_;
    g_gemm_calls++;
    if (getenv("E2_PROFILE")) {
        int i = 0;
        for (; i < g_nslots; i++)
            if (g_slots[i].M == M && g_slots[i].N == N && g_slots[i].K == K) break;
        if (i == g_nslots && g_nslots < 24) { g_slots[g_nslots++] = (struct aprof_slot){ M, N, K, 0, 0 }; }
        if (i < g_nslots) { g_slots[i].ms += t1 - t0_; g_slots[i].calls++; }
    }
}

/* ---------------------------------------------------------------- frontend
 * 16 kHz; 20 ms frames, 10 ms hop, 512-point FFT, 128 htk mel filters, log(mel + 1e-3), magnitude
 * (not power), periodic Hann, semicausal padding - i.e. Gemma4AudioFeatureExtractor. */
static void fft_complex(float *re, float *im, const e2_audio *a) {
    const int N = E2_A_FFT;
    for (int i = 0; i < N; i++) {
        const int j = a->fft_rev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= N; len <<= 1) {
        const int half = len >> 1, step = N / len;
        for (int i = 0; i < N; i += len) {
            for (int k = 0; k < half; k++) {
                const float wr = a->fft_cos[k * step], wi = -a->fft_sin[k * step];
                const int x = i + k, y = x + half;
                const float tr = re[y] * wr - im[y] * wi;
                const float ti = re[y] * wi + im[y] * wr;
                re[y] = re[x] - tr;
                im[y] = im[x] - ti;
                re[x] += tr;
                im[x] += ti;
            }
        }
    }
}

/* Frame i covers samples [i*hop - frame/2, i*hop + frame/2) of the input (semicausal), and the
 * extractor emits a frame only when its whole analysis window is real audio. */
static int mel_frames(int n_samples) {
    const int L = n_samples + E2_A_FRAME / 2;
    if (L < E2_A_FRAME + 1) return 0;
    return (L - (E2_A_FRAME + 1)) / E2_A_HOP + 1;
}

struct amel_task {
    e2_audio *a;
    const float *pcm;
    int n_samples, n_frames;
    _Atomic int next;
};

static void amel_worker(void *arg, int tid, int nthreads) {
    struct amel_task *t = arg;
    e2_audio *a = t->a;
    float re[E2_A_FFT], im[E2_A_FFT], mag[E2_A_BINS];
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int f0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (f0 >= t->n_frames) break;
        int f1 = f0 + 8;
        if (f1 > t->n_frames) f1 = t->n_frames;
        for (int f = f0; f < f1; f++) {
            const int start = f * E2_A_HOP - E2_A_FRAME / 2;
            for (int k = 0; k < E2_A_FRAME; k++) {
                const int s = start + k;
                re[k] = (s >= 0 && s < t->n_samples ? t->pcm[s] : 0.f) * a->window[k];
                im[k] = 0.f;
            }
            for (int k = E2_A_FRAME; k < E2_A_FFT; k++) { re[k] = 0.f; im[k] = 0.f; }
            fft_complex(re, im, a);
            for (int b = 0; b < E2_A_BINS; b++) mag[b] = sqrtf(re[b] * re[b] + im[b] * im[b]);
            float *out = a->mel + (size_t)f * E2_A_MEL;
            for (int c = 0; c < E2_A_MEL; c++) {
                const float *fb = a->melfb + (size_t)c * E2_A_BINS;
                float acc = 0.f;
                for (int b = 0; b < E2_A_BINS; b++) acc += mag[b] * fb[b];
                out[c] = logf(acc + 1e-3f);
            }
        }
    }
}

/* ---------------------------------------------------------------- subsample
 * Two 3x3 stride-2 convs over (time, frequency): mel[F][128] -> [F/2][64][128] -> [F/4][32][32],
 * each followed by a bias-free LayerNorm over the channels and a ReLU. The second output is
 * flattened frequency-major, channel-minor, which is the reference's permute(0,2,3,1)+reshape. */
struct aconv_task {
    e2_audio *a;
    int which; /* 0: mel -> c0_out, 1: c0_out -> c1_out */
    int Tin, T;
    _Atomic int next;
};

static void aconv_worker(void *arg, int tid, int nthreads) {
    struct aconv_task *t = arg;
    e2_audio *a = t->a;
    (void)tid;
    (void)nthreads;
    if (t->which == 0) {
        for (;;) {
            const int t0 = atomic_fetch_add_explicit(&t->next, 4, memory_order_relaxed);
            if (t0 >= t->T) break;
            int t1 = t0 + 4;
            if (t1 > t->T) t1 = t->T;
            for (int s = t0; s < t1; s++) {
                for (int f = 0; f < 64; f++) {
                    float *row = a->c0_out + ((size_t)s * 64 + f) * E2_A_C0;
                    for (int c = 0; c < E2_A_C0; c++) row[c] = 0.f;
                    for (int ki = 0; ki < E2_A_CONV_K; ki++) {
                        const int it = 2 * s - 1 + ki;
                        if (it < 0 || it >= t->Tin) continue;
                        const float *src = a->mel + (size_t)it * E2_A_MEL;
                        for (int kf = 0; kf < E2_A_CONV_K; kf++) {
                            const int jf = 2 * f - 1 + kf;
                            if (jf < 0 || jf >= E2_A_MEL) continue;
                            const float v = src[jf];
                            const float *wrow = a->c0_wt + ((size_t)ki * E2_A_CONV_K + kf) * E2_A_C0;
                            for (int c = 0; c < E2_A_C0; c++) row[c] += v * wrow[c];
                        }
                    }
                    float sum = 0.f, sq = 0.f;
                    for (int c = 0; c < E2_A_C0; c++) { sum += row[c]; sq += row[c] * row[c]; }
                    const float mu = sum / E2_A_C0;
                    const float inv = 1.0f / sqrtf(sq / E2_A_C0 - mu * mu + 1e-6f);
                    for (int c = 0; c < E2_A_C0; c++) {
                        const float v = (row[c] - mu) * inv * a->c0_norm[c];
                        row[c] = v > 0.f ? v : 0.f;
                    }
                }
            }
        }
    } else {
        for (;;) {
            const int t0 = atomic_fetch_add_explicit(&t->next, 2, memory_order_relaxed);
            if (t0 >= t->T) break;
            int t1 = t0 + 2;
            if (t1 > t->T) t1 = t->T;
            for (int s = t0; s < t1; s++) {
                for (int f = 0; f < 32; f++) {
                    float *row = a->c1_out + ((size_t)s * 32 + f) * E2_A_C1;
                    for (int c = 0; c < E2_A_C1; c++) row[c] = 0.f;
                    for (int ki = 0; ki < E2_A_CONV_K; ki++) {
                        const int it = 2 * s - 1 + ki;
                        if (it < 0 || it >= t->Tin) continue;
                        for (int kf = 0; kf < E2_A_CONV_K; kf++) {
                            const int jf = 2 * f - 1 + kf;
                            if (jf < 0 || jf >= 64) continue;
                            const float *src = a->c0_out + ((size_t)it * 64 + jf) * E2_A_C0;
                            const float *wt = a->c1_wt + ((size_t)(ki * E2_A_CONV_K + kf) * E2_A_C0) * E2_A_C1;
                            for (int ch = 0; ch < E2_A_C0; ch++) {
                                const float v = src[ch];
                                const float *wc = wt + (size_t)ch * E2_A_C1;
                                for (int c = 0; c < E2_A_C1; c++) row[c] += v * wc[c];
                            }
                        }
                    }
                    float sum = 0.f, sq = 0.f;
                    for (int c = 0; c < E2_A_C1; c++) { sum += row[c]; sq += row[c] * row[c]; }
                    const float mu = sum / E2_A_C1;
                    const float inv = 1.0f / sqrtf(sq / E2_A_C1 - mu * mu + 1e-6f);
                    for (int c = 0; c < E2_A_C1; c++) {
                        const float v = (row[c] - mu) * inv * a->c1_norm[c];
                        row[c] = v > 0.f ? v : 0.f;
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------- attention
 * One work item is one block of E2_A_CHUNK queries. For query p, key distance d (0 = the query
 * itself, up to E2_A_PAST-1 frames back): score = q_p . k_{p-d} + q_p . relk[d], soft-capped with
 * 50*tanh(s/50), softmaxed, then accumulated onto v_{p-d}. Everything is per-head, so a work item
 * is 12 queries x 8 heads over <= 12 x 128 values: no GEMM, no shared state. */
struct aatt_task {
    e2_audio *a;
    int S, nblocks;
    _Atomic int next;
};

static void aatt_worker(void *arg, int tid, int nthreads) {
    struct aatt_task *t = arg;
    e2_audio *a = t->a;
    const int D = E2_A_HEAD_DIM, HD = E2_A_HEADS * E2_A_HEAD_DIM;
    const float cap = 50.0f;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int b = atomic_fetch_add_explicit(&t->next, 1, memory_order_relaxed);
        if (b >= t->nblocks) break;
        const int p0 = b * E2_A_CHUNK;
        int rows = t->S - p0;
        if (rows > E2_A_CHUNK) rows = E2_A_CHUNK;
        for (int i = 0; i < rows; i++) {
            const int p = p0 + i;
            for (int h = 0; h < E2_A_HEADS; h++) {
                const size_t hoff = (size_t)h * D;
                const float *qp = a->q + (size_t)p * HD + hoff;
                float sc[E2_A_PAST];
                int nk = 0;
                const e2_kernels *K = e2_k();
                for (int d = 0; d < E2_A_PAST && d <= p; d++) {
                    const float *kp = a->k + (size_t)(p - d) * HD + hoff;
                    const float *rp = a->relk + (size_t)(E2_A_RELPOS - 1 - d) * HD + hoff;
                    sc[d] = K->dot_f32(qp, kp, (size_t)D) + K->dot_f32(qp, rp, (size_t)D);
                    nk = d + 1;
                }
                float mx = -INFINITY;
                for (int d = 0; d < nk; d++) {
                    const float s = cap * tanhf(sc[d] / cap);
                    sc[d] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.f;
                for (int d = 0; d < nk; d++) {
                    const float e = expf(sc[d] - mx);
                    sc[d] = e;
                    sum += e;
                }
                const float inv = 1.0f / sum;
                float *out = a->att + (size_t)p * HD + hoff;
                for (int e = 0; e < D; e++) out[e] = 0.f;
                for (int d = 0; d < nk; d++)
                    K->axpy(out, a->v + (size_t)(p - d) * HD + hoff, sc[d] * inv, D);
            }
        }
    }
}

/* ---------------------------------------------------------------- light conv */
struct alc_task {
    e2_audio *a;
    int rows;
    const float *src, *w; /* w is the transposed depthwise kernel: [tap][channel] */
    float *dst;
    _Atomic int next;
};

static void alc_worker(void *arg, int tid, int nthreads) {
    struct alc_task *t = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int t0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (t0 >= t->rows) break;
        int t1 = t0 + 8;
        if (t1 > t->rows) t1 = t->rows;
        for (int s = t0; s < t1; s++) {
            float *out = t->dst + (size_t)s * E2_A_HIDDEN;
            /* out[t] = sum_j w[j] * x[t + j - (k-1)]: the reference left-pads by k-1 and then runs an
             * unpadded convolution, so the tap with the smallest j is the oldest sample */
            for (int c = 0; c < E2_A_HIDDEN; c++) out[c] = 0.f;
            for (int j = 0; j < E2_A_LCONV_K; j++) {
                const int row = s - (E2_A_LCONV_K - 1 - j);
                if (row < 0) continue;
                const float *in = t->src + (size_t)row * E2_A_HIDDEN;
                const float *wj = t->w + (size_t)j * E2_A_HIDDEN;
                for (int c = 0; c < E2_A_HIDDEN; c++) out[c] += in[c] * wj[c];
            }
        }
    }
}

/* ---------------------------------------------------------------- layer pieces */
/* Elementwise rows on the pool. At 280 soft tokens the tower's activations are 280 x 4096, and the
 * SiLU / GLU / RMSNorm passes over them were ~40 % of the call while running on one core. */
struct amap_task {
    float *x, *b;
    int rows, D, mode, stride; /* 0: rmsnorm, 1: silu, 2: glu, 3: 0.5*x + b, 4: clamp */
    const float *gamma;
    float eps, lo, hi;
    _Atomic int next;
};

static void amap_worker(void *arg, int tid, int nthreads) {
    struct amap_task *t = arg;
    const e2_kernels *K = e2_k();
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int r0 = atomic_fetch_add_explicit(&t->next, 8, memory_order_relaxed);
        if (r0 >= t->rows) break;
        int rn = t->rows - r0;
        if (rn > 8) rn = 8;
        float *x = t->x + (size_t)r0 * t->stride; /* stride, not D: the GLU rows are 2*hidden wide */
        if (t->mode == 0) {
            K->rmsnorm(x, rn, t->D, t->gamma, t->eps);
        } else if (t->mode == 1) {
            K->silu(x, (size_t)rn * t->D);
        } else if (t->mode == 2) {
            for (int r = 0; r < rn; r++) K->glu(x + (size_t)r * t->stride, x + (size_t)r * t->stride + t->D, t->D);
        } else if (t->mode == 3) {
            K->scale(x, 0.5f, (size_t)rn * t->D);
            K->add(x, t->b + (size_t)r0 * t->D, (size_t)rn * t->D);
        }
    }
}

static void amap_rows(float *x, int rows, int D, int mode, const float *gamma, float eps) {
    struct amap_task t = { x, NULL, rows, D, mode, D, gamma, eps, 0.f, 0.f, 0 };
    if (rows <= 8 || e2_nthreads() <= 1) {
        if (mode == 0) e2_k()->rmsnorm(x, rows, D, gamma, eps);
        else e2_k()->silu(x, (size_t)rows * D);
        return;
    }
    atomic_init(&t.next, 0);
    e2_parallel(amap_worker, &t);
}

static void amap_glu(float *x, int rows, int stride, int D) {
    struct amap_task t = { x, NULL, rows, D, 2, stride, NULL, 0.f, 0.f, 0.f, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(amap_worker, &t);
}

/* x = 0.5 * x + res, in one region */
static void amap_half_add(float *x, const float *res, int rows, int D) {
    struct amap_task t = { x, (float *)res, rows, D, 3, D, NULL, 0.f, 0.f, 0.f, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(amap_worker, &t);
}

/* feed-forward block: residual, pre-norm, 1024->4096, SiLU, 4096->1024, post-norm, *0.5, +residual */
static void aff(e2_model *m, e2_audio *a, const e2_alin *a1, const e2_alin *a2, const float *npre,
                const float *npost, int S) {
    const int H = E2_A_HIDDEN, I = E2_A_INTER;
    memcpy(a->res, a->x, sizeof(float) * (size_t)S * H);
    amap_rows(a->x, S, H, 0, npre, 1e-6f);
    agemm_run(m, a, a->ff, a->x, a1, S, I, H);
    amap_rows(a->ff, S, I, 1, NULL, 0.f);
    agemm_run(m, a, a->tmp, a->ff, a2, S, H, I);
    amap_rows(a->tmp, S, H, 0, npost, 1e-6f);
    amap_half_add(a->tmp, a->res, S, H);
    memcpy(a->x, a->tmp, sizeof(float) * (size_t)S * H);
}

/* light conv: residual, pre-norm, GLU(2048), depthwise causal k=5, conv-norm, SiLU, 1024 projection */
static void alconv(e2_model *m, e2_audio *a, const e2_alayer *L, int S, int dump) {
    const int H = E2_A_HIDDEN;
    const e2_kernels *K = e2_k();
    memcpy(a->res, a->x, sizeof(float) * (size_t)S * H);
    K->rmsnorm(a->x, S, H, L->n_lc_pre, 1e-6f);
    if (dump) adump("l0_lc_norm", a->x, (size_t)S * H);
    agemm_run(m, a, a->lc, a->x, &L->lc_start, S, 2 * H, H);
    /* GLU into a compact [S][1024] block: the 2048-wide projection's two halves are the gate and the
     * value, and everything after this point is hidden-size wide */
    if (S > 8 && e2_nthreads() > 1) {
        /* in place on the first half, one region: the second half is read, never written */
        amap_glu(a->lc, S, 2 * H, H);
        for (int s = 0; s < S; s++)
            memmove(a->tmp + (size_t)s * H, a->lc + (size_t)s * 2 * H, sizeof(float) * H);
    } else {
        for (int s = 0; s < S; s++) {
            const float *row = a->lc + (size_t)s * 2 * H;
            float *dst = a->tmp + (size_t)s * H;
            memcpy(dst, row, sizeof(float) * H);
            K->glu(dst, row + H, H);
        }
    }
    if (dump) adump("l0_glu", a->tmp, (size_t)S * H);
    struct alc_task t;
    memset(&t, 0, sizeof t);
    t.a = a;
    t.rows = S;
    t.src = a->tmp;
    t.w = L->lc_conv_t;
    t.dst = a->att;
    atomic_init(&t.next, 0);
    e2_parallel(alc_worker, &t);
    if (dump) adump("l0_dw", a->att, (size_t)S * H);
    amap_rows(a->att, S, H, 0, L->n_lc_conv, 1e-6f);
    amap_rows(a->att, S, H, 1, NULL, 0.f);
    agemm_run(m, a, a->tmp, a->att, &L->lc_end, S, H, H);
    if (dump) adump("l0_lc_end", a->tmp, (size_t)S * H);
    e2_k()->add(a->tmp, a->res, (size_t)S * H);
    memcpy(a->x, a->tmp, sizeof(float) * (size_t)S * H);
}

/* ---------------------------------------------------------------- frontend + forward */
/* The front end is shared by the CPU tower and the GPU path: the GPU takes over at the [S,1024]
 * tensor that enters the first Conformer layer, so mel/FFT and the two subsampling convolutions stay
 * in one implementation and the two backends cannot drift apart. */
static int pcm_ok(const float *pcm, int n_samples) {
    if (!pcm || n_samples <= 0) { e2_set_error("empty audio input"); return -1; }
    const int n = n_samples > E2_A_MAX_SAMPLES ? E2_A_MAX_SAMPLES : n_samples;
    if (n <= E2_A_FRAME / 2) { e2_set_error("audio too short: %d samples", n); return -1; }
    if (!e2_valid_floats(pcm, (size_t)n, -1.f, 1.f)) {
        e2_set_error("audio PCM must be finite and in [-1,1]");
        return -1;
    }
    return 0;
}

int e2_audio_frontend(e2_model *m, const float *pcm, int n_samples, const float **x, int *n_tok) {
    e2_audio *a = (e2_audio *)m->audio;
    if (!a) { e2_set_error("audio tower not loaded"); return -1; }
    if (pcm_ok(pcm, n_samples) || e2_audio_prepare(m)) return -1;
    if (n_samples > E2_A_MAX_SAMPLES) n_samples = E2_A_MAX_SAMPLES;
    if (aws_alloc(a)) return -1;

    const int nf = mel_frames(n_samples);
    if (nf <= 0) { e2_set_error("audio too short: %d samples", n_samples); return -1; }
    if (asub_alloc(a, nf)) return -1;

    struct amel_task mt;
    memset(&mt, 0, sizeof mt);
    mt.a = a;
    mt.pcm = pcm;
    mt.n_samples = n_samples;
    mt.n_frames = nf;
    atomic_init(&mt.next, 0);
    e2_parallel(amel_worker, &mt);
    adump("mel", a->mel, (size_t)nf * E2_A_MEL);

    const int T1 = (nf + 1) / 2, T2 = (T1 + 1) / 2;
    if (T2 > E2_A_MAX_TOK) { e2_set_error("audio is longer than %d soft tokens", E2_A_MAX_TOK); return -1; }
    struct aconv_task ct;
    memset(&ct, 0, sizeof ct);
    ct.a = a;
    ct.which = 0;
    ct.Tin = nf;
    ct.T = T1;
    atomic_init(&ct.next, 0);
    e2_parallel(aconv_worker, &ct);
    ct.which = 1;
    ct.Tin = T1;
    ct.T = T2;
    atomic_init(&ct.next, 0);
    e2_parallel(aconv_worker, &ct);
    adump("conv", a->c1_out, (size_t)T2 * 1024);
    adump("flat", a->c1_out, (size_t)T2 * 1024);
    agemm_run(m, a, a->x, a->c1_out, &a->in_proj, T2, E2_A_HIDDEN, 1024);
    adump("conv", a->x, (size_t)T2 * 1024);
    *x = a->x;
    *n_tok = T2;
    return 0;
}

static int audio_forward(e2_model *m, const float *pcm, int n_samples, float *soft, int *n_tok) {
    e2_audio *a = (e2_audio *)m->audio;
    const e2_kernels *K = e2_k();
    if (!a) { e2_set_error("audio tower not loaded"); return -1; }
    const int prof = getenv("E2_PROFILE") != NULL;
    const double t0 = prof ? anow() : 0;
    const float *fx = NULL;
    int S = 0;
    if (e2_audio_frontend(m, pcm, n_samples, &fx, &S)) return -1;
    (void)fx;
    const int nf = mel_frames(n_samples > E2_A_MAX_SAMPLES ? E2_A_MAX_SAMPLES : n_samples);
    const double tf = prof ? anow() : 0;

    const int H = E2_A_HIDDEN, HD = E2_A_HEADS * E2_A_HEAD_DIM;
    const float kscale = (float)(log(1.0 + A_E) / log(2.0));

    for (int l = 0; l < E2_A_LAYERS; l++) {
        const e2_alayer *L = &a->layers[l];
        /* relative keys: relk[m] = rel_k_proj(pos[m]); nothing is clamped on this projection */
        e2_alin rel;
        memset(&rel, 0, sizeof rel);
        rel.w = L->rel_k;
        rel.q8 = L->q8_rel_k;
        rel.in_min = rel.out_min = -1e31f;
        rel.in_max = rel.out_max = 1e31f;
        agemm_run(m, a, a->relk, a->pos, &rel, E2_A_RELPOS, H, H);

        /* ff1 -> attn -> lconv -> ff2 -> norm_out, the reference's layer body */
        aff(m, a, &L->ff1a, &L->ff1b, L->n_ff1_pre, L->n_ff1_post, S);
        if (l == 0) adump("l0_ff1", a->x, (size_t)S * H);
        memcpy(a->res, a->x, sizeof(float) * (size_t)S * H);
        amap_rows(a->x, S, H, 0, L->n_pre_attn, 1e-6f);
        if (l == 0) adump("l0_attn_in", a->x, (size_t)S * H);
        agemm_run(m, a, a->q, a->x, &L->q, S, HD, H);
        agemm_run(m, a, a->k, a->x, &L->k, S, HD, H);
        agemm_run(m, a, a->v, a->x, &L->v, S, HD, H);
        for (int s = 0; s < S; s++) {
            float *q = a->q + (size_t)s * HD, *k = a->k + (size_t)s * HD;
            for (int h = 0; h < E2_A_HEADS; h++)
                for (int d = 0; d < E2_A_HEAD_DIM; d++) {
                    q[h * E2_A_HEAD_DIM + d] *= L->qmult[d];
                    k[h * E2_A_HEAD_DIM + d] *= kscale;
                }
        }
        struct aatt_task at;
        memset(&at, 0, sizeof at);
        at.a = a;
        at.S = S;
        at.nblocks = (S + E2_A_CHUNK - 1) / E2_A_CHUNK;
        atomic_init(&at.next, 0);
        e2_parallel(aatt_worker, &at);
        if (l == 0) adump("l0_attn_ctx", a->att, (size_t)S * HD);
        agemm_run(m, a, a->tmp, a->att, &L->post, S, H, HD);
        if (l == 0) adump("l0_attn_out", a->tmp, (size_t)S * H);
        amap_rows(a->tmp, S, H, 0, L->n_post_attn, 1e-6f);
        e2_k()->add(a->tmp, a->res, (size_t)S * H);
        memcpy(a->x, a->tmp, sizeof(float) * (size_t)S * H);
        if (l == 0) adump("l0_lconv_in", a->x, (size_t)S * H);

        alconv(m, a, L, S, l == 0);
        if (l == 0) adump("l0_lconv_out", a->x, (size_t)S * H);
        aff(m, a, &L->ff2a, &L->ff2b, L->n_ff2_pre, L->n_ff2_post, S);
        if (l == 0) adump("l0_ff2", a->x, (size_t)S * H);
        K->rmsnorm(a->x, S, H, L->n_out, 1e-6f);
        if (l == 0 || getenv("E2_DUMP")) {
            char nm[16];
            snprintf(nm, sizeof nm, "l%d", l);
            adump(nm, a->x, (size_t)S * H);
        }
    }
    const double tt = prof ? anow() : 0;

    /* output_proj (1024 -> 1536, with bias), then embed_audio: scale-free RMSNorm + 1536 -> 512 */
    agemm_run(m, a, a->tmp, a->x, &a->out_proj, S, E2_A_OUT, H);
    for (int s = 0; s < S; s++) {
        float *row = a->tmp + (size_t)s * E2_A_OUT;
        for (int d = 0; d < E2_A_OUT; d++) row[d] += e2_bf16f(a->out_proj_b[d]);
    }
    adump("out_proj", a->tmp, (size_t)S * E2_A_OUT);
    amap_rows(a->tmp, S, E2_A_OUT, 0, NULL, 1e-6f);
    agemm_run(m, a, soft, a->tmp, &a->embed, S, 512, E2_A_OUT);
    adump("soft", soft, (size_t)S * 512);
    *n_tok = S;

    if (prof) {
        const double te = anow();
        printf("[audio] frontend+subsample %.2f ms (%d frames -> %d tokens), tower %.2f ms "
               "(gemm %.2f ms in %ld regions), out+proj %.2f ms\n",
               tf - t0, nf, S, tt - tf, g_gemm_ms, g_gemm_calls, te - tt);
        for (int i = 0; i < g_nslots; i++) {
            const double macs = (double)g_slots[i].M * g_slots[i].N * g_slots[i].K;
            printf("   gemm M=%3d N=%4d K=%4d  %7.2f ms in %3d calls  %6.0f GMAC/s\n", g_slots[i].M,
                   g_slots[i].N, g_slots[i].K, g_slots[i].ms, g_slots[i].calls,
                   g_slots[i].calls * macs / g_slots[i].ms / 1e6);
        }
        g_nslots = 0;
        g_gemm_ms = 0;
        g_gemm_calls = 0;
    }
    return 0;
}

/* ---------------------------------------------------------------- load */
/* int8 weights are built on the first audio call: quantising 305 M weights at load would cost a
 * text-only process ~1 s and ~300 MB for a tower it never touches. With E2_Q8_WRITE they must exist
 * at load time (the cache file is written then), so that path stays eager. */
static int aq8(e2_model *m, e2_alin *lin, const char *name, int N, int K) {
    if (!m->use_int8 || !e2_weight_ok(lin->w)) return 0;
    return e2_q8w_load(m, &lin->q8, name, lin->w, N, K);
}

static void aset_bounds(e2_alin *lin) {
    lin->q8.out_min = lin->out_min;
    lin->q8.out_max = lin->out_max;
}

int e2_audio_prepare(e2_model *m) {
    e2_audio *a = (e2_audio *)m->audio;
    if (!a) { e2_set_error("audio tower not loaded"); return -1; }
    if (a->ready || !m->use_int8) return 0;
    char name[192];
    if (aq8(m, &a->in_proj, "audio_tower.subsample_conv_projection.input_proj_linear.weight", E2_A_HIDDEN, 1024) ||
        aq8(m, &a->out_proj, "audio_tower.output_proj.weight", E2_A_OUT, E2_A_HIDDEN) ||
        aq8(m, &a->embed, "embed_audio.embedding_projection.weight", 512, E2_A_OUT))
        return -1;
    aset_bounds(&a->in_proj);
    aset_bounds(&a->out_proj);
    aset_bounds(&a->embed);
    for (int l = 0; l < E2_A_LAYERS; l++) {
        e2_alayer *L = &a->layers[l];
        snprintf(name, sizeof name, "audio_tower.layers.%d", l);
        const char *suf[11] = { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.post",
                                "feed_forward1.ffw_layer_1", "feed_forward1.ffw_layer_2",
                                "feed_forward2.ffw_layer_1", "feed_forward2.ffw_layer_2",
                                "lconv1d.linear_start", "lconv1d.linear_end", "self_attn.relative_k_proj" };
        e2_alin *lin[10] = { &L->q, &L->k, &L->v, &L->post, &L->ff1a, &L->ff1b, &L->ff2a, &L->ff2b,
                             &L->lc_start, &L->lc_end };
        const int N[11] = { E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_INTER, E2_A_HIDDEN,
                            E2_A_INTER, E2_A_HIDDEN, 2 * E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN };
        const int K[11] = { E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_INTER,
                            E2_A_HIDDEN, E2_A_INTER, E2_A_HIDDEN, E2_A_HIDDEN, E2_A_HIDDEN };
        for (int i = 0; i < 10; i++) {
            char wn[256];
            snprintf(wn, sizeof wn, "%s.%s.linear.weight", name, suf[i]);
            if (aq8(m, lin[i], wn, N[i], K[i])) return -1;
            aset_bounds(lin[i]);
        }
        {
            char wn[256];
            snprintf(wn, sizeof wn, "%s.self_attn.relative_k_proj.weight", name);
            if (e2_q8w_load(m, &L->q8_rel_k, wn, L->rel_k, E2_A_HIDDEN, E2_A_HIDDEN)) return -1;
        }
    }
    a->ready = 1;
    return 0;
}

static int alat_init(e2_model *m, e2_alin *lin, const char *prefix, int N, int K,
                      const char *tensor_suffix, int clipped) {
    char wname[192], n_[192];
    snprintf(wname, sizeof wname, "%s.%s.linear.weight", prefix, tensor_suffix);
    const int64_t d[2] = { N, K };
    lin->w = e2_tensor_bf16(m, wname, 2, d);
    if (!e2_weight_ok(lin->w)) return -1;
    lin->in_min = lin->out_min = -1e31f;
    lin->in_max = lin->out_max = 1e31f;
    if (clipped) {
        static const char *b[4] = { "input_min", "input_max", "output_min", "output_max" };
        float *dst[4] = { &lin->in_min, &lin->in_max, &lin->out_min, &lin->out_max };
        for (int i = 0; i < 4; i++) {
            snprintf(n_, sizeof n_, "%s.%s.%s", prefix, tensor_suffix, b[i]);
            const uint16_t *p = e2_tensor_bf16(m, n_, 0, NULL); /* the clamp bounds are 0-d */
            if (!p) return -1; /* this model's learned clamp bounds are required */
            *dst[i] = e2_bf16f(p[0]);
        }
    }
    if ((e2_q8_is_write() || e2_q8_is_read()) && aq8(m, lin, wname, N, K)) return -1;
    lin->q8.out_min = lin->out_min;
    lin->q8.out_max = lin->out_max;
    return 0;
}

static float *acopy(const uint16_t *src, size_t n) {
    if (!src) return NULL;
    float *p = malloc(sizeof(float) * n);
    if (!p) e2_set_error("oom: audio tensor copy");
    else e2_bf16_to_f32(p, src, n);
    return p;
}

int e2_audio_init(e2_model *m) {
    e2_audio *a = calloc(1, sizeof *a);
    if (!a) { e2_set_error("oom"); return -1; }
    m->audio = a;

    /* the checkpoint is optional: a text+image checkpoint without an audio tower must still load,
     * so a missing audio_config tensor only disables the entry points */
    {
        const int64_t d1[1] = { 1 };
        if (!e2_tensor_exists(m, "audio_tower.output_proj.bias")) {
            free(a);
            m->audio = NULL;
            return 0;
        }
        (void)d1;
    }

    const int H = E2_A_HIDDEN;
    const int64_t dconv0[4] = { E2_A_C0, 1, E2_A_CONV_K, E2_A_CONV_K };
    const int64_t dconv1[4] = { E2_A_C1, E2_A_C0, E2_A_CONV_K, E2_A_CONV_K };
    const int64_t dout[2] = { E2_A_OUT, H };
    const int64_t demb[2] = { 512, E2_A_OUT };
    const int64_t dbias[1] = { E2_A_OUT };
    a->c0_w = e2_tensor_bf16(m, "audio_tower.subsample_conv_projection.layer0.conv.weight", 4, dconv0);
    a->c1_w = e2_tensor_bf16(m, "audio_tower.subsample_conv_projection.layer1.conv.weight", 4, dconv1);
    a->out_proj_b = e2_tensor_bf16(m, "audio_tower.output_proj.bias", 1, dbias);
    if (!a->c0_w || !a->c1_w || !a->out_proj_b) return -1;
    a->c0_norm = acopy(e2_tensor_bf16(m, "audio_tower.subsample_conv_projection.layer0.norm.weight", 1,
                                      (const int64_t[1]){ E2_A_C0 }),
                       E2_A_C0);
    a->c1_norm = acopy(e2_tensor_bf16(m, "audio_tower.subsample_conv_projection.layer1.norm.weight", 1,
                                      (const int64_t[1]){ E2_A_C1 }),
                       E2_A_C1);
    if (!a->c0_norm || !a->c1_norm) return -1;

    /* weight layouts the kernels want: conv3x3 as [tap][channel], depthwise as [tap][channel] */
    a->c0_wt = malloc(sizeof(float) * 9 * E2_A_C0);
    a->c1_wt = malloc(sizeof(float) * 9 * E2_A_C0 * E2_A_C1);
    if (!a->c0_wt || !a->c1_wt) { e2_set_error("oom"); return -1; }
    for (int k = 0; k < 9; k++)
        for (int c = 0; c < E2_A_C0; c++) a->c0_wt[k * E2_A_C0 + c] = e2_bf16f(a->c0_w[c * 9 + k]);
    for (int k = 0; k < 9; k++)
        for (int ch = 0; ch < E2_A_C0; ch++)
            for (int c = 0; c < E2_A_C1; c++)
                a->c1_wt[(k * E2_A_C0 + ch) * E2_A_C1 + c] = e2_bf16f(a->c1_w[((c * E2_A_C0 + ch) * 9) + k]);

    /* the three non-clipped projections (no clamp tensors in the checkpoint for these) */
    {
        struct { e2_alin *lin; const char *name; int N, K; } p[3] = {
            { &a->in_proj, "audio_tower.subsample_conv_projection.input_proj_linear.weight", H, 1024 },
            { &a->out_proj, "audio_tower.output_proj.weight", E2_A_OUT, H },
            { &a->embed, "embed_audio.embedding_projection.weight", 512, E2_A_OUT },
        };
        for (int i = 0; i < 3; i++) {
            const int64_t d[2] = { p[i].N, p[i].K };
            p[i].lin->w = e2_tensor_bf16(m, p[i].name, 2, d);
            p[i].lin->in_min = p[i].lin->out_min = -1e31f;
            p[i].lin->in_max = p[i].lin->out_max = 1e31f;
            if (!e2_weight_ok(p[i].lin->w)) return -1;
            if ((e2_q8_is_write() || e2_q8_is_read()) && aq8(m, p[i].lin, p[i].name, p[i].N, p[i].K)) return -1;
        }
    }
    (void)dout;
    (void)demb;

    char pre[160];
    for (int l = 0; l < E2_A_LAYERS; l++) {
        e2_alayer *L = &a->layers[l];
        snprintf(pre, sizeof pre, "audio_tower.layers.%d", l);
        if (alat_init(m, &L->q, pre, H, H, "self_attn.q_proj", 1) ||
            alat_init(m, &L->k, pre, H, H, "self_attn.k_proj", 1) ||
            alat_init(m, &L->v, pre, H, H, "self_attn.v_proj", 1) ||
            alat_init(m, &L->post, pre, H, H, "self_attn.post", 1) ||
            alat_init(m, &L->ff1a, pre, E2_A_INTER, H, "feed_forward1.ffw_layer_1", 1) ||
            alat_init(m, &L->ff1b, pre, H, E2_A_INTER, "feed_forward1.ffw_layer_2", 1) ||
            alat_init(m, &L->ff2a, pre, E2_A_INTER, H, "feed_forward2.ffw_layer_1", 1) ||
            alat_init(m, &L->ff2b, pre, H, E2_A_INTER, "feed_forward2.ffw_layer_2", 1) ||
            alat_init(m, &L->lc_start, pre, 2 * H, H, "lconv1d.linear_start", 1) ||
            alat_init(m, &L->lc_end, pre, H, H, "lconv1d.linear_end", 1)) return -1;
        {
            const int64_t drel[2] = { H, H };
            char n_[256];
            snprintf(n_, sizeof n_, "%s.self_attn.relative_k_proj.weight", pre);
            L->rel_k = e2_tensor_bf16(m, n_, 2, drel);
            if (!e2_weight_ok(L->rel_k)) return -1;
            if (e2_q8_is_write() || e2_q8_is_read()) {
                e2_alin tmp;
                memset(&tmp, 0, sizeof tmp);
                tmp.w = L->rel_k;
                if (aq8(m, &tmp, n_, H, H)) return -1;
                L->q8_rel_k = tmp.q8;
            }
            snprintf(n_, sizeof n_, "%s.self_attn.per_dim_scale", pre);
            const int64_t dscale[1] = { E2_A_HEAD_DIM };
            const uint16_t *pds = e2_tensor_bf16(m, n_, 1, dscale);
            if (!pds) return -1;
            L->qmult = malloc(sizeof(float) * E2_A_HEAD_DIM);
            if (!L->qmult) { e2_set_error("oom"); return -1; }
            /* q_scale = head_dim^-0.5 / log(2), and the reference scales by softplus(per_dim_scale) */
            const float qs = (1.0f / sqrtf((float)E2_A_HEAD_DIM)) / 0.6931471805599453f;
            for (int d = 0; d < E2_A_HEAD_DIM; d++) {
                const float v = e2_bf16f(pds[d]);
                L->qmult[d] = qs * (v > 30.f ? v : log1pf(expf(v)));
            }
            const int64_t dkern[3] = { H, 1, E2_A_LCONV_K };
            snprintf(n_, sizeof n_, "%s.lconv1d.depthwise_conv1d.weight", pre);
            L->lc_conv = e2_tensor_bf16(m, n_, 3, dkern);
            if (!L->lc_conv) return -1;
            L->lc_conv_t = malloc(sizeof(float) * E2_A_LCONV_K * H);
            if (!L->lc_conv_t) { e2_set_error("oom"); return -1; }
            for (int j = 0; j < E2_A_LCONV_K; j++)
                for (int c = 0; c < H; c++) L->lc_conv_t[j * H + c] = e2_bf16f(L->lc_conv[c * E2_A_LCONV_K + j]);
        }
        /* the eight RMSNorm scales */
        static const char *nsuf[7] = { "norm_pre_attn", "norm_post_attn", "norm_out",
                                       "feed_forward1.pre_layer_norm", "feed_forward1.post_layer_norm",
                                       "feed_forward2.pre_layer_norm", "feed_forward2.post_layer_norm" };
        float **ndst[7] = { &L->n_pre_attn, &L->n_post_attn, &L->n_out, &L->n_ff1_pre, &L->n_ff1_post,
                            &L->n_ff2_pre, &L->n_ff2_post };
        char n_[192];
        for (int i = 0; i < 7; i++) {
            snprintf(n_, sizeof n_, "%s.%s.weight", pre, nsuf[i]);
            *ndst[i] = acopy(e2_tensor_bf16(m, n_, 1, (const int64_t[1]){ H }), H);
            if (!*ndst[i]) return -1;
        }
        snprintf(n_, sizeof n_, "%s.lconv1d.pre_layer_norm.weight", pre);
        L->n_lc_pre = acopy(e2_tensor_bf16(m, n_, 1, (const int64_t[1]){ H }), H);
        snprintf(n_, sizeof n_, "%s.lconv1d.conv_norm.weight", pre);
        L->n_lc_conv = acopy(e2_tensor_bf16(m, n_, 1, (const int64_t[1]){ H }), H);
        if (!L->n_lc_pre || !L->n_lc_conv) return -1;
    }

    /* ---- feature frontend tables: periodic Hann, fft twiddles + bit reversal, htk mel bank */
    a->window = malloc(sizeof(float) * E2_A_FRAME);
    a->fft_cos = malloc(sizeof(float) * E2_A_FFT / 2);
    a->fft_sin = malloc(sizeof(float) * E2_A_FFT / 2);
    a->fft_rev = malloc(sizeof(int) * E2_A_FFT);
    a->melfb = malloc(sizeof(float) * E2_A_BINS * E2_A_MEL);
    a->pos = malloc(sizeof(float) * E2_A_RELPOS * H);
    if (!a->window || !a->fft_cos || !a->fft_sin || !a->fft_rev || !a->melfb || !a->pos) {
        e2_set_error("oom");
        return -1;
    }
    for (int n = 0; n < E2_A_FRAME; n++)
        a->window[n] = 0.5f - 0.5f * cosf(2.0f * (float)A_PI * (float)n / (float)E2_A_FRAME);
    for (int k = 0; k < E2_A_FFT / 2; k++) {
        a->fft_cos[k] = cosf(2.0f * (float)A_PI * (float)k / (float)E2_A_FFT);
        a->fft_sin[k] = sinf(2.0f * (float)A_PI * (float)k / (float)E2_A_FFT);
    }
    {
        int bits = 0;
        while ((1 << bits) < E2_A_FFT) bits++;
        for (int i = 0; i < E2_A_FFT; i++) {
            int r = 0;
            for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
            a->fft_rev[i] = r;
        }
    }
    /* htk mel: 2595*log10(1 + f/700), triangular filters linear in Hz, no area normalisation */
    {
        const int nfreq = E2_A_BINS;
        float lo = 0.f, hi = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
        float cf[E2_A_MEL + 2];
        for (int i = 0; i < E2_A_MEL + 2; i++) {
            const float m = lo + (hi - lo) * (float)i / (float)(E2_A_MEL + 1);
            cf[i] = 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
        }
        for (int b = 0; b < nfreq; b++) {
            const float f = 8000.0f * (float)b / (float)(nfreq - 1);
            for (int c = 0; c < E2_A_MEL; c++) {
                const float down = (f - cf[c]) / (cf[c + 1] - cf[c]);
                const float up = (cf[c + 2] - f) / (cf[c + 2] - cf[c + 1]);
                float w = down < up ? down : up;
                if (w < 0.f) w = 0.f;
                a->melfb[(size_t)c * nfreq + b] = w;
            }
        }
    }
    /* relative position table: position ids 12..0, [sin(inv*t) | cos(inv*t)] */
    for (int m = 0; m < E2_A_RELPOS; m++) {
        const float t = (float)(E2_A_RELPOS - 1 - m);
        for (int i = 0; i < H / 2; i++) {
            const float inc = logf(10000.0f) / (float)(H / 2 - 1);
            const float inv = expf(-inc * (float)i);
            a->pos[(size_t)m * H + i] = sinf(t * inv);
            a->pos[(size_t)m * H + H / 2 + i] = cosf(t * inv);
        }
    }
    return 0;
}

void e2_audio_free(e2_audio *a) {
    if (!a) return;
    e2_q8w_free(&a->in_proj.q8);
    e2_q8w_free(&a->out_proj.q8);
    e2_q8w_free(&a->embed.q8);
    for (int l = 0; l < E2_A_LAYERS; l++) {
        e2_alayer *L = &a->layers[l];
        /* the int8 weights are built lazily (e2_audio_prepare): free whatever exists */
        e2_q8w_free(&L->q.q8); e2_q8w_free(&L->k.q8); e2_q8w_free(&L->v.q8); e2_q8w_free(&L->post.q8);
        e2_q8w_free(&L->ff1a.q8); e2_q8w_free(&L->ff1b.q8);
        e2_q8w_free(&L->ff2a.q8); e2_q8w_free(&L->ff2b.q8);
        e2_q8w_free(&L->lc_start.q8); e2_q8w_free(&L->lc_end.q8);
        e2_q8w_free(&L->q8_rel_k);
        free(L->n_pre_attn); free(L->n_post_attn); free(L->n_out);
        free(L->n_ff1_pre); free(L->n_ff1_post); free(L->n_ff2_pre); free(L->n_ff2_post);
        free(L->n_lc_pre); free(L->n_lc_conv); free(L->qmult); free(L->lc_conv_t);
    }
    free(a->c0_norm); free(a->c1_norm); free(a->c0_wt); free(a->c1_wt);
    free(a->window); free(a->fft_cos); free(a->fft_sin); free(a->fft_rev); free(a->melfb); free(a->pos);
    free(a->mel); free(a->c0_out); free(a->c1_out);
    if (a->x) free(a->x); /* the workspace block starts at x */
    free(a);
}

/* ---------------------------------------------------------------- entry points */
/* The generic splice: one text-tower input block, with the image and/or audio soft tokens dropped
 * into their placeholder slots. */
static int mm_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pixels,
                    const int32_t *ppos, int n_patches, const float *pcm, int n_samples, float *out768) {
    e2_set_error("");
    char verr[256];
    if (!m) { e2_set_error("model is NULL"); return -1; }
    e2_audio *a = (e2_audio *)m->audio;
    if (!a) { e2_set_error("audio tower not loaded"); return -1; }
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) {
        e2_set_error("sequence length %d out of range", n_ids);
        return -1;
    }
    /* Everything below this point reads weights (the audio INT8 build, the vision tower, the token
     * table), so bad input is rejected here first. */
    if (!out768) { e2_set_error("output pointer is NULL"); return -1; }
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) { e2_set_error("%s", verr); return -1; }
    if (pixels && !ppos) { e2_set_error("position ids pointer is NULL"); return -1; }
    if (pixels && (n_patches <= 0 || n_patches > E2_V_MAX_PATCHES)) {
        e2_set_error("patch count %d out of range (max %d)", n_patches, E2_V_MAX_PATCHES);
        return -1;
    }
    if (pcm && pcm_ok(pcm, n_samples)) return -1;
    if (e2_audio_prepare(m) || aws_alloc(a)) return -1;

    const float *isoft = NULL;
    int n_img = 0;
    if (pixels) {
        n_img = e2_vision_soft_tokens(m, pixels, ppos, n_patches, &isoft);
        if (n_img < 0) return -1;
    }
    int n_aud = 0;
    if (pcm) {
        if (audio_forward(m, pcm, n_samples, a->soft, &n_aud)) return -1;
    }

    float *e = a->emb;
    int img = 0, aud = 0;
    for (int s = 0; s < n_ids; s++) {
        float *dst = e + (size_t)s * 512;
        if (ids[s] == I_TOKEN) {
            if (!isoft) { e2_set_error("<|image|> slot but no image given"); return -1; }
            if (img >= n_img) { e2_set_error("more image slots than soft tokens"); return -1; }
            memcpy(dst, isoft + (size_t)img++ * 512, sizeof(float) * 512);
        } else if (ids[s] == A_TOKEN) {
            if (!pcm) { e2_set_error("<|audio|> slot but no audio given"); return -1; }
            if (aud >= n_aud) { e2_set_error("more audio slots than soft tokens"); return -1; }
            memcpy(dst, a->soft + (size_t)aud++ * 512, sizeof(float) * 512);
        } else {
            e2_embed_token(m, ids[s], dst);
        }
    }
    if (isoft && img != n_img) { e2_set_error("%d soft tokens for %d image slots", n_img, img); return -1; }
    if (pcm && aud != n_aud) { e2_set_error("%d soft tokens for %d audio slots", n_aud, aud); return -1; }
    return e2_text_embed_embeds(m, e, n_ids, out768);
}

int e2_text_audio_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pcm, int n_samples,
                        float *out768) {
    return mm_embed(m, ids, n_ids, NULL, NULL, 0, pcm, n_samples, out768);
}

int e2_text_image_audio_embed(e2_model *m, const int32_t *ids, int n_ids, const float *pixels,
                              const int32_t *ppos, int n_patches, const float *pcm, int n_samples,
                              float *out768) {
    return mm_embed(m, ids, n_ids, pixels, ppos, n_patches, pcm, n_samples, out768);
}

int e2_audio_embed(e2_model *m, const float *pcm, int n_samples, float *out768) {
    e2_set_error("");
    if (!m) { e2_set_error("model is NULL"); return -1; }
    e2_audio *a = (e2_audio *)m->audio;
    if (!a) { e2_set_error("audio tower not loaded"); return -1; }
    if (pcm_ok(pcm, n_samples)) return -1;
    if (!out768) { e2_set_error("output pointer is NULL"); return -1; }
    if (e2_audio_prepare(m) || aws_alloc(a)) return -1;
    int n = 0;
    if (audio_forward(m, pcm, n_samples, a->soft, &n)) return -1;
    const int S = n + 4;
    if (S > E2_MAX_SEQ) { e2_set_error("audio layout of %d tokens exceeds the context", S); return -1; }
    float *e = a->emb;
    e2_embed_token(m, 2, e);                       /* <bos> */
    e2_embed_token(m, A_BOA, e + 512);             /* <|audio> */
    memcpy(e + (size_t)2 * 512, a->soft, sizeof(float) * (size_t)n * 512);
    e2_embed_token(m, A_EOA, e + (size_t)(2 + n) * 512); /* <audio|> */
    e2_embed_token(m, 1, e + (size_t)(3 + n) * 512);     /* <eos> */
    return e2_text_embed_embeds(m, e, S, out768);
}

/* Soft tokens a clip of n_samples at 16 kHz becomes: ceil(mel frames / 4), without running anything.
 * The reference calls this one soft token per 40 ms of audio. */
int e2_audio_tokens(int n_samples) {
    if (n_samples <= 0) return 0;
    const int nf = mel_frames(n_samples > E2_A_MAX_SAMPLES ? E2_A_MAX_SAMPLES : n_samples);
    return nf <= 0 ? 0 : (((nf + 1) / 2) + 1) / 2;
}
