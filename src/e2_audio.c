/* e2_audio.c - mel spectrogram + Gemma 4 audio conformer tower. */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>

#define AUDIO_MAX_SAMPLES 480000  /* feature extractor's default max_length */

/* ------------------------------------------------------------------ */
/* radix-2 FFT                                                        */
/* ------------------------------------------------------------------ */

static void fft512(const float *in, float *mag) {
    static int init = 0;
    static int rev[512];
    static float cs[256], sn[256];
    if (!init) {
        for (int i = 0; i < 512; i++) {
            int r = 0, x = i;
            for (int b = 0; b < 9; b++) { r = (r << 1) | (x & 1); x >>= 1; }
            rev[i] = r;
        }
        for (int i = 0; i < 256; i++) {
            double a = -2.0 * M_PI * i / 512.0;
            cs[i] = (float)cos(a);
            sn[i] = (float)sin(a);
        }
        init = 1;
    }
    float re[512], im[512];
    for (int i = 0; i < 512; i++) { re[rev[i]] = in[i]; im[rev[i]] = 0.0f; }
    for (int len = 2; len <= 512; len <<= 1) {
        int half = len >> 1, step = 512 / len;
        for (int i = 0; i < 512; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = cs[j * step], wi = sn[j * step];
                int a = i + j, b = a + half;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
            }
        }
    }
    for (int i = 0; i < 257; i++) mag[i] = sqrtf(re[i] * re[i] + im[i] * im[i]);
}

/* Features: [T,128] float, mask [T] (1 valid), zeroed for invalid frames. */

static double aud_prof[9];

/* 2 = AVX-512, 1 = AVX2 + FMA, 0 = scalar.  Every AVX-512 part has AVX2. */
static int aud_simd(void) {
#if defined(__x86_64__)
    static int v = -1;
    if (v < 0) {
        __builtin_cpu_init();
        v = __builtin_cpu_supports("avx512f") ? 2
          : (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
    }
    return v;
#else
    return 0;
#endif
}

/* 3x3 stride-2 conv, channel-last in/out, vectorised over output channels.
 * wT is [3][3][c_in][c_out_pad] so 32 output channels are four ymm loads.
 * The reduction order stays (ic, ky, kx), so results match the scalar loop. */
typedef struct {
    const float *in, *wT; float *out;
    int c_in, T_in, W_in, c_out, T_out, W_out, cpad;
} conv3_arg;

#if defined(__x86_64__)
static void conv3_rows_a2(const conv3_arg *a, int t0, int t1);
static void mel_bank_a2(const float *mag, const float *mel, float *dst, int n, float floor);
#endif

typedef struct {
    float *x; int T, W, C; const float *norm; float eps;
} ln_arg;

static void conv3_work(void *vp, int tid, int nt) {
    conv3_arg *a = vp;
    int cpad = a->cpad, c_out = a->c_out;
    int t0, t1;
    e2_split(a->T_out, tid, nt, &t0, &t1);
#if defined(__x86_64__)
    if (aud_simd() >= 1) { conv3_rows_a2(a, t0, t1); return; }
#endif
    float acc[128];
    for (int t = t0; t < t1; t++) {
        for (int w = 0; w < a->W_out; w++) {
            for (int i = 0; i < cpad; i++) acc[i] = 0.0f;
            for (int ic = 0; ic < a->c_in; ic++) {
                for (int ky = 0; ky < 3; ky++) {
                    int iy = t * 2 - 1 + ky;
                    if (iy < 0 || iy >= a->T_in) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = w * 2 - 1 + kx;
                        if (ix < 0 || ix >= a->W_in) continue;
                        float v = a->in[((size_t)iy * a->W_in + ix) * a->c_in + ic];
                        const float *wt = a->wT + ((size_t)(ky * 3 + kx) * a->c_in + ic) * cpad;
                        for (int j = 0; j < cpad; j++) acc[j] += v * wt[j];
                    }
                }
            }
            float *op = a->out + ((size_t)t * a->W_out + w) * c_out;
            for (int j = 0; j < c_out; j++) op[j] = acc[j];
        }
    }
}

/* layernorm over the last (channel) axis + relu, in place, one thread per pixel
 * row block; the reduction order over channels is unchanged. */
static void ln_relu_work(void *vp, int tid, int nt) {
    ln_arg *a = vp;
    int px = a->T * a->W, C = a->C;
    int p0, p1;
    e2_split(px, tid, nt, &p0, &p1);
    for (int p = p0; p < p1; p++) {
        float *v = a->x + (size_t)p * C;
        float mean = 0.0f, var = 0.0f;
        for (int c = 0; c < C; c++) mean += v[c];
        mean /= (float)C;
        for (int c = 0; c < C; c++) { float d = v[c] - mean; var += d * d; }
        var /= (float)C;
        float inv = 1.0f / sqrtf(var + a->eps);
        for (int c = 0; c < C; c++) {
            float y = (v[c] - mean) * inv * a->norm[c];
            v[c] = y > 0.0f ? y : 0.0f;
        }
    }
}

typedef struct {
    const float *samples; int n, T; float *feat; uint8_t *mask;
    const float *hann, *mel; float floor;
} mel_arg;

static void mel_work(void *vp, int tid, int nt) {
    mel_arg *a = vp;
    int t0, t1;
    e2_split(a->T, tid, nt, &t0, &t1);
    float frame[512];
    float mag[257];
    for (int t = t0; t < t1; t++) {
        /* padded waveform has 160 leading zeros; frame starts at t*160 */
        for (int i = 0; i < 320; i++) {
            int idx = t * 160 + i - 160;
            float v = (idx >= 0 && idx < a->n) ? a->samples[idx] : 0.0f;
            frame[i] = v * a->hann[i];
        }
        for (int i = 320; i < 512; i++) frame[i] = 0.0f;
        fft512(frame, mag);
#if defined(__x86_64__)
        if (aud_simd() >= 1) {
            mel_bank_a2(mag, a->mel, a->feat + (size_t)t * 128, 128, a->floor);
        } else
#endif
        {
            for (int c = 0; c < 128; c++) {
                float s = 0.0f;
                for (int f = 0; f < 257; f++) s += mag[f] * a->mel[f * 128 + c];
                a->feat[(size_t)t * 128 + c] = logf(s + a->floor);
            }
        }
        a->mask[t] = (t * 160 + 160) < a->n;
        if (!a->mask[t]) {
            float *p = a->feat + (size_t)t * 128;
            for (int c = 0; c < 128; c++) p[c] = 0.0f;
        }
    }
}

static int audio_mel(e2_ctx *ctx, const float *samples, int n,
                     float **feat_out, uint8_t **mask_out, int *T_out) {
    e2_model *m = &ctx->m;
    if (n <= 0) return e2_fail("empty audio");
    if (n > AUDIO_MAX_SAMPLES) n = AUDIO_MAX_SAMPLES;
    int npad = (n + 127) & ~127;
    int T = (npad - 161) / 160 + 1;
    if (T < 1) T = 1;
    float *feat = calloc((size_t)T * 128, sizeof(float));
    uint8_t *mask = calloc((size_t)T, 1);
    if (!feat || !mask) { free(feat); free(mask); return e2_fail("oom"); }
    mel_arg ma = {samples, n, T, feat, mask, m->hann, m->mel_filters,
                  ctx->m.meta.mel_floor};
    if (ctx->nthreads > 1 && T >= 32) e2_parallel(ctx, mel_work, &ma);
    else mel_work(&ma, 0, 1);
    {
        const char *dir = getenv("E2_DUMP");
        if (dir) {
            char path[512];
            snprintf(path, sizeof(path), "%s/aud_feat.bin", dir);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(feat, sizeof(float), (size_t)T * 128, f); fclose(f); }
        }
    }
    *feat_out = feat;
    *mask_out = mask;
    *T_out = T;
    return 0;
}

/* ------------------------------------------------------------------ */
/* layers                                                             */
/* ------------------------------------------------------------------ */

static void adump(const char *name, const float *p, size_t n) {
    const char *dir = getenv("E2_DUMP");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, sizeof(float), n, f);
    fclose(f);
}

static void audio_ff(e2_ctx *ctx, e2_audio_layer *L, float *x, int T,
                     int which, int H, int I) {
    e2_meta *c = &ctx->m.meta;
    float *h = e2_arena_alloc(&ctx->scratch, (size_t)T * I * sizeof(float));
    float *up = e2_arena_alloc(&ctx->scratch, (size_t)T * I * sizeof(float));
    e2_linear *a = which == 0 ? &L->ff1a : &L->ff2a;
    e2_linear *b = which == 0 ? &L->ff1b : &L->ff2b;
    const float *pre = which == 0 ? L->ff1_pre_norm : L->ff2_pre_norm;
    const float *post = which == 0 ? L->ff1_post_norm : L->ff2_post_norm;
    /* norm, two linears with SiLU, then the gated residual on x in place */
    e2_rmsnorm(ctx, h, x, pre, T, H, c->audio_eps);
    e2_linear_forward(ctx, up, h, T, a);
    e2_silu_ctx(ctx, up, T * I);
    e2_linear_forward(ctx, h, up, T, b);
    e2_rmsnorm(ctx, h, h, post, T, H, c->audio_eps);
    e2_vec_scale_ctx(ctx, h, T * H, c->audio_residual);
    e2_add_inplace_ctx(ctx, x, h, T * H);
}

static float softplus(float v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return expf(v);
    return log1pf(e2_fexp(v));
}

/* dot product, AVX-512 when available, else AVX2 (D is 128 here) */
#if defined(__x86_64__)
#include <immintrin.h>
__attribute__((target("avx512f")))
static float dot128_v(const float *a, const float *b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
    }
    for (; i + 16 <= n; i += 16)
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1),
                                                 _mm512_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

__attribute__((target("avx512f")))
static void axpy_v(float *o, const float *v, float w, int n) {
    __m512 vw = _mm512_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(vw, _mm512_loadu_ps(v + i),
                                                _mm512_loadu_ps(o + i)));
    for (; i < n; i++) o[i] += w * v[i];
}

/* AVX2 + FMA twins for hosts without AVX-512 (Arrow Lake, Alder Lake,
 * Zen 2/3/4): 8 lanes instead of 1. */
#define A2 __attribute__((target("avx2,fma")))

A2 static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

A2 static float dot128_a2(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float s = hsum8(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

A2 static void axpy_a2(float *o, const float *v, float w, int n) {
    __m256 vw = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(vw, _mm256_loadu_ps(v + i),
                                                _mm256_loadu_ps(o + i)));
    for (; i < n; i++) o[i] += w * v[i];
}

/* Mel filterbank: one pass over the 257 bins per 64 output channels, so the
 * sum for a channel still runs over f in ascending order. */
A2 static void mel_bank_a2(const float *mag, const float *mel, float *dst, int n,
                           float floor) {
    for (int c0 = 0; c0 < n; c0 += 64) {
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        __m256 s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        for (int f = 0; f < 257; f++) {
            const float *m = mel + (size_t)f * n + c0;
            __m256 v = _mm256_broadcast_ss(mag + f);
            s0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m), s0);
            s1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 8), s1);
            s2 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 16), s2);
            s3 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 24), s3);
            s4 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 32), s4);
            s5 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 40), s5);
            s6 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 48), s6);
            s7 = _mm256_fmadd_ps(v, _mm256_loadu_ps(m + 56), s7);
        }
        __m256 s[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int i = 0; i < 8; i++) {
            float buf[8];
            _mm256_storeu_ps(buf, s[i]);
            for (int j = 0; j < 8; j++) dst[c0 + i * 8 + j] = logf(buf[j] + floor);
        }
    }
}

/* 3x3 stride-2 conv over 32 output channels per pass: the accumulators stay in
 * four ymm registers instead of a memory array.  cpad is 32 or 64 here. */
A2 static void conv3_rows_a2(const conv3_arg *a, int t0, int t1) {
    for (int c0 = 0; c0 < a->cpad; c0 += 32) {
        for (int t = t0; t < t1; t++) {
            for (int w = 0; w < a->W_out; w++) {
                __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
                for (int ic = 0; ic < a->c_in; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = t * 2 - 1 + ky;
                        if (iy < 0 || iy >= a->T_in) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = w * 2 - 1 + kx;
                            if (ix < 0 || ix >= a->W_in) continue;
                            __m256 v = _mm256_set1_ps(
                                a->in[((size_t)iy * a->W_in + ix) * a->c_in + ic]);
                            const float *wt = a->wT +
                                ((size_t)(ky * 3 + kx) * a->c_in + ic) * a->cpad + c0;
                            s0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(wt), s0);
                            s1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(wt + 8), s1);
                            s2 = _mm256_fmadd_ps(v, _mm256_loadu_ps(wt + 16), s2);
                            s3 = _mm256_fmadd_ps(v, _mm256_loadu_ps(wt + 24), s3);
                        }
                    }
                }
                float *op = a->out + ((size_t)t * a->W_out + w) * a->c_out + c0;
                _mm256_storeu_ps(op, s0);
                _mm256_storeu_ps(op + 8, s1);
                _mm256_storeu_ps(op + 16, s2);
                _mm256_storeu_ps(op + 24, s3);
            }
        }
    }
}

/* row-wise scale by a D-wide pattern: the per-dim q scale and the k scale. */
A2 static void scale_rows_a2(float *p, int rows, int D, const float *s) {
    for (int r = 0; r < rows; r++) {
        float *x = p + (size_t)r * D;
        int d = 0;
        for (; d + 8 <= D; d += 8)
            _mm256_storeu_ps(x + d, _mm256_mul_ps(_mm256_loadu_ps(x + d),
                                                  _mm256_loadu_ps(s + d)));
        for (; d < D; d++) x[d] *= s[d];
    }
}
#endif
static void aud_qk_scale(float *q, float *k, const float *qs, const float *ks,
                         int rows, int D) {
#if defined(__x86_64__)
    if (aud_simd() >= 1) {
        scale_rows_a2(q, rows, D, qs);
        scale_rows_a2(k, rows, D, ks);
        return;
    }
#endif
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < D; d++) { q[(size_t)r * D + d] *= qs[d]; k[(size_t)r * D + d] *= ks[d]; }
}
static float dot128(const float *a, const float *b, int n) {
#if defined(__x86_64__)
    int lv = aud_simd();
    if (lv == 2) return dot128_v(a, b, n);
    if (lv >= 1) return dot128_a2(a, b, n);
#endif
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static void axpy(float *o, const float *v, float w, int n) {
#if defined(__x86_64__)
    int lv = aud_simd();
    if (lv == 2) { axpy_v(o, v, w, n); return; }
    if (lv >= 1) { axpy_a2(o, v, w, n); return; }
#endif
    for (int i = 0; i < n; i++) o[i] += w * v[i];
}

/* Chunked local attention.  Work items are (chunk, head) so the pool keeps
 * every core busy while each item stays cache resident. */
typedef struct {
    const float *q, *k, *v, *rk;
    float *out;
    int T, heads, D, chunk, ctx_left, ctxsz, pos_len, nb;
    const uint8_t *mask;
    float softcap, kscale;
} aud_att_arg;

static void aud_att_work(void *vp, int tid, int nt) {
    aud_att_arg *a = vp;
    int nh = a->nb * a->heads;
    float sc[128];
    for (int it = tid; it < nh; it += nt) {
        int b = it / a->heads, hh = it % a->heads;
        for (int i = 0; i < a->chunk; i++) {
            int qg = b * a->chunk + i;
            if (qg >= a->T || !a->mask[qg]) continue;
            const float *qp = a->q + ((size_t)qg * a->heads + hh) * a->D;
            float mx = -INFINITY;
            for (int j = 0; j < a->ctxsz; j++) {
                int kg = b * a->chunk - (a->ctx_left - 1) + j;
                float s = -INFINITY;
                if (kg >= 0 && kg < a->T && a->mask[kg]) {
                    int dist = i + (a->ctx_left - 1) - j;
                    if (dist >= 0 && dist < a->ctx_left - 1) {
                        const float *kp = a->k + ((size_t)kg * a->heads + hh) * a->D;
                        s = dot128(qp, kp, a->D);
                        int r = j - i;
                        if (r >= 0 && r < a->pos_len) {
                            const float *rp = a->rk + ((size_t)r * a->heads + hh) * a->D;
                            s += dot128(qp, rp, a->D);
                        }
                    }
                }
                sc[j] = s;
            }
            /* logit soft cap.  The mask lives in its own array: tanh maps
             * -inf to -1 (and used to map it to NaN), so it cannot double as
             * the sentinel. */
            uint8_t ok[128];
            int any = 0;
            for (int j = 0; j < a->ctxsz; j++) {
                ok[j] = sc[j] != -INFINITY;
                if (ok[j]) { sc[j] /= a->softcap; any = 1; }
            }
            if (!any) continue;
            e2_tanh_span(sc, a->ctxsz);
            mx = -INFINITY;
            for (int j = 0; j < a->ctxsz; j++) {
                if (!ok[j]) continue;
                sc[j] *= a->softcap;
                if (sc[j] > mx) mx = sc[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < a->ctxsz; j++) {
                if (!ok[j]) { sc[j] = 0.0f; continue; }
                sc[j] = e2_fexp(sc[j] - mx);
                sum += sc[j];
            }
            if (sum <= 0.0f) continue;
            float inv = 1.0f / sum;
            float *op = a->out + ((size_t)qg * a->heads + hh) * a->D;
            for (int j = 0; j < a->ctxsz; j++) {
                float w = sc[j] * inv;
                if (w == 0.0f) continue;
                int kg = b * a->chunk - (a->ctx_left - 1) + j;
                if (kg < 0 || kg >= a->T) continue;
                const float *vp2 = a->v + ((size_t)kg * a->heads + hh) * a->D;
                axpy(op, vp2, w, a->D);
            }
        }
    }
}

static void audio_attention(e2_ctx *ctx, e2_audio_layer *L, float *x, int T,
                            const uint8_t *mask, int H, int heads, int D,
                            const float *pe, int pos_len) {
    e2_meta *c = &ctx->m.meta;
    int nb = (T + c->audio_chunk - 1) / c->audio_chunk;
    int chunk = c->audio_chunk;
    int ctxsz = chunk + (c->audio_ctx_left - 1) + c->audio_ctx_right;
    float *q = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *k = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *v = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *out = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    if (!q || !k || !v || !out) return;
    double _a0 = e2_now_ms();
    {
        const e2_linear *ql[3] = {&L->q, &L->k, &L->v};
        float *yl[3] = {q, k, v};
        e2_linear_forward_multi(ctx, x, T, ql, yl, 3);
    }
    aud_prof[4] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    float qs[256], ks[256];
    for (int d = 0; d < D; d++) {
        qs[d] = c->audio_qscale * softplus(L->per_dim_scale[d]);
        ks[d] = c->audio_kscale;
    }
    const float kscale = c->audio_kscale;
    aud_qk_scale(q, k, qs, ks, T * heads, D);
    aud_prof[5] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    float *rk = e2_arena_alloc(&ctx->scratch, (size_t)pos_len * H * sizeof(float));
    e2_linear_forward(ctx, rk, pe, pos_len, &L->rel_k);
    aud_prof[6] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    memset(out, 0, (size_t)T * H * sizeof(float));
    aud_att_arg aa = {q, k, v, rk, out, T, heads, D, chunk, c->audio_ctx_left,
                      ctxsz, pos_len, nb, mask, c->audio_softcap, kscale};
    e2_parallel(ctx, aud_att_work, &aa);
    aud_prof[7] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    e2_linear_forward(ctx, x, out, T, &L->post);
    aud_prof[8] += e2_now_ms() - _a0;
}

#if defined(__x86_64__)
__attribute__((target("avx512f")))
static void dw_conv_add_v(float *op, const float *hp, const float *wp, int H) {
    int c0 = 0;
    for (; c0 + 16 <= H; c0 += 16)
        _mm512_storeu_ps(op + c0,
                         _mm512_fmadd_ps(_mm512_loadu_ps(hp + c0),
                                         _mm512_loadu_ps(wp + c0),
                                         _mm512_loadu_ps(op + c0)));
    for (; c0 < H; c0++) op[c0] += hp[c0] * wp[c0];
}

A2 static void dw_conv_add_a2(float *op, const float *hp, const float *wp, int H) {
    int c0 = 0;
    for (; c0 + 8 <= H; c0 += 8)
        _mm256_storeu_ps(op + c0,
                         _mm256_fmadd_ps(_mm256_loadu_ps(hp + c0),
                                         _mm256_loadu_ps(wp + c0),
                                         _mm256_loadu_ps(op + c0)));
    for (; c0 < H; c0++) op[c0] += hp[c0] * wp[c0];
}
#endif
static void dw_conv_add(float *op, const float *hp, const float *wp, int H) {
#if defined(__x86_64__)
    int lv = aud_simd();
    if (lv == 2) { dw_conv_add_v(op, hp, wp, H); return; }
    if (lv >= 1) { dw_conv_add_a2(op, hp, wp, H); return; }
#endif
    for (int c0 = 0; c0 < H; c0++) op[c0] += hp[c0] * wp[c0];
}

typedef struct {
    const float *h, *dwt; float *conv; int T, H, K;
} dwconv_arg;

/* Causal depthwise conv over time, split over (t, channel block). */
static void dwconv_work(void *vp, int tid, int nt) {
    dwconv_arg *a = vp;
    int nb = (a->H + 255) / 256, lo, hi;
    e2_split(a->T * nb, tid, nt, &lo, &hi);
    for (int it = lo; it < hi; it++) {
        int t = it / nb, c0 = (it % nb) * 256;
        int c1 = c0 + 256 > a->H ? a->H : c0 + 256;
        float *op = a->conv + (size_t)t * a->H + c0;
        for (int c = 0; c < c1 - c0; c++) op[c] = 0.0f;
        for (int j = 0; j < a->K; j++) {
            int tt = t - (a->K - 1) + j;
            if (tt < 0 || tt >= a->T) continue;
            dw_conv_add(op, a->h + (size_t)tt * 2 * a->H + c0, a->dwt + (size_t)j * a->H + c0,
                        c1 - c0);
        }
    }
}

static void audio_lconv(e2_ctx *ctx, e2_audio_layer *L, float *x, int T,
                        int H, int K) {
    e2_meta *c = &ctx->m.meta;
    float *h = e2_arena_alloc(&ctx->scratch, (size_t)T * 2 * H * sizeof(float));
    float *conv = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *dwt = e2_arena_alloc(&ctx->scratch, (size_t)K * H * sizeof(float));
    if (!h || !conv || !dwt) return;
    e2_rmsnorm(ctx, h, x, L->lc_pre_norm, T, H, c->audio_eps);
    e2_linear_forward(ctx, h, h, T, &L->lc_start);   /* [T, 2H] */
    e2_gated_silu(h, T, H);
    for (int ch = 0; ch < H; ch++)
        for (int j = 0; j < K; j++) dwt[j * H + ch] = L->dw_conv[ch * K + j];
    /* causal depthwise conv over time, vectorised over channels */
    {
        dwconv_arg da = {h, dwt, conv, T, H, K};
        e2_parallel(ctx, dwconv_work, &da);
    }
    e2_rmsnorm(ctx, conv, conv, L->lc_conv_norm, T, H, c->audio_eps);
    e2_silu_ctx(ctx, conv, T * H);
    e2_linear_forward(ctx, h, conv, T, &L->lc_end);
    e2_add_inplace_ctx(ctx, x, h, T * H);
}

/* ------------------------------------------------------------------ */

int e2_audio_forward(e2_ctx *ctx, const float *samples, int n, float *out, int *ntok) {
    e2_model *m = &ctx->m;
    e2_meta *c = &m->meta;
    if (!(c->flags & E2_MOD_AUDIO)) return e2_fail("model has no audio tower");
    int prof = getenv("E2_PROFILE") != NULL;
    double now0 = e2_now_ms();
    memset(aud_prof, 0, sizeof(aud_prof));
    float *feat = NULL;
    uint8_t *mask = NULL;
    int T = 0;
    if (audio_mel(ctx, samples, n, &feat, &mask, &T) != 0) return -1;
    double t_mel = e2_now_ms() - now0;

    const int H = c->audio_hidden, heads = c->audio_heads, D = c->audio_head_dim;
    const int I = c->audio_inter, K = c->audio_conv_k, sub0 = 128, sub1 = 32;
    e2_arena_reset(&ctx->scratch);

    double t_sub0 = e2_now_ms();
    /* ---- subsample conv (two 3x3 stride-2 convs), channel-last ---- */
    int T0 = (T - 1) / 2 + 1;
    int W0 = 64;
    int T1 = (T0 - 1) / 2 + 1;
    int W1 = 32;
    for (int t = 0; t < T; t++)
        if (!mask[t]) for (int f = 0; f < 128; f++) feat[(size_t)t * 128 + f] = 0.0f;
    /* weights transposed once per forward: [3][3][c_in][c_out] */
    float *w0t = e2_arena_alloc(&ctx->scratch, 3 * 3 * 1 * sub0 * sizeof(float));
    float *w1t = e2_arena_alloc(&ctx->scratch, 3 * 3 * sub0 * sub1 * sizeof(float));
    float *c0 = e2_arena_alloc(&ctx->scratch, (size_t)T0 * W0 * sub0 * sizeof(float));
    float *c1 = e2_arena_alloc(&ctx->scratch, (size_t)T1 * W1 * sub1 * sizeof(float));
    if (!w0t || !w1t || !c0 || !c1) { free(feat); free(mask); return e2_fail("oom"); }
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++)
            for (int oc = 0; oc < sub0; oc++)          /* conv0: 1 input channel */
                w0t[(ky * 3 + kx) * sub0 + oc] = m->aud_conv0[(size_t)oc * 9 + ky * 3 + kx];
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++)
            for (int ic = 0; ic < sub0; ic++)
                for (int oc = 0; oc < sub1; oc++)
                    w1t[((size_t)(ky * 3 + kx) * sub0 + ic) * sub1 + oc] =
                        m->aud_conv1[((size_t)oc * sub0 + ic) * 9 + ky * 3 + kx];
    {
        conv3_arg ca = {feat, w0t, c0, 1, T, 128, sub0, T0, W0, sub0};
        e2_parallel(ctx, conv3_work, &ca);
        ln_arg la = {c0, T0, W0, sub0, m->aud_sub0_norm, c->audio_eps};
        e2_parallel(ctx, ln_relu_work, &la);
    }
    uint8_t *m0 = e2_arena_alloc(&ctx->scratch, (size_t)T0);
    if (!m0) return e2_fail("oom");
    for (int t = 0; t < T0; t++) m0[t] = mask[t * 2];
    for (int t = 0; t < T0; t++)
        if (!m0[t]) for (int i = 0; i < W0 * sub0; i++) c0[(size_t)t * W0 * sub0 + i] = 0.0f;
    {
        conv3_arg ca = {c0, w1t, c1, sub0, T0, W0, sub1, T1, W1, sub1};
        e2_parallel(ctx, conv3_work, &ca);
        ln_arg la = {c1, T1, W1, sub1, m->aud_sub1_norm, c->audio_eps};
        e2_parallel(ctx, ln_relu_work, &la);
    }
    uint8_t *m1 = e2_arena_alloc(&ctx->scratch, (size_t)T1);
    if (!m1) return e2_fail("oom");
    for (int t = 0; t < T1; t++) m1[t] = m0[t * 2];
    for (int t = 0; t < T1; t++)
        if (!m1[t]) for (int i = 0; i < sub1 * W1; i++) c1[(size_t)t * sub1 * W1 + i] = 0.0f;
    /* c1 is already [T1][W1][c], i.e. the flattened [T1, 32*32] input */
    float *x = e2_arena_alloc(&ctx->scratch, (size_t)T1 * H * sizeof(float));
    if (!x) return e2_fail("oom");
    e2_linear linproj = {0}; linproj.w = m->aud_in_proj;
    e2_linear_forward(ctx, x, c1, T1, &linproj);

    double t_sub = e2_now_ms() - t_sub0;
    adump("aud_in", x, (size_t)T1 * H);

    /* relative position embedding table: same for every layer */
    float *pe = NULL;
    int pos_len = 0;
    {
        int chunk = c->audio_chunk;
        int ctxsz = chunk + (c->audio_ctx_left - 1) + c->audio_ctx_right;
        pos_len = ctxsz / 2 + 1;
        pe = e2_arena_alloc(&ctx->scratch, (size_t)pos_len * H * sizeof(float));
        int half = H / 2;
        float loginc = logf(10000.0f) / (float)(half - 1 > 0 ? half - 1 : 1);
        for (int p = 0; p < pos_len; p++) {
            int pos = pos_len - 1 - p;
            for (int i = 0; i < half; i++) {
                float ts = expf(-(float)i * loginc);
                float a = (float)pos * ts;
                pe[(size_t)p * H + i] = sinf(a);
                pe[(size_t)p * H + half + i] = cosf(a);
            }
        }
    }
    /* ---- conformer layers ---- */
    double t_layers0 = e2_now_ms();
    for (int li = 0; li < c->audio_layers; li++) {
        e2_audio_layer *L = &m->al[li];
        double _l0 = e2_now_ms();
        audio_ff(ctx, L, x, T1, 0, H, I);
        aud_prof[0] += e2_now_ms() - _l0;
        /* attention */
        float *hn = e2_arena_alloc(&ctx->scratch, (size_t)T1 * H * sizeof(float));
        e2_rmsnorm(ctx, hn, x, L->pre_attn_norm, T1, H, c->audio_eps);
        _l0 = e2_now_ms();
        audio_attention(ctx, L, hn, T1, m1, H, heads, D, pe, pos_len);
        aud_prof[1] += e2_now_ms() - _l0;
        e2_rmsnorm(ctx, hn, hn, L->post_attn_norm, T1, H, c->audio_eps);
        e2_add_inplace_ctx(ctx, x, hn, T1 * H);
        _l0 = e2_now_ms();
        audio_lconv(ctx, L, x, T1, H, K);
        aud_prof[2] += e2_now_ms() - _l0;
        _l0 = e2_now_ms();
        audio_ff(ctx, L, x, T1, 1, H, I);
        aud_prof[3] += e2_now_ms() - _l0;
        e2_rmsnorm(ctx, x, x, L->norm_out, T1, H, c->audio_eps);
        char nm[32]; snprintf(nm, sizeof(nm), "aud_l%d", li + 1);
        adump(nm, x, (size_t)T1 * H);
    }

    double t_layers = e2_now_ms() - t_layers0;
    /* ---- output projection + text projection ---- */
    float *aout = e2_arena_alloc(&ctx->scratch, (size_t)T1 * c->audio_output_dim * sizeof(float));
    float *soft = e2_arena_alloc(&ctx->scratch, (size_t)T1 * c->text_hidden * sizeof(float));
    if (!aout || !soft) return e2_fail("oom");
    e2_linear lout = {0}; lout.w = m->aud_out_proj;
    e2_linear_forward(ctx, aout, x, T1, &lout);
    adump("aud_out", aout, (size_t)T1 * c->audio_output_dim);
    e2_rmsnorm(ctx, aout, aout, NULL, T1, c->audio_output_dim, c->audio_eps);
    e2_linear lemb = {0}; lemb.w = m->aud_embed;
    e2_linear_forward(ctx, soft, aout, T1, &lemb);
    adump("aud_soft", soft, (size_t)T1 * c->text_hidden);

    int cnt = 0;
    for (int t = 0; t < T1; t++)
        if (m1[t]) {
            memcpy(out + (size_t)cnt * c->text_hidden, soft + (size_t)t * c->text_hidden,
                   (size_t)c->text_hidden * sizeof(float));
            cnt++;
        }
    *ntok = cnt;
    if (prof)
        fprintf(stderr, "audio: T=%d T1=%d H=%d I=%d L=%d mel %.2f sub %.2f layers %.2f "
                "[ff1 %.2f attn %.2f lconv %.2f ff2 %.2f | qkv %.2f scale %.2f pe %.2f loop %.2f post %.2f] out %.2f total %.2f\n",
                T, T1, H, I, c->audio_layers, t_mel, t_sub, t_layers,
                aud_prof[0], aud_prof[1], aud_prof[2], aud_prof[3],
                aud_prof[4], aud_prof[5], aud_prof[6], aud_prof[7], aud_prof[8],
                e2_now_ms() - t_layers0 - t_layers, e2_now_ms() - now0);
    free(feat);
    free(mask);
    return 0;
}
