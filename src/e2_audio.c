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

static double aud_prof[12];

/* 3x3 stride-2 conv, parallel over output channels. */
typedef struct {
    const float *in, *w;
    float *out;
    int c_in, T_in, W_in, c_out, T_out, W_out;
} conv_arg;

static void conv2_work(void *vp, int tid, int nt) {
    conv_arg *a = vp;
    int o0 = (int)((long)a->c_out * tid / nt), o1 = (int)((long)a->c_out * (tid + 1) / nt);
    for (int oc = o0; oc < o1; oc++)
        for (int t = 0; t < a->T_out; t++)
            for (int w = 0; w < a->W_out; w++) {
                float s = 0.0f;
                for (int ic = 0; ic < a->c_in; ic++) {
                    const float *wts = a->w + ((size_t)oc * a->c_in + ic) * 9;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = t * 2 - 1 + ky;
                        if (iy < 0 || iy >= a->T_in) continue;
                        const float *row = a->in + ((size_t)ic * a->T_in + iy) * a->W_in;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = w * 2 - 1 + kx;
                            if (ix < 0 || ix >= a->W_in) continue;
                            s += row[ix] * wts[ky * 3 + kx];
                        }
                    }
                }
                a->out[((size_t)oc * a->T_out + t) * a->W_out + w] = s;
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
    const float *hann = m->hann;
    const float *mel = m->mel_filters;
    float frame[512];
    float mag[257];
    for (int t = 0; t < T; t++) {
        /* padded waveform has 160 leading zeros; frame starts at t*160 */
        for (int i = 0; i < 320; i++) {
            int idx = t * 160 + i - 160;
            float v = (idx >= 0 && idx < n) ? samples[idx] : 0.0f;
            frame[i] = v * hann[i];
        }
        for (int i = 320; i < 512; i++) frame[i] = 0.0f;
        fft512(frame, mag);
        for (int c = 0; c < 128; c++) {
            float s = 0.0f;
            for (int f = 0; f < 257; f++) s += mag[f] * mel[f * 128 + c];
            feat[(size_t)t * 128 + c] = logf(s + ctx->m.meta.mel_floor);
        }
        mask[t] = (t * 160 + 160) < n;
        if (!mask[t]) {
            float *p = feat + (size_t)t * 128;
            for (int c = 0; c < 128; c++) p[c] = 0.0f;
        }
    }
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
    /* residual + clamp, norm, two linears with SiLU */
    float *res = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    memcpy(res, x, (size_t)T * H * sizeof(float));
    e2_rmsnorm(ctx, h, x, pre, T, H, c->audio_eps);
    e2_linear_forward(ctx, up, h, T, a);
    e2_silu_ctx(ctx, up, T * I);
    e2_linear_forward(ctx, h, up, T, b);
    e2_rmsnorm(ctx, h, h, post, T, H, c->audio_eps);
    for (size_t i = 0; i < (size_t)T * H; i++) x[i] = res[i] + h[i] * c->audio_residual;
}

static float softplus(float v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return expf(v);
    return log1pf(e2_fexp(v));
}

static void audio_attention(e2_ctx *ctx, e2_audio_layer *L, float *x, int T,
                            const uint8_t *mask, int H, int heads, int D,
                            const float *pe, int pos_len) {
    e2_meta *c = &ctx->m.meta;
    int nb = (T + c->audio_chunk - 1) / c->audio_chunk;
    int chunk = c->audio_chunk;
    int ctxsz = chunk + (c->audio_ctx_left - 1) + c->audio_ctx_right;
    /* rel pos embedding and its projection are recomputed per layer small */
    float *q = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *k = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *v = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    float *out = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    if (!q || !k || !v || !out) return;
    double _a0 = e2_now_ms();
    e2_linear_forward(ctx, q, x, T, &L->q);
    e2_linear_forward(ctx, k, x, T, &L->k);
    e2_linear_forward(ctx, v, x, T, &L->v);
    aud_prof[4] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    float qs[256];
    for (int d = 0; d < D; d++)
        qs[d] = c->audio_qscale * softplus(L->per_dim_scale[d]);
    const float kscale = c->audio_kscale;
    for (int t = 0; t < T; t++) {
        for (int hh = 0; hh < heads; hh++) {
            float *qp = q + ((size_t)t * heads + hh) * D;
            float *kp = k + ((size_t)t * heads + hh) * D;
            for (int d = 0; d < D; d++) { qp[d] *= qs[d]; kp[d] *= kscale; }
        }
    }
    aud_prof[5] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    float *rk = e2_arena_alloc(&ctx->scratch, (size_t)pos_len * H * sizeof(float));
    e2_linear_forward(ctx, rk, pe, pos_len, &L->rel_k);
    /* rk layout [pos_len, heads*D]; view as [pos_len][heads][D] */
    aud_prof[6] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    memset(out, 0, (size_t)T * H * sizeof(float));
    float scores[512];
    for (int b = 0; b < nb; b++) {
        for (int hh = 0; hh < heads; hh++) {
            for (int i = 0; i < chunk; i++) {
                int qg = b * chunk + i;
                if (qg >= T) break;
                if (!mask[qg]) continue;
                float *qp = q + ((size_t)qg * heads + hh) * D;
                float mx = -INFINITY;
                for (int j = 0; j < ctxsz; j++) {
                    int kg = b * chunk - (c->audio_ctx_left - 1) + j;
                    float s = 0.0f;
                    if (kg >= 0 && kg < T && mask[kg]) {
                        const float *kp = k + ((size_t)kg * heads + hh) * D;
                        for (int d = 0; d < D; d++) s += qp[d] * kp[d];
                        int r = j - i;
                        if (r >= 0 && r < pos_len) {
                            const float *rp = rk + ((size_t)r * heads + hh) * D;
                            for (int d = 0; d < D; d++) s += qp[d] * rp[d];
                        }
                        /* window: 0 <= q-kv < audio_ctx_left */
                        int dist = i + (c->audio_ctx_left - 1) - j;
                        if (dist < 0 || dist >= c->audio_ctx_left - 1) s = -INFINITY;
                    } else {
                        s = -INFINITY;
                    }
                    scores[j] = s;
                    if (s > mx) mx = s;
                }
                if (mx == -INFINITY) continue;
                /* the logit soft cap applies before softmax */
                mx = -INFINITY;
                for (int j = 0; j < ctxsz; j++) {
                    if (scores[j] == -INFINITY) continue;
                    scores[j] = tanhf(scores[j] / c->audio_softcap) * c->audio_softcap;
                    if (scores[j] > mx) mx = scores[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < ctxsz; j++) {
                    if (scores[j] == -INFINITY) { scores[j] = 0.0f; continue; }
                    scores[j] = e2_fexp(scores[j] - mx);
                    sum += scores[j];
                }
                if (sum <= 0.0f) continue;
                float inv = 1.0f / sum;
                float *op = out + ((size_t)qg * heads + hh) * D;
                for (int j = 0; j < ctxsz; j++) {
                    float w = scores[j] * inv;
                    if (w == 0.0f) continue;
                    int kg = b * chunk - (c->audio_ctx_left - 1) + j;
                    if (kg < 0 || kg >= T) continue;
                    const float *vp = v + ((size_t)kg * heads + hh) * D;
                    for (int d = 0; d < D; d++) op[d] += w * vp[d];
                }
            }
        }
    }
    aud_prof[7] += e2_now_ms() - _a0;
    _a0 = e2_now_ms();
    e2_linear_forward(ctx, x, out, T, &L->post);
    aud_prof[8] += e2_now_ms() - _a0;
}

static void audio_lconv(e2_ctx *ctx, e2_audio_layer *L, float *x, int T,
                        int H, int K) {
    e2_meta *c = &ctx->m.meta;
    float *h = e2_arena_alloc(&ctx->scratch, (size_t)T * 2 * H * sizeof(float));
    if (!h) return;
    e2_rmsnorm(ctx, h, x, L->lc_pre_norm, T, H, c->audio_eps);
    e2_linear_forward(ctx, h, h, T, &L->lc_start);   /* [T, 2H] */
    for (int t = 0; t < T; t++) {
        float *p = h + (size_t)t * 2 * H;
        for (int i = 0; i < H; i++) p[i] = p[i] / (1.0f + e2_fexp(-p[H + i]));
    }
    /* causal depthwise conv over time */
    float *conv = e2_arena_alloc(&ctx->scratch, (size_t)T * H * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int ch = 0; ch < H; ch++) {
            float s = 0.0f;
            for (int j = 0; j < K; j++) {
                int tt = t - (K - 1) + j;
                if (tt < 0 || tt >= T) continue;
                s += h[(size_t)tt * 2 * H + ch] * L->dw_conv[ch * K + j];
            }
            conv[(size_t)t * H + ch] = s;
        }
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
    /* ---- subsample conv (two 3x3 stride-2 convs) ---- */
    int T0 = (T - 1) / 2 + 1;
    int W0 = 64;
    int T1 = (T0 - 1) / 2 + 1;
    int W1 = 32;
    float *c0 = e2_arena_alloc(&ctx->scratch, (size_t)sub0 * T0 * W0 * sizeof(float));
    float *c1 = e2_arena_alloc(&ctx->scratch, (size_t)sub1 * T1 * W1 * sizeof(float));
    if (!c0 || !c1) { free(feat); free(mask); return e2_fail("oom"); }
    /* input masked */
    for (int t = 0; t < T; t++)
        if (!mask[t]) for (int f = 0; f < 128; f++) feat[(size_t)t * 128 + f] = 0.0f;
    /* layer0: in 1 channel, out 128, k=3, s=2, p=1 */
    {
        conv_arg ca = {feat, m->aud_conv0, c0, 1, T, 128, sub0, T0, W0};
        e2_parallel(ctx, conv2_work, &ca);
    }
    /* layernorm over channels at each (t,w), relu */
    for (int t = 0; t < T0; t++)
        for (int w = 0; w < W0; w++) {
            float mean = 0, var = 0;
            for (int ch = 0; ch < sub0; ch++) mean += c0[((size_t)ch * T0 + t) * W0 + w];
            mean /= sub0;
            for (int ch = 0; ch < sub0; ch++) {
                float d = c0[((size_t)ch * T0 + t) * W0 + w] - mean;
                var += d * d;
            }
            var /= sub0;
            float inv = 1.0f / sqrtf(var + c->audio_eps);
            for (int ch = 0; ch < sub0; ch++) {
                float v = (c0[((size_t)ch * T0 + t) * W0 + w] - mean) * inv * m->aud_sub0_norm[ch];
                c0[((size_t)ch * T0 + t) * W0 + w] = v > 0 ? v : 0;
            }
        }
    /* mask downsample, layer1 in 128 out 32 */
    uint8_t *m0 = e2_arena_alloc(&ctx->scratch, (size_t)T0);
    if (!m0) return e2_fail("oom");
    for (int t = 0; t < T0; t++) m0[t] = mask[t * 2];
    for (int t = 0; t < T0; t++)
        if (!m0[t]) for (int ch = 0; ch < sub0; ch++)
            for (int w = 0; w < W0; w++) c0[((size_t)ch * T0 + t) * W0 + w] = 0.0f;
    {
        conv_arg ca = {c0, m->aud_conv1, c1, sub0, T0, W0, sub1, T1, W1};
        e2_parallel(ctx, conv2_work, &ca);
    }
    uint8_t *m1 = e2_arena_alloc(&ctx->scratch, (size_t)T1);
    if (!m1) return e2_fail("oom");
    for (int t = 0; t < T1; t++) m1[t] = m0[t * 2];
    for (int t = 0; t < T1; t++)
        if (!m1[t]) for (int ch = 0; ch < sub1; ch++)
            for (int w = 0; w < W1; w++) c1[((size_t)ch * T1 + t) * W1 + w] = 0.0f;
    /* layernorm over 32 channels + relu */
    for (int t = 0; t < T1; t++)
        for (int w = 0; w < W1; w++) {
            float mean = 0, var = 0;
            for (int ch = 0; ch < sub1; ch++) mean += c1[((size_t)ch * T1 + t) * W1 + w];
            mean /= sub1;
            for (int ch = 0; ch < sub1; ch++) {
                float d = c1[((size_t)ch * T1 + t) * W1 + w] - mean;
                var += d * d;
            }
            var /= sub1;
            float inv = 1.0f / sqrtf(var + c->audio_eps);
            for (int ch = 0; ch < sub1; ch++) {
                float v = (c1[((size_t)ch * T1 + t) * W1 + w] - mean) * inv * m->aud_sub1_norm[ch];
                c1[((size_t)ch * T1 + t) * W1 + w] = v > 0 ? v : 0;
            }
        }
    /* flatten (t, w, ch) -> [T1, 32*32] and project */
    float *x = e2_arena_alloc(&ctx->scratch, (size_t)T1 * H * sizeof(float));
    float *flat = e2_arena_alloc(&ctx->scratch, (size_t)T1 * sub1 * W1 * sizeof(float));
    if (!x || !flat) return e2_fail("oom");
    for (int t = 0; t < T1; t++)
        for (int w = 0; w < W1; w++)
            for (int ch = 0; ch < sub1; ch++)
                flat[((size_t)t * W1 + w) * sub1 + ch] = c1[((size_t)ch * T1 + t) * W1 + w];
    e2_linear linproj = {0}; linproj.w = m->aud_in_proj;
    e2_linear_forward(ctx, x, flat, T1, &linproj);
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
        float *res = e2_arena_alloc(&ctx->scratch, (size_t)T1 * H * sizeof(float));
        float *hn = e2_arena_alloc(&ctx->scratch, (size_t)T1 * H * sizeof(float));
        memcpy(res, x, (size_t)T1 * H * sizeof(float));
        e2_rmsnorm(ctx, hn, x, L->pre_attn_norm, T1, H, c->audio_eps);
        _l0 = e2_now_ms();
        audio_attention(ctx, L, hn, T1, m1, H, heads, D, pe, pos_len);
        aud_prof[1] += e2_now_ms() - _l0;
        e2_rmsnorm(ctx, hn, hn, L->post_attn_norm, T1, H, c->audio_eps);
        e2_add_inplace_ctx(ctx, res, hn, T1 * H);
        memcpy(x, res, (size_t)T1 * H * sizeof(float));
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
