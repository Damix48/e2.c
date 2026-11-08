/* e2_attn.c - int8 attention, blocked over the key band a query block sees.
 *
 * scores = Q x K^T and out = P x V^T both run through the packed-int8 GEMM
 * (per-128-block u8 activation scales, per-row i8 weights, wsum correction),
 * so they get the same ~4x MAC rate as the linear layers.
 *
 * Per call:
 *   phase 1  pack K [keys][dim] and V^T [dim][keys] to int8, once per
 *            (sequence, kv head) -- the only O(t*dim) preprocessing.
 *   phase 2  one work item per (sequence, kv head, query head, query block).
 *            An item scores only the key band its queries can attend to
 *            (sliding layers see +/-512 keys), masks and softmaxes in place,
 *            and contracts with the same band.  The score tile is sized to
 *            stay in L2, so the passes over it never reach DRAM: walking a
 *            t x t score matrix was the vision tower's whole cost.
 *
 * Query blocks use single-threaded GEMMs; the item loop is the parallelism,
 * so a whole layer costs one dispatch instead of one per (head, block).
 */
#include "e2_internal.h"
#include "e2_gpu.h"

#include <math.h>
#include <stdlib.h>

#define ATT_MAXR 256

typedef struct {
    float *qg, *sc, *ob, *us;
    uint8_t *uq;
} attn_ws;

typedef struct {
    e2_ctx *ctx;
    float *out;
    const float *q, *k, *v;
    int t, tpad, heads, kv_heads, groups, head_dim, window, nseq;
    const int *seq_off;
    int ablk, R, qblk, width_max, nhd, nkey, kblk, vblk, tpb;
    long nitems;
    int nt, err;
    size_t kstride, qstride;
    /* buffer arrays, one entry per (sequence, kv head) */
    int8_t *kp; size_t kps;
    float *ksc; size_t kscs;
    int32_t *kwm; size_t kwms;
    int8_t *vp; size_t vps;
    float *vsc; size_t vscs;
    int32_t *vwm; size_t vwms;
    e2_qmat *km, *vm;
} attn_job;

/* ------------------------------------------------------------------ */
/* packing                                                             */
/* ------------------------------------------------------------------ */

/* Column max of V (one pass), then quantise+pack the transpose in 16x16
 * blocks so every source cache line is used fully.  The per-row weight sum
 * the zero-point correction needs is accumulated on the way. */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
static int avx512_ok(void) {
    static int v = -1;
    if (v < 0) {
        __builtin_cpu_init();
        v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512vl");
    }
    return v;
}
/* 16 (dim) x 4 (key) block: convert four float rows to int8 with a per-dim
 * scale and transpose them into the packed [dim][4 keys] layout, which is a
 * 4x16 byte transpose (three unpack levels).  The per-dim sums the zero-point
 * correction needs fall out of the transposed zmm. */
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void pack_vt_dims_v(attn_job *j, const float *v, int rs, int8_t *qw,
                           float *sc, int32_t *ws, int d0, int d1) {
    int t = j->t, tpad = j->tpad, tpb = j->tpb;
    size_t ts = (size_t)(tpad / 4) * 64;
    const __m512i sgn = _mm512_set1_epi32((int)0x80000000u);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    memset(ws + (size_t)d0 * tpb, 0, (size_t)(d1 - d0) * tpb * sizeof(int32_t));

    for (int db = d0; db < d1; db += 16) {
        __m512 mx = _mm512_set1_ps(sc[db]);      /* seeded to 1.0 by the caller */
        for (int jy = 0; jy < t; jy++) {
            __m512i a = _mm512_andnot_si512(sgn, _mm512_loadu_si512((const void *)(v + (size_t)jy * rs + db)));
            mx = _mm512_max_ps(mx, _mm512_castsi512_ps(a));
        }
        __m512 inv = _mm512_div_ps(_mm512_set1_ps(127.0f), mx);
        _mm512_storeu_ps(sc + db, _mm512_mul_ps(mx, _mm512_set1_ps(1.0f / 127.0f)));
        int jb = 0;
        for (; jb + 4 <= t; jb += 4) {
            __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(v + (size_t)(jb + 0) * rs + db), inv);
            __m512 v1 = _mm512_mul_ps(_mm512_loadu_ps(v + (size_t)(jb + 1) * rs + db), inv);
            __m512 v2 = _mm512_mul_ps(_mm512_loadu_ps(v + (size_t)(jb + 2) * rs + db), inv);
            __m512 v3 = _mm512_mul_ps(_mm512_loadu_ps(v + (size_t)(jb + 3) * rs + db), inv);
            __m128i a = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v0));
            __m128i b = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v1));
            __m128i c = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v2));
            __m128i d = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v3));
            __m128i ab_lo = _mm_unpacklo_epi8(a, b), ab_hi = _mm_unpackhi_epi8(a, b);
            __m128i cd_lo = _mm_unpacklo_epi8(c, d), cd_hi = _mm_unpackhi_epi8(c, d);
            __m512i z = _mm512_castsi128_si512(_mm_unpacklo_epi16(ab_lo, cd_lo));
            z = _mm512_inserti32x4(z, _mm_unpackhi_epi16(ab_lo, cd_lo), 1);
            z = _mm512_inserti32x4(z, _mm_unpacklo_epi16(ab_hi, cd_hi), 2);
            z = _mm512_inserti32x4(z, _mm_unpackhi_epi16(ab_hi, cd_hi), 3);
            int8_t *dst = qw + (size_t)(db / 16) * ts + (size_t)(jb / 4) * 64;
            _mm512_storeu_si512((void *)dst, z);
            __m512i s = _mm512_madd_epi16(_mm512_maddubs_epi16(ones8, z), ones16);
            int32_t sum[16];
            _mm512_storeu_si512((void *)sum, s);
            int blk = jb / 128;
            for (int d = 0; d < 16; d++) ws[(size_t)(db + d) * tpb + blk] += sum[d];
        }
        for (; jb < t; jb++) {                   /* tail: fewer than 4 keys */
            const float *vr = v + (size_t)jb * rs + db;
            int8_t *dst = qw + (size_t)(db / 16) * ts + (size_t)(jb / 4) * 64;
            int e = jb & 3;
            if (!e) memset(dst, 0, 64);
            for (int d = 0; d < 16; d++) {
                long x = lrintf(vr[d] * inv[d]);
                if (x < -127) x = -127;
                if (x > 127) x = 127;
                dst[d * 4 + e] = (int8_t)x;
                ws[(size_t)(db + d) * tpb + jb / 128] += (int32_t)x;
            }
        }
    }
    int j4 = (t + 3) >> 2;
    if (j4 < tpad / 4) {
        size_t nb = (size_t)(tpad / 4 - j4) * 64;
        for (int dt = d0; dt < d1; dt += 16)
            memset(qw + (size_t)(dt / 16) * ts + (size_t)j4 * 64, 0, nb);
    }
}
#endif

static void pack_vt_dims(attn_job *j, const float *v, int rs, int8_t *qw,
                         float *sc, int32_t *ws, int d0, int d1) {
#if defined(__x86_64__) || defined(__i386__)
    if (!getenv("E2_PACK_SCALAR") && avx512_ok()) {
        pack_vt_dims_v(j, v, rs, qw, sc, ws, d0, d1);
        return;
    }
#endif
    int t = j->t, tpad = j->tpad;
    int tpb = j->tpb, k4 = tpad / 4;
    size_t ts = (size_t)k4 * 64;
    int8_t blk[16][16];
    float inv[64];               /* one entry per dim in this 64-dim slab */
    memset(ws + (size_t)d0 * tpb, 0, (size_t)(d1 - d0) * tpb * sizeof(int32_t));

    for (int jy = 0; jy < t; jy++) {
        const float *vr = v + (size_t)jy * rs + d0;
        for (int d = 0; d < d1 - d0; d++) {
            float a = fabsf(vr[d]);
            if (a > sc[d0 + d]) sc[d0 + d] = a;
        }
    }
    for (int d = d0; d < d1; d++) {
        float s = sc[d] / 127.0f;
        if (s < 1e-12f) s = 1e-12f;
        sc[d] = s;
        inv[d - d0] = 1.0f / s;
    }

    for (int jb = 0; jb < t; jb += 16) {
        int jmax = t - jb; if (jmax > 16) jmax = 16;
        for (int db = d0; db < d1; db += 16) {
            int dmax = d1 - db; if (dmax > 16) dmax = 16;
            for (int jy = 0; jy < jmax; jy++) {
                const float *vr = v + (size_t)(jb + jy) * rs + db;
                for (int d = 0; d < dmax; d++) {
                    float f = vr[d] * inv[db - d0 + d];
                    int x = (int)lrintf(f);
                    if (x < -127) x = -127;
                    if (x > 127) x = 127;
                    blk[d][jy] = (int8_t)x;
                }
            }
            for (int jy = 0; jy < 16; jy += 4) {
                int8_t *dst = qw + (size_t)(db / 16) * ts
                                 + (size_t)((jb + jy) / 4) * 64;
                for (int d = 0; d < dmax; d++) {
                    int32_t *wp = ws + (size_t)(db + d) * tpb;
                    for (int e = 0; e < 4; e++) {
                        int8_t x = (jy + e < jmax) ? blk[d][jy + e] : 0;
                        dst[d * 4 + e] = x;
                        wp[(jb + jy + e) / 128] += x;
                    }
                }
            }
        }
    }
    /* Columns [t, tpad) are never written, but the P x V step still contracts
     * them (their activations are the zero point).  Left stale they add
     * 128 * w_old * scale, i.e. whatever the previous call's scratch held, so
     * the tail must be zeroed to make the zero-point correction exact. */
    int j4 = (t + 3) >> 2;
    if (j4 < tpad / 4) {
        size_t nb = (size_t)(tpad / 4 - j4) * 64;
        for (int dt = d0; dt < d1; dt += 16)
            memset(qw + (size_t)(dt / 16) * ts + (size_t)j4 * 64, 0, nb);
    }
}

static void pack_work(void *vp, int tid, int nt) {
    attn_job *j = vp;
    int hd = j->head_dim;
    int ktiles = (j->t + 15) / 16;              /* 16-key slabs of K */
    int vtiles = (j->nhd + 63) / 64;            /* 64-dim slabs of V^T */
    int per = ktiles + vtiles;
    long total = (long)j->nseq * j->kv_heads * per;
    int k4 = hd / 4;
    for (long it = tid; it < total; it += nt) {
        long w = it / per;
        int r = (int)(it % per);
        int s = (int)(w / j->kv_heads), kh = (int)(w % j->kv_heads);
        const float *ksrc = j->k + (size_t)j->seq_off[s] * j->kstride
                                 + (size_t)kh * hd;
        const float *vsrc = j->v + (size_t)j->seq_off[s] * j->kstride
                                 + (size_t)kh * hd;
        e2_qmat *km = &j->km[w], *vm = &j->vm[w];
        if (r < ktiles) {
            int row0 = r * 16;
            int nrow = j->t - row0; if (nrow > 16) nrow = 16;
            int blk = j->ablk;
            if (e2_pack_mat_into(NULL, ksrc + (size_t)row0 * j->kstride, nrow, hd,
                                 (int)j->kstride, 1,
                                 j->kp + w * j->kps + (size_t)row0 * hd,
                                 j->ksc + w * j->kscs + row0,
                                 j->kwm + w * j->kwms + (size_t)row0 * j->kblk,
                                 &blk)) {
                j->err = 1;
                return;
            }
            if (row0 == 0 && tid == 0) { }
        } else {
            int d0 = (r - ktiles) * 64;
            int d1 = d0 + 64; if (d1 > hd) d1 = hd;
            if (d1 - d0 > 64) d1 = d0 + 64;
            pack_vt_dims(j, vsrc, (int)j->kstride, j->vp + w * j->vps,
                         j->vsc + w * j->vscs, j->vwm + w * j->vwms, d0, d1);

        }
        (void)k4;
        (void)km; (void)vm;
    }
}

/* descriptors are filled in once, after packing */
static void pack_fill(attn_job *j) {
    int total = j->nseq * j->kv_heads;
    int hd = j->head_dim;
    for (int w = 0; w < total; w++) {
        e2_qmat *km = &j->km[w], *vm = &j->vm[w];
        km->q = j->kp + (size_t)w * j->kps;
        km->scale = j->ksc + (size_t)w * j->kscs;
        km->wsum = j->kwm + (size_t)w * j->kwms;
        km->n = j->t; km->k = hd; km->blk = j->ablk; km->packed = 1;
        km->tstride = (size_t)(hd / 4) * 64; km->wsmstride = j->kblk;
        vm->q = j->vp + (size_t)w * j->vps;
        vm->scale = j->vsc + (size_t)w * j->vscs;
        vm->wsum = j->vwm + (size_t)w * j->vwms;
        vm->n = hd; vm->k = j->tpad; vm->blk = j->vblk; vm->packed = 1;
        vm->tstride = (size_t)(j->tpad / 4) * 64; vm->wsmstride = j->tpb;
    }
}

/* ------------------------------------------------------------------ */
/* one score row inside a band                                         */
/* ------------------------------------------------------------------ */

/* sr[0..width) holds scores for keys [lo, lo+width); the row sees [lr, hr). */
static void band_softmax(float *sr, int width, int lr, int hr, float *invsum) {
    float mx = e2_span_max(sr + lr, hr - lr);
    float sum = e2_exp_sum(sr + lr, hr - lr, mx);
    if (lr) memset(sr, 0, (size_t)lr * sizeof(float));
    if (hr < width) memset(sr + hr, 0, (size_t)(width - hr) * sizeof(float));
    *invsum = sum > 0.0f ? 1.0f / sum : 0.0f;
}

/* ------------------------------------------------------------------ */
/* one work item                                                       */
/* ------------------------------------------------------------------ */

/* opt-in per-phase accounting (E2_ATTN_PROF) */
static int g_aprof;
static __thread double tl_tk[4];
static double g_tk[4];

static void attn_item(attn_job *j, attn_ws *ws, long it) {
    int per_seq = j->kv_heads * j->groups * j->qblk;
    int s = (int)(it / per_seq);
    int rem = (int)(it % per_seq);
    int kh = rem / (j->groups * j->qblk);
    rem -= kh * (j->groups * j->qblk);
    int g = rem / j->qblk;
    int blk = rem % j->qblk;
    int hd = j->head_dim, t = j->t;
    int R = j->R;
    int i0 = blk * R;
    int nq = t - i0; if (nq > R) nq = R;
    int h = kh * j->groups + g;
    size_t off = (size_t)j->seq_off[s];
    const float *qsrc = j->q + off * j->qstride + (size_t)h * hd;
    float *odst = j->out + off * j->qstride + (size_t)h * hd;
    size_t w = (size_t)s * j->kv_heads + kh;

    int lo = 0, hi = t;
    if (j->window > 0) {
        lo = i0 - j->window; if (lo < 0) lo = 0;
        hi = i0 + nq - 1 + j->window + 1; if (hi > t) hi = t;
    }
    int kb0 = lo / 128;
    lo = kb0 * 128;
    int keyend = (hi + 15) & ~15;
    if (keyend > j->nkey) keyend = j->nkey;
    int kb1 = (keyend + 127) / 128;
    int width = kb1 * 128 - lo;

    for (int r = 0; r < nq; r++)
        memcpy(ws->qg + (size_t)r * hd, qsrc + (size_t)(i0 + r) * j->qstride,
               (size_t)hd * sizeof(float));

    double _t = g_aprof ? e2_now_ms() : 0;
    e2_quant_rows_blk(ws->qg, nq, hd, ws->uq, ws->us, j->ablk);
    if (g_aprof) { tl_tk[0] += e2_now_ms() - _t; _t = e2_now_ms(); }
    e2_gemm_range(ws->sc, width, ws->uq, ws->us, hd / j->ablk, nq, hd,
                  &j->km[w], kb0 * 8, keyend / 16);
    if (g_aprof) { tl_tk[1] += e2_now_ms() - _t; _t = e2_now_ms(); }

    for (int r = 0; r < nq; r++) {
        int pos = i0 + r;
        int lr = 0, hr = t - lo;      /* keys [0,t) minus the band's left pad */
        if (hr > width) hr = width;
        if (j->window > 0) {
            lr = pos - j->window - lo; if (lr < 0) lr = 0;
            hr = pos + j->window + 1 - lo;
            /* the band may be padding past t: those keys have no score */
            if (hr > t - lo) hr = t - lo;
            if (hr > width) hr = width;
        }
        float inv;
        band_softmax(ws->sc + (size_t)r * width, width, lr, hr, &inv);
        ws->us[(size_t)r] = inv;
    }
    if (g_aprof) { tl_tk[2] += e2_now_ms() - _t; _t = e2_now_ms(); }
    e2_quant_rows_blk(ws->sc, nq, width, ws->uq, ws->us + ATT_MAXR, 128);
    e2_qmat vq = j->vm[w];
    vq.q += (size_t)kb0 * 32 * 64;
    vq.wsum += kb0;
    vq.k = width;
    if (g_aprof) { tl_tk[2] += e2_now_ms() - _t; _t = e2_now_ms(); }
    e2_gemm_range(ws->ob, hd, ws->uq, ws->us + ATT_MAXR, width / 128, nq, width,
                  &vq, 0, hd / 16);
    if (g_aprof) { tl_tk[3] += e2_now_ms() - _t; _t = e2_now_ms(); }
    for (int r = 0; r < nq; r++) {
        float iv = ws->us[(size_t)r];
        const float *op = ws->ob + (size_t)r * hd;
        float *dp = odst + (size_t)(i0 + r) * j->qstride;
        for (int d = 0; d < hd; d++) dp[d] = op[d] * iv;
    }
    if (g_aprof) tl_tk[2] += e2_now_ms() - _t;
}

typedef struct { attn_job *j; attn_ws *ws; } item_arg;

static void item_work(void *vp, int tid, int nt) {
    item_arg *a = vp;
    attn_job *j = a->j;
    for (long it = tid; it < j->nitems; it += nt)
        attn_item(j, &a->ws[tid], it);
    for (int i = 0; i < 4; i++) g_tk[i] += tl_tk[i];
}

/* ------------------------------------------------------------------ */
/* entry point                                                         */
/* ------------------------------------------------------------------ */

int e2_attn_int8(e2_ctx *ctx, float *out, const float *q, const float *k,
                 const float *v, int t, int heads, int kv_heads, int head_dim,
                 int window, int nseq, const int *seq_off) {
    static int aprof_c = -1;
    if (aprof_c < 0) { g_aprof = getenv("E2_ATTN_PROF") != NULL; aprof_c = g_aprof; }
    if (t <= 0 || heads <= 0 || kv_heads <= 0) return -1;
    if ((head_dim % 64) || head_dim > 512 || kv_heads > heads || (heads % kv_heads)
        || t > 8192)
        return -1;
    int ablk = (head_dim % 128 == 0) ? 128 : 64;
    int tpad = (t + 127) & ~127;

    int R = ATT_MAXR;
    int width_max = tpad;
    if (window > 0) {
        width_max = ((2 * window + 1 + R + 127) / 128) * 128;
        if (width_max > tpad) width_max = tpad;
    }
    while (R > 16 && (long)R * width_max * 5 > 200 * 1024) R >>= 1;
    if (R < 16) R = 16;
    const char *fr = getenv("E2_ATTN_R");
    if (fr) { R = atoi(fr); if (R < 4) R = 4; }

    int qblk = (t + R - 1) / R;
    long nitems = (long)nseq * kv_heads * (heads / kv_heads) * qblk;
    int nt = ctx->nthreads > 0 ? ctx->nthreads : 1;
    size_t nkv = (size_t)nseq * kv_heads;

    int nhd = (head_dim + 15) / 16 * 16;         /* V^T output rows (dims) */
    int nkey = (t + 15) & ~15;                   /* packed K rows (keys) */
    int kblk = head_dim / ablk, vblk = 128, tpb = tpad / 128;
    int nkp = nkey * head_dim;                   /* packed K byte count */
    int nvp = nhd * tpad;                        /* packed V^T byte count */

    size_t scstride = (size_t)R * (width_max > head_dim ? width_max : head_dim);
    size_t need = 0;
    int8_t *p8;
    float *pf;
    (void)p8; (void)pf;
    size_t off_kp = 0;                            size_t kps = (size_t)nkp;
    size_t off_ksc = off_kp + nkv * kps * sizeof(int8_t);
    size_t kscs = (size_t)nkey;
    size_t off_kwm = off_ksc + nkv * kscs * sizeof(float);
    size_t kwms = (size_t)nkey * kblk;
    size_t off_vp = off_kwm + nkv * kwms * sizeof(int32_t);
    size_t vps = (size_t)nvp;
    size_t off_vsc = off_vp + nkv * vps * sizeof(int8_t);
    size_t vscs = (size_t)nhd;
    size_t off_vwm = off_vsc + nkv * vscs * sizeof(float);
    size_t vwms = (size_t)nhd * tpb;
    size_t off_km = off_vwm + nkv * vwms * sizeof(int32_t);
    size_t off_ws = off_km + nkv * 2 * sizeof(e2_qmat);
    /* per-thread scratch: qg, sc (+ headroom), ob, us, uq */
    size_t off_qg = (off_ws + nkv * 0 + (size_t)nt * sizeof(attn_ws) + 63) & ~(size_t)63;
    size_t off_sc = off_qg + (size_t)nt * scstride * sizeof(float);
    size_t off_ob = off_sc + (size_t)nt * (scstride + tpad) * sizeof(float);
    size_t off_us = off_ob + (size_t)nt * scstride * sizeof(float);
    size_t off_uq = off_us + (size_t)nt * scstride * sizeof(float);
    need = off_uq + (size_t)nt * scstride;

    if (ctx->acap < need) {
        e2_dev_free(ctx, ctx->abuf);
        ctx->abuf = e2_dev_alloc(ctx, need);
        ctx->acap = ctx->abuf ? need : 0;
    }
    if (!ctx->abuf) return -1;
    unsigned char *base = ctx->abuf;

    attn_job jb;
    attn_job *j = &jb;
    memset(j, 0, sizeof(*j));
    j->ctx = ctx; j->out = out; j->q = q; j->k = k; j->v = v;
    j->t = t; j->tpad = tpad; j->heads = heads; j->kv_heads = kv_heads;
    j->groups = heads / kv_heads; j->head_dim = head_dim; j->window = window;
    j->nseq = nseq; j->seq_off = seq_off; j->ablk = ablk; j->R = R;
    j->qblk = qblk; j->width_max = width_max; j->nitems = nitems; j->nt = nt;
    j->nhd = nhd; j->nkey = nkey; j->kblk = kblk; j->vblk = vblk; j->tpb = tpb;
    j->kstride = (size_t)kv_heads * head_dim;
    j->qstride = (size_t)heads * head_dim;
    j->kp = (int8_t *)(base + off_kp);   j->kps = kps;
    j->ksc = (float *)(base + off_ksc);  j->kscs = kscs;
    j->kwm = (int32_t *)(base + off_kwm); j->kwms = kwms;
    j->vp = (int8_t *)(base + off_vp);   j->vps = vps;
    j->vsc = (float *)(base + off_vsc);  j->vscs = vscs;
    j->vwm = (int32_t *)(base + off_vwm); j->vwms = vwms;
    j->km = (e2_qmat *)(base + off_km);
    j->vm = j->km + nkv;
    for (size_t i = 0; i < nkv; i++)
        for (int d = 0; d < nhd; d++) j->vsc[i * vscs + d] = 1.0f;

    item_arg ia;
    ia.j = j;
    ia.ws = (attn_ws *)(base + off_ws);
    for (int i = 0; i < nt; i++) {
        ia.ws[i].qg = (float *)(base + off_qg) + (size_t)i * scstride;
        ia.ws[i].sc = (float *)(base + off_sc) + (size_t)i * (scstride + tpad) + tpad;
        ia.ws[i].ob = (float *)(base + off_ob) + (size_t)i * scstride;
        ia.ws[i].us = (float *)(base + off_us) + (size_t)i * scstride;
        ia.ws[i].uq = (uint8_t *)(base + off_uq) + (size_t)i * scstride;
    }

    int aprof = g_aprof;
    double t0 = e2_now_ms();
    if (nt > 1) e2_parallel(ctx, pack_work, j);
    else pack_work(j, 0, 1);
    double t1 = e2_now_ms();
    if (j->err) return -1;
    pack_fill(j);

    if (nt > 1 && nitems > 1) e2_parallel(ctx, item_work, &ia);
    else item_work(&ia, 0, 1);
    if (aprof) {
        fprintf(stderr, "attn t=%d hd=%d win=%d R=%d items=%ld nseq=%d "
                "pack %.3f items %.3f [qg %.3f qk %.3f sm %.3f pv %.3f]\n",
                t, head_dim, window, R, nitems, nseq, t1 - t0, e2_now_ms() - t1,
                g_tk[0], g_tk[1], g_tk[2], g_tk[3]);
        memset(g_tk, 0, sizeof(g_tk));
    }
    return 0;
}
