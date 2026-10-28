/* test_gpu_attn_recipe.c — experimental INT8 attention kernels against an independent recipe.
 * Long text uses this unsigned-probability recipe; shorter text uses FP16 attention.
 * The stage oracle complements, but does not replace, end-to-end embedding checks.
 *
 * The XMX attention recipe uses four kernel kinds (k_attn_packw for K, k_attn_packw for V^T,
 * k_attn_batch_gemm for the scores, k_attn_softmax, k_attn_i8_ctx for context/quantization) whose
 * arithmetic is documented in src/gpu/kernels/e2.cl. The end-to-end tests only look at a 768-element
 * embedding after 24 layers, so an indexing error in one attention block - or a scale read from the
 * wrong block - can hide behind the softmax of a nearly one-hot row. This test drives the four
 * kernels directly, with the host's exact grids and arguments, on a deterministic synthetic input and
 * compares every stage against the recipe computed here:
 *
 *   Q   per (token, head) row, one scale per 128-element block           (k_rowops)
 *   K   per (key, kv-head) row, ONE scale over the whole head dim, per-128-block bsq,
 *       keys past the real length written as zero bytes with scale 1     (k_attn_packw)
 *   P   per row, block max mb_b, byte = rte(255 e_k / mb_b), asc = mb_b / (255 l)  (k_attn_softmax)
 *   V   per (feature, kv-head) row of V^T, ONE scale over the real keys, per-128-key block bsq,
 *       keys past the real length are zero                              (k_attn_packw)
 *   ctx sum_b asc[m,b] * wsc[d] * dp4a (P has zero point 0)             (k_attn_i8_ctx)
 *
 * The comparisons are staged so a failure names the stage: the packers' scales (to 1e-6 relative - the
 * device's divide is a fast-math reciprocal), packed bytes (allowing adjacent rounding at a half-integer),
 * their exact block sums, the P bytes per real key (within
 * 0.1% of them, again tie flips, so this is a bound and not exact equality), the P block scales, and
 * last the context vector by cosine and by relative L2. Each product's oracle uses the validated packed
 * bytes, so permitted upstream rounding differences cannot masquerade as softmax errors.
 *
 * Recipe coverage: one sequence, all-valid rows and host-packed Q. A separate softmax check covers
 * query/key masks, padding and batch/head/query-tile offsets. End-to-end tests check embedding fidelity.
 *
 * Build: see the e2_test_gpu_attn_recipe target in CMakeLists.txt. No fixtures, no model.
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu/e2_tile.h"
#include "gpu/l0.h"

#define CHECK(c) \
    do { \
        if (!(c)) { \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #c, e2_l0_error(g)); \
            exit(2); \
        } \
    } while (0)

/* Skip only explicit absence messages, not module or initialization failures. */
static int no_gpu(const char *err) {
    return !strcmp(err, "no Level Zero driver") || !strcmp(err, "no Intel GPU device found");
}

static int failures;

/* A device-produced scale is used as a divisor and as a rte input, so it must be a positive finite
 * number: a NaN would make every comparison false and rte() undefined, which would read as "differs
 * from the recipe" rather than as the broken scale it is. */
static int scale_usable(int S, int D, int KV, const char *what, float sc) {
    if (isfinite(sc) && sc > 0.f) return 1;
    fprintf(stderr, "FAIL S=%d D=%d KV=%d: %s is %g\n", S, D, KV, what, (double)sc);
    failures++;
    return 0;
}

/* finite and non-degenerate before any cosine: a zero or NaN context must fail as itself, not as a
 * cosine of 0 */
static int usable(int S, int D, int KV, const char *what, const double *a, size_t n) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(a[i])) {
            fprintf(stderr, "FAIL S=%d D=%d KV=%d: %s is not finite at %zu\n", S, D, KV, what, i);
            failures++;
            return 0;
        }
        ss += a[i] * a[i];
    }
    if (!(ss > 0)) {
        fprintf(stderr, "FAIL S=%d D=%d KV=%d: %s is all zero\n", S, D, KV, what);
        failures++;
        return 0;
    }
    return 1;
}

static void fail(int S, int D, int KV, const char *what, double got, double limit) {
    fprintf(stderr, "FAIL S=%d D=%d KV=%d: %s %.9g (limit %.9g)\n", S, D, KV, what, got, limit);
    failures++;
}

static double now_cos(const double *a, const double *b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    if (!(aa > 0) || !(bb > 0)) return -1;
    return ab / sqrt(aa * bb);
}

static double now_rel(const double *a, const double *b, size_t n) {
    double d = 0, bb = 0;
    for (size_t i = 0; i < n; i++) {
        d += (a[i] - b[i]) * (a[i] - b[i]);
        bb += b[i] * b[i];
    }
    return bb > 0 ? sqrt(d / bb) : 1;
}

/* round half to even, like the kernels' convert_int4_rte */
static int rte(float v) {
    float r = rintf(v);
    if (r > 127.f) r = 127.f;
    if (r < -127.f) r = -127.f;
    return (int)r;
}

/* Fast reciprocal multiplication can choose the adjacent integer only at a rounding tie. */
static int packed_byte_ok(float value, int got) {
    const int expected = rte(value);
    return got == expected ||
           (abs(got - expected) == 1 && fabsf(value - (floorf(value) + .5f)) <= 1e-4f);
}

struct shape {
    int S, D, H, KV, window; /* window 0 = no mask (the 4 full layers), else the sliding window */
};

static void run_shape(e2_l0 *g, struct shape sh, unsigned seed) {
    const int before = failures;
    const int S = sh.S, D = sh.D, H = sh.H, KV = sh.KV, window = sh.window;
    const int sup = (S + 127) / 128 * 128;
    const int mpad = (S + E2_DM - 1) / E2_DM * E2_DM;
    const int tile = mpad;
    const int qstride = H * D, kvstride = KV * D, nbq = D / 128, nbv = sup / 128;
    const size_t nq = (size_t)S * qstride, nkv = (size_t)S * kvstride;
    const size_t nscr = (size_t)H * tile * sup;

    float *q = malloc(sizeof(float) * nq), *k = malloc(sizeof(float) * nkv);
    float *v = malloc(sizeof(float) * nkv);
    CHECK(q && k && v);
    /* unit-RMS rows, like the RMSNorm k_norm_rope3 leaves in these buffers, so the softmax has the
     * dynamic range the model's own scores have */
    for (int r = 0; r < S * H; r++) {
        double ss = 0;
        for (int i = 0; i < D; i++) q[(size_t)r * D + i] = (float)((int)(seed = seed * 1664525u + 1013904223u) >> 8 & 0xFFFF) / 32768.f - 1.f;
        for (int i = 0; i < D; i++) ss += (double)q[(size_t)r * D + i] * q[(size_t)r * D + i];
        for (int i = 0; i < D; i++) q[(size_t)r * D + i] *= (float)(1.0 / sqrt(ss / D));
    }
    for (int r = 0; r < S * KV; r++) {
        double ss = 0, sv = 0;
        for (int i = 0; i < D; i++) k[(size_t)r * D + i] = (float)((int)(seed = seed * 1664525u + 1013904223u) >> 8 & 0xFFFF) / 32768.f - 1.f;
        for (int i = 0; i < D; i++) v[(size_t)r * D + i] = (float)((int)(seed = seed * 1664525u + 1013904223u) >> 8 & 0xFFFF) / 32768.f - 1.f;
        for (int i = 0; i < D; i++) ss += (double)k[(size_t)r * D + i] * k[(size_t)r * D + i];
        for (int i = 0; i < D; i++) sv += (double)v[(size_t)r * D + i] * v[(size_t)r * D + i];
        for (int i = 0; i < D; i++) k[(size_t)r * D + i] *= (float)(1.0 / sqrt(ss / D));
        for (int i = 0; i < D; i++) v[(size_t)r * D + i] *= (float)(1.0 / sqrt(sv / D));
        /* Cover zero-context quantization: neutral bytes and scale 1. */
        if (S == 63 && KV == 2 && r % KV == 0) memset(v + (size_t)r * D, 0, (size_t)D * sizeof *v);
    }

    /* the host half of the recipe: k_rowops' Q quantization, per 128-element block */
    uint8_t *qb = malloc(nq);
    float *qsc = malloc(sizeof(float) * (size_t)S * H * nbq);
    CHECK(qb && qsc);
    for (int row = 0; row < S * H; row++)
        for (int b = 0; b < nbq; b++) {
            const float *src = q + (size_t)row * D + b * 128;
            float amax = 0.f;
            for (int i = 0; i < 128; i++) amax = fmaxf(amax, fabsf(src[i]));
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            qsc[(size_t)row * nbq + b] = sc;
            for (int i = 0; i < 128; i++) qb[(size_t)row * D + b * 128 + i] = (uint8_t)(rte(src[i] / sc) + 128);
        }

    void *kdev = e2_l0_alloc(g, sizeof(float) * nkv), *vdev = e2_l0_alloc(g, sizeof(float) * nkv);
    void *qpk = e2_l0_alloc(g, nq + 4096), *qps = e2_l0_alloc(g, sizeof(float) * (size_t)S * H * nbq + 64);
    void *kpk = e2_l0_alloc(g, (size_t)KV * sup * D + 4096);
    void *ksc_d = e2_l0_alloc(g, sizeof(float) * (size_t)KV * sup + 64);
    void *kbq_d = e2_l0_alloc(g, sizeof(int) * (size_t)KV * nbq * sup + 64);
    void *vpk = e2_l0_alloc(g, (size_t)KV * D * sup + 4096);
    void *vsc_d = e2_l0_alloc(g, sizeof(float) * (size_t)KV * D + 64);
    void *vbq_d = e2_l0_alloc(g, sizeof(int) * (size_t)KV * nbv * D + 64);
    void *scr_d = e2_l0_alloc(g, sizeof(float) * nscr), *pb_d = e2_l0_alloc(g, nscr);
    void *pbs_d = e2_l0_alloc(g, sizeof(float) * (size_t)H * tile * nbv);
    void *ctx_d = e2_l0_alloc(g, sizeof(float) * nq), *valid_d = e2_l0_alloc(g, (size_t)S + 64);
    uint8_t *ones = malloc((size_t)S);
    CHECK(ones);
    memset(ones, 1, (size_t)S);
    CHECK(!e2_l0_write(g, valid_d, (size_t)S, ones));
    CHECK(!e2_l0_write(g, kdev, sizeof(float) * nkv, k));
    CHECK(!e2_l0_write(g, vdev, sizeof(float) * nkv, v));
    CHECK(!e2_l0_write(g, qpk, nq, qb));
    CHECK(!e2_l0_write(g, qps, sizeof(float) * (size_t)S * H * nbq, qsc));

    const int gy = 4, one = 1, q0 = 0, wr = 1;
    e2_graph *gr = e2_l0_graph_begin(g, 1);
    CHECK(gr);
    CHECK(!e2_l0_append(g, gr, "k_attn_packw", (uint32_t)((sup + gy - 1) / gy), 1, (uint32_t)KV, 32, gy,
                        1, 13, &kdev, sizeof kdev, &kpk, sizeof kpk, &ksc_d, sizeof ksc_d, &kbq_d,
                        sizeof kbq_d, &sup, sizeof sup, &D, sizeof D, &kvstride, sizeof kvstride,
                        &one, sizeof one, &S, sizeof S, &D, sizeof D, &S, sizeof S, &KV, sizeof KV,
                        &valid_d, sizeof valid_d));
    CHECK(!e2_l0_append(g, gr, "k_attn_packw", (uint32_t)((D + gy - 1) / gy), 1, (uint32_t)KV, 32, gy,
                        1, 13, &vdev, sizeof vdev, &vpk, sizeof vpk, &vsc_d, sizeof vsc_d, &vbq_d,
                        sizeof vbq_d, &D, sizeof D, &sup, sizeof sup, &one, sizeof one, &kvstride,
                        sizeof kvstride, &D, sizeof D, &S, sizeof S, &S, sizeof S, &KV, sizeof KV,
                        &valid_d, sizeof valid_d));
    CHECK(!e2_l0_append(g, gr, "k_attn_batch_gemm", (uint32_t)(tile / E2_DM),
                        (uint32_t)((sup + E2_DN * gy - 1) / (E2_DN * gy)), (uint32_t)H, 8,
                        (uint32_t)gy, 1, 13, &kpk, sizeof kpk, &ksc_d, sizeof ksc_d, &kbq_d,
                        sizeof kbq_d, &qpk, sizeof qpk, &qps, sizeof qps, &scr_d, sizeof scr_d, &S,
                        sizeof S, &sup, sizeof sup, &D, sizeof D, &H, sizeof H, &KV, sizeof KV, &q0,
                        sizeof q0, &tile, sizeof tile));
    CHECK(!e2_l0_append(g, gr, "k_attn_softmax", (uint32_t)((S + gy - 1) / gy), 1, (uint32_t)H, 32,
                        gy, 1, 12, &scr_d, sizeof scr_d, &pb_d, sizeof pb_d, &pbs_d, sizeof pbs_d,
                        &valid_d, sizeof valid_d, &S, sizeof S, &sup, sizeof sup, &S, sizeof S,
                        &window, sizeof window, &H, sizeof H, &nbv, sizeof nbv, &q0, sizeof q0,
                        &tile, sizeof tile));
    CHECK(!e2_l0_append(g, gr, "k_attn_i8_ctx", (uint32_t)(tile / E2_DM),
                        (uint32_t)(D / 128), (uint32_t)H, 8, 8, 1, 16,
                        &vpk, sizeof vpk, &vsc_d, sizeof vsc_d, &vbq_d, sizeof vbq_d,
                        &pb_d, sizeof pb_d, &pbs_d, sizeof pbs_d, &ctx_d, sizeof ctx_d,
                        &S, sizeof S, &sup, sizeof sup, &D, sizeof D, &H, sizeof H, &KV, sizeof KV,
                        &q0, sizeof q0, &tile, sizeof tile, &qpk, sizeof qpk, &qps, sizeof qps,
                        &wr, sizeof wr));
    CHECK(!e2_l0_graph_close(g, gr));
    CHECK(!e2_l0_graph_run(g, gr));

    float *ctx = malloc(sizeof(float) * nq), *ksc_k = malloc(sizeof(float) * (size_t)KV * sup);
    float *vsc_k = malloc(sizeof(float) * (size_t)KV * D);
    int *kbq_k = malloc(sizeof(int) * (size_t)KV * nbq * sup);
    int *vbq_k = malloc(sizeof(int) * (size_t)KV * nbv * D);
    uint8_t *pb_k = malloc(nscr);
    float *pbs_k = malloc(sizeof(float) * (size_t)H * tile * nbv);
    int8_t *kbytes = malloc((size_t)KV * sup * D), *vbytes = malloc((size_t)KV * D * sup);
    CHECK(ctx && ksc_k && vsc_k && kbq_k && vbq_k && pb_k && pbs_k && kbytes && vbytes);
    CHECK(!e2_l0_read(g, kpk, (size_t)KV * sup * D, kbytes));
    CHECK(!e2_l0_read(g, vpk, (size_t)KV * D * sup, vbytes));
    CHECK(!e2_l0_read(g, ctx_d, sizeof(float) * nq, ctx));
    CHECK(!e2_l0_read(g, ksc_d, sizeof(float) * (size_t)KV * sup, ksc_k));
    CHECK(!e2_l0_read(g, kbq_d, sizeof(int) * (size_t)KV * nbq * sup, kbq_k));
    CHECK(!e2_l0_read(g, vsc_d, sizeof(float) * (size_t)KV * D, vsc_k));
    CHECK(!e2_l0_read(g, vbq_d, sizeof(int) * (size_t)KV * nbv * D, vbq_k));
    CHECK(!e2_l0_read(g, pb_d, nscr, pb_k));
    CHECK(!e2_l0_read(g, pbs_d, sizeof(float) * (size_t)H * tile * nbv, pbs_k));

    uint8_t *cq = malloc(nq);
    float *cas = malloc((nq / 128) * sizeof *cas);
    CHECK(cq && cas);
    CHECK(!e2_l0_read(g, qpk, nq, cq));
    CHECK(!e2_l0_read(g, qps, (nq / 128) * sizeof *cas, cas));
    for (size_t b = 0; b < nq / 128; b++) {
        float mx = 0.f;
        for (int j = 0; j < 128; j++) mx = fmaxf(mx, fabsf(ctx[b * 128 + j]));
        const float want = mx > 0.f ? mx / 127.f : 1.f, got = cas[b];
        CHECK(isfinite(got) && got > 0.f && fabsf(got - want) <= 1e-5f * want);
        for (int j = 0; j < 128; j++)
            CHECK(packed_byte_ok(ctx[b * 128 + j] / got, (int)cq[b * 128 + j] - 128));
    }
    free(cq); free(cas);

    /* ---------------------------------------------------------------- the recipe, on the host */
    /* packers: one scale per row, rte, padded rows zero with scale 1 */
    int scale_bad = 0, bsq_bad = 0, packed_bad = 0;
    double *kn = calloc((size_t)sup * KV * D, sizeof *kn);      /* [key][kv][d] */
    double *ksc = calloc((size_t)sup * KV, sizeof *ksc);
    double *kbs = calloc((size_t)sup * KV * nbq, sizeof *kbs);
    double *vn = calloc((size_t)D * KV * sup, sizeof *vn);      /* [feature][kv][key] */
    double *vsc = calloc((size_t)D * KV, sizeof *vsc);
    double *vbs = calloc((size_t)D * KV * nbv, sizeof *vbs);
    CHECK(kn && ksc && kbs && vn && vsc && vbs);
    for (int kvh = 0; kvh < KV; kvh++) {
        for (int n = 0; n < S; n++) {
            const float *xk = k + (size_t)n * kvstride + (size_t)kvh * D;
            float amax = 0.f;
            for (int i = 0; i < D; i++) amax = fmaxf(amax, fabsf(xk[i]));
            const float want = amax > 0.f ? amax / 127.f : 1.f;
            /* Validate the scale independently before checking each stored byte. */
            const float sc = ksc_k[kvh * sup + n];
            if (!scale_usable(S, D, KV, "K scale", sc)) continue;
            scale_bad += fabs((double)sc - (double)want) > 1e-6 * (double)want;
            ksc[n * KV + kvh] = sc;
            for (int i = 0; i < D; i++) {
                int byte = kbytes[(size_t)kvh * sup * D +
                    E2_DPAS_OFF(i / 4, n / 8, n % 8, i % 4, sup / 8)];
                packed_bad += !packed_byte_ok(xk[i] / sc, byte);
                kn[((size_t)n * KV + kvh) * D + i] = byte;
                kbs[(size_t)(n * KV + kvh) * nbq + i / 128] += byte;
            }
        }
        for (int n = S; n < sup; n++) {
            const float sc = ksc_k[kvh * sup + n];
            if (scale_usable(S, D, KV, "padded K scale", sc) && sc != 1.f)
                fail(S, D, KV, "padded K scale", sc, 1.0);
            ksc[n * KV + kvh] = 1.0;
            for (int i = 0; i < D; i++)
                packed_bad += kbytes[(size_t)kvh * sup * D +
                    E2_DPAS_OFF(i / 4, n / 8, n % 8, i % 4, sup / 8)] != 0;
        }
        for (int f = 0; f < D; f++) {
            float amax = 0.f;
            for (int n = 0; n < S; n++)
                amax = fmaxf(amax, fabsf(v[(size_t)n * kvstride + (size_t)kvh * D + f]));
            const float want = amax > 0.f ? amax / 127.f : 1.f;
            const float sc = vsc_k[kvh * D + f];
            if (!scale_usable(S, D, KV, "V scale", sc)) continue;
            scale_bad += fabs((double)sc - (double)want) > 1e-6 * (double)want;
            vsc[f * KV + kvh] = sc;
            for (int n = 0; n < S; n++) {
                int byte = vbytes[(size_t)kvh * D * sup +
                    E2_DPAS_OFF(n / 4, f / 8, f % 8, n % 4, D / 8)];
                packed_bad += !packed_byte_ok(v[(size_t)n * kvstride + (size_t)kvh * D + f] / sc, byte);
                vn[((size_t)f * KV + kvh) * sup + n] = byte;
                vbs[(size_t)(f * KV + kvh) * nbv + n / 128] += byte;
            }
            for (int n = S; n < sup; n++)
                packed_bad += vbytes[(size_t)kvh * D * sup +
                    E2_DPAS_OFF(n / 4, f / 8, f % 8, n % 4, D / 8)] != 0;
        }
    }
    /* Block sums must match the actual stored bytes exactly. */
    for (int kvh = 0; kvh < KV; kvh++)
        for (int b = 0; b < nbq; b++)
            for (int n = 0; n < sup; n++)
                bsq_bad += kbq_k[((size_t)kvh * nbq + b) * sup + n] !=
                           kbs[(size_t)(n * KV + kvh) * nbq + b];
    for (int kvh = 0; kvh < KV; kvh++)
        for (int b = 0; b < nbv; b++)
            for (int f = 0; f < D; f++)
                bsq_bad += vbq_k[((size_t)kvh * nbv + b) * D + f] !=
                           vbs[(size_t)(f * KV + kvh) * nbv + b];
    /* Packing and row-sum errors are separate from floating-point product error. */
    const long nscale = (long)KV * sup + (long)KV * D, nbsq = (long)KV * nbq * sup + (long)KV * nbv * D;
    if (scale_bad > 0) fail(S, D, KV, "packed scales off the recipe by more than 1e-6 rel (count)", scale_bad, 0);
    if (packed_bad) fail(S, D, KV, "packed bytes disagree beyond a rounding tie (count)", packed_bad, 0);
    if (bsq_bad) fail(S, D, KV, "packed zero-point sum mismatch (count)", bsq_bad, 0);

    /* scores, softmax, context */
    double *p8 = malloc(sizeof(double) * (size_t)S * H * sup);   /* [row][head][key] */
    double *pa = malloc(sizeof(double) * (size_t)S * H * nbv);
    double *oref = calloc(nq, sizeof *oref);
    double *eps = malloc(sizeof(double) * sup);
    CHECK(p8 && pa && oref && eps);
    memset(p8, 0, sizeof(double) * (size_t)S * H * sup);
    for (int h = 0; h < H; h++) {
        const int kvh = h * KV / H;
        double *s = malloc(sizeof(double) * (size_t)S * sup);
        CHECK(s);
        for (int m = 0; m < S; m++) {
            for (int n = 0; n < sup; n++) {
                double acc = 0;
                for (int b = 0; b < nbq; b++) {
                    long dot = 0;
                    for (int i = b * 128; i < (b + 1) * 128; i++)
                        dot += (long)qb[((size_t)m * H + h) * D + i] *
                               (long)kn[((size_t)n * KV + kvh) * D + i];
                    dot -= 128L * (long)kbs[(size_t)(n * KV + kvh) * nbq + b];
                    acc += (double)qsc[((size_t)m * H + h) * nbq + b] * ksc[n * KV + kvh] * (double)dot;
                }
                s[(size_t)m * sup + n] = acc;
            }
            /* k_attn_softmax's window: pos = q0 + row, [pos-window, pos+window] inclusive */
            const int pos = q0 + m;
            const int lo_w = (window > 0 && pos - window > 0) ? pos - window : 0;
            const int hi_w = (window > 0 && pos + window + 1 < S) ? pos + window + 1 : S;
            double m1 = -1e300;
            for (int n = lo_w; n < hi_w; n++) m1 = fmax(m1, s[(size_t)m * sup + n]);
            double l = 0;
            for (int n = 0; n < S; n++) {
                eps[n] = (n >= lo_w && n < hi_w) ? exp(s[(size_t)m * sup + n] - m1) : 0.0;
                l += eps[n];
                p8[((size_t)m * H + h) * sup + n] = 0; /* probabilities are unsigned, with zero point 0 */
            }
            for (int b = 0; b < nbv; b++) {
                const int lo = b * 128, hi = (b + 1) * 128 < S ? (b + 1) * 128 : S;
                double mb = 0;
                if (lo < S) {
                    for (int n = lo; n < hi; n++) mb = fmax(mb, eps[n]);
                }
                pa[((size_t)m * H + h) * nbv + b] = (mb > 0 && l > 0) ? mb / (255.0 * l) : 0.0;
                for (int n = lo; n < hi; n++)
                    p8[((size_t)m * H + h) * sup + n] =
                        (mb > 0 && eps[n] > 0) ? fminf(255.f, rintf((float)(eps[n] * (255.0 / mb)))) : 0;
            }
        }
        free(s);
    }
    /* P has zero point 0; unlike Q, the context product must not subtract 128*bsq. */
    for (int h = 0; h < H; h++) {
        const int kvh = h * KV / H;
        for (int b = 0; b < nbv; b++) {
            const int lo = b * 128, hi = (b + 1) * 128 < S ? (b + 1) * 128 : S;
            if (lo >= S) continue;
            for (int m = 0; m < S; m++)
                for (int d = 0; d < D; d++) {
                    long dot = 0;
                    for (int n = lo; n < hi; n++)
                        dot += (long)p8[((size_t)m * H + h) * sup + n] *
                               (long)vn[((size_t)d * KV + kvh) * sup + n];
                    oref[(size_t)m * qstride + (size_t)h * D + d] +=
                        pa[((size_t)m * H + h) * nbv + b] * vsc[d * KV + kvh] * (double)dot;
                }
        }
    }

    /* the device's P block scales are compared below and feed the context: check them first */
    int pdev_bad = 0;
    for (int h = 0; h < H && pdev_bad == 0; h++)
        for (int m = 0; m < S; m++)
            for (int b = 0; b < nbv; b++) {
                const double got = pbs_k[(size_t)(h * tile + m) * nbv + b];
                pdev_bad += !(isfinite(got) && got >= 0.0);
            }
    if (pdev_bad) fail(S, D, KV, "device P block scales not finite and non-negative (count)", pdev_bad, 0);
    /* P bytes and scales must match for every real key */
    long pbad = 0, pcmp = 0;
    double pasc_rel = 0, pasc_max = 0, pmax = 0;
    for (size_t i = 0; i < (size_t)S * H * nbv; i++) pmax = fmax(pmax, pa[i]);
    for (int h = 0; h < H; h++)
        for (int m = 0; m < S; m++) {
            const uint8_t *pk = pb_k + ((size_t)h * tile + m) * sup;
            for (int n = 0; n < S; n++, pcmp++)
                pbad += pk[n] != (uint8_t)p8[((size_t)m * H + h) * sup + n];
            for (int b = 0; b < nbv; b++) {
                const double got = pbs_k[(size_t)(h * tile + m) * nbv + b];
                const double want = pa[((size_t)m * H + h) * nbv + b];
                const double d = fabs(got - want);
                if (d > pasc_max) pasc_max = d;
                if (want > 1e-3 * pmax) pasc_rel = fmax(pasc_rel, d / want);
            }
        }
    if (pbad > pcmp / 1000) fail(S, D, KV, "P bytes differing from the recipe (count)", (double)pbad, pcmp / 1000.0);
    if (!usable(S, D, KV, "P block scales", pa, (size_t)S * H * nbv)) pasc_max = 1;
    if (pasc_max > 1e-2 * pmax) fail(S, D, KV, "P block scale absolute error / max", pasc_max / pmax, 1e-2);

    /* the context vector, both against the recipe and against fp32 attention */
    double *kd = malloc(sizeof(double) * nq), *od = malloc(sizeof(double) * (size_t)S);
    CHECK(kd && od);
    for (size_t i = 0; i < nq; i++) kd[i] = ctx[i];
    const int ok_k = usable(S, D, KV, "context vector", kd, nq);
    const int ok_o = usable(S, D, KV, "recipe context vector", oref, nq);
    const int ok = ok_k && ok_o;
    double c_kernel = ok ? now_cos(kd, oref, nq) : -1, r_kernel = ok ? now_rel(kd, oref, nq) : 1;
    if (!ok) {
        /* already reported by usable() */
    } else if (!isfinite(c_kernel) || c_kernel < 0.99999)
        fail(S, D, KV, "cos(kernel ctx, recipe ctx)", c_kernel, 0.99999);
    else if (r_kernel > 2e-3) fail(S, D, KV, "rel(kernel ctx, recipe ctx)", r_kernel, 2e-3);
    /* the recipe's own error against fp32 attention is the documented int8 cost: report it, and only
     * guard that it stays in the range the tower tolerates, so a change that makes attention *worse*
     * than int8 fails here rather than as a mysterious embedding regression */
    double *a32 = calloc(nq, sizeof *a32);
    CHECK(a32);
    for (int h = 0; h < H; h++) {
        const int kvh = h * KV / H;
        for (int m = 0; m < S; m++) {
            /* the same window the fused softmax path applies: [pos-window, pos+window] inclusive */
            const int pos = q0 + m;
            const int lo_w = (window > 0 && pos - window > 0) ? pos - window : 0;
            const int hi_w = (window > 0 && pos + window + 1 < S) ? pos + window + 1 : S;
            double m1 = -1e300, l = 0;
            for (int n = 0; n < S; n++) {
                double dot = 0;
                for (int i = 0; i < D; i++)
                    dot += (double)q[(size_t)m * qstride + (size_t)h * D + i] *
                           (double)k[(size_t)n * kvstride + (size_t)kvh * D + i];
                od[n] = dot;
                if (n >= lo_w && n < hi_w) m1 = fmax(m1, dot);
            }
            for (int n = 0; n < S; n++) {
                od[n] = (n >= lo_w && n < hi_w) ? exp(od[n] - m1) : 0.0;
                l += od[n];
            }
            for (int n = 0; n < S; n++) od[n] /= l;
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int n = 0; n < S; n++)
                    acc += od[n] * (double)v[(size_t)n * kvstride + (size_t)kvh * D + i];
                a32[(size_t)m * qstride + (size_t)h * D + i] = acc;
            }
        }
    }
    const double c_f32 = ok && usable(S, D, KV, "fp32 attention", a32, nq) ? now_cos(kd, a32, nq) : -1;
    if (ok && (!isfinite(c_f32) || c_f32 < 0.999))
        fail(S, D, KV, "cos(kernel ctx, fp32 attention)", c_f32, 0.999);

    printf("%-4s S=%-4d D=%-3d KV=%d win=%-4d pack scal/bsq %d/%d of %ld/%ld  P bytes %ld/%ld  "
           "P scale %.1e/%.1e  cos(kernel,recipe) %.9f  rel %.2e  cos(kernel,fp32) %.6f\n",
           failures != before ? "FAIL" : "ok", S, D, KV, window, scale_bad, bsq_bad, nscale, nbsq, pbad, pcmp,
           pasc_max / pmax, pasc_rel, c_kernel, r_kernel, c_f32);

    e2_l0_graph_free(g, gr);
    e2_l0_free(g, kdev); e2_l0_free(g, vdev);
    e2_l0_free(g, qpk); e2_l0_free(g, qps); e2_l0_free(g, kpk); e2_l0_free(g, ksc_d);
    e2_l0_free(g, kbq_d); e2_l0_free(g, vpk); e2_l0_free(g, vsc_d); e2_l0_free(g, vbq_d);
    e2_l0_free(g, scr_d); e2_l0_free(g, pb_d); e2_l0_free(g, pbs_d); e2_l0_free(g, ctx_d);
    e2_l0_free(g, valid_d);
    free(q); free(k); free(v); free(qb); free(qsc); free(ctx); free(ksc_k); free(vsc_k);
    free(kbq_k); free(vbq_k); free(pb_k); free(pbs_k); free(kn); free(ksc); free(kbs); free(vn);
    free(vsc); free(vbs); free(p8); free(pa); free(oref); free(eps); free(kd); free(od); free(a32);
    free(ones); free(kbytes); free(vbytes);
}

/* Equal logits make the unsigned probability bytes and scales exact, including query masks. */
static void check_softmax_masks(e2_l0 *g) {
    const int S = 129, N = 256, H = 4, B = 2, tile = 16, M = 2, nb = 2, q0 = 16, window = 0;
    const size_t count = (size_t)B * H * tile * N, nscale = (size_t)B * H * tile * nb;
    float *scores = calloc(count, sizeof *scores), *scales = malloc(nscale * 4);
    uint8_t *prob = malloc(count), valid[258] = {0};
    CHECK(scores && scales && prob);
    memset(valid, 1, (size_t)S);
    valid[16] = valid[128] = 0;
    void *sd = e2_l0_alloc(g, count * 4), *pd = e2_l0_alloc(g, count);
    void *ad = e2_l0_alloc(g, nscale * 4), *md = e2_l0_alloc(g, sizeof valid);
    CHECK(sd && pd && ad && md);
    CHECK(!e2_l0_write(g, sd, count * 4, scores));
    CHECK(!e2_l0_write(g, md, sizeof valid, valid));
    e2_graph *gr = e2_l0_graph_begin(g, 1);
    CHECK(gr);
    CHECK(!e2_l0_append(g, gr, "k_attn_softmax", 1, 1, B * H, 32, 4, 1, 12,
        &sd, sizeof sd, &pd, sizeof pd, &ad, sizeof ad, &md, sizeof md,
        &M, sizeof M, &N, sizeof N, &S, sizeof S, &window, sizeof window,
        &H, sizeof H, &nb, sizeof nb, &q0, sizeof q0, &tile, sizeof tile));
    CHECK(!e2_l0_graph_close(g, gr));
    CHECK(!e2_l0_graph_run(g, gr));
    CHECK(!e2_l0_read(g, pd, count, prob));
    CHECK(!e2_l0_read(g, ad, nscale * 4, scales));
    for (int z = 0; z < B * H; z++) for (int m = 0; m < M; m++) {
        const int live = z < H && m == 1;
        for (int j = 0; j < N; j++)
            CHECK(prob[((size_t)z * tile + m) * N + j] ==
                  (live && j < S && valid[j] ? 255 : 0));
        for (int b = 0; b < nb; b++) {
            const float want = live && b == 0 ? 1.f / (255.f * 127.f) : 0.f;
            const float got = scales[((size_t)z * tile + m) * nb + b];
            CHECK(isfinite(got) && fabsf(got - want) <= 1e-6f * want);
        }
    }
    e2_l0_graph_free(g, gr);
    e2_l0_free(g, sd); e2_l0_free(g, pd); e2_l0_free(g, ad); e2_l0_free(g, md);
    free(scores); free(scales); free(prob);
    puts("unsigned P: masked queries/keys, padding, batch/head and query-tile offsets passed");
}

int main(void) {
    assert(packed_byte_ok(.5f, 0) && packed_byte_ok(.5f, 1));
    assert(packed_byte_ok(-.5f, 0) && packed_byte_ok(-.5f, -1));
    assert(!packed_byte_ok(.49f, 1) && !packed_byte_ok(.5f, 2));
    char err[512];
    e2_l0 *g = e2_l0_create(err, sizeof err);
    if (!g) {
        if (no_gpu(err)) {
            printf("skip: %s\n", err);
            return 77;
        }
        fprintf(stderr, "FAIL create: %s\n", err[0] ? err : "(no message)");
        return 2;
    }
    printf("device: %s\n", e2_l0_device_name(g));
    /* every length boundary the towers change shape at, both layer types */
    /* E2_NUM_HEADS = 4 is the only head count the text tower has; KV 2 on the 20 sliding layers,
     * KV 1 on the 4 full ones. window 512 is config.sliding_window, which excludes nothing below 513
     * tokens - hence the 519/1024 cases and the short windows that isolate the mask at a boundary. */
    static const struct shape sizes[] = {
        { 63, 256, 4, 2, 0 },    { 64, 256, 4, 2, 0 },    { 65, 256, 4, 2, 0 },
        { 127, 256, 4, 2, 0 },   { 128, 256, 4, 2, 0 },   { 129, 256, 4, 2, 0 },
        { 130, 256, 4, 2, 0 },   { 196, 256, 4, 2, 0 },   { 255, 256, 4, 2, 0 },
        { 256, 256, 4, 2, 0 },   { 257, 256, 4, 2, 0 },   { 512, 256, 4, 2, 0 },
        { 513, 256, 4, 2, 0 },   { 129, 512, 4, 1, 0 },   { 257, 512, 4, 1, 0 },
        { 1024, 256, 4, 2, 0 },
        { 519, 256, 4, 2, 512 }, { 1024, 256, 4, 2, 512 }, { 129, 256, 4, 2, 64 },
        { 257, 256, 4, 2, 64 },  { 130, 256, 4, 2, 1 },    { 196, 256, 4, 2, 8 },
        { 512, 512, 4, 1, 512 }, { 1024, 512, 4, 1, 0 },
    };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        run_shape(g, sizes[i], 7u + (unsigned)i);
    check_softmax_masks(g);
    e2_l0_destroy(g);
    if (failures) {
        printf("FAILED: %d comparison(s)\n", failures);
        return 1;
    }
    printf("OK: %zu shapes\n", sizeof sizes / sizeof sizes[0]);
    return 0;
}
