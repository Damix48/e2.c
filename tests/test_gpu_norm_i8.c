/* Fused RMSNorm/RoPE -> Q quantization and packed K against a scalar FP64 oracle.
 * Check both head sizes, GQA, two sequences, masks, padded keys and debug-store modes. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu/e2_tile.h"
#include "gpu/l0.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, e2_l0_error(g)); exit(1); \
} } while (0)
enum { Q, K, V, GQ, GK, CS, SN, MASK, QA, AS, KP, KS, KB, COUNT };

static void reference(double *out, const float *x, int rows, int heads, int D, int S,
                      const float *gamma, const float *cs, const float *sn,
                      const uint8_t *valid, float eps) {
    for (int r = 0; r < rows; r++) {
        double ss = 0;
        for (int d = 0; d < D; d++) ss += (double)x[(size_t)r * D + d] * x[(size_t)r * D + d];
        const double inv = 1.0 / sqrt(ss / D + eps);
        const int pos = r / heads % S;
        for (int d = 0; d < D; d++) {
            const int p = d < D / 2 ? d + D / 2 : d - D / 2;
            double a = x[(size_t)r * D + d] * inv;
            if (gamma) {
                const double b = x[(size_t)r * D + p] * inv * gamma[p];
                a = a * gamma[d] * cs[pos * D + d] +
                    (d < D / 2 ? -b : b) * sn[pos * D + d];
            }
            out[(size_t)r * D + d] = valid[r / heads] ? a : 0.0;
        }
    }
}

static void run(e2_l0 *g, int S, int D, int KV, int wr) {
    const int B = 2, H = 4, SP = (S + 127) / 128 * 128, nb = D / 128;
    const int rq = B * S * H, rk = B * S * KV, kr = B * SP * KV;
    const int nr = rq > kr ? rq : kr;
    const float eps = 1e-6f;
    const size_t nq = (size_t)rq * D, nk = (size_t)rk * D;
    const size_t bytes[COUNT] = {nq * 4, nk * 4, nk * 4, D * 4, D * 4,
        (size_t)S * D * 4, (size_t)S * D * 4, (size_t)B * S, nq, (size_t)rq * nb * 4,
        (size_t)B * KV * SP * D, (size_t)B * KV * SP * 4, (size_t)B * KV * nb * SP * 4};
    void *host[COUNT], *dev[COUNT];
    for (int i = 0; i < COUNT; i++) {
        host[i] = malloc(bytes[i]); dev[i] = e2_l0_alloc(g, bytes[i]);
        CHECK(host[i] && dev[i]); memset(host[i], 0xa5, bytes[i]);
    }
    uint8_t *valid = host[MASK];
    for (int r = 0; r < B * S; r++) valid[r] = r % 11 != 0 && r < S + S / 2;
    valid[0] = 1; valid[S] = 0;
    for (int t = Q; t <= V; t++) {
        float *x = host[t]; const int heads = t == Q ? H : KV, rows = t == Q ? rq : rk;
        for (int r = 0; r < rows; r++) for (int d = 0; d < D; d++)
            x[(size_t)r * D + d] = valid[r / heads] ?
                sinf(.17f * d + .37f * r + t) + .6f * cosf(.03f * d - .19f * r) : 0.f;
    }
    float *gq = host[GQ], *gk = host[GK], *cs = host[CS], *sn = host[SN];
    for (int d = 0; d < D; d++) { gq[d] = .75f + .2f * sinf(d); gk[d] = 1.2f + .15f * cosf(d); }
    for (int p = 0; p < S; p++) for (int d = 0; d < D; d++) {
        const float angle = .01f * (p + 1) * (1 + d % (D / 2)) / (D / 2);
        cs[p * D + d] = cosf(angle); sn[p * D + d] = sinf(angle);
    }
    double *want[3] = {malloc(nq * sizeof(double)), malloc(nk * sizeof(double)), malloc(nk * sizeof(double))};
    CHECK(want[0] && want[1] && want[2]);
    reference(want[Q], host[Q], rq, H, D, S, gq, cs, sn, valid, eps);
    reference(want[K], host[K], rk, KV, D, S, gk, cs, sn, valid, eps);
    reference(want[V], host[V], rk, KV, D, S, NULL, cs, sn, valid, eps);
    for (int i = 0; i < COUNT; i++) CHECK(!e2_l0_write(g, dev[i], bytes[i], host[i]));
    void *gv = NULL;
    e2_graph *gr = e2_l0_graph_begin(g, 1); CHECK(gr);
    CHECK(!e2_l0_append(g, gr, "k_norm_rope3_i8", (nr + 1) / 2, 1, 1, 32, 2, 1, 24,
        &dev[Q], sizeof(void *), &dev[GQ], sizeof(void *), &rq, sizeof rq,
        &dev[K], sizeof(void *), &dev[GK], sizeof(void *), &rk, sizeof rk,
        &dev[V], sizeof(void *), &gv, sizeof gv, &rk, sizeof rk,
        &dev[CS], sizeof(void *), &dev[SN], sizeof(void *), &eps, sizeof eps,
        &H, sizeof H, &KV, sizeof KV, &D, sizeof D, &S, sizeof S,
        &dev[QA], sizeof(void *), &dev[AS], sizeof(void *), &dev[KP], sizeof(void *),
        &dev[KS], sizeof(void *), &dev[KB], sizeof(void *), &SP, sizeof SP,
        &dev[MASK], sizeof(void *), &wr, sizeof wr));
    CHECK(!e2_l0_graph_close(g, gr)); CHECK(!e2_l0_graph_run(g, gr));
    float *raw = malloc(bytes[Q]); CHECK(raw);
    for (int t = Q; t <= V; t++) {
        CHECK(!e2_l0_read(g, dev[t], bytes[t], raw));
        if (!wr && t != V) CHECK(!memcmp(raw, host[t], bytes[t]));
        else for (size_t i = 0; i < bytes[t] / 4; i++)
            CHECK(isfinite(raw[i]) && fabs(raw[i] - want[t][i]) <= 2e-5 * fmax(1.0, fabs(want[t][i])));
    }
    for (int t = QA; t < COUNT; t++) CHECK(!e2_l0_read(g, dev[t], bytes[t], host[t]));
    const uint8_t *qa = host[QA]; const float *as = host[AS], *ks = host[KS];
    const int8_t *pk = host[KP]; const int *bs = host[KB];
    for (int r = 0; r < rq; r++) for (int b = 0; b < nb; b++) {
        double mx = 0;
        for (int j = 0; j < 128; j++) mx = fmax(mx, fabs(want[Q][(size_t)r * D + b * 128 + j]));
        const double sc = mx > 0 ? mx / 127 : 1;
        const float got = as[r * nb + b];
        CHECK(isfinite(got) && got > 0 && fabs(got - sc) <= 2e-5 * sc);
        for (int j = 0; j < 128; j++)
            CHECK(fabs((int)qa[(size_t)r * D + b * 128 + j] - 128 -
                       want[Q][(size_t)r * D + b * 128 + j] / got) <= .501);
    }
    for (int s = 0; s < B; s++) for (int h = 0; h < KV; h++) for (int key = 0; key < SP; key++) {
        const int z = s * KV + h; const size_t base = (size_t)z * SP * D;
        const double *ref = key < S ? want[K] + ((size_t)(s * S + key) * KV + h) * D : NULL;
        double mx = 0;
        for (int d = 0; d < D; d++) mx = fmax(mx, ref ? fabs(ref[d]) : 0);
        const double sc = mx > 0 ? mx / 127 : 1; const float got = ks[z * SP + key];
        CHECK(isfinite(got) && got > 0 && fabs(got - sc) <= 2e-5 * sc);
        for (int b = 0; b < nb; b++) {
            int sum = 0;
            for (int j = 0; j < 128; j++) {
                const int d = b * 128 + j;
                const int v = pk[base + E2_DPAS_OFF(d / 4, key / 8, key % 8, d % 4, SP / 8)];
                CHECK(fabs(v - (ref ? ref[d] / got : 0)) <= .501); sum += v;
            }
            CHECK(bs[((size_t)z * nb + b) * SP + key] == sum);
        }
    }
    printf("norm/pack S=%d D=%d KV=%d wr=%d: FP64 values, Q/K bytes, scales and sums passed\n", S,D,KV,wr);
    e2_l0_graph_free(g, gr); free(raw);
    for (int i = 0; i < COUNT; i++) { free(host[i]); e2_l0_free(g, dev[i]); }
    for (int i = 0; i < 3; i++) free(want[i]);
}

int main(void) {
    char err[512]; e2_l0 *g = e2_l0_create(err, sizeof err);
    if (!g) {
        fprintf(stderr, "%s\n", err);
        return !strcmp(err,"no Level Zero driver") || !strcmp(err,"no Intel GPU device found") ? 77 : 1;
    }
    for (int wr = 0; wr < 2; wr++) for (int D = 256; D <= 512; D *= 2) {
        run(g, 65, D, D == 256 ? 2 : 1, wr);
        run(g, 196, D, D == 256 ? 2 : 1, wr);
    }
    e2_l0_destroy(g); return 0;
}
