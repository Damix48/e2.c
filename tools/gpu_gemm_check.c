/* gpu_gemm_check - device GEMM vs the CPU int8 kernel, on a real model matrix.
 *
 * usage: gpu_gemm_check MODEL TOK [layer] [m]
 *
 * Picks one packed matrix out of the loaded model, quantises a deterministic
 * activation block, runs it through e2_gpu_op_linear (the deferred device path)
 * and through e2_gemm_range (the CPU reference) and reports max error, cosine
 * and GMAC/s for both.  Not shipped; build with `make gpu-check`.
 */
#include "e2.h"
#include "src/e2_internal.h"
#include "src/e2_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL TOK [layer] [m]\n", argv[0]);
        return 1;
    }
    e2_ctx *ctx = e2_load(argv[1], argv[2]);
    if (!ctx) { fprintf(stderr, "load failed: %s\n", e2_model_info(ctx)); return 1; }
    if (e2_set_device(ctx, E2_DEVICE_GPU) != 0) {
        fprintf(stderr, "no gpu device: %s\n", e2_model_info(ctx));
        return 1;
    }
    int layer = argc > 3 ? atoi(argv[3]) : 0;
    int m = argc > 4 ? atoi(argv[4]) : 64;
    e2_text_layer *ly = &ctx->m.tl[layer];
    const e2_linear *ls[] = {&ly->q, &ly->k, &ly->v, &ly->o, &ly->gate, &ly->down,
                             &ly->ple_gate};
    const char *names[] = {"q", "k", "v", "o", "gate", "down", "ple_gate"};

    if (getenv("E2_CHECK_TEXT")) {
        const char *txt = getenv("E2_CHECK_TEXT");
        float eg[E2_EMBED_DIM], ec[E2_EMBED_DIM];
        e2_ctx *cg = ctx;
        e2_ctx *cc = e2_load(argv[1], argv[2]);
        int rg = 0, rcc = 0;
        printf("text=[%s]\n", txt);
        for (int i = 0; i < 4; i++) rg = e2_embed_text(cg, txt, eg, E2_EMBED_DIM);
        double tg = e2_last_ms(cg);
        for (int i = 0; i < 4; i++) rcc = e2_embed_text(cc, txt, ec, E2_EMBED_DIM);
        double tc = e2_last_ms(cc);
        double d = 0, aa = 0, bb = 0, mx = 0;
        for (int i = 0; i < E2_EMBED_DIM; i++) {
            d += (double)eg[i] * ec[i];
            aa += (double)eg[i] * eg[i];
            bb += (double)ec[i] * ec[i];
            double df = fabs((double)eg[i] - ec[i]);
            if (df > mx) mx = df;
        }
        printf("eg[0..2]=%.6f %.6f %.6f  ec[0..2]=%.6f %.6f %.6f\n", eg[0],eg[1],eg[2], ec[0],ec[1],ec[2]);
        printf("gpu %.2f ms (rc=%d, |e|=%.4f)  cpu %.2f ms (rc=%d, |e|=%.4f)  cos=%.7f maxdiff=%.4g\n",
               tg, rg, sqrt(aa), tc, rcc, sqrt(bb), d / (sqrt(aa * bb) + 1e-30), mx);
        e2_free(cc);
        e2_free(ctx);
        return 0;
    }
    if (getenv("E2_CHECK_SCALE")) {
        /* marginal cost of one launch inside a single submission */
        const e2_qmat *w = &ly->gate.w;
        int k = w->k, n = w->n;
        float *x = e2_dev_alloc(ctx, (size_t)m * k * sizeof(float));
        float *yd = e2_dev_alloc(ctx, (size_t)m * n * sizeof(float));
        for (int i = 0; i < m * k; i++) x[i] = sinf((float)i * 0.017f) * 2.0f;
        for (int rep = 0; rep < 3; rep++) {
            int counts[] = {1, 2, 4, 8, 16, 32, 64};
            for (int ci = 0; ci < 7; ci++) {
                int N = counts[ci];
                double t0 = e2_now_ms();
                for (int i = 0; i < N; i++)
                    if (e2_gpu_op_linear(ctx, yd, x, m, &ly->gate) != 0) return 1;
                e2_gpu_flush(ctx);
                double t = e2_now_ms() - t0;
                if (rep == 2)
                    printf("N=%3d  %8.3f ms total  %7.3f ms/gemm  %8.1f GMAC/s\n",
                           N, t, t / N, (double)m * n * k * N / t / 1e6);
            }
        }
        return 0;
    }
    int bad = 0;
    for (int li = 0; li < 7; li++) {
        const e2_qmat *w = &ls[li]->w;
        int k = w->k, n = w->n, nblk = k / 128;
        float *x = e2_dev_alloc(ctx, (size_t)m * k * sizeof(float));
        float *yd = e2_dev_alloc(ctx, (size_t)m * n * sizeof(float));
        float *yc = malloc((size_t)m * n * sizeof(float));
        uint8_t *xq = malloc((size_t)m * k);
        float *xs = malloc((size_t)m * nblk * sizeof(float));
        for (int i = 0; i < m * k; i++) x[i] = sinf((float)i * 0.017f) * 2.0f
                                            + 0.3f * cosf((float)i * 0.0011f);
        e2_quant_rows_blk(x, m, k, xq, xs, 128);
        int ntiles = (n + 15) / 16;
        double t0 = e2_now_ms();
        e2_gemm_range(yc, n, xq, xs, nblk, m, k, w, 0, ntiles);
        double tcpu = e2_now_ms() - t0;

        memset(yd, 0, (size_t)m * n * sizeof(float));
        t0 = e2_now_ms();
        for (int it = 0; it < 10; it++) {
            if (e2_gpu_op_linear(ctx, yd, x, m, ls[li]) != 0) {
                fprintf(stderr, "%s: device rejected the linear\n", names[li]);
                return 1;
            }
        }
        e2_gpu_flush(ctx);
        double tgpu = (e2_now_ms() - t0) / 10;

        printf("   yd[0..3]=%.5g %.5g %.5g   yc[0..3]=%.5g %.5g %.5g\n",
               yd[0], yd[1], yd[2], yc[0], yc[1], yc[2]);
        double d = 0, aa = 0, bb = 0, mx = 0;
        for (int i = 0; i < m * n; i++) {
            double diff = fabs((double)yd[i] - yc[i]);
            if (diff > mx) mx = diff;
            d += (double)yd[i] * yc[i];
            aa += (double)yd[i] * yd[i];
            bb += (double)yc[i] * yc[i];
        }
        double cos = d / (sqrt(aa * bb) + 1e-30);
        double mac = (double)m * n * k;
        printf("%-8s m=%4d k=%4d n=%5d  cos=%.7f maxerr=%.3g  cpu %6.2f M/s  gpu %6.2f M/s  (%.2fx)\n",
               names[li], m, k, n, cos, mx,
               mac / tcpu / 1e3, mac / tgpu / 1e3, tcpu / tgpu);
        if (cos < 0.9999) bad = 1;
        e2_dev_free(ctx, x);
        e2_dev_free(ctx, yd);
        free(yc); free(xq); free(xs);
    }
    e2_free(ctx);
    return bad;
}
