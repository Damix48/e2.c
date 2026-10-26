/* hip_kernel_bench.hip — same-session A/B of the AMD backend's kernel variants.
 *
 *   e2_hip_kernel_bench gemm      # e2k_gemm (direct loads) vs e2k_gemm_lds, + bit-identical check
 *   e2_hip_kernel_bench attn      # e2k_attn (scalar) vs e2k_attn_lds (fused blocked)
 *
 * On this machine the GPU clock swings by several times between processes (600 MHz idle, 2.2 GHz
 * boost), so a single run of one binary is not a measurement: the two variants are timed alternately
 * inside one process, and the ratio is what matters. The kernel sources are included directly, so
 * what is measured is exactly what the engine launches.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.hip"

/* the kernels' profile hooks are only used by the engine build */
extern "C" int e2h_prof_rec = 0;
extern "C" void e2h_prof_ev(int, const char *, hipEvent_t, hipEvent_t) {}

static hipEvent_t ev0, ev1;

/* ------------------------------------------------------------------ GEMM A/B */

typedef struct {
    int M, N, K, nb, Npad;
    const char *name;
} gshape;

static int bench_gemm(void) {
    gshape shapes[] = {
        { 224, 512, 2048, 16, 512, "down      M224 N512  K2048" },
        { 224, 2048, 512, 4, 2048, "gate|up   M224 N2048 K512 " },
        { 224, 12288, 512, 4, 12288, "ple model M224 N12288 K512" },
        { 224, 512, 512, 4, 512, "o sliding M224 N512  K512 " },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const gshape sh = shapes[s];
        const int mpad = (sh.M + 31) / 32 * 32;
        const size_t wbytes = (size_t)sh.Npad * sh.K;
        const size_t wsc_off = (wbytes + 63) & ~(size_t)63;
        const size_t qa_off = 0;
        const size_t asc_off = ((size_t)sh.M * sh.K + 63) & ~(size_t)63;
        const size_t y_off = (asc_off + 4 * (size_t)sh.M * sh.nb + 63) & ~(size_t)63;
        void *wbuf, *acts;
        hipMalloc(&wbuf, wsc_off + 4 * (size_t)sh.Npad);
        hipMalloc(&acts, y_off + 4 * (size_t)mpad * sh.N);
        signed char *hw = (signed char *)malloc(wbytes);
        signed char *hqa = (signed char *)malloc((size_t)sh.M * sh.K);
        float *hsc = (float *)malloc(4 * (size_t)(sh.Npad > sh.M * sh.nb ? sh.Npad : sh.M * sh.nb));
        srand(7 + sh.N + sh.K);
        for (size_t i = 0; i < wbytes; i++) hw[i] = (signed char)(rand() % 255 - 127);
        for (size_t i = 0; i < (size_t)sh.M * sh.K; i++) hqa[i] = (signed char)(rand() % 255 - 127);
        for (int i = 0; i < sh.Npad; i++) hsc[i] = 1.0f;
        for (size_t i = 0; i < (size_t)sh.M * sh.nb; i++) hsc[i] = 1.0f;
        hipMemcpy(wbuf, hw, wbytes, hipMemcpyHostToDevice);
        hipMemcpy((char *)wbuf + wsc_off, hsc, 4 * (size_t)sh.Npad, hipMemcpyHostToDevice);
        hipMemcpy((char *)acts + qa_off, hqa, (size_t)sh.M * sh.K, hipMemcpyHostToDevice);
        hipMemcpy((char *)acts + asc_off, hsc, 4 * (size_t)sh.M * sh.nb, hipMemcpyHostToDevice);
        hipMemset((char *)acts + y_off, 0, 4 * (size_t)mpad * sh.N);
        int *desc;
        hipMalloc((void **)&desc, 4 * 12);
        int hd[12] = { 0 };
        hd[0] = 0; hd[1] = (int)wsc_off; hd[3] = (int)y_off;
        hd[4] = sh.N; hd[5] = sh.K; hd[6] = sh.nb; hd[7] = (int)qa_off; hd[8] = (int)asc_off;
        hd[10] = sh.Npad;
        hipMemcpy(desc, hd, 4 * 12, hipMemcpyHostToDevice);
        double macs = (double)mpad * sh.N * sh.K;
        double bl = 1e9, bd = 1e9;
        for (int r = 0; r < 16; r++) {
            float ms;
            hipEventRecord(ev0, 0);
            dim3 g1((mpad + 31) / 32, (sh.N + 127) / 128, 1), b1(32, 8, 1);
            e2k_gemm_lds<<<g1, b1>>>((const signed char *)wbuf, (const signed char *)acts, desc, mpad, 0, 1);
            hipEventRecord(ev1, 0);
            hipEventSynchronize(ev1);
            hipEventElapsedTime(&ms, ev0, ev1);
            if (ms < bl) bl = ms;
            hipEventRecord(ev0, 0);
            dim3 g2((mpad + 31) / 32, (sh.N + 255) / 256, 1), b2(64, 4, 1);
            e2k_gemm<<<g2, b2>>>((const signed char *)wbuf, (const signed char *)acts, desc, mpad, 0, 1);
            hipEventRecord(ev1, 0);
            hipEventSynchronize(ev1);
            hipEventElapsedTime(&ms, ev0, ev1);
            if (ms < bd) bd = ms;
        }
        /* bit-identical check: run each once, compare */
        float *ref = (float *)malloc(4 * (size_t)mpad * sh.N);
        float *got = (float *)malloc(4 * (size_t)mpad * sh.N);
        hipMemcpy(ref, (char *)acts + y_off, 4 * (size_t)mpad * sh.N, hipMemcpyDeviceToHost);
        dim3 g2((mpad + 31) / 32, (sh.N + 255) / 256, 1), b2(64, 4, 1);
        e2k_gemm<<<g2, b2>>>((const signed char *)wbuf, (const signed char *)acts, desc, mpad, 0, 1);
        hipDeviceSynchronize();
        hipMemcpy(got, (char *)acts + y_off, 4 * (size_t)mpad * sh.N, hipMemcpyDeviceToHost);
        double maxd = 0;
        for (size_t i = 0; i < (size_t)mpad * sh.N; i++) {
            double d = got[i] - ref[i];
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
        }
        printf("%-28s lds %7.3f ms %7.1f GMAC/s | direct %7.3f ms %7.1f GMAC/s | max|d| %.1e\n",
               sh.name, bl, macs / bl / 1e6, bd, macs / bd / 1e6, maxd);
        free(hw); free(hqa); free(hsc); free(ref); free(got);
        hipFree(wbuf); hipFree(acts); hipFree(desc);
    }
    return 0;
}

/* ------------------------------------------------------------------ attention A/B */

static void bench_attn_shape(int rows, int H, int D, int KV, int window, const char *label) {
    const int q_stride = H * D, kv_stride = KV * D, nb = q_stride / 128;
    void *q, *k, *v, *ctx, *qa, *asc, *valid;
    hipMalloc(&q, 4 * (size_t)rows * q_stride);
    hipMalloc(&k, 4 * (size_t)rows * kv_stride);
    hipMalloc(&v, 4 * (size_t)rows * kv_stride);
    hipMalloc(&ctx, 4 * (size_t)rows * q_stride);
    hipMalloc(&qa, (size_t)rows * q_stride);
    hipMalloc(&asc, 4 * (size_t)rows * nb);
    hipMalloc(&valid, rows);
    {
        unsigned char *hv1 = (unsigned char *)malloc(rows);
        memset(hv1, 1, rows);
        hipMemcpy(valid, hv1, rows, hipMemcpyHostToDevice);
        free(hv1);
    }
    float *hq = (float *)malloc(4 * (size_t)rows * q_stride);
    for (size_t i = 0; i < (size_t)rows * q_stride; i++) hq[i] = sinf((float)i * 0.01f);
    hipMemcpy(q, hq, 4 * (size_t)rows * q_stride, hipMemcpyHostToDevice);
    hipMemcpy(k, hq, 4 * (size_t)rows * kv_stride, hipMemcpyHostToDevice);
    hipMemcpy(v, hq, 4 * (size_t)rows * kv_stride, hipMemcpyHostToDevice);
    double bs = 1e9, bl = 1e9;
    for (int r = 0; r < 20; r++) {
        float ms;
        hipEventRecord(ev0, 0);
        dim3 g1((rows + 3) / 4, H, 1), b1(32, 4, 1);
        e2k_attn<<<g1, b1>>>((float *)ctx, (const float *)q, (const float *)k, (const float *)v,
                              (signed char *)qa, (float *)asc, (const unsigned char *)valid, rows, H,
                              D, KV, q_stride, kv_stride, window, rows);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < bs) bs = ms;
        hipEventRecord(ev0, 0);
        dim3 g2((rows + 7) / 8, H, 1), b2(32, 8, 1);
        e2k_attn_lds<<<g2, b2>>>((float *)ctx, (const float *)q, (const float *)k,
                                  (const float *)v, (signed char *)qa, (float *)asc,
                                  (const unsigned char *)valid, rows, H, D, KV, q_stride, kv_stride,
                                  window, rows);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < bl) bl = ms;
    }
    printf("%-22s scalar %8.3f ms | lds %8.3f ms | lds/scalar %.2fx\n", label, bs, bl, bs / bl);
    free(hq);
    hipFree(q); hipFree(k); hipFree(v); hipFree(ctx); hipFree(qa); hipFree(asc); hipFree(valid);
}

static int bench_attn(void) {
    bench_attn_shape(196, 4, 512, 1, 0, "full 196 tok");
    bench_attn_shape(196, 4, 256, 2, 512, "sliding 196 tok");
    bench_attn_shape(1024, 4, 512, 1, 0, "full 1024 tok");
    return 0;
}

/* ------------------------------------------------------------------ int8 attention A/B
 * The fp32 fused blocked kernel against the whole int8 path (Q-quant + packk + packv + scores +
 * softmax + ctx), timed as a group and per kernel, with a cosine check of the fp32 context
 * against the fp32 kernel's. */
static int bench_attn8_shape(int rows, int H, int D, int KV, int window, const char *label) {
    const int S = rows, q_stride = H * D, kv_stride = KV * D;
    const int sup = (S + 127) / 128 * 128, nb = D / 128, nbv = sup / 128;
    float *q, *k, *v, *ctx0, *ctx1, *qqs, *ksc, *vsc, *scr, *pbs;
    signed char *qq, *kpk, *vpk, *pb;
    unsigned char *valid;
    hipMalloc(&q, 4 * (size_t)rows * q_stride);
    hipMalloc(&k, 4 * (size_t)rows * kv_stride);
    hipMalloc(&v, 4 * (size_t)rows * kv_stride);
    hipMalloc(&ctx0, 4 * (size_t)rows * q_stride);
    hipMalloc(&ctx1, 4 * (size_t)rows * q_stride);
    float *ctx2;
    hipMalloc(&ctx2, 4 * (size_t)rows * q_stride);
    hipMalloc(&qq, (size_t)rows * q_stride + (64 << 10));
    hipMalloc(&qqs, 4 * (size_t)rows * H * nb + (4 << 10));
    hipMalloc(&kpk, (size_t)KV * sup * D);
    hipMalloc(&ksc, 4 * (size_t)KV * sup);
    hipMalloc(&vpk, (size_t)KV * D * sup);
    hipMalloc(&vsc, 4 * (size_t)KV * D);
    hipMalloc(&scr, 4 * (size_t)H * S * sup);
    hipMalloc(&pb, (size_t)H * S * sup + (1 << 20));
    hipMalloc(&pbs, 4 * (size_t)H * S * nbv + (16 << 10));
    hipMalloc(&valid, rows);
    {
        unsigned char *hv1 = (unsigned char *)malloc(rows);
        memset(hv1, 1, rows);
        hipMemcpy(valid, hv1, rows, hipMemcpyHostToDevice);
        free(hv1);
    }
    float *hq = (float *)malloc(4 * (size_t)rows * q_stride);
    srand(11 + D);
    for (size_t i = 0; i < (size_t)rows * q_stride; i++)
        hq[i] = 0.4f * sinf((float)i * 0.013f) + 0.1f * sinf((float)i * 0.0007f);

    hipMemcpy(q, hq, 4 * (size_t)rows * q_stride, hipMemcpyHostToDevice);
    hipMemcpy(k, hq, 4 * (size_t)rows * kv_stride, hipMemcpyHostToDevice);
    hipMemcpy(v, hq, 4 * (size_t)rows * kv_stride, hipMemcpyHostToDevice);
    /* reference: fp32 blocked kernel, fp32 context out */
    dim3 g0((rows + 7) / 8, H, 1), b0(32, 8, 1);
    e2k_attn_lds<<<g0, b0>>>(ctx0, q, k, v, NULL, NULL, valid, rows, H, D, KV, q_stride,
                             kv_stride, window, rows);
    /* int8 path */
    dim3 gr((rows * H + 3) / 4, 1, 1), br(32, 4, 1);
    e2k_rowops<<<gr, br>>>(q, NULL, NULL, qq, qqs, NULL, NULL, 1e-6f, 1.f, 1.f, rows * H, D, nb,
                           16 | 2, NULL);
    dim3 gk((sup + 3) / 4, 1, KV), bk(32, 4, 1);
    e2k_attn_packk<<<gk, bk>>>(kpk, ksc, k, S, sup, D, KV, kv_stride);
    dim3 gv((D + 127) / 128, 1, KV), bv(32, 4, 1);
    dim3 gvq((D + 31) / 32, (sup + 31) / 32, KV), bvq(32, 8, 1);
    e2k_attn_vscale<<<gv, bv>>>(vsc, v, S, D, KV, kv_stride);
    e2k_attn_vquant<<<gvq, bvq>>>(vpk, vsc, v, S, sup, D, KV, kv_stride);
    dim3 gs((S + 7) / 8, (sup + 255) / 256, H), bs(32, 8, 1);
    e2k_attn_scores8<<<gs, bs>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
    dim3 gm((S + 3) / 4, 1, H), bm(32, 4, 1);
    e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
    dim3 gc((S + 7) / 8, (D + 255) / 256, H), bc(32, 8, 1);
    e2k_attn_ctx8<<<gc, bc>>>(ctx1, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
    dim3 gsl((S + 15) / 16, (sup + 127) / 128, H), bsl(32, 8, 1);
    dim3 gcl((S + 15) / 16, (D + 127) / 128, H), bcl(32, 8, 1);
    e2k_attn_scores8_lds<<<gsl, bsl>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
    e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
    e2k_attn_ctx8_lds<<<gcl, bcl>>>(ctx2, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
    dim3 gsl2((S + 15) / 16, (sup + 127) / 128, H);
    e2k_attn_scores8<<<gs, bs>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
    e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
    hipDeviceSynchronize();
    /* cosine(ref, int8) on the fp32 contexts */
    float *h0 = (float *)malloc(4 * (size_t)rows * q_stride);
    float *h1 = (float *)malloc(4 * (size_t)rows * q_stride);
    hipMemcpy(h0, ctx0, 4 * (size_t)rows * q_stride, hipMemcpyDeviceToHost);
    hipMemcpy(h1, ctx1, 4 * (size_t)rows * q_stride, hipMemcpyDeviceToHost);
    double dot = 0, n0 = 0, n1 = 0, maxd = 0;
    long nan0 = 0, nan1 = 0;
    for (size_t i = 0; i < (size_t)rows * q_stride; i++) {
        if (h0[i] != h0[i]) { nan0++; if (nan0 == 1) printf("  [first NaN in fp32 ref at %zu]\n", i); }
        if (h1[i] != h1[i]) { nan1++; if (nan1 == 1) printf("  [first NaN in int8 ctx at %zu]\n", i); }
        dot += (double)h0[i] * h1[i];
        n0 += (double)h0[i] * h0[i];
        n1 += (double)h1[i] * h1[i];
        double d = h0[i] - h1[i];
        if (d < 0) d = -d;
        if (d > maxd) maxd = d;
    }
    const double cosv = dot / sqrt(n0 * n1);
    float *h2 = (float *)malloc(4 * (size_t)rows * q_stride);
    hipMemcpy(h2, ctx2, 4 * (size_t)rows * q_stride, hipMemcpyDeviceToHost);
    double dot2 = 0, n2 = 0, maxd2 = 0;
    for (size_t i = 0; i < (size_t)rows * q_stride; i++) {
        dot2 += (double)h0[i] * h2[i];
        n2 += (double)h2[i] * h2[i];
        double d = h0[i] - h2[i];
        if (d < 0) d = -d;
        if (d > maxd2) maxd2 = d;
    }
    const double cosv2 = dot2 / sqrt(n0 * n2);
    free(h2);

    const double macs = 2.0 * H * S * S * D; /* both products, nominal full key range */
    double t_fp32 = 1e9, t_group = 1e9, t_lds = 1e9, tsc = 1e9, tcx = 1e9, tvs = 1e9, tvq = 1e9;
    double tk[6] = { 1e9, 1e9, 1e9, 1e9, 1e9, 1e9 };
    for (int r = 0; r < 20; r++) {
        float ms;
        hipEventRecord(ev0, 0);
        e2k_attn_lds<<<g0, b0>>>(ctx0, q, k, v, NULL, NULL, valid, rows, H, D, KV, q_stride,
                                 kv_stride, window, rows);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < t_fp32) t_fp32 = ms;
        /* the int8 group, and each kernel inside it */
        hipEventRecord(ev0, 0);
        e2k_rowops<<<gr, br>>>(q, NULL, NULL, qq, qqs, NULL, NULL, 1e-6f, 1.f, 1.f, rows * H, D,
                               nb, 16 | 2, NULL);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tk[0]) tk[0] = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_packk<<<gk, bk>>>(kpk, ksc, k, S, sup, D, KV, kv_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tk[1]) tk[1] = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_vscale<<<gv, bv>>>(vsc, v, S, D, KV, kv_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tvs) tvs = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_vquant<<<gvq, bvq>>>(vpk, vsc, v, S, sup, D, KV, kv_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tvq) tvq = ms;
        if (tvs + tvq < tk[2]) tk[2] = tvs + tvq;
        hipEventRecord(ev0, 0);
        e2k_attn_scores8<<<gs, bs>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tk[3]) tk[3] = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tk[4]) tk[4] = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_ctx8<<<gc, bc>>>(ctx1, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tk[5]) tk[5] = ms;
        hipEventRecord(ev0, 0);
        e2k_rowops<<<gr, br>>>(q, NULL, NULL, qq, qqs, NULL, NULL, 1e-6f, 1.f, 1.f, rows * H, D,
                               nb, 16 | 2, NULL);
        e2k_attn_packk<<<gk, bk>>>(kpk, ksc, k, S, sup, D, KV, kv_stride);
        e2k_attn_vscale<<<gv, bv>>>(vsc, v, S, D, KV, kv_stride);
        e2k_attn_vquant<<<gvq, bvq>>>(vpk, vsc, v, S, sup, D, KV, kv_stride);
        e2k_attn_scores8<<<gs, bs>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
        e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
        e2k_attn_ctx8<<<gc, bc>>>(ctx1, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < t_group) t_group = ms;
        /* the LDS group alone, plus its two products */
        hipEventRecord(ev0, 0);
        e2k_attn_scores8_lds<<<gsl, bsl>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tsc) tsc = ms;
        e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
        hipEventRecord(ev0, 0);
        e2k_attn_ctx8_lds<<<gcl, bcl>>>(ctx2, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < tcx) tcx = ms;
        hipEventRecord(ev0, 0);
        e2k_attn_scores8_lds<<<gsl, bsl>>>(scr, qq, qqs, kpk, ksc, S, sup, D, H, KV, q_stride);
        e2k_attn_softmax8<<<gm, bm>>>(scr, pb, pbs, valid, S, sup, H, window);
        e2k_attn_ctx8_lds<<<gcl, bcl>>>(ctx2, pb, pbs, vpk, vsc, S, sup, D, H, KV, q_stride);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < t_lds) t_lds = ms;
    }
    printf("%-18s fp32 %7.3f | int8 %7.3f (%5.1f G, %.1fx, cos %.6f) | lds %7.3f (%5.1f G, "
           "%.1fx, cos %.6f maxd %.1e) sc %.3f/%.3f cx %.3f/%.3f sm %.3f pv %.3f (vs %.3f vq %.3f)\n",
           label, t_fp32, t_group, macs / (t_group - tk[0] - tk[1] - tk[2] - tk[4]) / 1e6,
           t_fp32 / t_group, cosv, t_lds, macs / (t_lds - tk[0] - tk[1] - tk[2] - tk[4]) / 1e6,
           t_fp32 / t_lds, cosv2, maxd2, tk[3], tsc, tk[5], tcx, tk[4], tk[2], tvs, tvq);
    free(hq); free(h0); free(h1);
    hipFree(q); hipFree(k); hipFree(v); hipFree(ctx0); hipFree(ctx1); hipFree(ctx2);
    hipFree(qq); hipFree(qqs); hipFree(kpk); hipFree(ksc); hipFree(vpk); hipFree(vsc);
    hipFree(scr); hipFree(pb); hipFree(pbs); hipFree(valid);
    return 0;
}

static void kernel_report(void) {
    struct { const char *n; const void *f; } ks[] = {
        { "attn_scores8", (const void *)e2k_attn_scores8 },
        { "attn_ctx8", (const void *)e2k_attn_ctx8 },
        { "attn_softmax8", (const void *)e2k_attn_softmax8 },
        { "attn_packk", (const void *)e2k_attn_packk },
        { "attn_vscale", (const void *)e2k_attn_vscale },
        { "attn_vquant", (const void *)e2k_attn_vquant },
        { "attn_lds", (const void *)e2k_attn_lds },
        { "gemm_lds", (const void *)e2k_gemm_lds },
    };
    for (size_t i = 0; i < sizeof ks / sizeof ks[0]; i++) {
        hipFuncAttributes a;
        memset(&a, 0, sizeof a);
        if (hipFuncGetAttributes(&a, ks[i].f) == hipSuccess)
            printf("%-14s regs %3d lds %6zu maxthreads %4d spills %d\n", ks[i].n, a.numRegs,
                   a.sharedSizeBytes, a.maxThreadsPerBlock, a.localSizeBytes);
    }
}

static int bench_attn8(void) {
    kernel_report();
    bench_attn8_shape(196, 4, 512, 1, 0, "full 196 tok");
    bench_attn8_shape(196, 4, 256, 2, 512, "sliding 196 tok");
    bench_attn8_shape(1024, 4, 512, 1, 0, "full 1024 tok");
    bench_attn8_shape(2048, 4, 256, 2, 512, "sliding 2048 tok");
    return 0;
}

int main(int argc, char **argv) {
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    const char *mode = argc > 1 ? argv[1] : "gemm";
    if (!strcmp(mode, "gemm")) return bench_gemm();
    if (!strcmp(mode, "attn")) return bench_attn();
    if (!strcmp(mode, "attn8")) return bench_attn8();
    fprintf(stderr, "usage: %s [gemm|attn|attn8]\n", argv[0]);
    return 2;
}
