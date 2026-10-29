/* dp4a_bench.c — standalone OpenCL DP4A GEMM microbenchmark.
 *
 * Not part of the e2 build. It runs the dp4a projection GEMM from
 * src/gpu/kernels/e2.cl (k_gemm_tiled) at a chosen shape, checks it against a
 * scalar reference, and reports GMAC/s. The point is to decide whether a
 * DP4A-only backend variant is worth writing for GPUs without XMX/DPAS, such
 * as the N100's Xe-LP.
 *
 * Build (needs the Khronos OpenCL headers and the ICD loader, no OpenCL dev
 * package required beyond libOpenCL.so.1):
 *
 *   cc -O2 -std=c11 -I/path/to/OpenCL-Headers -I src/gpu \
 *      tools/bench/dp4a_bench.c -o /tmp/dp4a_bench \
 *      -L/usr/lib/x86_64-linux-gnu -l:libOpenCL.so.1
 *
 * Run:
 *
 *   /tmp/dp4a_bench tools/bench/dp4a_bench.cl src/gpu 64 512 2048 20
 *   /tmp/dp4a_bench tools/bench/dp4a_bench.cl src/gpu 64 2048 512 20
 *
 * M must be a multiple of E2_BM, N a multiple of 4, K a multiple of 128. The
 * kernel is the dp4a path, not the XMX one: it uses dot(uchar4, char4), so the
 * program is built with -cl-std=CL3.0 and needs cl_khr_integer_dot_product.
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e2_tile.h"

static void cl_ok(cl_int e, const char *what) {
    if (e != CL_SUCCESS) {
        fprintf(stderr, "dp4a_bench: %s: OpenCL error %d\n", what, (int)e);
        exit(1);
    }
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(2);
    }
    if (fseek(f, 0, SEEK_END) != 0) exit(2);
    long n = ftell(f);
    if (n < 0) exit(2);
    rewind(f);
    char *s = malloc((size_t)n + 1);
    if (!s || fread(s, 1, (size_t)n, f) != (size_t)n) exit(2);
    s[n] = 0;
    fclose(f);
    return s;
}

static size_t align64(size_t n) { return (n + 63u) & ~(size_t)63u; }

static uint32_t g_rng = 0x9e3779b9u;
static int rnd8(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (int)((g_rng >> 24) & 0xffu) - 128;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s kernel.cl include_dir M N K [reps]\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        usage(argv[0]);
        return 2;
    }
    const char *cl_path = argv[1], *inc_dir = argv[2];
    const int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]);
    const int reps = argc > 6 ? atoi(argv[6]) : 20;
    if (M <= 0 || N <= 0 || K <= 0 || reps <= 0 || reps > 10000) {
        usage(argv[0]);
        return 2;
    }
    if (M % E2_BM != 0 || N % 4 != 0 || K % 128 != 0) {
        fprintf(stderr, "dp4a_bench: need M %% %d == 0, N %% 4 == 0, K %% 128 == 0\n", E2_BM);
        return 2;
    }
    const int nb = K / 128;

    /* ---- host buffers: the exact layout k_gemm_tiled and the CPU reference use.
     * wbuf: quad-major weights [K/4][N][4], then wsc[N], then bsq[nb][N].
     * acts: q = a + 128 [M][K], then asc[M][nb], then y[M][N]. */
    const size_t wsc_off = align64((size_t)N * K);
    const size_t bsq_off = align64(wsc_off + (size_t)N * 4);
    const size_t wbuf_bytes = bsq_off + (size_t)nb * N * 4;
    const size_t asc_off = align64((size_t)M * K);
    const size_t y_off = align64(asc_off + (size_t)M * nb * 4);
    const size_t acts_bytes = y_off + (size_t)M * N * 4;

    uint8_t *wbuf = calloc(1, wbuf_bytes);
    uint8_t *acts = calloc(1, acts_bytes);
    int8_t *wrow = malloc((size_t)N * K);
    int32_t *bsq = malloc((size_t)nb * N * sizeof(int32_t));
    float *wsc = (float *)(wbuf + wsc_off);
    float *asc = (float *)(acts + asc_off);
    uint8_t *qa = acts;
    if (!wbuf || !acts || !wrow || !bsq) {
        fprintf(stderr, "dp4a_bench: out of memory\n");
        return 1;
    }
    memset(wrow, 0, (size_t)N * K);
    memset(bsq, 0, (size_t)nb * N * sizeof(int32_t));

    for (int n = 0; n < N; n++) wsc[n] = 0.0007f + 0.00007f * (float)(n % 11);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++)
            asc[(size_t)m * nb + b] = 0.0013f + 0.00011f * (float)(b % 7) + 0.00003f * (float)(m % 5);

    for (int q = 0; q < K / 4; q++)
        for (int n = 0; n < N; n++)
            for (int kk = 0; kk < 4; kk++) {
                int v = rnd8();
                wrow[(size_t)n * K + (size_t)q * 4 + kk] = (int8_t)v;
                wbuf[(size_t)q * N * 4 + (size_t)n * 4 + kk] = (uint8_t)(int8_t)v;
            }

    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            int s = 0;
            for (int k = b * 128; k < (b + 1) * 128; k++) s += wrow[(size_t)n * K + k];
            bsq[(size_t)b * N + n] = s;
        }
    memcpy(wbuf + bsq_off, bsq, (size_t)nb * N * sizeof(int32_t));

    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) qa[(size_t)m * K + k] = (uint8_t)(rnd8() + 128);

    /* Scalar reference, in double, over the biased q bytes. */
    float *ref = malloc((size_t)M * N * sizeof(float));
    if (!ref) return 1;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int b = 0; b < nb; b++) {
                int s = 0;
                const int8_t *wp = wrow + (size_t)n * K + b * 128;
                const uint8_t *qp = qa + (size_t)m * K + b * 128;
                for (int k = 0; k < 128; k++) s += (int)wp[k] * ((int)qp[k] - 128);
                acc += (double)asc[(size_t)m * nb + b] * (double)wsc[n] * (double)s;
            }
            ref[(size_t)m * N + n] = (float)acc;
        }

    int desc[16] = {0};
    desc[0] = 0;
    desc[1] = (int)wsc_off;
    desc[2] = (int)bsq_off;
    desc[3] = (int)y_off;
    desc[4] = N;
    desc[5] = K;
    desc[6] = nb;
    desc[7] = 0;
    desc[8] = (int)asc_off;

    /* ---- OpenCL: pick the first GPU platform. */
    cl_uint nplat = 0;
    cl_ok(clGetPlatformIDs(0, NULL, &nplat), "clGetPlatformIDs");
    if (!nplat) {
        fprintf(stderr, "dp4a_bench: no OpenCL platform\n");
        return 1;
    }
    cl_platform_id *plats = calloc(nplat, sizeof *plats);
    if (!plats) return 1;
    cl_ok(clGetPlatformIDs(nplat, plats, NULL), "clGetPlatformIDs");
    cl_device_id device = NULL;
    for (cl_uint i = 0; i < nplat && !device; i++) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) == CL_SUCCESS && nd) {
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        }
    }
    free(plats);
    if (!device) {
        fprintf(stderr, "dp4a_bench: no GPU device\n");
        return 1;
    }

    char dname[256] = {0};
    cl_uint cus = 0, mhz = 0;
    cl_ok(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof dname, dname, NULL), "device name");
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
    fprintf(stderr, "dp4a_bench: %s, %u CUs @ %u MHz (dp4a peak ~%.0f GMAC/s at 8 lanes/EU)\n",
            dname, cus, mhz, (double)cus * 8.0 * 4.0 * (double)mhz * 1e6 / 1e9);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_ok(err, "clCreateContext");
    const cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, qprops, &err);
    cl_ok(err, "clCreateCommandQueueWithProperties");

    char *src = slurp(cl_path);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
    cl_ok(err, "clCreateProgramWithSource");
    char opts[1200];
    snprintf(opts, sizeof opts, "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math -I%s", inc_dir);
    err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *log = malloc(len + 1);
        if (log) {
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
            log[len] = 0;
            fprintf(stderr, "dp4a_bench: build failed:\n%s\n", log);
            free(log);
        }
        return 1;
    }

    cl_kernel kern = clCreateKernel(prog, "k_gemm_tiled", &err);
    cl_ok(err, "clCreateKernel");
    cl_mem wm = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wbuf_bytes, wbuf, &err);
    cl_ok(err, "clCreateBuffer(weights)");
    cl_mem am = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, acts_bytes, acts, &err);
    cl_ok(err, "clCreateBuffer(acts)");
    cl_mem dm = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof desc, desc, &err);
    cl_ok(err, "clCreateBuffer(desc)");
    cl_ok(clSetKernelArg(kern, 0, sizeof wm, &wm), "arg0");
    cl_ok(clSetKernelArg(kern, 1, sizeof am, &am), "arg1");
    cl_ok(clSetKernelArg(kern, 2, sizeof dm, &dm), "arg2");
    cl_ok(clSetKernelArg(kern, 3, sizeof M, &M), "arg3");

    /* e2_l0_append takes group counts; OpenCL takes work-items. Multiply by the local size. */
    const size_t groups[3] = { (size_t)M / E2_BM, (size_t)(N + E2_BN - 1) / E2_BN, 1 };
    const size_t local[3] = { E2_GEMM_NG, E2_GEMM_MG, 1 };
    const size_t global[3] = { groups[0] * local[0], groups[1] * local[1], groups[2] * local[2] };
    for (int i = 0; i < 3; i++) {
        cl_ok(clEnqueueNDRangeKernel(queue, kern, 3, NULL, global, local, 0, NULL, NULL), "warmup");
        cl_ok(clFinish(queue), "warmup finish");
    }

    double *times = malloc((size_t)reps * sizeof(double));
    if (!times) return 1;
    for (int i = 0; i < reps; i++) {
        cl_event ev = NULL;
        cl_ok(clEnqueueNDRangeKernel(queue, kern, 3, NULL, global, local, 0, NULL, &ev), "enqueue");
        cl_ok(clWaitForEvents(1, &ev), "wait");
        cl_ulong t0 = 0, t1 = 0;
        cl_int pe = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        if (pe == CL_SUCCESS)
            pe = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        clReleaseEvent(ev);
        if (pe != CL_SUCCESS || t1 <= t0) {
            fprintf(stderr, "dp4a_bench: event profiling failed (%d)\n", (int)pe);
            return 1;
        }
        times[i] = (double)(t1 - t0) / 1e6;
    }
    double *sorted = malloc((size_t)reps * sizeof(double));
    if (!sorted) return 1;
    memcpy(sorted, times, (size_t)reps * sizeof(double));
    qsort(sorted, (size_t)reps, sizeof(double), cmp_double);
    const double best = sorted[0], median = sorted[reps / 2];
    free(sorted);
    free(times);

    float *out = malloc((size_t)M * N * sizeof(float));
    if (!out) return 1;
    cl_ok(clEnqueueReadBuffer(queue, am, CL_TRUE, y_off, (size_t)M * N * sizeof(float), out, 0, NULL,
                              NULL),
          "read output");

    double max_abs = 0.0, max_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        double d = fabs((double)out[i] - (double)ref[i]);
        if (d > max_abs) max_abs = d;
        if (fabs((double)ref[i]) > max_ref) max_ref = fabs((double)ref[i]);
    }
    const double max_rel = max_abs / (max_ref > 0.0 ? max_ref : 1.0);
    const int ok = max_rel < 1e-4;

    printf("{\"device\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"reps\":%d,"
           "\"best_ms\":%.6f,\"median_ms\":%.6f,\"gmac_s\":%.1f,"
           "\"max_rel\":%.3e,\"check\":\"%s\"}\n",
           dname, M, N, K, reps, best, median,
           (double)M * (double)N * (double)K / (best * 1e6), max_rel, ok ? "ok" : "FAIL");
    if (!ok) {
        int nz_out = 0, nz_ref = 0;
        for (int i = 0; i < M * N; i++) { nz_out += out[i] != 0.f; nz_ref += ref[i] != 0.f; }
        fprintf(stderr, "dp4a_bench: output does not match the scalar reference (out nonzero %d/%d, ref nonzero %d/%d)\n",
                nz_out, M * N, nz_ref, M * N);
        for (int i = 0; i < 8; i++) fprintf(stderr, "  [%d] out=%g ref=%g\n", i, out[i], ref[i]);
    }

    free(out);
    free(ref);
    free(wrow);
    free(bsq);
    free(acts);
    free(wbuf);
    free(src);
    clReleaseMemObject(dm);
    clReleaseMemObject(am);
    clReleaseMemObject(wm);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return ok ? 0 : 1;
}
