/* e2_util.c - arena, thread pool, misc. */
#define _GNU_SOURCE
#include "e2_internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

double e2_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Arena                                                              */
/* ------------------------------------------------------------------ */

struct e2_arena_block {
    struct e2_arena_block *next;
    size_t size, used;
    unsigned char data[];
};

#define ARENA_MIN (1u << 20)

/* Blocks are appended and refilled in allocation order, so a repeated forward
 * (same sizes every call) reuses the blocks it allocated last time instead of
 * growing the arena without bound -- which is what pushed audio (many 800 KB
 * buffers) into a page fault per 16 KB per layer. */
void *e2_arena_alloc(e2_arena *a, size_t n) {
    n = (n + 63) & ~(size_t)63;
    for (e2_arena_block *b = a->cur; b; b = b->next)
        if (b->used + n <= b->size) {
            void *p = b->data + b->used;
            b->used += n;
            a->cur = b;
            a->used += n;
            return p;
        }
    size_t sz = n > ARENA_MIN ? n : ARENA_MIN;
    if (sz < a->cap) sz = a->cap;
    e2_arena_block *b = malloc(sizeof(*b) + sz);
    if (!b) return NULL;
    b->size = sz;
    b->used = n;
    b->next = NULL;
    if (a->cur) a->cur->next = b;
    else a->head = b;
    a->cur = b;
    a->cap = sz;
    a->used += n;
    return b->data;
}

void e2_arena_reset(e2_arena *a) {
    for (e2_arena_block *b = a->head; b; b = b->next) b->used = 0;
    a->cur = a->head;
    a->used = 0;
}

void e2_arena_free(e2_arena *a) {
    e2_arena_block *b = a->head;
    while (b) {
        e2_arena_block *n = b->next;
        free(b);
        b = n;
    }
    memset(a, 0, sizeof(*a));
}

/* ------------------------------------------------------------------ */
/* Thread pool: main thread is tid 0, workers are 1..n-1.             */
/* Workers spin briefly for the next dispatch; this keeps the latency  */
/* of the ~200 small GEMMs per forward at a few microseconds.          */
/* ------------------------------------------------------------------ */

struct worker_arg { struct e2_pool *pool; int tid; };

struct e2_pool {
    int n;
    pthread_t *th;
    struct worker_arg *wa;
    _Alignas(64) _Atomic unsigned gen;
    _Alignas(64) _Atomic int done;
    _Alignas(64) _Atomic int stop;
    _Alignas(64) _Atomic int ready;
    e2_work_fn fn;
    void *arg;
};

static void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    sched_yield();
#endif
}

/* Cores a hybrid part marks as "low power" (Arrow Lake: the 2 LP-E cores,
 * ~8x slower than a P-core for this code) are left out entirely: one straggler
 * on a barrier costs more than the extra parallelism is worth. */
static unsigned char g_skip[256];

static void load_skip_list(void) {
    FILE *f = fopen("/sys/devices/cpu_lowpower/cpus", "r");
    if (!f) return;
    char buf[512];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return; }
    fclose(f);
    char *p = buf;
    while (*p) {
        int a = (int)strtol(p, &p, 10), b = a;
        if (*p == '-') { p++; b = (int)strtol(p, &p, 10); }
        for (int c = a; c <= b && c < 256; c++) g_skip[c] = 1;
        if (*p == ',') p++; else break;
    }
}

/* Usable CPUs, primary hardware thread of each core first (SMT siblings only
 * add contention for these kernels), LP cores dropped. */
static int g_cpus[512], g_ncpu = -1;

static void build_cpu_list(void) {
    if (g_ncpu >= 0) return;
    cpu_set_t all;
    CPU_ZERO(&all);
    if (sched_getaffinity(0, sizeof(all), &all) != 0) { g_ncpu = 0; return; }
    load_skip_list();
    int nn = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nn > 256) nn = 256;
    for (int attempt = 0; attempt < 2; attempt++) {
        int keep_skip = attempt == 0;
        g_ncpu = 0;
        for (int pass = 0; pass < 2; pass++)
            for (int c = 0; c < nn && g_ncpu < 512; c++) {
                if (!CPU_ISSET(c, &all)) continue;
                if (keep_skip && g_skip[c]) continue;
                char path[160];
                int first = c;
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
                FILE *f = fopen(path, "r");
                if (f) { if (fscanf(f, "%d", &first) != 1) first = c; fclose(f); }
                if ((pass == 0) == (first == c)) g_cpus[g_ncpu++] = c;
            }
        if (g_ncpu > 0) return;   /* else: every core was in the skip list */
    }
}

/* Per-thread relative throughput, measured once when the thread is pinned.
 * Hybrid cores differ by more than their clock (an E-core issues half the
 * 256-bit VNNI work of a P-core), so clock alone is not a good weight. */
static int g_tw[256];

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static float calib_vec(void) {
    __m256 x = _mm256_set1_ps(1.000001f), c = _mm256_set1_ps(1e-7f);
    __m256 a0 = x, a1 = x, a2 = x, a3 = x, a4 = x, a5 = x, a6 = x, a7 = x;
    for (int i = 0; i < 50000; i++) {
        a0 = _mm256_fmadd_ps(a0, x, c); a1 = _mm256_fmadd_ps(a1, x, c);
        a2 = _mm256_fmadd_ps(a2, x, c); a3 = _mm256_fmadd_ps(a3, x, c);
        a4 = _mm256_fmadd_ps(a4, x, c); a5 = _mm256_fmadd_ps(a5, x, c);
        a6 = _mm256_fmadd_ps(a6, x, c); a7 = _mm256_fmadd_ps(a7, x, c);
    }
    a0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
                       _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
    float out[8];
    _mm256_storeu_ps(out, a0);
    return out[0] + out[3] + out[7];
#else
static float calib_vec(void) { return 1.0f; }
#endif
}

/* Calibrated one thread at a time: measuring while the other workers spin up
 * makes the faster cores look slow and skews every later work split. */
static _Atomic int g_calib_turn;

static void measure_speed(int tid) {
    if (tid < 0 || tid > 255 || g_tw[tid]) return;
    while (atomic_load_explicit(&g_calib_turn, memory_order_acquire) != tid)
        cpu_relax();
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        double dt = e2_gemm_calib();
        if (dt <= 0.0) break;
        if (dt < best) best = dt;   /* a preempted run is not a slow core */
    }
    if (best >= 1e30) {
        /* no packed kernel yet: fall back to the raw instruction rate */
        best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t0 = e2_now_ms();
            volatile float v = calib_vec();
            (void)v;
            double dt = e2_now_ms() - t0;
            if (dt < best) best = dt;
        }
    }
    /* inverse time in micro-units; only the ratios between threads matter */
    g_tw[tid] = best > 0.0 ? (int)(1e6 / best) : 1;
    atomic_store_explicit(&g_calib_turn, tid + 1, memory_order_release);
    if (getenv("E2_DEBUG_TW"))
        fprintf(stderr, "tid %d -> cpu %d speed %d\n", tid,
                g_ncpu > 0 ? g_cpus[tid % g_ncpu] : -1, g_tw[tid]);
}

static void pin_cpu(int tid) {
    build_cpu_list();
    if (g_ncpu <= 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(g_cpus[tid % g_ncpu], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    measure_speed(tid);
}

/* Split [0,n) over nt workers in proportion to their measured speed; a barrier
 * ends at the slowest worker, so equal shares on a hybrid CPU waste the fast
 * ones. */
void e2_split(int n, int tid, int nt, int *lo, int *hi) {
    if (nt <= 1 || tid < 0 || tid >= nt) { *lo = 0; *hi = n; return; }
    long tot = 0, start = 0;
    for (int i = 0; i < nt; i++) {
        int w = (i < 256 && g_tw[i] > 0) ? g_tw[i] : 1;
        tot += w;
        if (i < tid) start += w;
    }
    int w = (tid < 256 && g_tw[tid] > 0) ? g_tw[tid] : 1;
    /* round to the nearest unit: truncating turned a 1% weight difference into
     * a whole extra tile for one thread (3 vs 5 tiles at n=32/nt=8), which cost
     * 15-25% on a uniform CPU. */
    *lo = (int)(((long)n * start + tot / 2) / tot);
    *hi = (int)(((long)n * (start + w) + tot / 2) / tot);
}

static void set_ftz_w(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned csr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;
    __asm__ __volatile__("ldmxcsr %0" :: "m"(csr));
#endif
}

/* Spins before the first nanosleep.  Tunable because the cost of a wakeup
 * depends on the host (a loaded box needs the spin, a quiet one pays for it
 * in power and shared-cache traffic). */
static int g_spin = 20000;
static void spin_init(void) {
    const char *e = getenv("E2_POOL_SPIN");
    if (e) g_spin = atoi(e);
    if (g_spin < 0) g_spin = 0;
}

static void *worker(void *vp) {
    set_ftz_w();
    struct worker_arg *wa = vp;
    struct e2_pool *pool = wa->pool;
    int tid = wa->tid;
    /* Pin and calibrate before the first dispatch, so every split sees the
     * same weights on every call (a weight that appears after the first
     * dispatch would change the row boundaries and the rounding with them). */
    pin_cpu(tid);
    atomic_fetch_add_explicit(&pool->ready, 1, memory_order_release);
    unsigned seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool->gen, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) return NULL;
            if (++spins < g_spin) cpu_relax();
            else { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); }
        }
        if (atomic_load_explicit(&pool->stop, memory_order_acquire)) return NULL;
        seen = atomic_load_explicit(&pool->gen, memory_order_relaxed);
        e2_work_fn fn = pool->fn;
        void *arg = pool->arg;
        fn(arg, tid, pool->n);
        atomic_fetch_add_explicit(&pool->done, 1, memory_order_acq_rel);
    }
}

int e2_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 1;
    /* Unique physical cores, LP cores excluded: SMT siblings only add
     * contention for these kernels and an LP core is a straggler. */
    build_cpu_list();
    int cores = 0;
    for (int i = 0; i < g_ncpu; i++) {
        char path[160];
        int core = -1, pkg = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", g_cpus[i]);
        FILE *f = fopen(path, "r");
        if (!f) { cores = 0; break; }
        if (fscanf(f, "%d", &core) != 1) { fclose(f); cores = 0; break; }
        fclose(f);
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", g_cpus[i]);
        f = fopen(path, "r");
        if (f) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        int dup = 0;
        for (int k = 0; k < i && !dup; k++) {
            int c2 = -1, p2 = 0;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", g_cpus[k]);
            f = fopen(path, "r");
            if (f) { if (fscanf(f, "%d", &c2) != 1) c2 = -1; fclose(f); }
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", g_cpus[k]);
            f = fopen(path, "r");
            if (f) { if (fscanf(f, "%d", &p2) != 1) p2 = 0; fclose(f); }
            if (c2 == core && p2 == pkg) dup = 1;
        }
        if (!dup) cores++;
    }
    if (cores > 0) return cores;
    return (int)n;
}

int e2_pool_init(e2_ctx *ctx, int n) {
    if (n < 1) n = 1;
    struct e2_pool *p = calloc(1, sizeof(*p));
    if (!p) return -1;
    spin_init();
    p->n = n;
    p->th = calloc((size_t)n, sizeof(pthread_t));
    p->wa = calloc((size_t)n, sizeof(*p->wa));
    for (int i = 0; i < n; i++) { p->wa[i].pool = p; p->wa[i].tid = i; }
    pin_cpu(0);
    for (int i = 1; i < n; i++) {
        if (pthread_create(&p->th[i], NULL, worker, &p->wa[i]) != 0) {
            p->n = i;
            break;
        }
    }
    while (atomic_load_explicit(&p->ready, memory_order_acquire) < p->n - 1) {
        struct timespec ts = {0, 100000};
        nanosleep(&ts, NULL);
    }
    ctx->pool = p;
    ctx->nthreads = p->n;
    return 0;
}

void e2_pool_free(e2_ctx *ctx) {
    struct e2_pool *p = ctx->pool;
    if (!p) return;
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
    free(p->th);
    free(p->wa);
    free(p);
    ctx->pool = NULL;
}

void e2_parallel(e2_ctx *ctx, e2_work_fn fn, void *arg) {
    struct e2_pool *p = ctx->pool;
    if (!p || p->n <= 1) {
        fn(arg, 0, 1);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);

    fn(arg, 0, p->n);

    int spins = 0;
    while (atomic_load_explicit(&p->done, memory_order_acquire) < p->n - 1) {
        if (++spins < 20000) cpu_relax();
        else { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); }
    }
}
