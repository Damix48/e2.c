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

void e2_arena_stat(const e2_arena *a, long *blocks, size_t *bytes) {
    long n = 0; size_t b = 0;
    for (e2_arena_block *x = a->head; x; x = x->next) { n++; b += x->size; }
    *blocks = n; *bytes = b;
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

/* Pin thread tid to a physical core: one thread per core beats SMT sharing
 * for these throughput-bound kernels. */
static void pin_cpu(int tid) {
    cpu_set_t set;
    CPU_ZERO(&set);
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) return;
    /* CPU ids of the first thread of each core, in order */
    int want = tid, placed = 0;
    for (int c = 0; c < ncpu && placed <= want; c++) {
        char path[160];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *f = fopen(path, "r");
        if (!f) return;
        int first = -1;
        if (fscanf(f, "%d", &first) != 1) first = c;
        fclose(f);
        if (first == c) {
            if (placed == want) { CPU_SET(c, &set); pthread_setaffinity_np(pthread_self(), sizeof(set), &set); return; }
            placed++;
        }
    }
}

static void set_ftz_w(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned csr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;
    __asm__ __volatile__("ldmxcsr %0" :: "m"(csr));
#endif
}

static void *worker(void *vp) {
    set_ftz_w();
    struct worker_arg *wa = vp;
    struct e2_pool *pool = wa->pool;
    int tid = wa->tid;
    unsigned seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool->gen, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&pool->stop, memory_order_relaxed)) return NULL;
            if (++spins < 20000) cpu_relax();
            else { struct timespec ts = {0, 50000}; nanosleep(&ts, NULL); }
        }
        if (atomic_load_explicit(&pool->stop, memory_order_acquire)) return NULL;
        seen = atomic_load_explicit(&pool->gen, memory_order_relaxed);
        if (seen == 1) pin_cpu(tid);
        e2_work_fn fn = pool->fn;
        void *arg = pool->arg;
        fn(arg, tid, pool->n);
        atomic_fetch_add_explicit(&pool->done, 1, memory_order_acq_rel);
    }
}

int e2_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 1;
    if (n > 256) n = 1;
    /* Count unique physical cores when sysfs is available: SMT siblings only
     * add contention for these bandwidth-bound GEMMs. */
    unsigned long long keys[256];
    int cores = 0;
    for (int i = 0; i < (int)n && cores < 256; i++) {
        char path[160];
        int core = -1, pkg = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE *f = fopen(path, "r");
        if (!f) { cores = 0; break; }
        if (fscanf(f, "%d", &core) != 1) { fclose(f); cores = 0; break; }
        fclose(f);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = fopen(path, "r");
        if (f) { if (fscanf(f, "%d", &pkg) != 1) pkg = 0; fclose(f); }
        unsigned long long key = ((unsigned long long)(pkg + 1) << 32) | (unsigned)core;
        int dup = 0;
        for (int k = 0; k < cores; k++) if (keys[k] == key) { dup = 1; break; }
        if (!dup) keys[cores++] = key;
    }
    if (cores > 0) return cores;
    return (int)n;
}

int e2_pool_init(e2_ctx *ctx, int n) {
    if (n < 1) n = 1;
    struct e2_pool *p = calloc(1, sizeof(*p));
    if (!p) return -1;
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
