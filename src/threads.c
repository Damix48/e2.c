/* threads.c — minimal persistent worker pool.
 *
 * The engine calls e2_parallel() once per parallel region (a GEMM column range, a group of attention
 * rows). Workers are created once at load time: spawning threads per region would cost more than the
 * work in regions this small (~170 regions per text embedding).
 *
 * Synchronisation is atomics plus a short spin, then a futex sleep. The mutex+condvar version this
 * replaced cost 31 us for a *no-op* region with 16 threads: every worker had to take the same mutex
 * twice per region (once to read the generation, once to report completion) and the caller blocked
 * on a condition variable guarded by it. On a 16-core box that was the single largest cost in a
 * short-text embedding. A spin-then-futex barrier is ~1 us and still idles at zero CPU.
 */
#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#define E2_PAUSE() _mm_pause()
#else
#define E2_PAUSE() ((void)0)
#endif

/* Pause iterations before falling back to a futex. ~50 us of spinning: consecutive regions are
 * back-to-back inside one embedding, so the spin almost always wins. */
#define E2_SPIN 1000000

struct worker_arg {
    struct e2_pool *pool;
    int tid;
};

struct e2_pool {
    pthread_t *threads;
    struct worker_arg *wargs;
    int nthreads;
    e2_task_fn fn;
    void *arg;
    _Atomic unsigned generation;
    _Atomic int nfinished;
    _Atomic int shutdown;
};

static e2_pool *g_pool; /* one model, one pool, for the process lifetime */

#if defined(__linux__)
static void futex_wait(_Atomic unsigned *addr, unsigned expected) {
    syscall(SYS_futex, (unsigned *)addr, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
}
static void futex_wake(_Atomic unsigned *addr) {
    syscall(SYS_futex, (unsigned *)addr, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
}
#else
static void futex_wait(_Atomic unsigned *addr, unsigned expected) {
    (void)addr; (void)expected; sched_yield();
}
static void futex_wake(_Atomic unsigned *addr) { (void)addr; }
#endif

static void *worker(void *p) {
    struct worker_arg *wa = p;
    e2_pool *pool = wa->pool;
    const int tid = wa->tid;
    unsigned seen = 0; /* not the current generation: a worker that starts late must not skip the
                        * region the caller is already blocked on */
    for (;;) {
        unsigned g = seen;
        for (int spins = 0;; spins++) {
            g = atomic_load_explicit(&pool->generation, memory_order_acquire);
            if (atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) return NULL;
            if (g != seen) break;
            if (spins < E2_SPIN) E2_PAUSE();
            else futex_wait(&pool->generation, seen);
        }
        seen = g;
        pool->fn(pool->arg, tid, pool->nthreads);
        atomic_fetch_add_explicit(&pool->nfinished, 1, memory_order_release);
    }
}

e2_pool *e2_pool_create(int nthreads) {
    if (nthreads <= 1) return NULL;
    e2_pool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->nthreads = nthreads;
    p->threads = calloc((size_t)(nthreads - 1), sizeof(pthread_t));
    p->wargs = calloc((size_t)(nthreads - 1), sizeof(struct worker_arg));
    atomic_init(&p->generation, 0);
    atomic_init(&p->nfinished, 0);
    atomic_init(&p->shutdown, 0);
    for (int i = 0; i < nthreads - 1; i++) {
        p->wargs[i].pool = p;
        p->wargs[i].tid = i + 1;
        if (pthread_create(&p->threads[i], NULL, worker, &p->wargs[i])) { p->nthreads = i + 1; break; }
    }
    return p;
}

void e2_pool_run(e2_pool *p, e2_task_fn fn, void *arg) {
    if (!p) { fn(arg, 0, 1); return; }
    p->fn = fn;
    p->arg = arg;
    atomic_store_explicit(&p->nfinished, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    futex_wake(&p->generation);

    fn(arg, 0, p->nthreads); /* the calling thread takes part */

    while (atomic_load_explicit(&p->nfinished, memory_order_acquire) < p->nthreads - 1) {
        for (int spins = 0; spins < E2_SPIN; spins++) {
            if (atomic_load_explicit(&p->nfinished, memory_order_acquire) >= p->nthreads - 1) return;
            E2_PAUSE();
        }
        sched_yield();
    }
}

void e2_pool_destroy(e2_pool *p) {
    if (!p) return;
    atomic_store_explicit(&p->shutdown, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    futex_wake(&p->generation);
    for (int i = 0; i < p->nthreads - 1; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->wargs);
    free(p);
}

int e2_nthreads(void) { return g_pool ? g_pool->nthreads : 1; }

int e2_pool_init(int nthreads) { return e2_pool_global_init(nthreads); }

/* Hybrid CPUs (P-cores, E-cores, low-power E-cores) publish a peak frequency per core in sysfs. The
 * pool keeps only the tiers within 75% of the fastest: a low-power core is ~2x slower, and since
 * every parallel region ends at a barrier the slowest thread sets the latency of all ~170 regions in
 * an embedding. On the reference machine including the two low-power cores costs 40-70% of the
 * document throughput. SMT siblings are dropped as well: the engine is bandwidth- and uop-bound, so
 * a second logical thread on the same core (AMD "16 cores" with 8 physical) measured 1.7x slower on
 * a document. Falls back to the online CPU count wherever the files are missing. */
static int primary_thread(int i) {
    char path[160];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    char buf[64];
    int first = i;
    if (fgets(buf, sizeof buf, f)) first = atoi(buf); /* "0,8" / "0-1": first id is the representative */
    fclose(f);
    return first == i;
}

static int default_threads(void) {
    const int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    enum { MAXC = 256 };
    long freq[MAXC];
    if (n > MAXC) return n;
    long top = 0;
    for (int i = 0; i < n; i++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *f = fopen(path, "r");
        if (!f) return n;
        if (fscanf(f, "%ld", &freq[i]) != 1) { fclose(f); return n; }
        fclose(f);
        if (freq[i] > top) top = freq[i];
    }
    int keep = 0;
    for (int i = 0; i < n; i++)
        if (primary_thread(i) && freq[i] * 4 >= top * 3) keep++;
    return keep > 0 ? keep : n;
}

int e2_pool_global_init(int nthreads) {
    if (g_pool) return g_pool->nthreads;
    const char *env = getenv("E2_THREADS"); /* E2_THREADS=1 disables the pool (profiling) */
    if (env) nthreads = atoi(env);
    if (nthreads <= 0) nthreads = default_threads();
    if (nthreads > 64) nthreads = 64;
    g_pool = e2_pool_create(nthreads);
    return g_pool ? g_pool->nthreads : 1;
}

void e2_pool_global_free(void) {
    e2_pool_destroy(g_pool);
    g_pool = NULL;
}

void e2_parallel(e2_task_fn fn, void *arg) { e2_pool_run(g_pool, fn, arg); }
