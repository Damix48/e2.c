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
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "threads.h"
#include <sched.h>

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
#elif defined(_WIN32)
#include <windows.h>
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
#elif defined(_WIN32)
/* A lock and a condition variable, which is the Win32 spelling of "sleep while *addr == expected".
 * WaitOnAddress would be the literal translation but it is not in every Windows SDK's import
 * library, and this costs nothing measurable: there is one pool for the process lifetime, so one
 * condition variable is all that is needed, and the callers re-test the generation in a loop, so a
 * spurious wake is harmless. Parking matters - the pool parks ~170 times per embedding. */
static SRWLOCK g_park_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE g_park_cv = CONDITION_VARIABLE_INIT;

static void futex_wait(_Atomic unsigned *addr, unsigned expected) {
    AcquireSRWLockExclusive(&g_park_lock);
    while (atomic_load_explicit(addr, memory_order_acquire) == expected)
        SleepConditionVariableSRW(&g_park_cv, &g_park_lock, INFINITE, 0);
    ReleaseSRWLockExclusive(&g_park_lock);
}
static void futex_wake(_Atomic unsigned *addr) {
    (void)addr; /* one pool, one condition variable */
    AcquireSRWLockExclusive(&g_park_lock);
    WakeAllConditionVariable(&g_park_cv);
    ReleaseSRWLockExclusive(&g_park_lock);
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
    if (!p->threads || !p->wargs) {
        free(p->threads);
        free(p->wargs);
        free(p);
        return NULL; /* the global caller safely falls back to serial execution */
    }
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

/* cgroup v2 quotas may be fractional CPU equivalents; round up, bounded by affinity. */
#if defined(__linux__)
static int quota_threads(FILE *f) {
    long long quota, period;
    if (fscanf(f, "%lld %lld", &quota, &period) != 2 || quota <= 0 || period <= 0) return 64;
    long long n = quota / period + (quota % period != 0);
    return n < 64 ? (int)n : 64;
}
static int cpu_budget(void) {
    int limit = 64;
    cpu_set_t allowed;
    if (!sched_getaffinity(0, sizeof allowed, &allowed)) {
        int n = CPU_COUNT(&allowed);
        if (n > 0 && n < limit) limit = n;
    }
    char path[2048] = "/sys/fs/cgroup", line[2048];
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "0::/", 4) && !strstr(line + 3, "..")) {
                line[strcspn(line, "\n")] = 0;
                if (strlen(line + 3) + strlen(path) < sizeof path)
                    strcat(path, line + 3);
                break;
            }
        }
        fclose(f);
    }
    const size_t root = strlen("/sys/fs/cgroup");
    for (;;) {
        char file[2080];
        snprintf(file, sizeof file, "%s/cpu.max", path);
        f = fopen(file, "r");
        if (f) { int n = quota_threads(f); if (n < limit) limit = n; fclose(f); }
        size_t n = strlen(path);
        if (n <= root) break;
        char *slash = strrchr(path, '/');
        if (!slash || (size_t)(slash - path) < root) break;
        *slash = 0;
    }
    return limit;
}
#endif

int e2_pool_global_init(int nthreads) {
    if (g_pool) return g_pool->nthreads;
    const char *env = getenv("E2_THREADS"); /* E2_THREADS=1 disables the pool (profiling) */
    if (env) nthreads = atoi(env);
    if (nthreads <= 0) {
        nthreads = default_threads();
#if defined(__linux__)
        int budget = cpu_budget();
        if (nthreads > budget) nthreads = budget;
#endif
    }
    if (nthreads > 64) nthreads = 64;
    g_pool = e2_pool_create(nthreads);
    return g_pool ? g_pool->nthreads : 1;
}

void e2_pool_global_free(void) {
    e2_pool_destroy(g_pool);
    g_pool = NULL;
}

void e2_parallel(e2_task_fn fn, void *arg) { e2_pool_run(g_pool, fn, arg); }
