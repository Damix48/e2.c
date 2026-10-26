/* threads.h — persistent worker pool (see threads.c). */
#ifndef E2_THREADS_H
#define E2_THREADS_H

/* A spin hint for the two-phase regions in text.c/vision.c (a barrier inside a region would
 * otherwise have to yield). */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
static inline void e2_pause(void) { _mm_pause(); }
#else
static inline void e2_pause(void) {}
#endif

typedef struct e2_pool e2_pool;

/* fn is called once per worker; the worker splits its own work by (tid, nthreads). */
typedef void (*e2_task_fn)(void *arg, int tid, int nthreads);

e2_pool *e2_pool_create(int nthreads);
void e2_pool_run(e2_pool *p, e2_task_fn fn, void *arg);
void e2_pool_destroy(e2_pool *p);

int e2_pool_global_init(int nthreads); /* 0 = hardware concurrency */
void e2_pool_global_free(void);
void e2_parallel(e2_task_fn fn, void *arg);
int e2_nthreads(void);

#endif /* E2_THREADS_H */
