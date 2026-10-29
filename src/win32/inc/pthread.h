/* <pthread.h> for a native Windows build — see e2win.h.
 *
 * threads.c only creates and joins workers; every other synchronisation in the engine is C11 atomics
 * plus a park (futex on Linux, WaitOnAddress on Windows, see threads.c). So this is the whole of it.
 */
#ifndef E2W_PTHREAD_H
#define E2W_PTHREAD_H

typedef struct {
    void *h;
} pthread_t;

int pthread_create(pthread_t *t, const void *attr, void *(*fn)(void *), void *arg);
int pthread_join(pthread_t t, void **ret);

#endif /* E2W_PTHREAD_H */
