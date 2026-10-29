/* e2win.h — force-included (-include) into every translation unit of a native Windows build.
 *
 * This file and the headers next to it (unistd.h, sys/mman.h, pthread.h) are the whole Windows port:
 * the engine is POSIX by construction, so rather than sprinkle #ifdef _WIN32 through it, the build
 * puts src/win32/inc on the include path and the sources keep including the names they always did.
 * The implementations live in src/win32/compat.c; nothing here is used on Linux or aarch64.
 *
 * UCRT keeps the POSIX file API (open/close/read/write/unlink/fdopen) as long as
 * _CRT_NONSTDC_NO_DEPRECATE is defined, which the build does. What it does not have is the handful
 * of declarations below.
 */
#ifndef E2W_FORCE_H
#define E2W_FORCE_H

#ifdef _WIN32

#include <direct.h> /* _rmdir */
#include <time.h>

/* UCRT has no clock_gettime. e2win maps CLOCK_MONOTONIC onto QueryPerformanceCounter, which is the
 * same monotonic source the engine wants on Linux (clock_gettime is ~19 call sites: the profiler,
 * the benches and e2_test's timing). */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
int clock_gettime(int clk, struct timespec *ts);

/* POSIX environment mutation, used by the tests to force the packed-cache and bf16 paths off. */
int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);

/* threads.c sizes the pool from sysconf(_SC_NPROCESSORS_ONLN). On Windows this reports the number of
 * *physical* cores, not logical ones: the pool deliberately drops SMT siblings, and its own comment
 * records that a second logical thread per core measured 1.7x slower. */
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 1
#endif
long sysconf(int name);

/* The rest of what the tests use and UCRT does not spell the same way. */
char *mkdtemp(char *tmpl);
/* test_st truncates a copy of the header to check the parser's bounds. */
int truncate(const char *path, long len);
void e2w_usleep(unsigned usec);
#define usleep(us) e2w_usleep(us)
#define rmdir _rmdir
#define strtok_r strtok_s

#endif /* _WIN32 */
#endif /* E2W_FORCE_H */
