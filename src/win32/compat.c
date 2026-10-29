/* compat.c — the Win32 side of the port whose declarations are in src/win32/inc.
 *
 * Everything here is a mechanical mapping onto the Win32 API; there is no engine logic and nothing
 * here runs on Linux or aarch64:
 *
 *   mmap/munmap      CreateFileMapping(PAGE_READONLY) + MapViewOfFile, closed immediately (the view
 *                    keeps the section alive). The engine only ever maps read-only.
 *   pthread_create   _beginthreadex through a thunk, so no function-pointer cast is involved.
 *   sched_yield      SwitchToThread.
 *   mkstemp          _open(_O_CREAT|_O_EXCL) over a name derived from the counter, tick count and pid.
 *   clock_gettime    QueryPerformanceCounter, which is the monotonic source the engine wants.
 *   setenv/unsetenv  _putenv_s.
 *
 * The one thing that is NOT a straight mapping is the file size: st.c and q8cache.c read it from
 * fstat, and UCRT's struct stat has a 32-bit st_size, so those two call sites take the _fstat64 form
 * under #ifdef _WIN32 rather than being truncated at 2 GB (the checkpoint is 1.4 GB).
 */
#ifdef _WIN32

#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> /* _S_IREAD */
#include <windows.h>

#include "pthread.h"
#include "sys/mman.h"

void *e2w_mmap_ro(int fd, size_t len) {
    const intptr_t os = _get_osfhandle(fd);
    if (os == -1 || len == 0) return MAP_FAILED;
    HANDLE map = CreateFileMappingW((HANDLE)os, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!map) return MAP_FAILED;
    void *view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(map); /* the mapped view holds the section open */
    return view ? view : MAP_FAILED;
}

int e2w_munmap(void *addr, size_t len) {
    (void)len;
    return (addr && UnmapViewOfFile(addr)) ? 0 : -1;
}

int e2w_sched_yield(void) {
    SwitchToThread();
    return 0;
}

int mkstemp(char *tmpl) {
    static unsigned counter;
    const size_t n = strlen(tmpl);
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX")) {
        errno = EINVAL;
        return -1;
    }
    for (unsigned attempt = 0; attempt < 1000; attempt++) {
        unsigned v = (unsigned)GetTickCount() ^ (++counter * 7919u) ^ ((unsigned)_getpid() << 16);
        for (int i = 0; i < 6; i++) {
            tmpl[n - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[v % 36];
            v /= 36;
        }
        const int fd = _open(tmpl, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
        if (fd >= 0) return fd;
        if (errno != EEXIST) return -1;
    }
    return -1;
}

char *mkdtemp(char *tmpl) {
    static unsigned counter;
    const size_t n = strlen(tmpl);
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX")) {
        errno = EINVAL;
        return 0;
    }
    for (unsigned attempt = 0; attempt < 1000; attempt++) {
        unsigned v = (unsigned)GetTickCount() ^ (++counter * 104729u) ^ ((unsigned)_getpid() << 16);
        for (int i = 0; i < 6; i++) {
            tmpl[n - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[v % 36];
            v /= 36;
        }
        if (_mkdir(tmpl) == 0) return tmpl;
        if (errno != EEXIST) return 0;
    }
    return 0;
}

int truncate(const char *path, long len) {
    const int fd = _open(path, _O_RDWR | _O_BINARY);
    if (fd < 0) return -1;
    const int rc = _chsize_s(fd, (long long)len) ? -1 : 0;
    _close(fd);
    return rc;
}

void e2w_usleep(unsigned usec) {
    Sleep(usec < 1000 ? 1 : usec / 1000);
}

int clock_gettime(int clk, struct timespec *ts) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    (void)clk;
    if (!freq.QuadPart && !QueryPerformanceFrequency(&freq)) return -1;
    QueryPerformanceCounter(&now);
    ts->tv_sec = (time_t)(now.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((now.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}

long sysconf(int name) {
    (void)name;
    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len) &&
        GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        char *buf = malloc(len);
        if (buf) {
            long cores = 0;
            if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                 (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf,
                                                 &len)) {
                for (DWORD off = 0; off < len;) {
                    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e =
                        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
                    cores++;
                    off += e->Size;
                }
            }
            free(buf);
            if (cores > 0) return cores;
        }
    }
    return (long)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
}

int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value) ? -1 : 0;
}

int unsetenv(const char *name) {
    return _putenv_s(name, "") ? -1 : 0;
}

struct spawn {
    void *(*fn)(void *);
    void *arg;
};

static unsigned __stdcall spawn_trampoline(void *p) {
    struct spawn *s = p;
    void *(*fn)(void *) = s->fn;
    void *arg = s->arg;
    free(s);
    fn(arg);
    return 0;
}

int pthread_create(pthread_t *t, const void *attr, void *(*fn)(void *), void *arg) {
    (void)attr;
    struct spawn *s = malloc(sizeof *s);
    if (!s) return -1;
    s->fn = fn;
    s->arg = arg;
    t->h = (void *)_beginthreadex(NULL, 0, spawn_trampoline, s, 0, NULL);
    if (!t->h) {
        free(s);
        return -1;
    }
    return 0;
}

int pthread_join(pthread_t t, void **ret) {
    (void)ret;
    if (!t.h) return -1;
    WaitForSingleObject((HANDLE)t.h, INFINITE);
    CloseHandle((HANDLE)t.h);
    return 0;
}

#endif /* _WIN32 */
