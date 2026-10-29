/* <sys/mman.h> for a native Windows build — see e2win.h.
 *
 * Only the read-only private mapping the safetensors view and the packed-cache reader use is
 * implemented: PROT_READ | MAP_PRIVATE over CreateFileMapping/MapViewOfFile. The engine never asks for
 * anything else on the CPU path, and the GPU backends are not built on Windows.
 */
#ifndef E2W_SYS_MMAN_H
#define E2W_SYS_MMAN_H

#include <stddef.h>

#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED ((void *)-1)

void *e2w_mmap_ro(int fd, size_t len);
int e2w_munmap(void *addr, size_t len);

#define mmap(addr, len, prot, flags, fd, off) e2w_mmap_ro((fd), (len))
#define munmap(addr, len) e2w_munmap((addr), (len))

#endif /* E2W_SYS_MMAN_H */
