/* l0.h — the engine's Level Zero layer: one context on the Intel GPU, one module built from the
 * embedded SPIR-V, cached kernel handles, device memory, and recorded command lists.
 *
 * Why Level Zero and not OpenCL: measured on this iGPU (Arrow Lake-H Arc, NEO), a kernel dispatched
 * through the OpenCL queue costs ~14 us of driver time even when a thousand are queued back to back,
 * while the same launch inside a Level Zero command list costs 0.8 us. The forward pass of one
 * embedding is a few hundred kernels, so with OpenCL the queue alone would cost more than OpenVINO's
 * whole inference. The command list also lets a fixed shape be recorded once and replayed: the graph
 * of an embedding depends on nothing but the sequence length.
 *
 * One context and queue; every submission waits and checks a fresh completion cookie. No images.
 */
#ifndef E2_L0_H
#define E2_L0_H

#include <stddef.h>
#include <stdint.h>

#include <level_zero/ze_api.h>

typedef struct e2_l0 e2_l0;
typedef struct e2_graph e2_graph; /* one recorded command list */

e2_l0 *e2_l0_create(char *err, size_t errn);
void e2_l0_destroy(e2_l0 *g);
/* Execution/output failures retire the context. A pending submission's resources live until exit. */
int e2_l0_ready(e2_l0 *g);
int e2_l0_invalidate(e2_l0 *g, const char *reason);

/* Fixed upper bound per queue wait; a supervisor must also bound blocking driver calls. */
#define E2_L0_TIMEOUT_NS UINT64_C(30000000000)

const char *e2_l0_device_name(e2_l0 *g);
uint32_t e2_l0_eu_count(e2_l0 *g);
uint32_t e2_l0_clock_mhz(e2_l0 *g);
ze_context_handle_t e2_l0_context(e2_l0 *g);

/* Kernel handle by name from the embedded module (created once, cached). NULL on failure. */
ze_kernel_handle_t e2_l0_kernel(e2_l0 *g, const char *name);

/* Device memory (an iGPU's "device" memory is ordinary RAM, but this keeps the intent clear and
 * works unchanged on a discrete card). */
void *e2_l0_alloc(e2_l0 *g, size_t bytes);
void e2_l0_free(e2_l0 *g, void *p);
/* Memory the CPU writes or reads every call (token ids, the output vector) and the kernels also
 * touch, so it must be shared rather than host-only. */
void *e2_l0_alloc_shared(e2_l0 *g, size_t bytes);
/* Synchronous copy into/out of a device allocation. */
int e2_l0_write(e2_l0 *g, void *dev_dst, size_t bytes, const void *src);
int e2_l0_read(e2_l0 *g, const void *dev_src, size_t bytes, void *dst);

/* ---- recording
 * `groups` is the number of work groups, `local` the work group size (like every other GPU API the
 * work is expressed as groups x group size). Kernel arguments are passed as (pointer, size) pairs,
 * the same convention as the OpenCL launcher: a buffer argument is &ptr, sizeof ptr. */
/* `barrier` inserts a barrier after every kernel. It is required, not an option: kernels appended to
 * one command list may overlap, and without the barrier a GEMM reads the previous kernel's output
 * while it is still being written (measured: the GEMM was off by a few percent, which looks exactly
 * like a quantization difference and took a while to pin down). */
e2_graph *e2_l0_graph_begin(e2_l0 *g, int barrier);
int e2_l0_append(e2_l0 *g, e2_graph *gr, const char *kernel, uint32_t gx, uint32_t gy, uint32_t gz,
                 uint32_t lx, uint32_t ly, uint32_t lz, int nargs, ...);
int e2_l0_graph_close(e2_l0 *g, e2_graph *gr);
/* Submit, wait at most 30 s, and verify completion. Failure retires the context; never retry it. */
int e2_l0_graph_run(e2_l0 *g, e2_graph *gr);
void e2_l0_graph_free(e2_l0 *g, e2_graph *gr);

/* E2_GPU_PROFILE=1 makes append() bracket every kernel with device timestamp writes; report()
 * prints the per-kernel-name totals of the last run. Diagnostic only. */
void e2_l0_graph_report(e2_l0 *g, e2_graph *gr);

/* The embedded SPIR-V module (generated from src/gpu/kernels/e2.cl by ocloc at build time). */
const unsigned char *e2_l0_spirv(size_t *bytes);
/* Force a rebuild of every kernel in the module (used after a module reload). */
int e2_l0_reload(e2_l0 *g);

const char *e2_l0_error(e2_l0 *g);
void e2_l0_set_error(e2_l0 *g, const char *fmt, ...);

#endif /* E2_L0_H */
