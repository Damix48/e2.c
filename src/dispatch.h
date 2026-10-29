/* dispatch.h — the one place that decides which attention precision a backend uses.
 *
 * Two layers, per DISPATCH.md:
 *   caps  — what a backend has (int8 GEMM, quantizer, fused attention kernel, fp16 units);
 *   rules — what it should prefer per tower, as {kind, min_keys, below}.
 *
 * Everything here is inert data. Nothing probes a kernel by calling it: HARDWARE.md records that
 * e2_i8_gemm_vnni on the Zen 4 minipc is a STATUS_ILLEGAL_INSTRUCTION, so a backend is identified by
 * the enum e2_cpu_init() selects and never by invoking its kernels. That is also what lets a test
 * assert every row of the matrix on any host.
 */
#ifndef E2_DISPATCH_H
#define E2_DISPATCH_H

typedef enum {
    E2_BK_CPU_SCALAR = 0, /* generic C: no quantizer, so no int8 anywhere */
    E2_BK_CPU_AVX2,       /* AVX2 int8 (vpmaddubsw); AVX-512 without VNNI lands here too */
    E2_BK_CPU_VNNI,       /* AVX-VNNI: 256-bit vpdpbusd */
    E2_BK_CPU_AVX512,     /* AVX512-VNNI: 512-bit vpdpbusd */
    E2_BK_CPU_NEON,       /* aarch64 + dotprod (vdotq_s32) */
    E2_BK_INTEL_GPU,      /* Level Zero */
    E2_BK_HIP,            /* ROCm/HIP */
    E2_BK_COUNT
} e2_backend;

typedef enum { E2_TOWER_TEXT = 0, E2_TOWER_IMAGE, E2_TOWER_AUDIO, E2_TOWER_COUNT } e2_tower;

typedef enum { E2_ATTN_F32 = 0, E2_ATTN_F16, E2_ATTN_I8 } e2_attn_kind;

/* Kernel-generation switch, deliberately a different namespace from precision (§2.3). Only HIP
 * consumes it today: `direct` selects the direct-load int8 attention products, `lds` the LDS-staged
 * ones, and `legacy` the pre-fusion scalar per-row fp32 kernel, which is fp32-only by construction.
 * `auto` is the per-backend default. */
typedef enum {
    E2_ATTN_STAGE_AUTO = 0,
    E2_ATTN_STAGE_DIRECT,
    E2_ATTN_STAGE_LDS,
    E2_ATTN_STAGE_LEGACY
} e2_attn_stage;

e2_attn_stage e2_attn_stage_override(void);

/* What a backend has. No lengths and no numbers in here - those are the rules' business.
 *
 * There is deliberately no ATTN_I8_GENERIC flag: text int8 attention is the ordinary GEMM with
 * quantized activations, so GEMM_I8 + QUANT_I8 already says it. SOFTMAX_Q_I8 is separate because the
 * fused softmax is an optimization, not a prerequisite: vision.c falls back to plain softmax plus
 * e2_i8_quantize when it is missing. */
enum {
    E2_CAP_GEMM_I8 = 1u << 0,       /* int8 weight GEMM */
    E2_CAP_QUANT_I8 = 1u << 1,      /* fp32 -> int8 activation quantizer */
    E2_CAP_SOFTMAX_Q_I8 = 1u << 2,  /* fused softmax + probability quantizer */
    E2_CAP_ATTN_I8_FUSED = 1u << 3, /* packed-operand attention kernel (images) */
    E2_CAP_ATTN_F16 = 1u << 4       /* fp16 matrix units (XMX) */
};

/* A key count no rule thresholds above: asking with this is how a caller says "the preferred kind,
 * with the length gate ignored" - the form an override like E2_ATTN_I8=1 needs. */
#define E2_ATTN_ANY_KEYS (1 << 30)

/* The CPU backend e2_cpu_init() chose. Kept separate from e2_cpu_name(), which reports the fp32 vtable
 * and returns "scalar" on aarch64 while NEON int8 runs. */
void e2_dispatch_set_cpu_backend(e2_backend bk);
e2_backend e2_cpu_backend(void);

/* The capability bitmask above. Asserted by tests/test_dispatch.c; the profile line in phase 5 is the
 * intended second reader. */
unsigned e2_backend_caps(e2_backend bk);

/* Runtime overrides. E2_ATTN_TEXT and E2_ATTN_IMAGE take auto|f32|f16|i8 and win over the table; the
 * old names (E2_ATTN_I8=0/1, E2_VATTN_OFF) are accepted as deprecated aliases only where the new one
 * said nothing. E2_ATTN_AUDIO is deliberately not parsed: audio has no selectable path (§1.4 of
 * DISPATCH.md), and a variable that can never change the answer is worse than none.
 *
 * The environment is read once, from e2_cpu_init(); use e2_attn_override() rather than setenv to flip
 * a decision later in the process. */
void e2_attn_env_reload(void);
/* Force one tower's kind without the environment; -1 (or an out-of-range value) restores the table. */
void e2_attn_override(e2_tower tower, int kind);

/* "f32", "f16", "i8" - for the E2_PROFILE decision line and for test failure messages. */
const char *e2_attn_kind_name(e2_attn_kind kind);

/* Length threshold for (backend, tower); 0 means no length gate. */
int e2_attn_min_keys(e2_backend bk, e2_tower tower);

/* The one query. Picks the rule for (backend, tower), applies its length gate, and degrades
 * I8 -> F16 -> F32 when the backend lacks the capability the rule asked for. Callers keep their own
 * overrides (E2_ATTN_I8, E2_VATTN_OFF) and shape-dependent fallbacks (HIP's pack region). */
e2_attn_kind e2_attn_pick(e2_backend bk, e2_tower tower, int n_keys);

#endif /* E2_DISPATCH_H */
