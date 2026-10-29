/* test_dispatch.c — DISPATCH.md's matrix, asserted as data on every host.
 *
 * The table is inert, so asking about a backend this machine does not have is a plain lookup rather
 * than a kernel call. That is deliberate: HARDWARE.md records that calling the AVX-VNNI kernel on the
 * Zen 4 minipc is a STATUS_ILLEGAL_INSTRUCTION, so a test that "tries" each backend would crash
 * instead of failing softly. This one only reads the rules.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "dispatch.h"
#include "int8.h"
#include "kernels.h"

static const char *bk_name(e2_backend bk) {
    switch (bk) {
    case E2_BK_CPU_SCALAR: return "cpu/scalar";
    case E2_BK_CPU_AVX2: return "cpu/avx2";
    case E2_BK_CPU_VNNI: return "cpu/vnni";
    case E2_BK_CPU_AVX512: return "cpu/avx512";
    case E2_BK_CPU_NEON: return "cpu/neon";
    case E2_BK_INTEL_GPU: return "intel-gpu";
    case E2_BK_HIP: return "hip";
    default: return "?";
    }
}

static const char *tower_name(e2_tower t) {
    switch (t) {
    case E2_TOWER_TEXT: return "text";
    case E2_TOWER_IMAGE: return "image";
    case E2_TOWER_AUDIO: return "audio";
    default: return "?";
    }
}

static int g_fail;

static void expect(e2_backend bk, e2_tower t, int keys, e2_attn_kind want) {
    const e2_attn_kind got = e2_attn_pick(bk, t, keys);
    if (got != want) {
        fprintf(stderr, "FAIL %s/%s at %d keys: got %s, want %s\n", bk_name(bk), tower_name(t), keys,
                e2_attn_kind_name(got), e2_attn_kind_name(want));
        g_fail++;
    }
}

int main(void) {
    /* The matrix below is asserted with no override in play, whatever the caller's environment is. */
    unsetenv("E2_ATTN_TEXT");
    unsetenv("E2_ATTN_IMAGE");
    unsetenv("E2_ATTN_I8");
    unsetenv("E2_VATTN_OFF");
    e2_attn_env_reload();

    static const e2_backend k_cpu_simd[] = { E2_BK_CPU_AVX2, E2_BK_CPU_VNNI, E2_BK_CPU_AVX512,
                                             E2_BK_CPU_NEON };
    const int n_cpu_simd = (int)(sizeof k_cpu_simd / sizeof k_cpu_simd[0]);

    /* ---- text: the packer amortises at 128 keys on every SIMD CPU ---------------------------- */
    expect(E2_BK_CPU_SCALAR, E2_TOWER_TEXT, 0, E2_ATTN_F32);
    /* Portable C has the GEMM and the quantizer but no rule row, so the table keeps it on fp32;
     * E2_ATTN_TEXT=i8 can still force it (checked further down). */
    expect(E2_BK_CPU_SCALAR, E2_TOWER_TEXT, 2048, E2_ATTN_F32);
    for (int i = 0; i < n_cpu_simd; i++) {
        expect(k_cpu_simd[i], E2_TOWER_TEXT, 0, E2_ATTN_F32);
        expect(k_cpu_simd[i], E2_TOWER_TEXT, 63, E2_ATTN_F32);
        expect(k_cpu_simd[i], E2_TOWER_TEXT, 127, E2_ATTN_F32); /* just under the threshold */
        expect(k_cpu_simd[i], E2_TOWER_TEXT, 128, E2_ATTN_I8);  /* exactly at it */
        expect(k_cpu_simd[i], E2_TOWER_TEXT, 512, E2_ATTN_I8);
        expect(k_cpu_simd[i], E2_TOWER_TEXT, E2_ATTN_ANY_KEYS, E2_ATTN_I8);
    }

    /* Intel GPU text: int8 from 512 up, and fp16 *below* it - a preference, not a degradation. */
    expect(E2_BK_INTEL_GPU, E2_TOWER_TEXT, 0, E2_ATTN_F16);
    expect(E2_BK_INTEL_GPU, E2_TOWER_TEXT, 511, E2_ATTN_F16);
    expect(E2_BK_INTEL_GPU, E2_TOWER_TEXT, 512, E2_ATTN_I8);
    expect(E2_BK_INTEL_GPU, E2_TOWER_TEXT, 2048, E2_ATTN_I8);

    /* HIP text: 64 keys. */
    expect(E2_BK_HIP, E2_TOWER_TEXT, 0, E2_ATTN_F32);
    expect(E2_BK_HIP, E2_TOWER_TEXT, 63, E2_ATTN_F32);
    expect(E2_BK_HIP, E2_TOWER_TEXT, 64, E2_ATTN_I8);
    expect(E2_BK_HIP, E2_TOWER_TEXT, 2048, E2_ATTN_I8);

    /* ---- image: no length gate; the fused kernel is the only gate ---------------------------- */
    for (int i = 0; i < n_cpu_simd; i++) {
        /* AVX2 asks for int8 like everyone else and gets fp32 because it has no fused kernel.
         * That split - rule says I8, cap says no - is the whole point of the two layers. */
        const e2_attn_kind want = k_cpu_simd[i] == E2_BK_CPU_AVX2 ? E2_ATTN_F32 : E2_ATTN_I8;
        expect(k_cpu_simd[i], E2_TOWER_IMAGE, 0, want);
        expect(k_cpu_simd[i], E2_TOWER_IMAGE, 2520, want);
    }
    expect(E2_BK_CPU_SCALAR, E2_TOWER_IMAGE, 2520, E2_ATTN_F32);
    expect(E2_BK_INTEL_GPU, E2_TOWER_IMAGE, 2520, E2_ATTN_I8);
    expect(E2_BK_HIP, E2_TOWER_IMAGE, 2520, E2_ATTN_I8);

    /* ---- audio: fp32 everywhere, at any length --------------------------------------------- */
    for (int bk = 0; bk < E2_BK_COUNT; bk++) {
        expect((e2_backend)bk, E2_TOWER_AUDIO, 0, E2_ATTN_F32);
        expect((e2_backend)bk, E2_TOWER_AUDIO, 750, E2_ATTN_F32);
        expect((e2_backend)bk, E2_TOWER_AUDIO, 2048, E2_ATTN_F32);
    }

    /* ---- thresholds are named per (backend, tower), never at a call site -------------------- */
    assert(e2_attn_min_keys(E2_BK_CPU_VNNI, E2_TOWER_TEXT) == 128);
    assert(e2_attn_min_keys(E2_BK_CPU_AVX2, E2_TOWER_TEXT) == 128);
    assert(e2_attn_min_keys(E2_BK_CPU_NEON, E2_TOWER_TEXT) == 128);
    assert(e2_attn_min_keys(E2_BK_INTEL_GPU, E2_TOWER_TEXT) == 512);
    assert(e2_attn_min_keys(E2_BK_HIP, E2_TOWER_TEXT) == 64);
    assert(e2_attn_min_keys(E2_BK_CPU_SCALAR, E2_TOWER_TEXT) == 0);
    for (int bk = 0; bk < E2_BK_COUNT; bk++) {
        assert(e2_attn_min_keys((e2_backend)bk, E2_TOWER_IMAGE) == 0);
        assert(e2_attn_min_keys((e2_backend)bk, E2_TOWER_AUDIO) == 0);
    }

    /* ---- the capability bitmask is the other half of the split ------------------------------ */
    /* the portable backend now runs the int8 weights: GEMM + quantizer, but no fused attention
     * kernel and no fused softmax, so the image and audio towers stay fp32 by capability. */
    assert((e2_backend_caps(E2_BK_CPU_SCALAR) & (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8)) ==
           (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8));
    assert(!(e2_backend_caps(E2_BK_CPU_SCALAR) & E2_CAP_ATTN_I8_FUSED));
    assert(!(e2_backend_caps(E2_BK_CPU_SCALAR) & E2_CAP_SOFTMAX_Q_I8));
    for (int i = 0; i < n_cpu_simd; i++) {
        const unsigned caps = e2_backend_caps(k_cpu_simd[i]);
        assert(caps & (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8 | E2_CAP_SOFTMAX_Q_I8));
        /* AVX2 has the quantizer and the fused softmax and still no fused attention kernel; every
         * other SIMD CPU has all four. No CPU has fp16 matrix units. */
        if (k_cpu_simd[i] == E2_BK_CPU_AVX2) assert(!(caps & E2_CAP_ATTN_I8_FUSED));
        else assert(caps & E2_CAP_ATTN_I8_FUSED);
        assert(!(caps & E2_CAP_ATTN_F16));
    }
    assert(e2_backend_caps(E2_BK_INTEL_GPU) & E2_CAP_ATTN_I8_FUSED);
    assert(e2_backend_caps(E2_BK_INTEL_GPU) & E2_CAP_ATTN_F16);
    assert(e2_backend_caps(E2_BK_HIP) & E2_CAP_ATTN_I8_FUSED);
    assert(!(e2_backend_caps(E2_BK_HIP) & E2_CAP_ATTN_F16));
    assert(e2_backend_caps((e2_backend)99) == 0);
    assert(E2_ATTN_ANY_KEYS >= 2048); /* above every threshold in the table */

    /* ---- the active CPU backend is capability id, and it round-trips ------------------------- */
    const e2_backend saved = e2_cpu_backend();
    e2_dispatch_set_cpu_backend(E2_BK_CPU_NEON);
    assert(e2_cpu_backend() == E2_BK_CPU_NEON);
    e2_dispatch_set_cpu_backend(E2_BK_CPU_SCALAR);
    assert(e2_cpu_backend() == E2_BK_CPU_SCALAR);
    e2_dispatch_set_cpu_backend((e2_backend)99); /* invalid: falls back, never indexes off the end */
    assert(e2_cpu_backend() == E2_BK_CPU_SCALAR);
    e2_dispatch_set_cpu_backend(saved);

    /* ---- overrides: new names, deprecated aliases, precedence, typos ------------------------ */
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 2048, E2_ATTN_I8); /* no override: the table */

    setenv("E2_ATTN_TEXT", "i8", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 0, E2_ATTN_I8);    /* the override drops the length gate */
    expect(E2_BK_CPU_SCALAR, E2_TOWER_TEXT, 0, E2_ATTN_I8); /* the portable kernel can do it */

    setenv("E2_ATTN_TEXT", "f32", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 2048, E2_ATTN_F32);

    setenv("E2_ATTN_TEXT", "auto", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 2048, E2_ATTN_I8);
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 0, E2_ATTN_F32);

    /* the new name wins over the deprecated one, so a script can migrate one variable at a time */
    setenv("E2_ATTN_I8", "0", 1);
    setenv("E2_ATTN_TEXT", "i8", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 0, E2_ATTN_I8);

    /* ...and the deprecated names still work on their own */
    unsetenv("E2_ATTN_TEXT");
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 2048, E2_ATTN_F32); /* E2_ATTN_I8=0 */
    setenv("E2_ATTN_I8", "1", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 0, E2_ATTN_I8); /* E2_ATTN_I8=1 */
    setenv("E2_VATTN_OFF", "1", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_IMAGE, 2520, E2_ATTN_F32); /* E2_VATTN_OFF */

    /* E2_ATTN_AUDIO is deliberately not parsed: nothing can be selected for audio */
    setenv("E2_ATTN_AUDIO", "i8", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_AUDIO, 2048, E2_ATTN_F32);

    /* an in-process override beats the environment, and -1 restores the table */
    setenv("E2_ATTN_IMAGE", "i8", 1);
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_IMAGE, 2520, E2_ATTN_I8);
    e2_attn_override(E2_TOWER_IMAGE, E2_ATTN_F32);
    expect(E2_BK_CPU_VNNI, E2_TOWER_IMAGE, 2520, E2_ATTN_F32);
    e2_attn_override(E2_TOWER_IMAGE, -1);
    expect(E2_BK_CPU_VNNI, E2_TOWER_IMAGE, 2520, E2_ATTN_I8);

    /* a typo is ignored with a warning, not treated as a kind */
    setenv("E2_ATTN_TEXT", "i8x", 1);
    unsetenv("E2_ATTN_I8");
    e2_attn_env_reload();
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 2048, E2_ATTN_I8); /* back to the table */
    expect(E2_BK_CPU_VNNI, E2_TOWER_TEXT, 0, E2_ATTN_F32);

    unsetenv("E2_ATTN_TEXT");
    unsetenv("E2_ATTN_IMAGE");
    unsetenv("E2_ATTN_AUDIO");
    unsetenv("E2_VATTN_OFF");
    e2_attn_env_reload();

    /* ---- E2_ATTN_STAGE: a generation switch, not a precision one ------------------------------ */
    unsetenv("E2_ATTN_STAGE");
    unsetenv("E2_HIP_ATTN");
    unsetenv("E2_HIP_ATTN_I8");
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_AUTO);
    assert(e2_attn_min_keys(E2_BK_HIP, E2_TOWER_TEXT) == 64); /* the table's gate */

    setenv("E2_ATTN_STAGE", "direct", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_DIRECT);
    setenv("E2_ATTN_STAGE", "lds", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_LDS);
    setenv("E2_ATTN_STAGE", "legacy", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_LEGACY);
    setenv("E2_ATTN_STAGE", "bogus", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_AUTO); /* typo ignored with a warning */

    /* the old HIP names map onto it, including "old/lds meant no int8" */
    unsetenv("E2_ATTN_STAGE");
    setenv("E2_HIP_ATTN", "old", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_LEGACY);
    expect(E2_BK_HIP, E2_TOWER_TEXT, E2_ATTN_ANY_KEYS, E2_ATTN_F32);
    setenv("E2_HIP_ATTN", "lds", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_AUTO);
    expect(E2_BK_HIP, E2_TOWER_TEXT, E2_ATTN_ANY_KEYS, E2_ATTN_F32);
    unsetenv("E2_HIP_ATTN");
    setenv("E2_HIP_ATTN_I8", "direct", 1);
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_DIRECT);
    setenv("E2_ATTN_STAGE", "legacy", 1); /* the new name wins */
    e2_attn_env_reload();
    assert(e2_attn_stage_override() == E2_ATTN_STAGE_LEGACY);

    /* an override drops the length gate the table would have applied */
    unsetenv("E2_ATTN_STAGE");
    unsetenv("E2_HIP_ATTN_I8");
    setenv("E2_ATTN_TEXT", "i8", 1);
    e2_attn_env_reload();
    assert(e2_attn_min_keys(E2_BK_HIP, E2_TOWER_TEXT) == 0);
    expect(E2_BK_HIP, E2_TOWER_TEXT, 15, E2_ATTN_I8); /* int8 at 15 keys, not only from 64 */
    setenv("E2_ATTN_TEXT", "auto", 1);
    e2_attn_env_reload();
    assert(e2_attn_min_keys(E2_BK_HIP, E2_TOWER_TEXT) == 64);
    expect(E2_BK_HIP, E2_TOWER_TEXT, 15, E2_ATTN_F32); /* the table gates it again */

    unsetenv("E2_ATTN_TEXT");
    unsetenv("E2_ATTN_STAGE");
    unsetenv("E2_HIP_ATTN");
    unsetenv("E2_HIP_ATTN_I8");
    e2_attn_env_reload();

    /* ---- E2_CPU_BACKEND: may downgrade, never invent an ISA --------------------------------- */
    unsetenv("E2_CPU_BACKEND");
    e2_cpu_init();
    const e2_backend best = e2_cpu_backend(); /* whatever this host selects on its own */
    assert((unsigned)best < (unsigned)E2_BK_COUNT);

    setenv("E2_CPU_BACKEND", "scalar", 1);
    e2_cpu_init();
    assert(e2_cpu_backend() == E2_BK_CPU_SCALAR);
    /* the portable row has both halves of the int8 path, so the model loads int8 weights even here */
    assert(e2_i8_gemm != NULL && e2_i8_quantize != NULL);
    /* ...and no stale kernel may survive from the best row that was selected first */
    assert(e2_i8_attn_gemm == NULL && e2_softmax_quant_i8 == NULL);
    assert(e2_attn_pick(e2_cpu_backend(), E2_TOWER_TEXT, 2048) == E2_ATTN_F32);

    setenv("E2_CPU_BACKEND", "not-a-backend", 1);
    e2_cpu_init(); /* unknown name: reported and ignored, so the best backend stays in place */
    assert(e2_cpu_backend() == best);

    /* The other architecture's name is either unknown or unsupported here; both are refused. */
    setenv("E2_CPU_BACKEND",
#if defined(__aarch64__)
           "avx2",
#else
           "neon",
#endif
           1);
    e2_cpu_init();
    assert(e2_cpu_backend() == best); /* never simulate an ISA that is not there */

    unsetenv("E2_CPU_BACKEND");
    e2_cpu_init();
    assert(e2_cpu_backend() == best);

    if (g_fail) {
        fprintf(stderr, "test_dispatch: %d mismatch(es)\n", g_fail);
        return 1;
    }
    printf("dispatch matrix: OK\n");
    return 0;
}
