/* dispatch.c — capabilities and attention-precision rules, one table each.
 *
 * This is DISPATCH.md Part 1 turned into data: the caps say what a backend has, the rules say what it
 * should prefer, and e2_attn_pick() is the only place the two are combined. Length thresholds are
 * numbers in the rule table and nowhere else; no tower file carries a literal.
 */
#include "dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E2_CPU_I8_CAPS (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8 | E2_CAP_SOFTMAX_Q_I8)

static const unsigned k_caps[E2_BK_COUNT] = {
    /* Portable C has the int8 GEMM and the quantizer, so it runs the int8 weights, but no fused
     * attention kernel and no fused softmax: text attention stays fp32 unless E2_ATTN_TEXT forces
     * it, image attention stays fp32 because that one needs the packed-operand kernel. */
    [E2_BK_CPU_SCALAR] = E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8,
    /* AVX2 has the quantizer and the fused softmax, but no maddubs attention kernel: attention on
     * images stays fp32 there, which is a capability gap rather than a tuning choice. */
    [E2_BK_CPU_AVX2] = E2_CPU_I8_CAPS,
    /* The fused attention kernel exists for VNNI at both widths and, since Attempt 23, for NEON. */
    [E2_BK_CPU_VNNI] = E2_CPU_I8_CAPS | E2_CAP_ATTN_I8_FUSED,
    [E2_BK_CPU_AVX512] = E2_CPU_I8_CAPS | E2_CAP_ATTN_I8_FUSED,
    [E2_BK_CPU_NEON] = E2_CPU_I8_CAPS | E2_CAP_ATTN_I8_FUSED,
    /* Level Zero only exists where the backend builds, and that is the XMX devices; the n100's Xe-LP
     * has no matrix units and is not a Level Zero target at all. */
    [E2_BK_INTEL_GPU] = E2_CPU_I8_CAPS | E2_CAP_ATTN_I8_FUSED | E2_CAP_ATTN_F16,
    [E2_BK_HIP] = E2_CPU_I8_CAPS | E2_CAP_ATTN_I8_FUSED,
};

/* A preference rule: what to use at or above min_keys, what to use below it. min_keys == 0 means no
 * length gate and `below` is unused. */
typedef struct {
    e2_attn_kind kind;
    e2_attn_kind below;
    int min_keys;
} e2_attn_rule;

/* DISPATCH.md Part 1's matrix, row for row. Anything unlisted is {F32, F32, 0} - fp32 at any length.
 *
 * Every threshold here is the same trade-off in different units: attention costs S*S*D against an
 * S*D pack per head, so packing amortises once S clears the packer's constant. The constant is a
 * property of the machine, not of the maths, which is why the same 128 appears four times and 64 and
 * 512 appear once each. */
static const e2_attn_rule k_rules[E2_BK_COUNT][E2_TOWER_COUNT] = {
    /* text, CPU: packer amortises at 128 on both x86-64 and aarch64 (PERFORMANCE.md Attempt 22). */
    [E2_BK_CPU_AVX2][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F32, 128 },
    [E2_BK_CPU_VNNI][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F32, 128 },
    [E2_BK_CPU_AVX512][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F32, 128 },
    [E2_BK_CPU_NEON][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F32, 128 },
    /* text, Intel GPU: XMX fp16 beats int8 attention below 512 keys. This is a preference, not a
     * degradation - it is spelled out in `below` so it does not depend on the I8 -> F16 -> F32 order. */
    [E2_BK_INTEL_GPU][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F16, 512 },
    /* text, HIP: v_dot4 pays from 64 keys up. */
    [E2_BK_HIP][E2_TOWER_TEXT] = { E2_ATTN_I8, E2_ATTN_F32, 64 },

    /* image: 2520 patches are effectively fixed, so S^2 dwarfs the S*D pack from the first call - no
     * length gate anywhere. AVX2 is listed even though it has no fused kernel: the rule still says
     * int8, and the capability is what turns it into fp32. That split is the point of the two layers. */
    [E2_BK_CPU_AVX2][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },
    [E2_BK_CPU_VNNI][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },
    [E2_BK_CPU_AVX512][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },
    [E2_BK_CPU_NEON][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },
    [E2_BK_INTEL_GPU][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },
    [E2_BK_HIP][E2_TOWER_IMAGE] = { E2_ATTN_I8, E2_ATTN_F32, 0 },

    /* audio: a 12-key local window with a relative-position term, ~36*D MACs per token. An int8 pack
     * over 128-wide blocks would cost more than the dots it replaced, at any length - F32 by
     * construction, not by omission. Every audio entry stays unlisted and therefore F32. */
};

static e2_backend g_cpu_backend = E2_BK_CPU_SCALAR;

/* Per-tower overrides: -1 = auto (use the rule table), otherwise an e2_attn_kind. Resolved from the
 * environment once, by e2_cpu_init()/e2_attn_env_reload(), so no tower reads getenv in a loop. */
static int g_override[E2_TOWER_COUNT] = { -1, -1, -1 };
static e2_attn_stage g_stage = E2_ATTN_STAGE_AUTO;

void e2_dispatch_set_cpu_backend(e2_backend bk) {
    g_cpu_backend = (unsigned)bk < (unsigned)E2_BK_COUNT ? bk : E2_BK_CPU_SCALAR;
}

e2_backend e2_cpu_backend(void) { return g_cpu_backend; }

unsigned e2_backend_caps(e2_backend bk) {
    return (unsigned)bk < (unsigned)E2_BK_COUNT ? k_caps[bk] : 0;
}

/* "auto", "f32", "f16", "i8" -> an override; -1 for auto or anything unrecognised (with one warning,
 * because silently ignoring a typo is how an A/B run reports a confidently wrong result). */
static int parse_override(const char *s, const char *name) {
    if (!s || !s[0] || strcmp(s, "auto") == 0) return -1;
    if (strcmp(s, "f32") == 0) return E2_ATTN_F32;
    if (strcmp(s, "f16") == 0) return E2_ATTN_F16;
    if (strcmp(s, "i8") == 0) return E2_ATTN_I8;
    fprintf(stderr, "e2: ignoring %s=%s (expected auto|f32|f16|i8)\n", name, s);
    return -1;
}

static e2_attn_stage parse_stage(const char *s, const char *name) {
    if (!s || !s[0] || strcmp(s, "auto") == 0) return E2_ATTN_STAGE_AUTO;
    if (strcmp(s, "direct") == 0) return E2_ATTN_STAGE_DIRECT;
    if (strcmp(s, "lds") == 0) return E2_ATTN_STAGE_LDS;
    if (strcmp(s, "legacy") == 0 || strcmp(s, "old") == 0) return E2_ATTN_STAGE_LEGACY;
    fprintf(stderr, "e2: ignoring %s=%s (expected auto|direct|lds|legacy)\n", name, s);
    return E2_ATTN_STAGE_AUTO;
}

void e2_attn_env_reload(void) {
    g_override[E2_TOWER_TEXT] = parse_override(getenv("E2_ATTN_TEXT"), "E2_ATTN_TEXT");
    g_override[E2_TOWER_IMAGE] = parse_override(getenv("E2_ATTN_IMAGE"), "E2_ATTN_IMAGE");
    g_override[E2_TOWER_AUDIO] = -1; /* no selectable audio path to override */
    g_stage = parse_stage(getenv("E2_ATTN_STAGE"), "E2_ATTN_STAGE");
    /* Deprecated aliases, lowest precedence so a script can migrate one variable at a time. */
    if (g_override[E2_TOWER_TEXT] < 0) {
        const char *e = getenv("E2_ATTN_I8");
        if (e && e[0] == '0') g_override[E2_TOWER_TEXT] = E2_ATTN_F32;
        else if (e && e[0] == '1') g_override[E2_TOWER_TEXT] = E2_ATTN_I8;
    }
    if (g_override[E2_TOWER_IMAGE] < 0 && getenv("E2_VATTN_OFF"))
        g_override[E2_TOWER_IMAGE] = E2_ATTN_F32;
    /* HIP's two old switches: E2_HIP_ATTN=old|lds picked a *generation* (and, for lds, quietly turned
     * int8 off), E2_HIP_ATTN_I8=direct picked the int8 staging. Both map onto the one stage value. */
    if (g_stage == E2_ATTN_STAGE_AUTO) {
        const char *ae = getenv("E2_HIP_ATTN");
        const char *ai = getenv("E2_HIP_ATTN_I8");
        if (ae && strcmp(ae, "old") == 0) g_stage = E2_ATTN_STAGE_LEGACY;
        else if (ai && strcmp(ai, "direct") == 0) g_stage = E2_ATTN_STAGE_DIRECT;
        /* old/lds both meant "not int8": keep that meaning when the new precision name is silent. */
        if (g_override[E2_TOWER_TEXT] < 0 && ae && (strcmp(ae, "old") == 0 || strcmp(ae, "lds") == 0))
            g_override[E2_TOWER_TEXT] = E2_ATTN_F32;
    }
}

e2_attn_stage e2_attn_stage_override(void) { return g_stage; }

void e2_attn_override(e2_tower tower, int kind) {
    if ((unsigned)tower >= (unsigned)E2_TOWER_COUNT) return;
    g_override[tower] = (kind < 0 || kind > E2_ATTN_I8) ? -1 : kind;
}

const char *e2_attn_kind_name(e2_attn_kind kind) {
    switch (kind) {
    case E2_ATTN_F32: return "f32";
    case E2_ATTN_F16: return "f16";
    case E2_ATTN_I8: return "i8";
    }
    return "?";
}

int e2_attn_min_keys(e2_backend bk, e2_tower tower) {
    if ((unsigned)bk >= (unsigned)E2_BK_COUNT || (unsigned)tower >= (unsigned)E2_TOWER_COUNT) return 0;
    /* An override replaces the rule and its gate in one step, so the gate it reports is none. On HIP
     * that is what makes `E2_ATTN_TEXT=i8` mean "int8 at 15 keys too" rather than "int8 above 64". */
    if (g_override[tower] >= 0) return 0;
    return k_rules[bk][tower].min_keys;
}

/* Whether a backend can execute a kind for a tower. Deliberately a data lookup: a capability is never
 * inferred from a non-NULL function pointer, and never from calling a kernel. */
static int attn_kind_ok(e2_backend bk, e2_tower tower, e2_attn_kind kind) {
    const unsigned caps = k_caps[bk];
    switch (kind) {
    case E2_ATTN_F32: return 1;
    case E2_ATTN_F16: return (caps & E2_CAP_ATTN_F16) != 0;
    case E2_ATTN_I8:
        if (tower == E2_TOWER_TEXT) {
            /* text int8 attention reuses the weight GEMM on quantized Q and probabilities */
            return (caps & (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8)) ==
                   (E2_CAP_GEMM_I8 | E2_CAP_QUANT_I8);
        }
        /* image int8 needs the fused packed-operand kernel; audio has no int8 form at all */
        return tower == E2_TOWER_IMAGE && (caps & E2_CAP_ATTN_I8_FUSED) != 0;
    }
    return 0;
}

e2_attn_kind e2_attn_pick(e2_backend bk, e2_tower tower, int n_keys) {
    if ((unsigned)bk >= (unsigned)E2_BK_COUNT || (unsigned)tower >= (unsigned)E2_TOWER_COUNT)
        return E2_ATTN_F32;
    e2_attn_kind kind;
    if (g_override[tower] >= 0) {
        /* An override replaces the rule and its length gate in one step: "i8" means int8 at any
         * length, "f32" means never - the two things an A/B run needs. */
        kind = (e2_attn_kind)g_override[tower];
    } else {
        const e2_attn_rule *r = &k_rules[bk][tower];
        kind = (r->min_keys > 0 && n_keys < r->min_keys) ? r->below : r->kind;
    }
    for (int guard = 0; guard < 3; guard++) {
        if (attn_kind_ok(bk, tower, kind)) return kind;
        kind = kind == E2_ATTN_I8 ? E2_ATTN_F16 : E2_ATTN_F32;
    }
    return E2_ATTN_F32;
}
