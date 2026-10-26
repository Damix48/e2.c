/* hip.c — the AMD/HIP host layer: device selection, memory, graph capture, and the e2_hip_* API.
 *
 * The structure mirrors the Intel backend (src/gpu/gpu.c) on purpose: the model is uploaded once,
 * every activation offset is a compile-time constant of the fixed region layout, and the kernel
 * sequence for a shape is recorded into a hipGraph the first time that shape is seen. A call is
 * then: write the pinned ids, launch the graph, synchronize, read the pinned embeddings. The
 * capture is done with hipStreamBeginCapture on the backend's own stream; the H2D/D2H copies live
 * inside the graph, so replay uploads the ids and downloads the output without any extra host work.
 *
 * E2_HIP_NOGRAPH=1 skips the capture and launches the kernels directly on every call, which is the
 * debugging path (a mis-recorded graph is otherwise hard to bisect). E2_HIP_LAYERS=n truncates the
 * text tower after n layers and E2_HIP_LIMIT=n stops after n kernels, the same diagnostics the
 * Intel backend exposes, and E2_HIP_DUMP=dir writes the activation regions after the first call.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "validation.h"
#include "hip.h"

static char g_err[1024];

const char *e2_hip_last_error(void) { return g_err; }

double e2h_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

void e2h_set_error(e2_hip *g, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    if (g) snprintf(g->err, sizeof g->err, "%s", g_err);
}

int e2h_ok(e2_hip *g, hipError_t e, const char *what) {
    if (e == hipSuccess) return 0;
    e2h_set_error(g, "%s: %s", what, hipGetErrorString(e));
    return -1;
}

/* ---- per-kernel profiling (E2_HIP_PROFILE=1 + E2_HIP_NOGRAPH=1)
 * The launchers record an event pair per kernel into a buffer while `e2h_prof_rec` is set (the first
 * call only); run_chunk flushes it after the call has synchronized, so the times are the real
 * in-stream kernel durations with no per-launch stall. */
int e2h_prof_on;
int e2h_prof_rec;
#define E2H_PROF_SLOTS 32
#define E2H_PROF_MAXEV 8192
static struct { const char *name; double ms; long n; } g_prof[E2H_PROF_SLOTS];
static int g_prof_used;
static struct { int slot; const char *name; hipEvent_t a, b; } g_pev[E2H_PROF_MAXEV];
static int g_pev_n;

void e2h_prof_add(int slot, const char *name, double ms) {
    if (slot < 0 || slot >= E2H_PROF_SLOTS) return;
    if (!g_prof[slot].name) {
        g_prof[slot].name = name;
        if (slot >= g_prof_used) g_prof_used = slot + 1;
    }
    g_prof[slot].ms += ms;
    g_prof[slot].n++;
}

void e2h_prof_ev(int slot, const char *name, hipEvent_t a, hipEvent_t b) {
    if (g_pev_n >= E2H_PROF_MAXEV) {
        hipEventDestroy(a);
        hipEventDestroy(b);
        return;
    }
    g_pev[g_pev_n].slot = slot;
    g_pev[g_pev_n].name = name;
    g_pev[g_pev_n].a = a;
    g_pev[g_pev_n].b = b;
    g_pev_n++;
}

void e2h_prof_flush(void) {
    for (int i = 0; i < g_pev_n; i++) {
        float ms = 0.f;
        hipEventElapsedTime(&ms, g_pev[i].a, g_pev[i].b);
        e2h_prof_add(g_pev[i].slot, g_pev[i].name, ms);
        hipEventDestroy(g_pev[i].a);
        hipEventDestroy(g_pev[i].b);
    }
    g_pev_n = 0;
}

void e2h_prof_report(void) {
    if (!e2h_prof_on) return;
    double total = 0;
    for (int i = 0; i < g_prof_used; i++) total += g_prof[i].ms;
    printf("[hip] kernel profile, one call (ms, total %.2f):\n", total);
    for (int i = 0; i < g_prof_used; i++)
        if (g_prof[i].n)
            printf("[hip]   %-14s %8.3f ms  %6ld calls  %6.3f ms/call  %5.1f%%\n", g_prof[i].name,
                   g_prof[i].ms, g_prof[i].n, g_prof[i].ms / g_prof[i].n,
                   total > 0 ? 100.0 * g_prof[i].ms / total : 0.0);
}

/* ---------------------------------------------------------------- public API */

e2_hip *e2_hip_create(e2_model *m) {
    if (!m) { e2h_set_error(NULL, "model is NULL"); return NULL; }
    e2_hip *g = calloc(1, sizeof *g);
    if (!g) { e2h_set_error(NULL, "oom"); return NULL; }
    g->m = m;
    g->nograph = getenv("E2_HIP_NOGRAPH") != NULL;
    /* E2_HIP_PROFILE also works with graphs: the event pairs become event-record nodes and are
     * flushed after the call, which is the only way to get true per-kernel times on this setup
     * (in stream mode the host launch gap is attributed to the kernel). */
    e2h_prof_on = getenv("E2_HIP_PROFILE") != NULL;
    g->gemm_lds = 1;
    { const char *ge = getenv("E2_HIP_GEMM"); if (ge && !strcmp(ge, "direct")) g->gemm_lds = 0; }
    g->attn_lds = 1;
    g->attn_i8 = 1;
    {
        const char *ae = getenv("E2_HIP_ATTN");
        if (ae && strcmp(ae, "old") == 0) { g->attn_lds = 0; g->attn_i8 = 0; }
        if (ae && strcmp(ae, "lds") == 0) g->attn_i8 = 0;
    }
    g->attn_i8_min = E2_HIP_ATTN_I8_MIN;
    { const char *am = getenv("E2_HIP_ATTN_I8_MIN"); if (am) g->attn_i8_min = atoi(am); }
    g->attn_i8_lds = 1;
    { const char *ai = getenv("E2_HIP_ATTN_I8"); if (ai && strcmp(ai, "direct") == 0) g->attn_i8_lds = 0; }
    const char *ll = getenv("E2_HIP_LAYERS");
    g->layer_limit = ll ? atoi(ll) : -1;
    g->limit = getenv("E2_HIP_LIMIT") ? atoi(getenv("E2_HIP_LIMIT")) : 0;
    g->dump_dir = getenv("E2_HIP_DUMP");
    g->dump_call = getenv("E2_HIP_DUMP_CALL") ? atoi(getenv("E2_HIP_DUMP_CALL")) : 1;
    { const char *v = getenv("E2_HIP_VLAYERS"); g->vlayer_limit = v ? atoi(v) : -1; }

    int ndev = 0;
    if (e2h_ok(g, hipGetDeviceCount(&ndev), "hipGetDeviceCount") || ndev <= 0) {
        if (ndev <= 0) e2h_set_error(g, "no HIP device visible (check /dev/kfd permissions)");
        free(g);
        return NULL;
    }
    const char *dev = getenv("E2_HIP_DEVICE");
    g->device = dev ? atoi(dev) : 0;
    if (g->device < 0 || g->device >= ndev) {
        e2h_set_error(g, "E2_HIP_DEVICE=%d out of range (0..%d)", g->device, ndev - 1);
        free(g);
        return NULL;
    }
    if (e2h_ok(g, hipSetDevice(g->device), "hipSetDevice")) goto bad;
    if (e2h_ok(g, hipStreamCreateWithFlags(&g->stream, hipStreamNonBlocking), "hipStreamCreate"))
        goto bad;
    if (!m->use_int8) {
        e2h_set_error(g, "the HIP backend needs the int8 weights (E2_INT8=0 disables them)");
        goto bad;
    }
    if (e2h_forward_build(g)) goto bad;
    return g;
bad:
    e2_hip_destroy(g);
    return NULL;
}

void e2_hip_destroy(e2_hip *g) {
    if (!g) return;
    e2h_prof_report();
    for (int i = 0; i < g->nslots; i++) {
        if (g->slots[i].exec) hipGraphExecDestroy(g->slots[i].exec);
        if (g->slots[i].graph) hipGraphDestroy(g->slots[i].graph);
    }
    if (g->stream) {
        hipFree(g->wbuf);
        hipFree(g->acts);
        hipFree(g->part);
        hipFree(g->desc);
        hipFree(g->toktab);
        hipFree(g->d_ids);
        hipFree(g->d_valid);
        hipFree(g->d_embout);
        hipHostFree(g->ids);
        hipHostFree(g->valid);
        hipFree(g->vpix);
        hipFree(g->vpos);
        hipFree(g->vorder);
        hipFree(g->vgpatch);
        hipFree(g->vgstart);
        hipFree(g->vrope);
        hipFree(g->vpool_ng);
        hipStreamDestroy(g->stream);
    }
    free(g->hdesc);
    free(g);
}

const char *e2_hip_device(e2_hip *g) {
    static char name[1024];
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, g->device) != hipSuccess) return "AMD GPU";
    snprintf(name, sizeof name, "%.128s (%.64s, %d CUs, %d MHz)", p.name, p.gcnArchName,
             p.multiProcessorCount, p.clockRate / 1000);
    return name;
}

int e2_hip_stats(e2_hip *g, long *calls, long *kernels_last, double *last_ms) {
    if (calls) *calls = g->calls;
    if (kernels_last) *kernels_last = g->kernels_last;
    if (last_ms) *last_ms = g->last_ms;
    return 0;
}

/* ---------------------------------------------------------------- graph slots */

/* Every graph launch is synchronized before returning to the caller. */
static e2_hip_slot *new_slot(e2_hip *g) {
    if (g->nslots == (int)(sizeof g->slots / sizeof g->slots[0])) {
        g->slots[0].from_embeds = -1; /* do not replay a partially destroyed entry on failure */
        if (g->slots[0].exec &&
            e2h_ok(g, hipGraphExecDestroy(g->slots[0].exec), "evict graph exec")) return NULL;
        g->slots[0].exec = NULL;
        if (e2h_ok(g, hipGraphDestroy(g->slots[0].graph), "evict graph")) return NULL;
        g->nslots--;
        memmove(g->slots, g->slots + 1, (size_t)g->nslots * sizeof g->slots[0]);
    }
    e2_hip_slot *sl = &g->slots[g->nslots];
    memset(sl, 0, sizeof *sl);
    return sl;
}

static e2_hip_slot *slot_for(e2_hip *g, int rows, int seq_len, int from_embeds) {
    for (int i = 0; i < g->nslots; i++)
        if (g->slots[i].rows == rows && g->slots[i].seq_len == seq_len &&
            g->slots[i].from_embeds == from_embeds)
            return &g->slots[i];
    e2_hip_slot *sl = new_slot(g);
    if (!sl) return NULL;
    sl->rows = rows;
    sl->seq_len = seq_len;
    sl->from_embeds = from_embeds;
    if (e2h_record(g, rows, seq_len, from_embeds, sl)) return NULL;
    g->nslots++;
    return sl;
}

/* ---------------------------------------------------------------- forward */

static int run_chunk(e2_hip *g, const int32_t *ids, const uint8_t *valid, int rows, int seq_len,
                     int from_embeds, float *out) {
    /* The pinned host buffers are what the recorded graph's H2D copy nodes read: fill them first,
     * then launch. The result comes back with a synchronous copy after the stream syncs. */
    if (ids) memcpy(g->ids, ids, sizeof(int32_t) * (size_t)rows);
    /* The mask covers more rows than the batch has (the PLE normalization walks rows*n_layers), so
     * everything past `rows` must read as valid rather than keep a previous call's padding. */
    memset(g->valid, 1, E2_HIP_MAX_ROWS);
    if (valid) memcpy(g->valid, valid, (size_t)rows);

    double t0 = e2h_now_ms();
    if (g->nograph) {
        if (e2h_launch_all(g, rows, seq_len, from_embeds)) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
    } else {
        e2_hip_slot *sl = slot_for(g, rows, seq_len, from_embeds);
        if (!sl) return -1;
        if (e2h_ok(g, hipGraphLaunch(sl->exec, g->stream), "graph launch")) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
        g->kernels_last = sl->kernels;
    }
    g->last_ms = e2h_now_ms() - t0;
    const int nseq = seq_len > 0 ? rows / seq_len : 1;
    if (e2h_ok(g, hipMemcpy(out, g->d_embout, sizeof(float) * 768 * (size_t)nseq,
                            hipMemcpyDeviceToHost),
               "result download"))
        return -1;
    g->calls++;
    if (g->dump_dir && g->calls == g->dump_call) e2h_dump_regions(g, rows);
    if (e2h_prof_rec) {
        e2h_prof_rec = 0;
        e2h_prof_flush();
    }
    return 0;
}

int e2_hip_text_embed(e2_hip *g, const int32_t *ids, int n_ids, float *out768) {
    char verr[256];
    if (!g) {
        e2h_set_error(NULL, "HIP context is NULL");
        return -1;
    }
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) {
        e2h_set_error(g, "sequence length %d out of range", n_ids);
        return -1;
    }
    if (!out768) {
        e2h_set_error(g, "output pointer is NULL");
        return -1;
    }
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) {
        e2h_set_error(g, "%s", verr);
        return -1;
    }
    return run_chunk(g, ids, NULL, n_ids, n_ids, 0, out768);
}

int e2_hip_text_embed_batch(e2_hip *g, const int32_t *ids, const uint8_t *valid, int batch,
                            int seq_len, float *out) {
    char verr[256];
    if (!g) {
        e2h_set_error(NULL, "HIP context is NULL");
        return -1;
    }
    if (batch <= 0 || seq_len <= 0 || seq_len > E2_MAX_SEQ ||
        (size_t)batch > SIZE_MAX / (size_t)seq_len / sizeof(*ids) ||
        (size_t)batch > SIZE_MAX / E2_EMBED_DIM / sizeof(*out)) {
        e2h_set_error(g, "bad batch geometry");
        return -1;
    }
    if (!out) {
        e2h_set_error(g, "output pointer is NULL");
        return -1;
    }
    /* The pad rows are gathered too, so every id in the block has to be a vocabulary entry. */
    if (e2_valid_ids(ids, (size_t)batch * (size_t)seq_len, verr, sizeof verr)) {
        e2h_set_error(g, "%s", verr);
        return -1;
    }
    /* Sequences beyond what one pass holds are processed in chunks of whole sequences. */
    int per = E2_HIP_MAX_ROWS / seq_len;
    if (per < 1) per = 1;
    for (int b0 = 0; b0 < batch;) {
        const int n = batch - b0 < per ? batch - b0 : per;
        const int rows = n * seq_len;
        if (run_chunk(g, ids + (size_t)b0 * seq_len, valid ? valid + (size_t)b0 * seq_len : NULL,
                      rows, seq_len, 0, out + (size_t)b0 * 768))
            return -1;
        b0 += n; /* by n, not per: b0 + per can leave int range on the last chunk */
    }
    return 0;
}

/* ---------------------------------------------------------------- vision tower
 * Runs the Gemma4 encoder on the device and leaves the soft tokens in `soft` (n_soft x 512 host
 * floats). The host walks the position ids once to build the valid-patch gather and the pooler's CSR
 * group lists, and builds the axial RoPE table per distinct coordinate; everything else is device
 * work recorded into a graph keyed by (patch count, group count). */

static void vrope_build(e2_hip *g, const int32_t *pos, const int32_t *order, int P, float *cs,
                        float *sn) {
    e2_vision *v = (e2_vision *)g->m->vision;
    static uint8_t seen[2][E2_V_POS];
    static float cx[E2_V_POS][16], sx[E2_V_POS][16], cy[E2_V_POS][16], sy[E2_V_POS][16];
    memset(seen, 0, sizeof seen);
    for (int p = 0; p < P; p++) {
        const int x = pos[2 * order[p]], y = pos[2 * order[p] + 1];
        if (!seen[0][x]) {
            seen[0][x] = 1;
            for (int i = 0; i < 16; i++) {
                const float a = (float)x * v->rope_inv[i];
                cx[x][i] = cosf(a);
                sx[x][i] = sinf(a);
            }
        }
        if (!seen[1][y]) {
            seen[1][y] = 1;
            for (int i = 0; i < 16; i++) {
                const float a = (float)y * v->rope_inv[i];
                cy[y][i] = cosf(a);
                sy[y][i] = sinf(a);
            }
        }
        float *c = cs + (size_t)p * 64, *s = sn + (size_t)p * 64;
        for (int i = 0; i < 16; i++) {
            c[i] = c[i + 16] = cx[x][i];
            s[i] = s[i + 16] = sx[x][i];
            c[32 + i] = c[48 + i] = cy[y][i];
            s[32 + i] = s[48 + i] = sy[y][i];
        }
    }
}

static e2_hip_slot *vgraph_for(e2_hip *g, int P) {
    const int key = g->vgroups;
    for (int i = 0; i < g->nslots; i++)
        if (g->slots[i].from_embeds == 2 && g->slots[i].rows == P && g->slots[i].seq_len == key)
            return &g->slots[i];
    e2_hip_slot *sl = new_slot(g);
    if (!sl) return NULL;
    sl->rows = P;
    sl->seq_len = key;
    sl->from_embeds = 2;
    if (e2h_record_vision(g, P, key, sl)) return NULL;
    g->nslots++;
    return sl;
}

static int vision_soft_tokens(e2_hip *g, const float *pixels, const int32_t *position_ids,
                              int n_patches, float *soft, int *n_soft_out) {
    if (!g->m->vision) {
        e2h_set_error(g, "the model has no vision tower");
        return -1;
    }
    if (n_patches <= 0 || n_patches > E2_V_MAX_PATCHES) {
        e2h_set_error(g, "patch count %d out of range (max %d)", n_patches, E2_V_MAX_PATCHES);
        return -1;
    }
    if (!pixels) {
        e2h_set_error(g, "pixels pointer is NULL");
        return -1;
    }
    /* Positions bound the axial table and the pool grid the kernels write to (E2_HIP_VGROUPS rows);
     * both are settled before the gather builds the group lists. */
    char verr[256];
    int cols = 0, grows = 0;
    if (e2_valid_positions(position_ids, n_patches, E2_HIP_VGROUPS, &cols, &grows, verr, sizeof verr)) {
        e2h_set_error(g, "%s", verr);
        return -1;
    }
    const int groups = cols * grows;
    const int nscr = 2 * (size_t)E2_V_MAX_PATCHES + 4 * (size_t)(E2_HIP_VGROUPS + 1);
    int32_t *scr = malloc(sizeof(int32_t) * nscr);
    if (!scr) {
        e2h_set_error(g, "oom for the image scratch");
        return -1;
    }
    int32_t *order = scr, *gpatch = scr + E2_V_MAX_PATCHES;
    int32_t *cnt = gpatch + E2_V_MAX_PATCHES;
    int32_t *gmap = cnt + E2_HIP_VGROUPS + 1;
    int32_t *gstart = gmap + E2_HIP_VGROUPS + 1;
    int32_t *wpos = gstart + E2_HIP_VGROUPS + 1;

    int P = 0;
    for (int i = 0; i < n_patches; i++)
        if (position_ids[2 * i] >= 0) order[P++] = i; /* the only negative pair is (-1,-1) padding */

    const int pool = E2_V_POOL;
    memset(cnt, 0, sizeof(int32_t) * (size_t)(groups + 1));
    for (int p = 0; p < P; p++) {
        const int i = order[p];
        cnt[position_ids[2 * i] / pool + cols * (position_ids[2 * i + 1] / pool)]++;
    }
    int ng = 0, acc = 0;
    for (int gg = 0; gg < groups; gg++) {
        if (cnt[gg]) {
            gmap[gg] = ng;
            gstart[ng++] = acc;
            acc += cnt[gg];
        } else {
            gmap[gg] = -1;
        }
    }
    gstart[ng] = acc;
    memcpy(wpos, gstart, sizeof(int32_t) * (size_t)(ng + 1));
    for (int p = 0; p < P; p++) {
        const int i = order[p];
        const int gg = position_ids[2 * i] / pool + cols * (position_ids[2 * i + 1] / pool);
        gpatch[wpos[gmap[gg]]++] = p;
    }

    float *rope = malloc(sizeof(float) * 2 * (size_t)E2_HIP_VP * 64);
    if (!rope) {
        free(scr);
        e2h_set_error(g, "oom for the rope tables");
        return -1;
    }
    vrope_build(g, position_ids, order, P, rope, rope + (size_t)E2_HIP_VP * 64);

    g->vgroups = ng;
    e2_hip_slot *sl = NULL;
    if (!g->nograph) {
        sl = vgraph_for(g, P);
        if (!sl) {
            free(scr);
            free(rope);
            return -1;
        }
    }
    if (e2h_ok(g, hipMemcpy(g->vpix, pixels, sizeof(float) * (size_t)n_patches * E2_V_PATCH,
                            hipMemcpyHostToDevice),
               "pixel upload") ||
        e2h_ok(g, hipMemcpy(g->vpos, position_ids, sizeof(int32_t) * (size_t)n_patches * 2,
                            hipMemcpyHostToDevice),
               "position upload") ||
        e2h_ok(g, hipMemcpy(g->vorder, order, sizeof(int32_t) * (size_t)P, hipMemcpyHostToDevice),
               "order upload") ||
        e2h_ok(g, hipMemcpy(g->vgstart, gstart, sizeof(int32_t) * (size_t)(ng + 1),
                            hipMemcpyHostToDevice),
               "group start upload") ||
        e2h_ok(g, hipMemcpy(g->vgpatch, gpatch, sizeof(int32_t) * (size_t)P,
                            hipMemcpyHostToDevice),
               "group patch upload") ||
        e2h_ok(g, hipMemcpy(g->vpool_ng, &ng, sizeof(int32_t), hipMemcpyHostToDevice),
               "group count upload") ||
        e2h_ok(g, hipMemcpy(g->vrope, rope, sizeof(float) * 2 * (size_t)E2_HIP_VP * 64,
                            hipMemcpyHostToDevice),
               "rope upload")) {
        free(scr);
        free(rope);
        return -1;
    }
    free(scr);
    free(rope);

    double t0 = e2h_now_ms();
    if (g->nograph) {
        if (e2h_launch_vision(g, P, ng)) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
    } else {
        if (e2h_ok(g, hipGraphLaunch(sl->exec, g->stream), "graph launch")) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
        g->kernels_last = sl->kernels;
    }
    g->vlast_ms = e2h_now_ms() - t0;
    g->vkernels_last = g->kernels_last;
    if (e2h_prof_rec) {
        e2h_prof_rec = 0;
        e2h_prof_flush();
    }
    if (g->vlayer_limit >= 0 || g->limit) {
        *n_soft_out = 0; /* diagnostic graph: the tail never ran */
        return 0;
    }
    if (e2h_ok(g, hipMemcpy(soft, (char *)g->acts + g->reg[HA_VSOFT].off,
                            sizeof(float) * (size_t)ng * 512, hipMemcpyDeviceToHost),
               "soft token read"))
        return -1;
    *n_soft_out = ng;
    return 0;
}

static int run_embeds(e2_hip *g, const float *embeds, int rows, float *out) {
    if (e2h_ok(g, hipMemcpy((char *)g->acts + g->reg[HA_VEMBIN].off, embeds,
                            sizeof(float) * (size_t)rows * 512, hipMemcpyHostToDevice),
               "embedding upload"))
        return -1;
    memset(g->valid, 1, E2_HIP_MAX_ROWS);
    double t0 = e2h_now_ms();
    if (g->nograph) {
        if (e2h_launch_all(g, rows, rows, 1)) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
    } else {
        e2_hip_slot *sl = slot_for(g, rows, rows, 1);
        if (!sl) return -1;
        if (e2h_ok(g, hipGraphLaunch(sl->exec, g->stream), "graph launch")) return -1;
        if (e2h_ok(g, hipStreamSynchronize(g->stream), "synchronize")) return -1;
        g->kernels_last = sl->kernels;
    }
    g->last_ms = e2h_now_ms() - t0;
    if (e2h_ok(g, hipMemcpy(out, g->d_embout, sizeof(float) * 768, hipMemcpyDeviceToHost),
               "result download"))
        return -1;
    g->calls++;
    if (g->dump_dir && g->calls == g->dump_call) e2h_dump_regions(g, rows);
    return 0;
}

int e2_hip_image_embed(e2_hip *g, const float *pixels, const int32_t *position_ids, int n_patches,
                       float *out768) {
    if (!g) {
        e2h_set_error(NULL, "HIP context is NULL");
        return -1;
    }
    if (!out768) {
        e2h_set_error(g, "output pointer is NULL");
        return -1;
    }
    float *soft = malloc(sizeof(float) * (size_t)E2_HIP_VGROUPS * 512);
    if (!soft) {
        e2h_set_error(g, "oom");
        return -1;
    }
    int n_soft = 0;
    if (vision_soft_tokens(g, pixels, position_ids, n_patches, soft, &n_soft)) {
        free(soft);
        return -1;
    }
    if (n_soft <= 0) {
        free(soft);
        return 0; /* E2_HIP_VLAYERS: diagnostic run */
    }
    const int S = n_soft + 4;
    float *e = calloc((size_t)S * 512, sizeof(float));
    if (!e) {
        free(soft);
        e2h_set_error(g, "oom");
        return -1;
    }
    /* <bos> <|image> <|image|> x n_soft <image|> <eos>, the layout src/vision.c builds */
    e2_embed_token(g->m, 2, e);
    e2_embed_token(g->m, 255999, e + 512);
    memcpy(e + 2 * 512, soft, sizeof(float) * (size_t)n_soft * 512);
    e2_embed_token(g->m, 258882, e + (size_t)(2 + n_soft) * 512);
    e2_embed_token(g->m, 1, e + (size_t)(3 + n_soft) * 512);
    free(soft);
    const int rc = run_embeds(g, e, S, out768);
    free(e);
    return rc;
}

int e2_hip_text_image_embed(e2_hip *g, const int32_t *ids, int n_ids, const float *pixels,
                            const int32_t *position_ids, int n_patches, float *out768) {
    char verr[256];
    if (!g) {
        e2h_set_error(NULL, "HIP context is NULL");
        return -1;
    }
    if (n_ids <= 0 || n_ids > E2_MAX_SEQ) {
        e2h_set_error(g, "sequence length %d out of range", n_ids);
        return -1;
    }
    if (!out768) {
        e2h_set_error(g, "output pointer is NULL");
        return -1;
    }
    if (e2_valid_ids(ids, (size_t)n_ids, verr, sizeof verr)) {
        e2h_set_error(g, "%s", verr);
        return -1;
    }
    float *soft = malloc(sizeof(float) * (size_t)E2_HIP_VGROUPS * 512);
    float *e = calloc((size_t)n_ids * 512, sizeof(float));
    if (!soft || !e) {
        free(soft);
        free(e);
        e2h_set_error(g, "oom");
        return -1;
    }
    int n_soft = 0;
    if (vision_soft_tokens(g, pixels, position_ids, n_patches, soft, &n_soft)) {
        free(soft);
        free(e);
        return -1;
    }
    if (n_soft <= 0) {
        free(soft);
        free(e);
        return 0;
    }
    int slot = 0;
    for (int s = 0; s < n_ids; s++) {
        if (ids[s] == 258880) { /* <|image|> */
            if (slot >= n_soft) {
                free(soft);
                free(e);
                e2h_set_error(g, "more <|image|> slots than soft tokens");
                return -1;
            }
            memcpy(e + (size_t)s * 512, soft + (size_t)slot++ * 512, sizeof(float) * 512);
        } else {
            e2_embed_token(g->m, ids[s], e + (size_t)s * 512);
        }
    }
    free(soft);
    if (slot != n_soft) {
        free(e);
        e2h_set_error(g, "%d soft tokens for %d slots", n_soft, slot);
        return -1;
    }
    const int rc = run_embeds(g, e, n_ids, out768);
    free(e);
    return rc;
}

int e2_hip_vision_stats(e2_hip *g, long *kernels, double *last_ms) {
    if (kernels) *kernels = g->vkernels_last;
    if (last_ms) *last_ms = g->vlast_ms;
    return 0;
}
