/* imageproc.c — the HF Gemma4ImageProcessor in C: aspect-ratio-preserving bicubic resize, 16x16
 * patchify, position ids, right-padding.
 *
 * The resize has to agree with the reference bit for bit (the engine's image fixture is the processor
 * output), and the reference path is torchvision's `resize(..., BICUBIC, antialias=True)` on a uint8
 * tensor. `_do_native_uint8_resize_on_cpu` returns true for bicubic, so that is ATen's uint8
 * antialias kernel - which for our 512x512 fixture matches Pillow's `Image.resize(BICUBIC)` on
 * 99.9977% of pixels (41 of 1769472 differ by 1 LSB) and matches nothing else we tried. So this is
 * Pillow's `ImagingResample` for 8bpc RGB: bounds truncated at the image edge with the coefficients
 * renormalized over the surviving taps, INT32 accumulation against 16.16... (22-bit) fixed-point
 * coefficients, one rounding + clip to uint8 per pass, horizontal pass first.
 *
 * See the README for what this is not (JPEG/WebP, 16-bit, interlaced PNG, other resample filters).
 *
 * Both passes and the patchify are work-claiming regions on the engine's pool (threads.h), and the
 * pair runs ~10x faster on 14 threads than on one. The numbers are identical to the serial version
 * because a claim only ever writes what no other claim reads.
 */
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "../include/e2.h"
#include "imagedec.h"
#include "e2png.h"
#include "threads.h"

void e2_set_error(const char *fmt, ...);

#define PATCH 16
#define POOL 3
#define MAX_SOFT 280                              /* processor_config.json: max_soft_tokens */
#define MAX_PATCHES (MAX_SOFT * POOL * POOL)      /* 2520 */
#define SIDE_MULT (POOL * PATCH)                  /* 48: the resize snaps to a multiple of this */
#define RESCALE_FACTOR 0.00392156862745098f       /* processor_config.json: rescale_factor */

/* Pillow's PRECISION_BITS and its bicubic kernel (a = -0.5, the same one torchvision uses for the
 * antialias path; the a = -0.75 "Keys" variant that plain bicubic uses is visibly different here). */
#define PREC 22
#define ROUND_HALF (1 << (PREC - 1))

static double bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/* Resampling coefficients for one axis: for each output index, [first, first+count) source indices
 * and their fixed-point weights. `bounds` is 2 ints per output index. */
typedef struct {
    int ksize;
    int *bounds;
    int32_t *coeff;
} coeffs;

static void coeffs_free(coeffs *c) {
    free(c->bounds);
    free(c->coeff);
    c->bounds = NULL;
    c->coeff = NULL;
}

static int coeffs_build(coeffs *c, int in_size, int out_size) {
    double scale = (double)in_size / out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 2.0 * filterscale;
    double inv = 1.0 / filterscale;
    double *w;
    c->ksize = (int)ceil(support) * 2 + 1;
    c->bounds = malloc(sizeof(int) * 2 * (size_t)out_size);
    c->coeff = malloc(sizeof(int32_t) * (size_t)out_size * (size_t)c->ksize);
    w = malloc(sizeof(double) * (size_t)c->ksize);
    if (!c->bounds || !c->coeff || !w) {
        free(w);
        coeffs_free(c);
        return -1;
    }
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale;
        double ww = 0.0;
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);
        int n, i;
        if (xmin < 0) xmin = 0;
        if (xmax > in_size) xmax = in_size;
        n = xmax - xmin;
        for (i = 0; i < n; i++) {
            double v = bicubic((i + xmin - center + 0.5) * inv);
            w[i] = v;
            ww += v;
        }
        /* round half away from zero, exactly like normalize_coeffs_8bpc */
        for (i = 0; i < n; i++) {
            double v = ww != 0.0 ? w[i] / ww : 0.0;
            v *= (double)(1 << PREC);
            c->coeff[xx * c->ksize + i] = (int32_t)(v < 0 ? v - 0.5 : v + 0.5);
        }
        for (; i < c->ksize; i++) c->coeff[xx * c->ksize + i] = 0;
        c->bounds[xx * 2 + 0] = xmin;
        c->bounds[xx * 2 + 1] = n;
    }
    free(w);
    return 0;
}

/* Passes 1 and 2 are run as work-claiming regions, and so is the patchify. Each output row (or
 * patch) is computed by exactly one claim and depends on nothing another claim writes, so the
 * numbers are identical to the serial version - only the schedule changes. The pool is the engine's
 * (threads.c), so `E2_THREADS` governs this too; the region cost (~2.5 us) is why a small image
 * falls back to a serial run instead.
 *
 * The claim grain is a few rows because a row is only 3-10 us of work: on a hybrid CPU the frames
 * differ by ~2x and the region cannot finish before its last claim does. */
#define ROW_GRAIN 4
#define PATCH_GRAIN 16

typedef struct {
    const uint8_t *in;
    uint8_t *out;
    int in_rowbytes, out_rowbytes;
    int out_rows;
    const coeffs *c;
    int vertical;
    _Atomic int next;
} passjob;

/* Horizontal: `dst[x] = sum_i k[i] * src[xmin+i]` per channel, one input row to one output row. The
 * taps of one output pixel have to be walked in order (the caller compares this against the
 * reference kernel bit for bit), but the three channel chains are independent and each is split
 * into two partials to break the add-multiply dependency chain. Integer addition is exact here, so
 * the split cannot move a bit. */
static void hrow(const passjob *j, int y) {
    const uint8_t *src = j->in + (size_t)y * j->in_rowbytes;
    uint8_t *dst = j->out + (size_t)y * j->out_rowbytes;
    const coeffs *c = j->c;
    const int ow = j->out_rowbytes / 3;
    for (int x = 0; x < ow; x++, dst += 3) {
        const int32_t *k = c->coeff + (size_t)x * c->ksize;
        const uint8_t *s = src + (size_t)c->bounds[x * 2] * 3;
        int n = c->bounds[x * 2 + 1];
        int32_t r0 = 0, r1 = 0, g0 = 0, g1 = 0, b0 = 0, b1 = 0;
        int i = 0;
        for (; i + 1 < n; i += 2, s += 6) {
            int32_t k0 = k[i], k1 = k[i + 1];
            r0 += s[0] * k0;
            r1 += s[3] * k1;
            g0 += s[1] * k0;
            g1 += s[4] * k1;
            b0 += s[2] * k0;
            b1 += s[5] * k1;
        }
        if (i < n) {
            int32_t k0 = k[i];
            r0 += s[0] * k0;
            g0 += s[1] * k0;
            b0 += s[2] * k0;
        }
        int32_t r = (r0 + r1 + ROUND_HALF) >> PREC;
        int32_t g = (g0 + g1 + ROUND_HALF) >> PREC;
        int32_t b = (b0 + b1 + ROUND_HALF) >> PREC;
        dst[0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
        dst[1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
        dst[2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
    }
}

/* Vertical: every input row contributes the same weight to a whole output row, so the loop is
 * turned inside out - accumulate the row in blocks that stay in L1 and keep the inner loop a
 * contiguous int32 MAC, which vectorizes. Same additions, same order per output value. */
#define VBLK 1024
static void vrow(const passjob *j, int y) {
    const coeffs *c = j->c;
    const int rowlen = j->out_rowbytes;
    const int ymin = c->bounds[y * 2];
    const int n = c->bounds[y * 2 + 1];
    const int32_t *k = c->coeff + (size_t)y * c->ksize;
    uint8_t *dst = j->out + (size_t)y * rowlen;
    int32_t acc[VBLK];
    for (int x0 = 0; x0 < rowlen; x0 += VBLK) {
        int nx = rowlen - x0 < VBLK ? rowlen - x0 : VBLK;
        for (int i = 0; i < nx; i++) acc[i] = ROUND_HALF;
        for (int i = 0; i < n; i++) {
            const uint8_t *s = j->in + (size_t)(ymin + i) * rowlen + x0;
            int32_t ki = k[i];
            for (int t = 0; t < nx; t++) acc[t] += (int32_t)s[t] * ki;
        }
        for (int i = 0; i < nx; i++) {
            int32_t v = acc[i] >> PREC;
            dst[x0 + i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}

static void pass_rows(const passjob *j, int y0, int y1) {
    if (j->vertical) {
        for (int y = y0; y < y1; y++) vrow(j, y);
    } else {
        for (int y = y0; y < y1; y++) hrow(j, y);
    }
}

static void pass_worker(void *arg, int tid, int nthreads) {
    passjob *j = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int y0 = atomic_fetch_add_explicit(&j->next, ROW_GRAIN, memory_order_relaxed);
        if (y0 >= j->out_rows) break;
        int y1 = y0 + ROW_GRAIN;
        if (y1 > j->out_rows) y1 = j->out_rows;
        pass_rows(j, y0, y1);
    }
}

static void pass_run(passjob *j) {
    if (j->out_rows <= 2 * ROW_GRAIN) { /* not enough work to pay for a region */
        pass_rows(j, 0, j->out_rows);
        return;
    }
    atomic_init(&j->next, 0);
    e2_parallel(pass_worker, j);
}

/* u8 * rescale_factor, precomputed: exactly the same products as a per-pixel multiply, but a load
 * instead of a convert + multiply in the patchify loop. */
static float g_scale[256];
static int g_scale_ready;
static void scale_init(void) {
    if (g_scale_ready) return;
    for (int i = 0; i < 256; i++) g_scale[i] = (float)i * RESCALE_FACTOR;
    g_scale_ready = 1;
}

/* The processor's get_aspect_ratio_preserving_size. */
static void target_size(int height, int width, int *th, int *tw) {
    double factor = sqrt((double)MAX_PATCHES * PATCH * PATCH / ((double)height * width));
    int h = (int)floor(factor * height / SIDE_MULT) * SIDE_MULT;
    int w = (int)floor(factor * width / SIDE_MULT) * SIDE_MULT;
    if (h == 0) {
        h = SIDE_MULT;
        w = (int)floor((double)width / height) * SIDE_MULT;
        if (w > (MAX_PATCHES / (POOL * POOL)) * SIDE_MULT) w = (MAX_PATCHES / (POOL * POOL)) * SIDE_MULT;
    } else if (w == 0) {
        w = SIDE_MULT;
        h = (int)floor((double)height / width) * SIDE_MULT;
        if (h > (MAX_PATCHES / (POOL * POOL)) * SIDE_MULT) h = (MAX_PATCHES / (POOL * POOL)) * SIDE_MULT;
    }
    *th = h;
    *tw = w;
}

/* patchify: (ph, ps, pw, ps, 3), the reference's permute(1,3,2,4,0).reshape(n, 768), i.e. a patch
 * is row-major over [y][x][c] - the layout the vision tower's patch projection expects. Claimed in
 * patch units: a patch is 768 bytes of source read and 3 KiB of float written, so the region stays
 * bandwidth-bound and the balance is fine at any size. */
typedef struct {
    const uint8_t *res;
    float *pixels;
    int32_t *pos;
    int tw, pw, n_patches;
    _Atomic int next;
} patchjob;

static void patch_one(const patchjob *j, int i) {
    const int pr = i / j->pw, pc = i - pr * j->pw;
    const uint8_t *s = j->res + (((size_t)pr * PATCH) * j->tw + (size_t)pc * PATCH) * 3;
    float *d = j->pixels + (size_t)i * 768;
    for (int y = 0; y < PATCH; y++) {
        const uint8_t *row = s + (size_t)y * j->tw * 3;
        float *o = d + y * PATCH * 3;
        for (int x = 0; x < PATCH * 3; x++) o[x] = g_scale[row[x]];
    }
    j->pos[i * 2] = pc;
    j->pos[i * 2 + 1] = pr;
}

static void patch_worker(void *arg, int tid, int nthreads) {
    patchjob *j = arg;
    (void)tid;
    (void)nthreads;
    for (;;) {
        const int i0 = atomic_fetch_add_explicit(&j->next, PATCH_GRAIN, memory_order_relaxed);
        if (i0 >= j->n_patches) break;
        int i1 = i0 + PATCH_GRAIN;
        if (i1 > j->n_patches) i1 = j->n_patches;
        for (int i = i0; i < i1; i++) patch_one(j, i);
    }
}

static void patch_run(const uint8_t *res, int tw, float *pixels, int32_t *pos, int pw, int n) {
    patchjob j;
    j.res = res;
    j.pixels = pixels;
    j.pos = pos;
    j.tw = tw;
    j.pw = pw;
    j.n_patches = n;
    if (n <= 2 * PATCH_GRAIN) {
        for (int i = 0; i < n; i++) patch_one(&j, i);
        return;
    }
    atomic_init(&j.next, 0);
    e2_parallel(patch_worker, &j);
}

int e2_image_preprocess_rgb(const uint8_t *rgb, int width, int height, float *pixels,
                            int32_t *position_ids) {
    int th, tw, ph, pw, n;
    uint8_t *tmp = NULL, *res = NULL;
    const uint8_t *img = rgb;
    coeffs ch, cv;
    int ok = -1;
    memset(&ch, 0, sizeof ch);
    memset(&cv, 0, sizeof cv);
    if (!rgb || !pixels || !position_ids || width <= 0 || height <= 0) {
        e2_set_error("image preprocessing: bad arguments");
        return -1;
    }
    scale_init();
    e2_pool_global_init(0); /* idempotent: no-op once a model, or a previous call, built the pool */
    target_size(height, width, &th, &tw);
    if (th <= 0 || tw <= 0 || (long)th * tw > (long)MAX_PATCHES * PATCH * PATCH) {
        e2_set_error("image preprocessing: %dx%d has no valid resize target", width, height);
        return -1;
    }
    if (th != height || tw != width) {
        passjob hj, vj;
        if (coeffs_build(&ch, width, tw) || coeffs_build(&cv, height, th)) goto done;
        tmp = malloc((size_t)height * tw * 3);
        res = malloc((size_t)th * tw * 3);
        if (!tmp || !res) goto done;
        hj.in = rgb;
        hj.out = tmp;
        hj.in_rowbytes = width * 3;
        hj.out_rowbytes = tw * 3;
        hj.out_rows = height;
        hj.c = &ch;
        hj.vertical = 0;
        pass_run(&hj);
        /* the barrier inside pass_run is what makes the horizontal buffer complete here */
        vj.in = tmp;
        vj.out = res;
        vj.in_rowbytes = tw * 3;
        vj.out_rowbytes = tw * 3;
        vj.out_rows = th;
        vj.c = &cv;
        vj.vertical = 1;
        pass_run(&vj);
        img = res;
    }
    ph = th / PATCH;
    pw = tw / PATCH;
    n = ph * pw;
    patch_run(img, tw, pixels, position_ids, pw, n);
    /* right-pad to MAX_PATCHES with a zero patch and (-1,-1): the tower strips exactly these */
    for (int i = n; i < MAX_PATCHES; i++) {
        memset(pixels + (size_t)i * 768, 0, 768 * sizeof(float));
        position_ids[i * 2] = -1;
        position_ids[i * 2 + 1] = -1;
    }
    ok = n;
done:
    free(res);
    free(tmp);
    coeffs_free(&ch);
    coeffs_free(&cv);
    if (ok < 0) e2_set_error("image preprocessing: out of memory");
    return ok;
}

/* Decode a container to RGB, apply its EXIF orientation, then resize + patchify. PNG goes through
 * the in-tree reader (byte-exact against the reference, no dependency); JPEG and WebP go through the
 * system libraries (see imagedec.c). Orientation is applied here, before the resize, because that is
 * the order the reference uses (load_image does exif_transpose, then the processor resizes). */
static int preprocess_container(const uint8_t *data, size_t len, float *pixels, int32_t *position_ids,
                               int png_only) {
    uint8_t *rgb;
    char msg[256];
    int w = 0, h = 0, o = 1, rc;
    if (png_only && !(len >= 8 && data[0] == 137 && !memcmp(data, "\x89PNG\r\n\x1a\n", 8))) {
        e2_set_error("png: not a PNG");
        return -1;
    }
    rgb = e2_image_decode(data, len, &w, &h, &o, msg, sizeof msg);
    if (!rgb) {
        e2_set_error("%s", msg);
        return -1;
    }
    if (o > 1) {
        rgb = e2_apply_orientation(rgb, o, &w, &h);
        if (!rgb) {
            e2_set_error("image: out of memory applying EXIF orientation");
            return -1;
        }
    }
    rc = e2_image_preprocess_rgb(rgb, w, h, pixels, position_ids);
    free(rgb);
    return rc;
}

int e2_image_preprocess(const uint8_t *data, size_t len, float *pixels, int32_t *position_ids) {
    if (!data) {
        e2_set_error("image: no data");
        return -1;
    }
    return preprocess_container(data, len, pixels, position_ids, 0);
}

int e2_image_preprocess_png(const uint8_t *png, size_t len, float *pixels, int32_t *position_ids) {
    if (!png) {
        e2_set_error("png: no data");
        return -1;
    }
    return preprocess_container(png, len, pixels, position_ids, 1);
}

int e2_image_supported_formats(void) { return e2_image_formats(); }

int e2_image_max_patches(void) { return MAX_PATCHES; }
int e2_image_soft_tokens(int n_patches) { return n_patches / (POOL * POOL); }
