/* e2_image.c - image decode, aspect-preserving bicubic resize, patchify. */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>

#ifdef E2_USE_SYSTEM_IMAGE
#include <png.h>
#include <jpeglib.h>
#include <setjmp.h>
#else
#define PNG_IMPLEMENTATION
#include "../vendor/png.h"
#undef PNG_IMPLEMENTATION
#define JPEG_IMPLEMENTATION
#include "../vendor/jpeg.h"
#undef JPEG_IMPLEMENTATION
#endif
#ifdef E2_USE_WEBP
#include <webp/decode.h>
#endif

static float cubic_a(float x) {
    /* Keys cubic, a=-0.5: the filter PyTorch uses for antialiased bicubic. */
    const float A = -0.5f;
    x = fabsf(x);
    if (x < 1.0f) return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
    return 0.0f;
}

typedef struct { float *w; int *min, *size; int cap; } taps_t;

static void taps_build(taps_t *t, int in, int out) {
    float scale = (float)in / (float)out;
    float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
    float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    int max_interp = (int)ceilf(support) * 2 + 1;
    t->cap = max_interp;
    t->w = calloc((size_t)out * max_interp, sizeof(float));
    t->min = malloc((size_t)out * sizeof(int));
    t->size = malloc((size_t)out * sizeof(int));
    for (int i = 0; i < out; i++) {
        float center = scale * (i + 0.5f);
        int xmin = (int)(center - support + 0.5f);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5f);
        if (xmax > in) xmax = in;
        int xsize = xmax - xmin;
        if (xsize < 0) xsize = 0;
        if (xsize > max_interp) xsize = max_interp;
        float total = 0.0f;
        float *w = t->w + (size_t)i * max_interp;
        for (int j = 0; j < xsize; j++) {
            float v = cubic_a((j + xmin - center + 0.5f) * invscale);
            w[j] = v;
            total += v;
        }
        if (total != 0.0f)
            for (int j = 0; j < xsize; j++) w[j] /= total;
        t->min[i] = xmin;
        t->size[i] = xsize;
    }
}

static void taps_free(taps_t *t) { free(t->w); free(t->min); free(t->size); }

static uint8_t clamp_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return (uint8_t)(v + 0.5f);
}

/* Bicubic antialiased resize, RGB8 in, RGB8 out. */
static void resize_rgb(const uint8_t *in, int iw, int ih, uint8_t *out, int ow, int oh) {
    taps_t tx, ty;
    taps_build(&tx, iw, ow);
    taps_build(&ty, ih, oh);
    float *tmp = malloc((size_t)ih * ow * 3 * sizeof(float));
    /* horizontal */
    for (int y = 0; y < ih; y++) {
        const uint8_t *row = in + (size_t)y * iw * 3;
        for (int x = 0; x < ow; x++) {
            float r = 0, g = 0, b = 0;
            int xm = tx.min[x];
            for (int j = 0; j < tx.size[x]; j++) {
                float w = tx.w[(size_t)x * tx.cap + j];
                const uint8_t *p = row + (size_t)(xm + j) * 3;
                r += w * p[0]; g += w * p[1]; b += w * p[2];
            }
            float *o = tmp + ((size_t)y * ow + x) * 3;
            o[0] = r; o[1] = g; o[2] = b;
        }
    }
    /* vertical */
    for (int y = 0; y < oh; y++) {
        int ym = ty.min[y];
        for (int x = 0; x < ow; x++) {
            float r = 0, g = 0, b = 0;
            for (int j = 0; j < ty.size[y]; j++) {
                float w = ty.w[(size_t)y * ty.cap + j];
                const float *p = tmp + ((size_t)(ym + j) * ow + x) * 3;
                r += w * p[0]; g += w * p[1]; b += w * p[2];
            }
            uint8_t *o = out + ((size_t)y * ow + x) * 3;
            o[0] = clamp_u8(r); o[1] = clamp_u8(g); o[2] = clamp_u8(b);
        }
    }
    free(tmp);
    taps_free(&tx);
    taps_free(&ty);
}

#ifdef E2_USE_SYSTEM_IMAGE
static int decode_png(const char *path, uint8_t **rgb, int *w, int *h) {
    png_image img;
    memset(&img, 0, sizeof(img));
    img.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&img, path))
        return e2_fail("png: %s", img.message);
    img.format = PNG_FORMAT_RGB;
    size_t sz = PNG_IMAGE_SIZE(img);
    uint8_t *buf = malloc(sz ? sz : 1);
    if (!buf) { png_image_free(&img); return e2_fail("oom"); }
    if (!png_image_finish_read(&img, NULL, buf, 0, NULL)) {
        free(buf);
        e2_fail("png: %s", img.message);
        png_image_free(&img);
        return -1;
    }
    *w = (int)img.width;
    *h = (int)img.height;
    *rgb = buf;
    png_image_free(&img);
    return 0;
}

struct e2_jerr {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
};

static void e2_jpeg_exit(j_common_ptr c) {
    struct e2_jerr *e = (struct e2_jerr *)c->err;
    longjmp(e->jb, 1);
}

static int decode_jpeg(const char *path, uint8_t **rgb, int *w, int *h) {
    struct jpeg_decompress_struct cinfo;
    struct e2_jerr jerr;
    FILE *f = NULL;
    uint8_t *buf = NULL;
    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = e2_jpeg_exit;
    if (setjmp(jerr.jb)) {
        free(buf);
        if (f) fclose(f);
        jpeg_destroy_decompress(&cinfo);
        return e2_fail("jpeg decode failed: %s", path);
    }
    jpeg_create_decompress(&cinfo);
    f = fopen(path, "rb");
    if (!f) { jpeg_destroy_decompress(&cinfo); return e2_fail("cannot open image: %s", path); }
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    int iw = (int)cinfo.output_width, ih = (int)cinfo.output_height;
    int comps = cinfo.output_components;
    buf = malloc((size_t)iw * ih * 3);
    if (!buf) { fclose(f); jpeg_destroy_decompress(&cinfo); return e2_fail("oom"); }
    int stride = iw * 3;
    while ((int)cinfo.output_scanline < ih) {
        JSAMPROW row = buf + (size_t)cinfo.output_scanline * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    *w = iw; *h = ih; *rgb = buf;
    (void)comps;
    return 0;
}
#else
static int decode_png(const char *path, uint8_t **rgb, int *w, int *h) {
    png_image *im = png_load(path);
    if (!im) return e2_fail("png decode failed: %s", path);
    *w = im->width; *h = im->height;
    *rgb = malloc((size_t)(*w) * (*h) * 3);
    if (!*rgb) { png_free(im); return e2_fail("oom"); }
    for (int i = 0, n = (*w) * (*h); i < n; i++) {
        const uint8_t *p = im->data + (size_t)i * im->channels;
        uint8_t *o = *rgb + (size_t)i * 3;
        if (im->channels >= 3) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; }
        else { o[0] = o[1] = o[2] = p[0]; }
    }
    png_free(im);
    return 0;
}

static int decode_jpeg(const char *path, uint8_t **rgb, int *w, int *h) {
    jpeg_image *im = jpeg_load(path);
    if (!im) return e2_fail("jpeg decode failed: %s", path);
    *w = im->width; *h = im->height;
    *rgb = malloc((size_t)(*w) * (*h) * 3);
    if (!*rgb) { jpeg_free(im); return e2_fail("oom"); }
    for (int i = 0, n = (*w) * (*h); i < n; i++) {
        const uint8_t *p = im->data + (size_t)i * im->channels;
        uint8_t *o = *rgb + (size_t)i * 3;
        o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
    }
    jpeg_free(im);
    return 0;
}
#endif

#ifdef E2_USE_WEBP
static int decode_webp(const char *path, uint8_t **rgb, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return e2_fail("cannot open image: %s", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data); fclose(f); return e2_fail("cannot read image: %s", path);
    }
    fclose(f);
    int iw = 0, ih = 0;
    if (!WebPGetInfo(data, (size_t)n, &iw, &ih)) { free(data); return e2_fail("webp decode failed"); }
    uint8_t *buf = malloc((size_t)iw * ih * 3);
    if (!buf || !WebPDecodeRGBInto(data, (size_t)n, buf, (size_t)iw * ih * 3, iw * 3)) {
        free(buf); free(data); return e2_fail("webp decode failed: %s", path);
    }
    free(data);
    *w = iw; *h = ih; *rgb = buf;
    return 0;
}
#endif

static int load_rgb_file(const char *path, uint8_t **rgb, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return e2_fail("cannot open image: %s", path);
    uint8_t magic[12];
    size_t got = fread(magic, 1, 12, f);
    fclose(f);
    if (got < 12) return e2_fail("short image: %s", path);

    if (magic[0] == 0x89 && magic[1] == 'P' && magic[2] == 'N' && magic[3] == 'G')
        return decode_png(path, rgb, w, h);
    if (magic[0] == 0xFF && magic[1] == 0xD8)
        return decode_jpeg(path, rgb, w, h);
#ifdef E2_USE_WEBP
    if (!memcmp(magic, "RIFF", 4) && !memcmp(magic + 8, "WEBP", 4))
        return decode_webp(path, rgb, w, h);
#endif
    return e2_fail("unsupported image format: %s", path);
}

int e2_image_load_rgb(const char *path, uint8_t **rgb, int *w, int *h) {
    return load_rgb_file(path, rgb, w, h);
}

/* Aspect-ratio-preserving target size, matching Gemma4ImageProcessor. */
static void target_size(int w, int h, int patch, int max_patches, int pool,
                        int *tw, int *th) {
    double total_px = (double)w * h;
    double target_px = (double)max_patches * patch * patch;
    double factor = sqrt(target_px / total_px);
    double ideal_h = factor * h, ideal_w = factor * w;
    int mult = pool * patch;
    int th_ = (int)floor(ideal_h / mult) * mult;
    int tw_ = (int)floor(ideal_w / mult) * mult;
    int max_side = (max_patches / (pool * pool)) * mult;
    if (th_ == 0 && tw_ == 0) { th_ = mult; tw_ = mult; }
    else if (th_ == 0) { th_ = mult; tw_ = (int)floor((double)w / h) * mult; if (tw_ > max_side) tw_ = max_side; if (tw_ < mult) tw_ = mult; }
    else if (tw_ == 0) { tw_ = mult; th_ = (int)floor((double)h / w) * mult; if (th_ > max_side) th_ = max_side; if (th_ < mult) th_ = mult; }
    *tw = tw_; *th = th_;
}

/* Build [P, 3*patch*patch] float patches and [P,2] int positions. */
int e2_image_prepare(e2_ctx *ctx, const uint8_t *src, int w, int h, int max_soft,
                     float **patches_out, int **pos_out, int *npatches_out,
                     int *pw_out, int *ph_out) {
    int patch = ctx->m.meta.vision_patch;
    int pool = ctx->m.meta.vision_pool;
    int max_patches = max_soft * pool * pool;
    int tw, th;
    target_size(w, h, patch, max_patches, pool, &tw, &th);
    uint8_t *resized = NULL;
    const uint8_t *img = src;
    if (tw != w || th != h) {
        resized = malloc((size_t)tw * th * 3);
        resize_rgb(src, w, h, resized, tw, th);
        img = resized;
    }
    int pw = tw / patch, ph = th / patch;
    int P = pw * ph;
    int psz = 3 * patch * patch;
    /* arena (shared USM on the device path): the patch projection reads it */
    float *pat = e2_arena_alloc(&ctx->scratch, (size_t)P * psz * sizeof(float));
    int *pos = e2_arena_alloc(&ctx->scratch, (size_t)P * 2 * sizeof(int));
    if (!pat || !pos) { free(resized); return -1; }
    int n = 0;
    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            float *o = pat + (size_t)n * psz;
            int idx = 0;
            for (int yy = 0; yy < patch; yy++) {
                const uint8_t *row = img + ((size_t)(py * patch + yy) * tw + px * patch) * 3;
                for (int xx = 0; xx < patch; xx++) {
                    const uint8_t *c = row + xx * 3;
                    /* 2*(v/255 - 0.5) folded into one scale */
                    o[idx++] = c[0] * (2.0f / 255.0f) - 1.0f;
                    o[idx++] = c[1] * (2.0f / 255.0f) - 1.0f;
                    o[idx++] = c[2] * (2.0f / 255.0f) - 1.0f;
                }
            }
            pos[n * 2 + 0] = px;
            pos[n * 2 + 1] = py;
            n++;
        }
    }
    free(resized);
    *patches_out = pat;
    *pos_out = pos;
    *npatches_out = n;
    *pw_out = pw;
    *ph_out = ph;
    return 0;
}
