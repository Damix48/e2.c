/* imagedec.c — JPEG and WebP through the system libraries, plus the EXIF orientation the processor
 * has to apply before resizing.
 *
 * Why dependencies here and nowhere else: the engine's rule is that everything it does itself is
 * something it can verify to the reference's standard, and a JPEG or WebP decoder is neither small
 * nor verifiable by hand - it is thousands of lines of entropy decoding and prediction over
 * attacker-controlled bytes. libjpeg-turbo and libwebp are also exactly what the HF reference goes
 * through, so linking them is what makes the pixels match rather than merely look close. They are
 * optional (see E2_JPEG / E2_WEBP in CMakeLists.txt): with neither found the engine builds and PNG
 * and raw RGB keep working, which is why this file is behind #ifdefs rather than a hard dependency.
 *
 * Orientation is parsed here rather than taken from the decoders because all three containers carry
 * the same EXIF blob in different wrappers, and because the reference applies it (PIL's
 * exif_transpose) before the resize, which is where it has to happen here too.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imagedec.h"
#include "e2png.h"

void e2_set_error(const char *fmt, ...); /* e2.c */

#if defined(E2_HAVE_LIBPNG)
#include <png.h> /* libpng's, not the in-tree e2png.h */
#include <setjmp.h>
#endif

#if defined(E2_HAVE_JPEG)
#include <jerror.h>
#include <jpeglib.h>
#include <setjmp.h>
#endif

#if defined(E2_HAVE_WEBP)
#include <webp/decode.h>
#endif

/* ---------------------------------------------------------------- EXIF */
/* The orientation tag lives in IFD0 of the TIFF structure that follows the "Exif\0\0" header: a byte
 * order mark, 0x2A, the offset of IFD0, then 12-byte entries of (tag, type, count, value). */
static int rd16(const uint8_t *p, int le) { return le ? (p[0] | (p[1] << 8)) : ((p[0] << 8) | p[1]); }
static uint32_t rd32(const uint8_t *p, int le) {
    if (le) return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int tiff_orientation(const uint8_t *p, size_t n) {
    int le, count, i;
    size_t off;
    if (n < 14) return 1;
    if (p[0] == 'I' && p[1] == 'I') {
        le = 1;
    } else if (p[0] == 'M' && p[1] == 'M') {
        le = 0;
    } else {
        return 1;
    }
    if (rd16(p + 2, le) != 0x2A) return 1;
    off = rd32(p + 4, le);
    if (off + 2 > n) return 1;
    count = rd16(p + off, le);
    for (i = 0; i < count; i++) {
        const uint8_t *e = p + off + 2 + (size_t)i * 12;
        if (off + 2 + (size_t)(i + 1) * 12 > n) break;
        if (rd16(e, le) == 0x0112) {
            int v = rd16(e + 8, le);
            return (v >= 1 && v <= 8) ? v : 1;
        }
    }
    return 1;
}

/* Walk the JPEG markers looking for the APP1 segment the EXIF blob starts in. */
static int jpeg_orientation(const uint8_t *d, size_t n) {
    size_t p = 2; /* past SOI */
    if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return 1;
    while (p + 4 <= n) {
        size_t seglen;
        int marker;
        if (d[p] != 0xFF) break;
        marker = d[p + 1];
        if (marker == 0xFF) { /* fill byte */
            p++;
            continue;
        }
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) {
            p += 2; /* standalone marker, no payload */
            continue;
        }
        if (marker == 0xDA || marker == 0xD9) break; /* scan data: no metadata past here */
        seglen = ((size_t)d[p + 2] << 8) | d[p + 3];
        if (seglen < 2 || p + 2 + seglen > n) break;
        if (marker == 0xE1 && seglen >= 8 && !memcmp(d + p + 4, "Exif\0\0", 6))
            return tiff_orientation(d + p + 10, seglen - 8);
        p += 2 + seglen;
    }
    return 1;
}

/* WebP is RIFF: "RIFF" size "WEBP" then (fourcc, size, payload) chunks padded to even. */
static int webp_orientation(const uint8_t *d, size_t n) {
    size_t p = 12;
    if (n < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WEBP", 4)) return 1;
    while (p + 8 <= n) {
        size_t sz = (size_t)d[p + 4] | ((size_t)d[p + 5] << 8) | ((size_t)d[p + 6] << 16) |
                    ((size_t)d[p + 7] << 24);
        if (p + 8 + sz > n) break;
        if (!memcmp(d + p, "EXIF", 4)) return tiff_orientation(d + p + 8, sz);
        p += 8 + sz + (sz & 1);
    }
    return 1;
}

/* PNG: 4-byte length, 4-byte type, payload, 4-byte CRC. */
static int png_orientation(const uint8_t *d, size_t n) {
    static const uint8_t sig[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
    size_t p = 8;
    if (n < 8 || memcmp(d, sig, 8)) return 1;
    while (p + 12 <= n) {
        size_t sz = ((size_t)d[p] << 24) | ((size_t)d[p + 1] << 16) | ((size_t)d[p + 2] << 8) | d[p + 3];
        if (p + 12 + sz > n) break;
        if (!memcmp(d + p + 4, "eXIf", 4)) return tiff_orientation(d + p + 8, sz);
        if (!memcmp(d + p + 4, "IDAT", 4)) break; /* metadata precedes the pixels */
        p += 12 + sz;
    }
    return 1;
}

int e2_container_orientation(const uint8_t *data, size_t len) {
    if (!data || len < 12) return 1;
    if (!memcmp(data, "RIFF", 4)) return webp_orientation(data, len);
    if (data[0] == 0xFF && data[1] == 0xD8) return jpeg_orientation(data, len);
    if (data[0] == 137 && data[1] == 'P') return png_orientation(data, len);
    return 1;
}

uint8_t *e2_apply_orientation(uint8_t *rgb, int orientation, int *w, int *h) {
    int W = *w, H = *h, ow, oh, x, y;
    uint8_t *out;
    if (orientation <= 1 || orientation > 8 || !rgb) return rgb;
    ow = (orientation >= 5) ? H : W;
    oh = (orientation >= 5) ? W : H;
    out = malloc((size_t)ow * oh * 3);
    if (!out) {
        free(rgb);
        return NULL;
    }
    for (y = 0; y < oh; y++) {
        for (x = 0; x < ow; x++) {
            int sx, sy; /* source pixel for output (x, y) */
            switch (orientation) {
                case 2: sx = W - 1 - x; sy = y; break;               /* mirror horizontally */
                case 3: sx = W - 1 - x; sy = H - 1 - y; break;       /* rotate 180 */
                case 4: sx = x; sy = H - 1 - y; break;               /* mirror vertically */
                case 5: sx = y; sy = x; break;                       /* transpose */
                case 6: sx = y; sy = H - 1 - x; break;               /* rotate 90 clockwise */
                case 7: sx = W - 1 - y; sy = H - 1 - x; break;       /* transverse */
                default: sx = W - 1 - y; sy = x; break;              /* 8: rotate 90 counter-clockwise */
            }
            memcpy(out + ((size_t)y * ow + x) * 3, rgb + ((size_t)sy * W + sx) * 3, 3);
        }
    }
    free(rgb);
    *w = ow;
    *h = oh;
    return out;
}

/* ---------------------------------------------------------------- JPEG */
#if defined(E2_HAVE_JPEG)
struct jerr {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
    char msg[JMSG_LENGTH_MAX];
    int truncated;
};

/* libjpeg's default handler writes to stderr and then exits the process, which is not an option for
 * a decoder reading untrusted bytes: every error unwinds to the caller instead. */
static void jerr_exit(j_common_ptr cinfo) {
    struct jerr *e = (struct jerr *)cinfo->err;
    (*cinfo->err->format_message)(cinfo, e->msg);
    longjmp(e->jb, 1);
}

/* Warnings are dropped rather than printed (a library has no business writing to the application's
 * stderr, and a malformed file must not be able to spam it), but a truncated image is remembered so
 * the caller can refuse it: libjpeg happily returns the scanlines it managed to decode with the rest
 * filled in, and PIL - and therefore the reference - raises on exactly that. */
static void jerr_emit(j_common_ptr cinfo, int msg_level) {
    struct jerr *e = (struct jerr *)cinfo->err;
    if (msg_level < 0 && cinfo->err->msg_code == JWRN_JPEG_EOF) {
        e->truncated = 1;
        (*cinfo->err->format_message)(cinfo, e->msg);
    }
}

static uint8_t *jpeg_decode(const uint8_t *data, size_t len, int *w, int *h, char *err, size_t errcap) {
    struct jpeg_decompress_struct cinfo;
    struct jerr jerr;
    /* volatile because these are written between setjmp and longjmp: without it the compiler is
     * free to keep them in registers and the cleanup below reads stale values, which leaked the RGB
     * buffer and the whole libjpeg memory manager (~12 KiB) on every malformed JPEG. */
    uint8_t *volatile buf = NULL;
    volatile int created = 0;
    int w1 = 0, h1 = 0;

    memset(&cinfo, 0, sizeof cinfo);
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jerr_exit;
    jerr.pub.emit_message = jerr_emit;
    jerr.msg[0] = 0;
    jerr.truncated = 0;
    if (setjmp(jerr.jb)) {
        free((void *)buf);
        if (created) jpeg_destroy_decompress(&cinfo);
        snprintf(err, errcap, "jpeg: %s", jerr.msg[0] ? jerr.msg : "decode failed");
        return NULL;
    }
    jpeg_create_decompress(&cinfo);
    created = 1;
    jpeg_mem_src(&cinfo, data, (unsigned long)len);
    jpeg_read_header(&cinfo, TRUE);
    /* the same conversion the reference relies on: gray and CMYK go to RGB, no ICC transform */
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    w1 = (int)cinfo.output_width;
    h1 = (int)cinfo.output_height;
    if (w1 <= 0 || h1 <= 0 || (uint64_t)w1 * h1 > (1u << 28)) {
        if (created) jpeg_destroy_decompress(&cinfo);
        snprintf(err, errcap, "jpeg: bad dimensions %dx%d", w1, h1);
        return NULL;
    }
    buf = malloc((size_t)w1 * h1 * 3);
    if (!buf) {
        if (created) jpeg_destroy_decompress(&cinfo);
        snprintf(err, errcap, "jpeg: out of memory");
        return NULL;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = (uint8_t *)buf + (size_t)cinfo.output_scanline * w1 * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    if (jerr.truncated) {
        free((void *)buf);
        snprintf(err, errcap, "jpeg: %s", jerr.msg[0] ? jerr.msg : "truncated image");
        return NULL;
    }
    *w = w1;
    *h = h1;
    return (uint8_t *)buf;
}
#endif /* E2_HAVE_JPEG */

/* ---------------------------------------------------------------- WebP */
#if defined(E2_HAVE_WEBP)
static uint8_t *webp_decode(const uint8_t *data, size_t len, int *w, int *h, char *err, size_t errcap) {
    int ww = 0, hh = 0;
    uint8_t *buf;
    if (!WebPGetInfo(data, len, &ww, &hh) || ww <= 0 || hh <= 0) {
        snprintf(err, errcap, "webp: not a WebP image");
        return NULL;
    }
    if ((uint64_t)ww * hh > (1u << 28)) {
        snprintf(err, errcap, "webp: image too large (%dx%d)", ww, hh);
        return NULL;
    }
    buf = WebPDecodeRGB(data, len, &ww, &hh);
    if (!buf) {
        snprintf(err, errcap, "webp: decode failed");
        return NULL;
    }
    *w = ww;
    *h = hh;
    return buf;
}
#endif /* E2_HAVE_WEBP */

/* ---------------------------------------------------------------- libpng */
#if defined(E2_HAVE_LIBPNG)
/* libpng's own defaults are used for everything that is not a decision this engine has to make
 * (CRC policy, chunk limits, unknown chunks), which is the point of using it. What is pinned here is
 * only the output format the rest of the pipeline expects: RGB8, with the alpha channel and the low
 * byte of a 16-bit sample dropped. `png_set_gamma` and `png_set_alpha_mode` are deliberately NOT
 * called - the reference does no colour management either, and turning either on would silently move
 * every pixel. */
struct pctx {
    char msg[256];
    const uint8_t *data;
    size_t len, pos;
};

static void png_read_cb(png_structp p, png_bytep out, png_size_t n) {
    struct pctx *c = (struct pctx *)png_get_io_ptr(p);
    if (c->pos + n > c->len) png_error(p, "unexpected end of PNG data");
    memcpy(out, c->data + c->pos, n);
    c->pos += n;
}

static void png_err_cb(png_structp p, png_const_charp msg) {
    struct pctx *c = (struct pctx *)png_get_error_ptr(p);
    snprintf(c->msg, sizeof c->msg, "%s", msg);
    longjmp(png_jmpbuf(p), 1);
}

/* Warnings are dropped: a library has no business writing to the application's stderr and a
 * malformed file must not be able to spam it. libpng's default CRC policy is left alone, so a bad
 * CRC discards an ancillary chunk with a warning but fails a critical one - stricter than the
 * in-tree reader, which ignores CRCs entirely, and closer to what Pillow does. */
static void png_warn_cb(png_structp p, png_const_charp msg) {
    (void)p;
    (void)msg;
}

static uint8_t *libpng_decode(const uint8_t *data, size_t len, int *w, int *h, char *err, size_t errcap) {
    png_structp png = NULL;
    png_infop info = NULL;
    struct pctx ctx;
    uint8_t *volatile buf = NULL;
    uint32_t ww = 0, hh = 0;
    int color, depth, passes, y;

    memset(&ctx, 0, sizeof ctx);
    ctx.data = data;
    ctx.len = len;
    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &ctx, png_err_cb, png_warn_cb);
    if (!png) {
        snprintf(err, errcap, "png: cannot create a libpng reader");
        return NULL;
    }
    info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        snprintf(err, errcap, "png: cannot create a libpng info struct");
        return NULL;
    }
    if (setjmp(png_jmpbuf(png))) {
        free((void *)buf);
        png_destroy_read_struct(&png, &info, NULL);
        snprintf(err, errcap, "png: %s", ctx.msg[0] ? ctx.msg : "decode failed");
        return NULL;
    }
    /* the same budget the in-tree reader enforces, applied before libpng allocates anything: per
     * side, then the total, so a hostile IHDR cannot ask for an 800 MB buffer */
    png_set_user_limits(png, 1u << 20, 1u << 20);
    /* Match Pillow, which is what the reference goes through: it sets warn-and-use and never calls
     * png_read_end, so a bad CRC on a critical chunk is used anyway (a corrupted IDAT *byte* is still
     * caught by zlib's adler32) and a missing or truncated IEND is not an error. Being stricter here
     * was measurably wrong: it rejected two malformed files that Pillow accepts. */
    png_set_crc_action(png, PNG_CRC_WARN_USE, PNG_CRC_DEFAULT);
    png_set_read_fn(png, &ctx, png_read_cb);
    png_read_info(png, info);
    ww = png_get_image_width(png, info);
    hh = png_get_image_height(png, info);
    if (!ww || !hh || (uint64_t)ww * hh > (1u << 28)) {
        png_destroy_read_struct(&png, &info, NULL);
        snprintf(err, errcap, "png: image too large (%ux%u)", ww, hh);
        return NULL;
    }
    color = png_get_color_type(png, info);
    depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png); /* high byte, as the in-tree reader does */
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (color & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    passes = png_set_interlace_handling(png); /* Adam7: the in-tree reader refuses it, this does not */
    png_read_update_info(png, info);
    if (png_get_channels(png, info) != 3 || png_get_bit_depth(png, info) != 8) {
        png_destroy_read_struct(&png, &info, NULL);
        snprintf(err, errcap, "png: could not normalise to RGB8");
        return NULL;
    }
    buf = malloc((size_t)ww * hh * 3);
    if (!buf) {
        png_destroy_read_struct(&png, &info, NULL);
        snprintf(err, errcap, "png: out of memory");
        return NULL;
    }
    for (int pass = 0; pass < passes; pass++) {
        for (y = 0; y < (int)hh; y++) {
            uint8_t *row = (uint8_t *)buf + (size_t)y * ww * 3;
            png_read_row(png, row, NULL);
        }
    }
    /* png_read_end is deliberately not called: the pixels are complete after the last row, and
     * Pillow does not call it either, so a file whose IEND is missing or truncated decodes here
     * exactly as it does there. */
    png_destroy_read_struct(&png, &info, NULL);
    *w = (int)ww;
    *h = (int)hh;
    return (uint8_t *)buf;
}
#endif /* E2_HAVE_LIBPNG */

/* ---------------------------------------------------------------- dispatch */
const char *e2_png_decoder(void) {
#if defined(E2_HAVE_LIBPNG)
    return "libpng";
#else
    return "in-tree";
#endif
}
int e2_image_formats(void) {
    int f = E2_FMT_PNG;
#if defined(E2_HAVE_JPEG)
    f |= E2_FMT_JPEG;
#endif
#if defined(E2_HAVE_WEBP)
    f |= E2_FMT_WEBP;
#endif
    return f;
}

uint8_t *e2_image_decode(const uint8_t *data, size_t len, int *w, int *h, int *orientation, char *err,
                         size_t errcap) {
    uint8_t *rgb = NULL;
    if (!data || len < 12) {
        snprintf(err, errcap, "not an image: too short");
        return NULL;
    }
    if (!memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WEBP", 4)) {
#if defined(E2_HAVE_WEBP)
        rgb = webp_decode(data, len, w, h, err, errcap);
#else
        snprintf(err, errcap, "webp: this build has no WebP support (rebuild with -DE2_WEBP=ON)");
#endif
    } else if (data[0] == 0xFF && data[1] == 0xD8) {
#if defined(E2_HAVE_JPEG)
        rgb = jpeg_decode(data, len, w, h, err, errcap);
#else
        snprintf(err, errcap, "jpeg: this build has no JPEG support (rebuild with -DE2_JPEG=ON)");
#endif
    } else if (data[0] == 137 && len >= 8) {
#if defined(E2_HAVE_LIBPNG)
        rgb = libpng_decode(data, len, w, h, err, errcap);
#else
        e2_png_error perr = E2_PNG_OK;
        rgb = e2_png_decode(data, len, w, h, &perr); /* in-tree, no dependency */
        if (!rgb) snprintf(err, errcap, "png: %s", e2_png_strerror(perr));
#endif
    } else {
        snprintf(err, errcap, "unrecognised container (want PNG, JPEG or WebP)");
        return NULL;
    }
    if (!rgb) return NULL;
    *orientation = e2_container_orientation(data, len);
    return rgb;
}
