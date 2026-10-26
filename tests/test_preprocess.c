/* test_preprocess.c — correctness + timing for the two front-end modules: the tokenizer
 * (tokenizer.json) and the image preprocessing (PNG -> paddle-free patch tensor).
 *
 * Usage: e2_test_pre <model_dir> <fixture_dir>
 * Neither module needs the 1.4 GB checkpoint, so this runs in a second.
 *
 * Fixtures (see tools/dump_tokenizer_fixture.py, tools/dump_image_fixture_pre.py):
 *   tokenizer.bin     u32 n, then per case: u32 nbytes, bytes, u32 nids, i32 ids[nids]
 *   tokenizer_mm.bin  u32 n, then per case: u32 nbytes, bytes, u32 nimg, i32 soft[nimg],
 *                                            u32 nids, i32 ids[nids]
 *   image_pre.bin     u32 n, then per image: i32 w, i32 h, u32 pnglen, u8 png[],
 *                                            i32 tw, i32 th, u32 nres, u8 resized[nres],
 *                                            i32 n_patches, u32 full,
 *                                            [f32 pixels[2520*768], i32 pos[2520*2]]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "e2.h"
#include "imagedec.h" /* decode + orientation, for the codec matrix */
#include "e2png.h"   /* the in-tree reader, for the format matrix below */

#define RESCALE_FACTOR 0.00392156862745098f

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int fails;

static uint32_t rd_u32(FILE *f) {
    uint32_t v = 0;
    if (fread(&v, 4, 1, f) != 1) {
        fprintf(stderr, "truncated fixture\n");
        exit(2);
    }
    return v;
}
static int32_t rd_i32(FILE *f) { return (int32_t)rd_u32(f); }

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "oom (%zu)\n", n);
        exit(2);
    }
    return p;
}

/* ---------------------------------------------------------------- tokenizer */
static void bench_text(e2_tokenizer *t, const char *text, long nbytes, const char *label) {
    static int32_t ids[65536];
    int reps = 2000;
    int n = 0;
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double a = now_ms();
        n = e2_encode(t, text, ids, 65536);
        double d = now_ms() - a;
        if (d < best) best = d;
    }
    if (n < 0) {
        fprintf(stderr, "encode failed: %s\n", e2_last_error());
        fails++;
        return;
    }
    printf("  %-22s %6ld B -> %4d ids  %8.2f us/encode  %6.2f MB/s\n", label, nbytes, n, best * 1e3,
           nbytes / best / 1e3);
}

static void test_tokenizer(const char *model_dir, const char *fixture_dir) {
    char path[1024];
    double t0 = now_ms();
    e2_tokenizer *t = e2_tokenizer_load(model_dir);
    if (!t) {
        fprintf(stderr, "tokenizer load failed: %s\n", e2_last_error());
        fails++;
        return;
    }
    printf("tokenizer loaded in %.0f ms\n", now_ms() - t0);

    /* corpus: exact id match against the HF fast tokenizer */
    snprintf(path, sizeof path, "%s/tokenizer.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/dump_tokenizer_fixture.py)\n", path);
        fails++;
        return;
    }
    uint32_t ncases = rd_u32(f);
    int bad = 0;
    long total_bytes = 0;
    int32_t *ids = xmalloc(sizeof(int32_t) * 70000);
    int32_t *ref = xmalloc(sizeof(int32_t) * 70000);
    for (uint32_t c = 0; c < ncases; c++) {
        uint32_t nb = rd_u32(f);
        char *txt = xmalloc(nb + 1);
        if (fread(txt, 1, nb, f) != nb) exit(2);
        txt[nb] = 0;
        uint32_t nids = rd_u32(f);
        if (fread(ref, 4, nids, f) != nids) exit(2);
        int n = e2_encode(t, txt, ids, 70000);
        total_bytes += nb;
        if (n != (int)nids || memcmp(ids, ref, 4 * (size_t)nids)) {
            if (bad < 3) {
                fprintf(stderr, "  MISMATCH '%s': got %d ids, want %u", txt, n, nids);
                for (int i = 0; i < n && i < (int)nids && ids[i] == ref[i]; i++)
                    fprintf(stderr, " (first difference at %d)\n", i), i = n;
                fprintf(stderr, "\n");
                if (n > 0) {
                    fprintf(stderr, "    got ");
                    for (int i = 0; i < n && i < 12; i++) fprintf(stderr, "%d ", ids[i]);
                    fprintf(stderr, "\n    want ");
                    for (uint32_t i = 0; i < nids && i < 12; i++) fprintf(stderr, "%d ", ref[i]);
                    fprintf(stderr, "\n");
                }
            }
            bad++;
        }
        free(txt);
    }
    fclose(f);
    printf("  corpus: %u cases, %ld bytes, %d mismatches  %s\n", ncases, total_bytes, bad,
           bad ? "FAIL" : "PASS");
    if (bad) fails++;

    /* the multimodal layout: <|image|> expanded for one 256-soft-token image */
    snprintf(path, sizeof path, "%s/tokenizer_mm.bin", fixture_dir);
    f = fopen(path, "rb");
    if (f) {
        uint32_t n = rd_u32(f);
        for (uint32_t c = 0; c < n; c++) {
            uint32_t nb = rd_u32(f);
            char *txt = xmalloc(nb + 1);
            if (fread(txt, 1, nb, f) != nb) exit(2);
            txt[nb] = 0;
            uint32_t nimg = rd_u32(f);
            int32_t *soft = xmalloc(sizeof(int32_t) * (nimg ? nimg : 1));
            for (uint32_t i = 0; i < nimg; i++) soft[i] = rd_i32(f);
            uint32_t nids = rd_u32(f);
            if (fread(ref, 4, nids, f) != nids) exit(2);
            int got = e2_encode_mm(t, txt, soft, (int)nimg, ids, 70000);
            int ok = got == (int)nids && !memcmp(ids, ref, 4 * (size_t)nids);
            printf("  mm layout: %u image(s), %d soft tokens -> %d ids  %s\n", nimg,
                   nimg ? soft[0] : 0, got, ok ? "PASS" : "FAIL");
            if (!ok) {
                if (got != (int)nids) fprintf(stderr, "    got %d ids, want %u\n", got, nids);
                else {
                    for (uint32_t i = 0; i < nids; i++)
                        if (ids[i] != ref[i]) {
                            fprintf(stderr, "    first difference at %u: got %d want %d\n", i, ids[i],
                                    ref[i]);
                            break;
                        }
                }
                fails++;
            }
            free(txt);
            free(soft);
        }
        fclose(f);
    }

    bench_text(t, "task: search result | query: What causes the northern lights?", 61, "query (15 tok)");
    {
        char *big = xmalloc(8001);
        for (int i = 0; i < 8000; i++) big[i] = "The quick brown fox jumps over the lazy dog. "[i % 45];
        big[8000] = 0;
        bench_text(t, big, 8000, "8000-byte document");
        free(big);
    }
    free(ids);
    free(ref);
    e2_tokenizer_free(t);
}

/* ---------------------------------------------------------------- image */
static void test_images(const char *fixture_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/image_pre.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/dump_image_fixture_pre.py)\n", path);
        fails++;
        return;
    }
    uint32_t n = rd_u32(f);
    float *pixels = xmalloc(sizeof(float) * E2_MAX_PATCHES * E2_PATCH_DIM);
    float *refpx = xmalloc(sizeof(float) * E2_MAX_PATCHES * E2_PATCH_DIM);
    int32_t *pos = xmalloc(sizeof(int32_t) * E2_MAX_PATCHES * 2);
    int32_t *refpos = xmalloc(sizeof(int32_t) * E2_MAX_PATCHES * 2);
    for (uint32_t i = 0; i < n; i++) {
        int32_t w = rd_i32(f), h = rd_i32(f);
        uint32_t pnglen = rd_u32(f);
        uint8_t *png = xmalloc(pnglen);
        if (fread(png, 1, pnglen, f) != pnglen) exit(2);
        int32_t tw = rd_i32(f), th = rd_i32(f);
        uint32_t nres = rd_u32(f);
        uint8_t *resized = xmalloc(nres);
        if (fread(resized, 1, nres, f) != nres) exit(2);
        int32_t npv = rd_i32(f);
        uint32_t full = rd_u32(f);
        if (full) {
            if (fread(refpx, 4, (size_t)E2_MAX_PATCHES * E2_PATCH_DIM, f) !=
                (size_t)E2_MAX_PATCHES * E2_PATCH_DIM)
                exit(2);
            if (fread(refpos, 4, (size_t)E2_MAX_PATCHES * 2, f) != (size_t)E2_MAX_PATCHES * 2)
                exit(2);
        }

        int got = -1;
        double t = now_ms();
        int reps = 200;
        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            double a = now_ms();
            got = e2_image_preprocess_png(png, pnglen, pixels, pos);
            double d = now_ms() - a;
            if (d < best) best = d;
        }
        t = best;
        printf("  %4dx%-4d png %6u B -> %4dx%-4d  %4d patches  %8.3f ms  %s\n", w, h, pnglen, tw, th,
               got, t, got == npv ? "PASS" : "FAIL");
        if (got != npv) {
            fprintf(stderr, "    %s\n", e2_last_error());
            fails++;
        }

        /* 1. every real patch must reproduce the reference resized pixels. The reference path is
         * torchvision's uint8 antialias kernel (ATen's AVX2 upsample), which is Pillow's algorithm
         * to within a rounding tie: it differs from Pillow itself on ~41 of 1.77M pixels of this
         * fixture. A different summation order cannot be bit-identical to it, so the bar is "every
         * value within 1 LSB, plus a couple of isolated ties at 2" and the counts are reported. */
        {
            int ph = th / 16, pw = tw / 16;
            long exact = 0, off_by_one = 0, worse = 0;
            int first[2] = { -1, -1 };
            long first_want = 0;
            double worst = 0;
            for (int pr = 0; pr < ph; pr++) {
                for (int pc = 0; pc < pw; pc++) {
                    const float *d = pixels + ((size_t)pr * pw + pc) * 768;
                    for (int y = 0; y < 16; y++) {
                        for (int x = 0; x < 48; x++) {
                            /* compare in the byte domain: the float round-trip through
                             * RESCALE_FACTOR is lossy in the last bit and would mask the result */
                            size_t si = ((size_t)(pr * 16 + y) * tw + pc * 16) * 3 + x;
                            long v = lround((double)d[y * 48 + x] / RESCALE_FACTOR);
                            long diff = labs(v - resized[si]);
                            if (diff == 0) exact++;
                            else if (diff == 1) off_by_one++;
                            else {
                                if (!worse) {
                                    first[0] = pr * pw + pc;
                                    first[1] = y * 48 + x;
                                    first_want = resized[si];
                                }
                                worse++;
                            }
                            if (diff > worst) worst = (double)diff;
                        }
                    }
                }
            }
            printf("    resize vs reference: %ld exact, %ld within 1 LSB, %ld worse, max %.0f LSB\n",
                   exact, off_by_one, worse, worst);
            if (worse > 8 || worst > 2.0) {
                fprintf(stderr, "    first worse-than-1-LSB at patch %d offset %d: want %ld\n",
                        first[0], first[1], first_want);
                fails++;
            }
        }
        /* 2. position ids: (patch_col, patch_row) row-major, then (-1,-1) up to 2520 */
        {
            int ph = th / 16, pw = tw / 16, bad = 0;
            for (int pr = 0; pr < ph; pr++)
                for (int pc = 0; pc < pw; pc++) {
                    int idx = pr * pw + pc;
                    if (pos[idx * 2] != pc || pos[idx * 2 + 1] != pr) bad++;
                }
            for (int k = got; k < E2_MAX_PATCHES; k++)
                if (pos[k * 2] != -1 || pos[k * 2 + 1] != -1) bad++;
            for (int k = got * 768; k < E2_MAX_PATCHES * 768; k++)
                if (pixels[k] != 0.0f) bad++;
            if (bad) {
                fprintf(stderr, "    position id / padding mismatch (%d)\n", bad);
                fails++;
            }
        }
        /* 3. the full processor tensor, when the fixture carries it */
        if (full) {
            long bad = 0;
            double worst = 0;
            for (int k = 0; k < E2_MAX_PATCHES * 768; k++) {
                long a = lround((double)pixels[k] / RESCALE_FACTOR);
                long b = lround((double)refpx[k] / RESCALE_FACTOR);
                long diff = labs(a - b);
                if (diff > 1 && bad++ == 0)
                    fprintf(stderr, "    pixel_values mismatch at %d: got %ld want %ld\n", k, a, b);
                if (diff > worst) worst = (double)diff;
            }
            for (int k = 0; k < E2_MAX_PATCHES * 2; k++)
                if (pos[k] != refpos[k]) bad++;
            printf("    full processor tensor: max %.1f LSB, %s\n", worst, bad ? "FAIL" : "PASS");
            if (bad) fails++;
        }
        free(png);
        free(resized);
    }
    fclose(f);
    free(pixels);
    free(refpx);
    free(pos);
    free(refpos);
}

/* Truncated and mutated PNGs must be rejected, never crash, never hang and never allocate without a
 * bound: a malformed stream that inflates forever is the one failure mode a decoder cannot recover
 * from, so the reader caps the output at what the image header can describe. */
static void test_png_robustness(const char *fixture_dir) {
    char path[1024];
    uint8_t buf[65536];
    snprintf(path, sizeof path, "%s/image_pre.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint32_t n = rd_u32(f);
    uint32_t pnglen = 0;
    for (uint32_t i = 0; i < n && i < 1; i++) {
        rd_i32(f);
        rd_i32(f);
        pnglen = rd_u32(f);
        if (pnglen > sizeof buf || fread(buf, 1, pnglen, f) != pnglen) pnglen = 0;
    }
    fclose(f);
    if (!pnglen) return;

    float *pixels = xmalloc(sizeof(float) * E2_MAX_PATCHES * E2_PATCH_DIM);
    int32_t *pos = xmalloc(sizeof(int32_t) * E2_MAX_PATCHES * 2);
    int accepted = 0, rejected = 0;
    for (uint32_t cut = 0; cut <= pnglen; cut += 13) {
        if (e2_image_preprocess_png(buf, cut, pixels, pos) < 0) rejected++;
        else accepted++;
    }
    uint32_t seed = 1;
    for (int it = 0; it < 2000; it++) {
        uint8_t copy[65536];
        seed = seed * 1103515245 + 12345;
        uint32_t len = (seed >> 8) % (pnglen + 1);
        memcpy(copy, buf, len);
        for (int k = 0, muts = 1 + (int)(seed % 6); k < muts && len; k++) {
            seed = seed * 1103515245 + 12345;
            copy[(seed >> 8) % len] = (uint8_t)(seed >> 16);
        }
        if (e2_image_preprocess_png(copy, len, pixels, pos) < 0) rejected++;
        else accepted++;
    }
    printf("  malformed input: %d accepted, %d rejected, no crash/hang  PASS\n", accepted, rejected);
    free(pixels);
    free(pos);
}

/* Every colour type and bit depth the reader supports, with each of the five scanline filters forced
 * onto row 0 and then cycled through the remaining rows. Run through BOTH decoders when the build has
 * libpng, so this doubles as the cross-check between them; the in-tree reader is the only one when
 * libpng is off. This is the test that catches a NULL previous row: the filter bytes live inside the
 * compressed stream, so a byte-mutation fuzzer cannot reach them and every ordinary file the
 * processor writes happens to start with Sub or None. */
static void test_png_formats(const char *fixture_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/png.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/dump_png_fixture.py)\n", path);
        fails++;
        return;
    }
    int with_libpng = !strcmp(e2_png_decoder(), "libpng");
    uint32_t n = rd_u32(f);
    long checked = 0, bad = 0, bad_prod = 0;
    for (uint32_t c = 0; c < n; c++) {
        int32_t w = rd_i32(f), h = rd_i32(f);
        uint32_t pnglen = rd_u32(f);
        uint8_t *png = xmalloc(pnglen);
        if (fread(png, 1, pnglen, f) != pnglen) exit(2);
        uint32_t rgblen = rd_u32(f);
        uint8_t *ref = xmalloc(rgblen);
        if (fread(ref, 1, rgblen, f) != rgblen) exit(2);

        for (int which = 0; which < (with_libpng ? 2 : 1); which++) {
            uint8_t *got;
            int gw = 0, gh = 0;
            long *pbad = which ? &bad_prod : &bad;
            if (which) { /* the production path: e2_image_decode -> libpng */
                char msg[256];
                int go = 1;
                got = e2_image_decode(png, pnglen, &gw, &gh, &go, msg, sizeof msg);
            } else {
                e2_png_error err = E2_PNG_OK;
                got = e2_png_decode(png, pnglen, &gw, &gh, &err);
            }
            if (!got) {
                fprintf(stderr, "  case %u (%s): decode failed\n", c, which ? "libpng" : "in-tree");
                (*pbad)++;
            } else if (gw != w || gh != h || rgblen != (uint32_t)(w * h * 3)) {
                fprintf(stderr, "  case %u (%s): decoded %dx%d, wanted %dx%d\n", c,
                        which ? "libpng" : "in-tree", gw, gh, w, h);
                (*pbad)++;
                free(got);
            } else {
                int diff = 0, first = -1, worst = 0;
                for (int32_t i = 0; i < w * h * 3; i++) {
                    int d = got[i] - ref[i];
                    if (d) {
                        if (first < 0) first = i;
                        if (d < 0) d = -d;
                        if (d > worst) worst = d;
                        diff++;
                    }
                }
                if (diff) {
                    fprintf(stderr, "  case %u (%s, %dx%d): %d bytes differ, max %d (got %d want %d)\n", c,
                            which ? "libpng" : "in-tree", w, h, diff, worst, got[first], ref[first]);
                    (*pbad)++;
                }
                free(got);
            }
        }
        checked++;
        free(png);
        free(ref);
    }
    fclose(f);
    printf("  format matrix: %ld cases (5 filters x row 0 & interior, 8/16-bit, all colour types)\n",
           checked);
    printf("    in-tree reader: %s", bad ? "FAIL" : "PASS");
    if (with_libpng) printf("    libpng (as used): %s", bad_prod ? "FAIL" : "PASS");
    else printf("    libpng: not in this build");
    printf("\n");
    if (bad || bad_prod) fails++;
}

/* Decode+orientation agreement with the reference path, for every container the build supports, and
 * a clean refusal for the corrupt ones. Skipped formats report why instead of failing, so this test
 * is meaningful both with and without libjpeg/libwebp. */
static void test_codecs(const char *fixture_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/codec.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run tools/dump_codec_fixture.py)\n", path);
        fails++;
        return;
    }
    int have = e2_image_supported_formats();
    uint32_t n = rd_u32(f);
    long ok = 0, skipped = 0, bad = 0;
    float *pixels = xmalloc(sizeof(float) * E2_MAX_PATCHES * E2_PATCH_DIM);
    int32_t *pos = xmalloc(sizeof(int32_t) * E2_MAX_PATCHES * 2);
    uint8_t *first[3] = { NULL, NULL, NULL }; /* one decodable sample per container, for the bench */
    uint32_t firstlen[3] = { 0, 0, 0 };
    for (uint32_t c = 0; c < n; c++) {
        uint32_t fmt = rd_u32(f), flen = rd_u32(f);
        uint8_t *file = xmalloc(flen);
        if (fread(file, 1, flen, f) != flen) exit(2);
        int32_t orientation = rd_i32(f);
        int32_t w = rd_i32(f), h = rd_i32(f);
        uint32_t rgblen = rd_u32(f);
        uint8_t *ref = xmalloc(rgblen ? rgblen : 1);
        if (rgblen && fread(ref, 1, rgblen, f) != rgblen) exit(2);

        uint8_t *got = NULL;
        int gw = 0, gh = 0, go = 1;
        char msg[256];
        if (fmt & have) {
            got = e2_image_decode(file, flen, &gw, &gh, &go, msg, sizeof msg);
            /* the expectation is post-orientation, which is what the production path produces, so
             * the test applies it exactly as e2_image_preprocess does */
            if (got && go > 1) {
                got = e2_apply_orientation(got, go, &gw, &gh);
                if (!got) exit(2);
            }
        }
        if (!(fmt & have)) {
            /* the codec is not in this build: the entry point must say so, not crash */
            if (e2_image_preprocess(file, flen, pixels, pos) >= 0) {
                fprintf(stderr, "  case %u: an unsupported format was accepted\n", c);
                bad++;
            } else {
                skipped++;
            }
        } else if (got && orientation != go) {
            fprintf(stderr, "  case %u: orientation %d decoded, EXIF says %d\n", c, go, orientation);
            bad++;
        } else if (!got) {
            if (rgblen) {
                fprintf(stderr, "  case %u: %s\n", c, msg);
                bad++;
            } else {
                ok++; /* a corrupt file refused, which is all that was asked of it */
            }
        } else if (!rgblen) {
            fprintf(stderr, "  case %u: corrupt input decoded instead of being refused\n", c);
            bad++;
        } else if (gw != w || gh != h || (uint32_t)(w * h * 3) != rgblen) {
            fprintf(stderr, "  case %u: got %dx%d, wanted %dx%d\n", c, gw, gh, w, h);
            bad++;
        } else {
            long diff = 0;
            int worst = 0, first = -1;
            for (int32_t i = 0; i < w * h * 3; i++) {
                int d = got[i] - ref[i];
                if (d) {
                    if (d < 0) d = -d;
                    if (d > worst) worst = d;
                    if (first < 0) first = i;
                    diff++;
                }
            }
            if (worst > 1) {
                fprintf(stderr, "  case %u (%dx%d, fmt %u): %ld bytes differ, max %d at %d (got %d want %d)\n",
                        c, w, h, fmt, diff, worst, first, got[first], ref[first]);
                bad++;
            } else {
                ok++;
            }
        }
        free(got);
        if (rgblen && (fmt & have)) { /* remember one good sample per format for the bench */
            int k = (fmt == E2_IMAGE_PNG) ? 0 : (fmt == E2_IMAGE_JPEG) ? 1 : 2;
            if (!first[k]) {
                first[k] = file;
                firstlen[k] = flen;
                free(ref);
                continue;
            }
        }
        free(file);
        free(ref);
    }
    fclose(f);
    printf("  codecs: %ld agreed with the reference (decode + EXIF orientation), %ld skipped "
           "(codec not in this build), %ld bad  %s\n",
           ok, skipped, bad, bad ? "FAIL" : "PASS");
    if (bad) fails++;

    {
        static const char *names[3] = { "PNG", "JPEG (libjpeg-turbo)", "WebP (libwebp)" };
        for (int k = 0; k < 3; k++) {
            if (!first[k]) continue;
            double best = 1e18;
            int np = 0;
            for (int r = 0; r < 200; r++) {
                double a = now_ms();
                np = e2_image_preprocess(first[k], firstlen[k], pixels, pos);
                double d = now_ms() - a;
                if (d < best) best = d;
            }
            if (np > 0)
                printf("  %-22s %6u B -> %4d patches  %7.3f ms\n",
                       k == 0 ? (strcmp(e2_png_decoder(), "libpng") ? "PNG (in-tree)" : "PNG (libpng)")
                              : names[k],
                       firstlen[k], np, best);
            free(first[k]);
        }
    }
    free(pixels);
    free(pos);
}

/* Same contract as the PNG fuzzer above, but for the container sniffing and the codec front ends:
 * \x89 truncated and mutated JPEG/WebP/PNG bytes must all be refused, never crash or hang. This is
 * the only part of the new code paths that parses attacker-controlled bytes itself. */
static void test_codec_robustness(const char *fixture_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/codec.bin", fixture_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    int have = e2_image_supported_formats();
    uint32_t n = rd_u32(f);
    uint8_t *samples[3] = { NULL, NULL, NULL };
    uint32_t lens[3] = { 0, 0, 0 };
    for (uint32_t c = 0; c < n; c++) {
        uint32_t fmt = rd_u32(f), flen = rd_u32(f);
        uint8_t *file = xmalloc(flen);
        if (fread(file, 1, flen, f) != flen) exit(2);
        rd_i32(f);
        rd_i32(f);
        rd_i32(f);
        uint32_t rgblen = rd_u32(f);
        if (rgblen) fseek(f, (long)rgblen, SEEK_CUR);
        if (rgblen && !samples[fmt == E2_IMAGE_PNG ? 0 : fmt == E2_IMAGE_JPEG ? 1 : 2]) {
            int k = fmt == E2_IMAGE_PNG ? 0 : fmt == E2_IMAGE_JPEG ? 1 : 2;
            samples[k] = file;
            lens[k] = flen;
            continue;
        }
        free(file);
    }
    fclose(f);

    float *pixels = xmalloc(sizeof(float) * E2_MAX_PATCHES * E2_PATCH_DIM);
    int32_t *pos = xmalloc(sizeof(int32_t) * E2_MAX_PATCHES * 2);
    long accepted = 0, rejected = 0;
    uint32_t seed = 7;
    for (int k = 0; k < 3; k++) {
        if (!samples[k]) continue;
        int fmt = (k == 0) ? E2_IMAGE_PNG : (k == 1) ? E2_IMAGE_JPEG : E2_IMAGE_WEBP;
        uint8_t *copy = xmalloc(lens[k]);
        for (int it = 0; it < 1500; it++) {
            seed = seed * 1103515245 + 12345;
            uint32_t len = (lens[k] ? (seed >> 9) % lens[k] : 0);
            memcpy(copy, samples[k], len);
            for (int m = 0, muts = 1 + (int)(seed % 5); m < muts && len; m++) {
                seed = seed * 1103515245 + 12345;
                copy[(seed >> 8) % len] = (uint8_t)(seed >> 17);
            }
            int rc = e2_image_preprocess(copy, len, pixels, pos);
            if (rc < 0) rejected++;
            else accepted++;
            (void)fmt;
        }
        free(copy);
    }
    printf("  codec fuzz: %ld accepted, %ld rejected over truncated+mutated %s inputs  PASS\n", accepted,
           rejected, (have & E2_IMAGE_JPEG) ? "PNG/JPEG/WebP" : "PNG");
    free(pixels);
    free(pos);
    for (int k = 0; k < 3; k++) free(samples[k]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <fixture_dir>\n", argv[0]);
        return 2;
    }
    printf("== tokenizer ==\n");
    test_tokenizer(argv[1], argv[2]);
    printf("== image preprocessing ==\n");
    test_images(argv[2]);
    test_png_formats(argv[2]);
    test_codecs(argv[2]);
    test_png_robustness(argv[2]);
    test_codec_robustness(argv[2]);
    printf("%s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
