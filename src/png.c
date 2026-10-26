/* png.c — minimal PNG reader for the image front end.
 *
 * Contains its own DEFLATE decoder (stored / fixed / dynamic Huffman, RFC 1951) because the engine
 * has no dependencies beyond libc, and PNG because it is the one container a lossless, exactly
 * reproducible fixture can be frozen in. Not supported, and rejected with E2_PNG_UNSUPPORTED:
 * Adam7 interlacing, bit depths 1/2/4, and colour types other than gray/RGB/palette/+alpha.
 * CRC32s are not verified (the payload is validated by the decoder itself).
 *
 * Four things carry its speed, each worth more than 2x on the fixtures:
 *   - a 9-bit first-level Huffman table, so a symbol costs a mask, a load and a shift, with the
 *     canonical walk kept as the fallback for longer codes and for the tail of the stream;
 *   - matches copied as whole blocks, with the written region doubling each time when the distance
 *     is shorter than the match - a flat image is distance 3 with runs of hundreds of bytes, and
 *     copying that byte by byte was the single largest cost in the whole module;
 *   - literals through an inlined one-compare store (a photographic image is mostly literals, and a
 *     call plus two range checks per byte cost more than everything else in that case);
 *   - the output allocated once from IHDR and the scanline filters undone in place, with the switch
 *     hoisted out of the per-byte loop and three interleaved chains for Sub and Paeth (their chain
 *     has a distance of `bpp`, which the compiler cannot see because bpp is a runtime value).
 * The output is capped at what IHDR describes, so a corrupt stream cannot inflate without bound.
 */
#include "e2png.h"

#include <stdlib.h>
#include <string.h>

const char *e2_png_strerror(e2_png_error err) {
    switch (err) {
        case E2_PNG_OK: return "ok";
        case E2_PNG_SIGNATURE: return "not a PNG";
        case E2_PNG_UNSUPPORTED: return "unsupported PNG variant (interlaced, <8-bit, or bad colour type)";
        case E2_PNG_TOO_LARGE: return "image too large (limit is 2^20 per side and 2^28 pixels)";
        case E2_PNG_CORRUPT: return "corrupt PNG stream";
        default: return "out of memory";
    }
}

/* ---------------------------------------------------------------- inflate */
#define MAXBITS 15
#define MAXLCODES 286
#define MAXDCODES 30
#define MAXCODES (MAXLCODES + MAXDCODES)
#define FIXLCODES 288
#define FASTBITS 9 /* first-level decode table: DEFLATE's most frequent codes are <= 8 bits */

typedef struct {
    int16_t count[MAXBITS + 1];
    int16_t symbol[MAXCODES];
    /* fast[len <= FASTBITS]: bit i = stream bit i, so a canonical code of length L occupies
     * bitrev_L(code) plus any multiple of 1<<L - a lookup is one mask and one load. 0 = miss. */
    uint32_t fast[1 << FASTBITS];
} huff;

typedef struct {
    const uint8_t *in;
    size_t in_len, in_pos;
    uint32_t bitbuf;
    int bitcnt;
    uint8_t *out;
    size_t out_len, out_cap, max_out; /* a malformed stream can inflate without bound */
    e2_png_error err;
} infs;

static int bits_ensure(infs *s, int need) {
    while (s->bitcnt < need) {
        if (s->in_pos >= s->in_len) return -1;
        s->bitbuf |= (uint32_t)s->in[s->in_pos++] << s->bitcnt;
        s->bitcnt += 8;
    }
    return 0;
}

static int bits(infs *s, int need) {
    int v;
    if (bits_ensure(s, need)) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    v = (int)(s->bitbuf & ((1u << need) - 1));
    s->bitbuf >>= need;
    s->bitcnt -= need;
    return v;
}

static inline void bits_drop(infs *s, int n) {
    s->bitbuf >>= n;
    s->bitcnt -= n;
}

static int bitrev(int v, int n) {
    int r = 0;
    for (int i = 0; i < n; i++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

/* Fill the first-level table from the canonical code lengths. */
static void huff_fast(huff *h) {
    int code = 0, k = 0;
    memset(h->fast, 0, sizeof h->fast);
    for (int len = 1; len <= MAXBITS; len++) {
        for (int n = 0; n < h->count[len]; n++, k++) {
            if (len <= FASTBITS) {
                uint32_t e = ((uint32_t)len << 16) | (uint32_t)(uint16_t)h->symbol[k];
                int step = 1 << len;
                for (int idx = bitrev(code, len); idx < (1 << FASTBITS); idx += step) h->fast[idx] = e;
            }
            code++;
        }
        code <<= 1;
    }
}

static int huff_build(huff *h, const int16_t *length, int n) {
    int symbol, len, left;
    int16_t offs[MAXBITS + 1];
    for (len = 0; len <= MAXBITS; len++) h->count[len] = 0;
    for (symbol = 0; symbol < n; symbol++) h->count[length[symbol]]++;
    if (h->count[0] == n) return 0; /* incomplete, but allowed for a zero-length code */
    left = 1;
    for (len = 1; len <= MAXBITS; len++) {
        left <<= 1;
        left -= h->count[len];
        if (left < 0) return left; /* over-subscribed */
    }
    offs[1] = 0;
    for (len = 1; len < MAXBITS; len++) offs[len + 1] = (int16_t)(offs[len] + h->count[len]);
    for (symbol = 0; symbol < n; symbol++)
        if (length[symbol] != 0) h->symbol[offs[length[symbol]]++] = (int16_t)symbol;
    huff_fast(h);
    return left;
}

static int huff_decode(infs *s, const huff *h) {
    int len, code = 0, first = 0, index = 0, count, b;
    uint32_t e;
    /* one load resolves any code of at most FASTBITS bits; a short tail at the end of the stream
     * falls through to the walk below, which needs only the bits that are actually there */
    if (bits_ensure(s, FASTBITS) == 0 && (e = h->fast[s->bitbuf & ((1u << FASTBITS) - 1)]) != 0) {
        bits_drop(s, (int)(e >> 16));
        return (int)(e & 0xFFFF);
    }
    for (len = 1; len <= MAXBITS; len++) {
        b = bits(s, 1);
        if (b < 0) return -1;
        code |= b;
        count = h->count[len];
        if (code - count < first) return h->symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    s->err = E2_PNG_CORRUPT;
    return -1;
}

/* Grow the output to hold `need` bytes total. */
static int out_grow(infs *s, size_t need) {
    size_t cap = s->out_cap ? s->out_cap : (1u << 16);
    uint8_t *p;
    while (cap < need) cap *= 2;
    p = realloc(s->out, cap);
    if (!p) {
        s->err = E2_PNG_OOM;
        return -1;
    }
    s->out = p;
    s->out_cap = cap;
    return 0;
}

/* Literals go through an inlined one-compare store: a photographic image is mostly literals, and a
 * call plus two range checks per byte was 4 ms of the 6 ms that image took. */
static inline int out_byte(infs *s, uint8_t v) {
    if (s->out_len >= s->out_cap || s->out_len >= s->max_out) {
        if (s->out_len >= s->max_out) {
            s->err = E2_PNG_CORRUPT; /* more data than the image header can describe */
            return -1;
        }
        if (out_grow(s, s->out_len + 1)) return -1;
    }
    s->out[s->out_len++] = v;
    return 0;
}

/* Reserve room for `n` bytes without deciding where they come from (matches and stored blocks). */
static int out_reserve(infs *s, size_t n) {
    if (s->out_len + n > s->max_out) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    if (s->out_len + n > s->out_cap && out_grow(s, s->out_len + n)) return -1;
    return 0;
}

static int codes(infs *s, const huff *lencode, const huff *distcode) {
    static const int16_t lens[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                      35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
    static const int16_t lext[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                      3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
    static const int16_t dists[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
                                       193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
                                       6145, 8193, 12289, 16385, 24577 };
    static const int16_t dext[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                      6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
    int symbol, len, dist, b;
    for (;;) {
        symbol = huff_decode(s, lencode);
        if (symbol < 0) return -1;
        if (symbol == 256) return 0;
        if (symbol < 256) {
            if (out_byte(s, (uint8_t)symbol)) return -1;
            continue;
        }
        symbol -= 257;
        if (symbol >= 29) {
            s->err = E2_PNG_CORRUPT;
            return -1;
        }
        b = bits(s, lext[symbol]);
        if (b < 0) return -1;
        len = lens[symbol] + b;
        symbol = huff_decode(s, distcode);
        if (symbol < 0 || symbol >= 30) {
            s->err = E2_PNG_CORRUPT;
            return -1;
        }
        b = bits(s, dext[symbol]);
        if (b < 0) return -1;
        dist = dists[symbol] + b;
        if ((size_t)dist > s->out_len) {
            s->err = E2_PNG_CORRUPT;
            return -1;
        }
        /* one capacity check and a logarithmic number of block copies per match. A match whose
         * distance is smaller than its length repeats the `dist` bytes just written (dist = 3 is
         * the whole of a flat RGB image), so copying byte by byte was the single largest cost in
         * inflate: here the written region doubles each time, each copy is non-overlapping, and a
         * 700 KiB run at distance 3 takes ~20 memcpys instead of 700k byte moves. */
        if (out_reserve(s, (size_t)len)) return -1;
        {
            uint8_t *d = s->out + s->out_len;
            size_t done, n;
            if ((size_t)dist >= (size_t)len) {
                memcpy(d, d - dist, (size_t)len);
            } else {
                n = (size_t)dist < (size_t)len ? (size_t)dist : (size_t)len;
                memcpy(d, d - dist, n);
                done = n;
                while (done < (size_t)len) {
                    size_t take = done < (size_t)len - done ? done : (size_t)len - done;
                    memcpy(d + done, d, take);
                    done += take;
                }
            }
            s->out_len += (size_t)len;
        }
    }
}

static int fixed_block(infs *s) {
    int i;
    int16_t lengths[FIXLCODES];
    huff lencode, distcode;
    for (i = 0; i < 144; i++) lengths[i] = 8;
    for (; i < 256; i++) lengths[i] = 9;
    for (; i < 280; i++) lengths[i] = 7;
    for (; i < FIXLCODES; i++) lengths[i] = 8;
    if (huff_build(&lencode, lengths, FIXLCODES) < 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    for (i = 0; i < MAXDCODES; i++) lengths[i] = 5;
    if (huff_build(&distcode, lengths, MAXDCODES) < 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    return codes(s, &lencode, &distcode);
}

static int dynamic_block(infs *s) {
    static const int16_t order[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
    int nlen, ndist, ncode, index, b;
    int16_t lengths[MAXCODES];
    huff lencode, distcode;
    nlen = bits(s, 5);
    ndist = bits(s, 5);
    ncode = bits(s, 4);
    if (nlen < 0 || ndist < 0 || ncode < 0) return -1;
    nlen += 257;
    ndist += 1;
    ncode += 4;
    if (nlen > MAXLCODES || ndist > MAXDCODES) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    for (index = 0; index < ncode; index++) {
        b = bits(s, 3);
        if (b < 0) return -1;
        lengths[order[index]] = (int16_t)b;
    }
    for (; index < 19; index++) lengths[order[index]] = 0;
    if (huff_build(&lencode, lengths, 19) < 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    index = 0;
    while (index < nlen + ndist) {
        int symbol = huff_decode(s, &lencode), repeat;
        if (symbol < 0) return -1;
        if (symbol < 16) {
            lengths[index++] = (int16_t)symbol;
            continue;
        }
        repeat = 0;
        if (symbol == 16) {
            if (index == 0) {
                s->err = E2_PNG_CORRUPT;
                return -1;
            }
            repeat = lengths[index - 1];
            b = bits(s, 2);
            if (b < 0) return -1;
            symbol = 3 + b;
        } else if (symbol == 17) {
            b = bits(s, 3);
            if (b < 0) return -1;
            symbol = 3 + b;
        } else {
            b = bits(s, 7);
            if (b < 0) return -1;
            symbol = 11 + b;
        }
        if (index + symbol > nlen + ndist) {
            s->err = E2_PNG_CORRUPT;
            return -1;
        }
        while (symbol--) lengths[index++] = (int16_t)repeat;
    }
    if (lengths[256] == 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    if (huff_build(&lencode, lengths, nlen) < 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    /* the distance code may be incomplete, which huff_build allows */
    if (huff_build(&distcode, lengths + nlen, ndist) < 0) {
        s->err = E2_PNG_CORRUPT;
        return -1;
    }
    return codes(s, &lencode, &distcode);
}

static int inflate(infs *s) {
    int last, type;
    s->bitbuf = 0;
    s->bitcnt = 0;
    do {
        last = bits(s, 1);
        type = bits(s, 2);
        if (last < 0 || type < 0) return -1;
        if (type == 0) {
            /* stored: skip to a byte boundary, then copy LEN bytes */
            s->bitbuf = 0;
            s->bitcnt = 0;
            if (s->in_pos + 4 > s->in_len) {
                s->err = E2_PNG_CORRUPT;
                return -1;
            }
            int n = s->in[s->in_pos] | (s->in[s->in_pos + 1] << 8);
            int nlen = s->in[s->in_pos + 2] | (s->in[s->in_pos + 3] << 8);
            if (nlen != (~n & 0xFFFF)) { /* LEN is followed by its 16-bit complement */
                s->err = E2_PNG_CORRUPT;
                return -1;
            }
            s->in_pos += 4;
            if (s->in_pos + (size_t)n > s->in_len) {
                s->err = E2_PNG_CORRUPT;
                return -1;
            }
            if (out_reserve(s, (size_t)n)) return -1;
            memcpy(s->out + s->out_len, s->in + s->in_pos, (size_t)n);
            s->out_len += (size_t)n;
            s->in_pos += (size_t)n;
        } else if (type == 1) {
            if (fixed_block(s)) return -1;
        } else if (type == 2) {
            if (dynamic_block(s)) return -1;
        } else {
            s->err = E2_PNG_CORRUPT;
            return -1;
        }
    } while (!last);
    return 0;
}

/* ---------------------------------------------------------------- chunks */
static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

uint8_t *e2_png_decode(const uint8_t *data, size_t len, int *w, int *h, e2_png_error *err) {
    static const uint8_t sig[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
    uint32_t width = 0, height = 0;
    int depth = 0, ctype = 0;
    uint8_t pal[256 * 3];
    int pal_n = 0;
    uint8_t *idat = NULL;
    size_t idat_len = 0, idat_cap = 0;
    uint8_t *rgb = NULL;
    size_t pos = 8;
    infs inf;
    memset(&inf, 0, sizeof inf);

    *err = E2_PNG_OK;
    if (len < 8 || memcmp(data, sig, 8)) {
        *err = E2_PNG_SIGNATURE;
        return NULL;
    }
    while (pos + 12 <= len) {
        uint32_t clen = be32(data + pos);
        const uint8_t *type = data + pos + 4;
        const uint8_t *body = data + pos + 8;
        if (pos + 12 + clen > len) break;
        if (!memcmp(type, "IHDR", 4)) {
            if (clen != 13) goto bad;
            width = be32(body);
            height = be32(body + 4);
            depth = body[8];
            ctype = body[9];
            if (body[10] != 0 || body[11] != 0 || body[12] != 0) goto unsup; /* compression/filter/interlace */
            if (depth != 8 && depth != 16) goto unsup;
            if (ctype != 0 && ctype != 2 && ctype != 3 && ctype != 4 && ctype != 6) goto unsup;
        } else if (!memcmp(type, "PLTE", 4)) {
            if (clen % 3 || clen > sizeof pal) goto bad;
            memcpy(pal, body, clen);
            pal_n = (int)(clen / 3);
        } else if (!memcmp(type, "IDAT", 4)) {
            if (idat_len + clen > idat_cap) {
                size_t cap = idat_cap ? idat_cap : (1u << 16);
                uint8_t *p;
                while (cap < idat_len + clen) cap *= 2;
                p = realloc(idat, cap);
                if (!p) {
                    *err = E2_PNG_OOM;
                    goto done;
                }
                idat = p;
                idat_cap = cap;
            }
            memcpy(idat + idat_len, body, clen);
            idat_len += clen;
        } else if (!memcmp(type, "IEND", 4)) {
            break;
        }
        pos += 12 + clen;
    }
    if (!width || !height || !idat) goto bad;
    if (width > (1u << 20) || height > (1u << 20) || (uint64_t)width * height > (1u << 28)) goto toobig;
    if (ctype == 3 && !pal_n) goto bad;
    /* the concatenated IDAT is one zlib stream: 2-byte header, deflate body, 4-byte adler32 */
    if (idat_len < 6 || (idat[0] & 0x0F) != 8 || (idat[0] >> 4) > 7 ||
        ((idat[0] << 8) | idat[1]) % 31 != 0)
        goto bad;

    {
        int channels = ctype == 0 || ctype == 3 ? 1 : ctype == 2 ? 3 : ctype == 4 ? 2 : 4;
        int step = depth / 8;
        size_t stride = (size_t)width * channels * step;
        inf.max_out = (stride + 1) * height;
        /* The exact size is known from IHDR, so allocate it once: growing from 64 KiB would
         * realloc-chain through 4 sizes and copy ~1 MiB every call. One extra stride of zeroed
         * space follows it, and that is the "previous row" of row 0 - the Up/Average/Paeth filters
         * read it, and keeping it materialised is what lets those loops stay branch-free. */
        if (out_grow(&inf, inf.max_out + stride)) {
            *err = E2_PNG_OOM;
            goto done;
        }
        memset(inf.out + inf.max_out, 0, stride);
    }
    inf.in = idat + 2;
    inf.in_len = idat_len - 2;
    if (inflate(&inf)) {
        *err = inf.err;
        goto done;
    }

    {
        int channels = ctype == 0 || ctype == 3 ? 1 : ctype == 2 ? 3 : ctype == 4 ? 2 : 4;
        int step = depth / 8;
        int bpp = channels * step;
        size_t stride = (size_t)width * channels * step;
        if (inf.out_len < (stride + 1) * height) goto bad;
        /* Undo the per-scanline filters in place (PNG 9.2). Dropping each row's filter byte moves the
         * row down by `y + 1` bytes, so the result is contiguous with no second buffer: a row's
         * source is always above everything written so far, and inside a row the read runs one byte
         * ahead of the write. The switch on the filter type is hoisted out of the loop, so it is one
         * tight loop per scanline instead of a branch per byte. */
        for (uint32_t y = 0; y < height; y++) {
            uint8_t *cur = inf.out + (size_t)y * stride;
            const uint8_t *src = inf.out + (size_t)y * (stride + 1) + 1;
            const uint8_t *prev = y ? cur - stride : inf.out + inf.max_out;
            size_t x = 0;
            switch (src[-1]) {
                case 0:
                    memmove(cur, src, stride);
                    break;
                case 1:
                    for (; x < (size_t)bpp; x++) cur[x] = src[x];
                    if (bpp == 3) {
                        /* `cur[x] += cur[x-3]` is a chain with a distance of bpp, so three of them
                         * can run at once; bpp is a runtime value, so gcc cannot see that */
                        for (; x + 2 < stride; x += 3)
                            for (int k = 0; k < 3; k++)
                                cur[x + k] = (uint8_t)(src[x + k] + cur[x + k - 3]);
                    }
                    for (; x < stride; x++) cur[x] = (uint8_t)(src[x] + cur[x - bpp]);
                    break;
                case 2:
                    for (; x < stride; x++) cur[x] = (uint8_t)(src[x] + prev[x]);
                    break;
                case 3:
                    for (; x < (size_t)bpp; x++) cur[x] = (uint8_t)(src[x] + (prev[x] >> 1));
                    for (; x < stride; x++)
                        cur[x] = (uint8_t)(src[x] + ((cur[x - bpp] + prev[x]) >> 1));
                    break;
                case 4:
                    for (; x < (size_t)bpp; x++) cur[x] = (uint8_t)(src[x] + prev[x]);
                    if (bpp == 3) {
                        /* same distance-3 chains as Sub: the predictor reads cur[x-bpp], not cur[x-1] */
                        for (; x + 2 < stride; x += 3) {
                            for (int k = 0; k < 3; k++) {
                                int a = cur[x + k - 3], b = prev[x + k], c = prev[x + k - 3];
                                int p = a + b - c;
                                int pa = p > a ? p - a : a - p;
                                int pb = p > b ? p - b : b - p;
                                int pc = p > c ? p - c : c - p;
                                int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                                cur[x + k] = (uint8_t)(src[x + k] + pred);
                            }
                        }
                    }
                    for (; x < stride; x++) {
                        int a = cur[x - bpp], b = prev[x], c = prev[x - bpp];
                        int p = a + b - c;
                        int pa = p > a ? p - a : a - p;
                        int pb = p > b ? p - b : b - p;
                        int pc = p > c ? p - c : c - p;
                        int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                        cur[x] = (uint8_t)(src[x] + pred);
                    }
                    break;
                default:
                    goto bad;
            }
        }
        if (ctype == 2 && step == 1) {
            /* 8-bit RGB is the common case and the tightest: the unfiltered buffer is already the
             * answer, so ownership moves to the caller */
            rgb = inf.out;
            inf.out = NULL;
        } else {
            /* expand to RGB8, dropping 16-bit low bytes and alpha */
            rgb = malloc((size_t)width * height * 3);
            if (!rgb) {
                *err = E2_PNG_OOM;
                goto done;
            }
            for (uint32_t y = 0; y < height; y++) {
                const uint8_t *cur = inf.out + stride * y;
                uint8_t *o = rgb + (size_t)y * width * 3;
                for (uint32_t x = 0; x < width; x++) {
                    const uint8_t *p = cur + (size_t)x * channels * step;
                    switch (ctype) {
                        case 0: o[0] = o[1] = o[2] = p[0]; break;
                        case 4: o[0] = o[1] = o[2] = p[0]; break;
                        case 2: o[0] = p[0]; o[1] = p[step]; o[2] = p[2 * step]; break;
                        case 6: o[0] = p[0]; o[1] = p[step]; o[2] = p[2 * step]; break;
                        default: {
                            unsigned i = p[0];
                            if ((int)i >= pal_n) goto bad;
                            o[0] = pal[i * 3];
                            o[1] = pal[i * 3 + 1];
                            o[2] = pal[i * 3 + 2];
                            break;
                        }
                    }
                    o += 3;
                }
            }
        }
        *w = (int)width;
        *h = (int)height;
    }
    goto done;

unsup:
    *err = E2_PNG_UNSUPPORTED;
    goto done;
toobig:
    *err = E2_PNG_TOO_LARGE;
    goto done;
bad:
    *err = E2_PNG_CORRUPT;
done:
    free(idat);
    free(inf.out);
    if (*err != E2_PNG_OK) {
        free(rgb);
        return NULL;
    }
    return rgb;
}
