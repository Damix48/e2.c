#include "e2.h"
#include "threads.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void be32(uint8_t *p, uint32_t v) {
    p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
}
static size_t chunk(uint8_t *p, const char *type, const uint8_t *data, size_t n) {
    be32(p, (uint32_t)n); memcpy(p + 4, type, 4); memcpy(p + 8, data, n);
    uint32_t crc = ~0u;
    for (size_t i = 4; i < n + 8; i++) {
        crc ^= p[i];
        for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
    }
    be32(p + n + 8, ~crc);
    return n + 12;
}
static size_t png(uint8_t *p, int color, int transparent) {
    memcpy(p, "\x89PNG\r\n\x1a\n", 8); size_t n = 8;
    uint8_t ihdr[13] = {0,0,0,1,0,0,0,1,8,0,0,0,0}; ihdr[9] = (uint8_t)color;
    n += chunk(p + n, "IHDR", ihdr, 13);
    uint8_t rgb[] = {200,40,80}, trns[] = {0,200,0,40,0,80};
    if (color == 3) n += chunk(p + n, "PLTE", rgb, 3);
    if (transparent) {
        if (color == 0) { trns[0] = 0; trns[1] = 80; }
        if (color == 3) trns[0] = 0;
        n += chunk(p + n, "tRNS", trns, color == 2 ? 6 : color == 0 ? 2 : 1);
    }
    uint8_t z[20] = {0x78,0x01,1};
    int len = color == 2 ? 4 : 2;
    z[3] = (uint8_t)len; z[4] = 0; z[5] = (uint8_t)~len; z[6] = 255;
    z[7] = 0;
    if (color == 2) memcpy(z + 8, rgb, 3); else z[8] = color == 0 ? 80 : 0;
    uint32_t a = 1, b = 0;
    for (int i = 0; i < len; i++) { a = (a + z[7+i]) % 65521; b = (b + a) % 65521; }
    be32(z + 7 + len, (b << 16) | a);
    n += chunk(p + n, "IDAT", z, (size_t)len + 11);
    n += chunk(p + n, "IEND", z, 0);
    return n;
}
int main(void) {
    size_t size = (size_t)E2_MAX_PATCHES * E2_PATCH_DIM * sizeof(float);
    float *opaque = malloc(size), *transparent = malloc(size);
    int32_t pos[E2_MAX_PATCHES * 2], refpos[E2_MAX_PATCHES * 2];
    assert(opaque && transparent);
    uint8_t dummy = 0;
    assert(e2_image_preprocess_rgb(&dummy, 2147483647, 1, opaque, pos) == -1);
    assert(e2_image_preprocess_rgb(&dummy, 1, 2147483647, opaque, pos) == -1);
    const int colors[] = {0,2,3};
    for (int i = 0; i < 3; i++) {
        uint8_t bytes[256];
        size_t n = png(bytes, colors[i], 0);
        int count = e2_image_preprocess(bytes, n, opaque, refpos); assert(count > 0);
        assert(e2_image_preprocess(NULL, 0, transparent, pos) == -1 && e2_last_error()[0]);
        n = png(bytes, colors[i], 1);
        assert(e2_image_preprocess(bytes, n, transparent, pos) == count && !e2_last_error()[0]);
        assert(!memcmp(opaque, transparent, size)); assert(!memcmp(pos, refpos, sizeof pos));
    }
    e2_pool_global_free(); free(opaque); free(transparent);
    puts("gray/RGB/palette tRNS transparency OK");
    return 0;
}
