/* e2png.h — the in-tree PNG reader (png.c): 8/16-bit, non-interlaced, gray / RGB / palette /
 * +alpha, no dependencies beyond libc (inflate is in png.c).
 *
 * Renamed away from png.h because libpng's own <png.h> is on the include path in a build that uses
 * it, and one would shadow the other. This reader is the production PNG path only when the build has
 * no libpng (see E2_WITH_PNG); otherwise it is the fallback and the cross-check for libpng. */
#ifndef E2_E2PNG_H
#define E2_E2PNG_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    E2_PNG_OK = 0,
    E2_PNG_SIGNATURE,
    E2_PNG_UNSUPPORTED,
    E2_PNG_TOO_LARGE,
    E2_PNG_CORRUPT,
    E2_PNG_OOM
} e2_png_error;

/* Decodes to tightly packed 8-bit RGB (w*h*3, malloc'd, caller frees). NULL on failure. */
uint8_t *e2_png_decode(const uint8_t *data, size_t len, int *w, int *h, e2_png_error *err);
const char *e2_png_strerror(e2_png_error err);

#endif /* E2_E2PNG_H */
