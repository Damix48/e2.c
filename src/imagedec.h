/* imagedec.h — container sniffing, the JPEG/WebP front ends, and EXIF orientation.
 *
 * The image decoders the processor needs are commodities: libjpeg-turbo and libwebp are the same
 * libraries the HF reference goes through, so linking them is how the engine gets bit-identical
 * pixels instead of a hand-written JPEG/WebP parser (several thousand lines of security-sensitive
 * code). PNG stays in-tree because it is already byte-exact against the reference and needs no
 * dependency to be. All of this is optional: with neither library found, the engine still builds and
 * PNG/raw-RGB keep working (see the E2_JPEG / E2_WEBP options in CMakeLists.txt).
 */
#ifndef E2_IMAGEDEC_H
#define E2_IMAGEDEC_H

#include <stddef.h>
#include <stdint.h>

/* Container formats the build understands. */
#define E2_FMT_PNG 1
#define E2_FMT_JPEG 2
#define E2_FMT_WEBP 4

/* Which of E2_FMT_* are decodable in this build. */
int e2_image_formats(void);

/* Which PNG decoder the build selects: "libpng" or "in-tree". Diagnostic, and the tests use it to
 * say which path they exercised. */
const char *e2_png_decoder(void);

/* Sniff the container, decode to tightly packed 8-bit RGB (malloc'd, caller frees) and return the
 * EXIF orientation in *orientation (1 = as stored). NULL on failure with a message in err. */
uint8_t *e2_image_decode(const uint8_t *data, size_t len, int *w, int *h, int *orientation, char *err,
                         size_t errcap);

/* The EXIF orientation of a container's metadata (1 when absent or unparsable). Exposed separately
 * because the in-tree PNG reader decodes the pixels itself and only needs the tag. */
int e2_container_orientation(const uint8_t *data, size_t len);

/* Rotate/flip an RGB8 buffer to its EXIF orientation. Takes ownership of `rgb` and returns the
 * buffer to use (a new one when the dimensions swap or a flip is needed), updating w/h in place. */
uint8_t *e2_apply_orientation(uint8_t *rgb, int orientation, int *w, int *h);

#endif /* E2_IMAGEDEC_H */
