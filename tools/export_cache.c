/* export_cache — build the model and write it to a packed int8 cache, nothing else.
 *
 * usage: export_cache MODEL_DIR
 *
 * The cache (E2_Q8_WRITE) is written by e2_load, so this tool is one load and one free: no fixtures,
 * no tokenizer, no embedding. Which towers end up in the file is chosen with E2_Q8_TOWERS
 * (text, text+image, text+audio, all); tools/export_models.sh drives the four combinations.
 *
 * Prints one line per export on success, so the caller can collect sizes:
 *   wrote <path> <bytes> bytes, text[+image][+audio]
 */
#include <stdio.h>
#include <stdlib.h>

#include "e2.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    const char *out = getenv("E2_Q8_WRITE");
    if (!out || !out[0]) {
        fprintf(stderr, "%s: E2_Q8_WRITE=<file> is required\n", argv[0]);
        return 2;
    }
    e2_model *m = e2_load(argv[1]);
    if (!m) {
        fprintf(stderr, "%s\n", e2_last_error());
        return 1;
    }
    e2_free(m);
    printf("wrote %s\n", out);
    return 0;
}
