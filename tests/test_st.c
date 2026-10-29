#include "test_tmpdir.h"
#include "st.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void file(const char *path, const char *json, size_t payload) {
    FILE *f = fopen(path, "wb"); assert(f);
    uint64_t n = (strlen(json) + 7) & ~(uint64_t)7;
    assert(fwrite(&n, 8, 1, f) == 1);
    assert(fwrite(json, 1, strlen(json), f) == strlen(json));
    for (size_t i = strlen(json); i < n; i++) fputc(' ', f);
    for (size_t i = 0; i < payload; i++) fputc(0, f);
    assert(!fclose(f));
}
int main(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/e2-st-XXXXXX", e2_test_tmpdir());
    int fd = mkstemp(path); assert(fd >= 0); close(fd);
    const char *valid = "{\"__metadata__\":{\"format\":\"pt\"},\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    file(path, valid, 8);
    st_file f;
    assert(!st_open(&f, path));
    assert(f.count == 1 && st_numel(st_find(&f, "x")) == 4);
    assert(st_bf16(&f, st_find(&f, "x")) != NULL);
    size_t full = f.size;
    st_close(&f);
    for (size_t cut = 0; cut < full; cut++) {
        file(path, valid, 8);
        assert(!truncate(path, (off_t)cut));
        assert(st_open(&f, path) == -1 && st_error()[0]);
        assert(f.fd == -1 && !f.base && !f.tensors);
    }
    const char *bad[] = {
        "{", "{\"x\":", "{\"x\":{\"shape\":[", "{\"x\":{\"dtype\":\"BF16\"}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[-1],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[9223372036854775807,8],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[999999999999999999999],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[8,0]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,10]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[1,9]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,6]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4,],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]},\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}",
        "{\"x\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]},\"y\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}",
        "{} trailing", "{\"__metadata__\":{\"x\":true"
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        file(path, bad[i], 8);
        assert(st_open(&f, path) == -1);
        st_close(&f);
    }
    FILE *out = fopen(path, "wb"); assert(out);
    uint64_t huge = UINT64_MAX; assert(fwrite(&huge, 8, 1, out) == 1); fclose(out);
    assert(st_open(&f, path) == -1);
    char nested[256]; memset(nested, '[', 100); memset(nested + 100, ']', 100); nested[200] = 0;
    char header[300]; snprintf(header, sizeof header, "{\"__metadata__\":%s}", nested);
    file(path, header, 0); assert(st_open(&f, path) == -1);
    file(path, "{\"x\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}", 4);
    assert(!st_open(&f, path)); assert(st_f32(&f, st_find(&f, "x"))); st_close(&f);
    unlink(path);
    puts("safetensors bounds and malformed headers OK");
    return 0;
}
