#include "../src/tokenizer.c"
#include <assert.h>

int main(void) {
    const char *bad[] = { "\xc2" "A", "\xc0\x80", "\xc1\xbf", "\xe0\x80\xaf",
        "\xed\xa0\x80", "\xf0\x80\x80\x80", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80",
        "\x80", "\xc2", "\xe2\x82", "\xf0\x9f\x98", "\xe2" "AB", "\xf0\x9f" "AB" };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        uint32_t cp; int len;
        assert(utf8_decode(bad[i], bad[i] + strlen(bad[i]), &cp, &len) == -1);
    }
    const uint32_t valid[] = { 0x41, 0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff, 0x10000, 0x10ffff };
    for (size_t i = 0; i < sizeof valid / sizeof valid[0]; i++) {
        char b[4]; int n = utf8_encode(valid[i], b), len; uint32_t cp;
        assert(!utf8_decode(b, b + n, &cp, &len) && cp == valid[i] && len == n);
        for (int cut = 0; cut < n; cut++) assert(utf8_decode(b, b + cut, &cp, &len) == -1);
    }
    const char digits[] = {'1','2','3'}; long value;
    js j = {digits, digits + sizeof digits, digits};
    assert(!js_int(&j, &value) && value == 123 && j.p == j.end);
    const char huge[] = "999999999999999999999999999999999999";
    j = (js){huge, huge + sizeof huge - 1, huge};
    assert(js_int(&j, &value) == -1);
    int32_t ids[8];
    assert(e2_encode(NULL, "x", ids, 8) == -1 && e2_last_error()[0]);
    assert(e2_tokenizer_load(NULL) == NULL && e2_last_error()[0]);
    e2_tokenizer t = { .bos = 2, .eos = 1 };
    assert(e2_encode(&t, "", ids, 1) == -1 && e2_last_error()[0]);
    assert(e2_encode(&t, "", ids, 8) == 2 && !e2_last_error()[0]);
    int32_t negative = -1;
    assert(e2_encode_mm(&t, "", &negative, 1, ids, 8) == -1 && e2_last_error()[0]);
    assert(e2_encode_mma(&t, "", NULL, 0, NULL, 1, ids, 8) == -1 && e2_last_error()[0]);
    assert(e2_encode_mm(&t, "", NULL, 0, ids, 8) == 2 && !e2_last_error()[0]);
    puts("UTF-8 boundaries and tokenizer error contract OK");
    return 0;
}
