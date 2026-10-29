/* File decoding, fallback parity, anti-aliasing and malformed inputs; no checkpoint needed. */
#include "test_tmpdir.h"
#include "e2.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(E2_HAVE_LIBSNDFILE)
#include <sndfile.h>
#endif

#define E2_TEST_PI 3.14159265358979323846
static void put16(uint8_t *p, unsigned v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { put16(p, v); put16(p + 2, v >> 16); }
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint8_t *make_wav(int format, int bits, int rate, int channels, int frames, int ext,
                          const double *samples, size_t *len) {
    size_t bytes = (size_t)frames * channels * (bits / 8);
    int fmt_len = ext ? 40 : format == 3 ? 18 : 16;
    *len = 12 + 8 + fmt_len + 12 + (format == 3 ? 12 : 0) + 8 + bytes + (bytes & 1);
    uint8_t *w = calloc(*len, 1); assert(w);
    memcpy(w, "RIFF", 4); put32(w + 4, (uint32_t)*len - 8); memcpy(w + 8, "WAVEfmt ", 8);
    put32(w + 16, fmt_len); put16(w + 20, ext ? 0xfffe : (unsigned)format);
    put16(w + 22, channels); put32(w + 24, rate); put32(w + 28, rate * channels * (bits / 8));
    put16(w + 32, channels * (bits / 8)); put16(w + 34, bits);
    if (ext) {
        static const uint8_t tail[12] = {0,0,16,0,128,0,0,170,0,56,155,113};
        put16(w + 36, 22); put16(w + 38, bits); put32(w + 44, format); memcpy(w + 48, tail, 12);
    }
    size_t at = 20 + fmt_len;
    memcpy(w + at, "JUNK", 4); put32(w + at + 4, 3); at += 12; /* odd chunk + pad */
    if (format == 3) { memcpy(w + at, "fact", 4); put32(w + at + 4, 4); put32(w + at + 8, frames); at += 12; }
    memcpy(w + at, "data", 4); put32(w + at + 4, (uint32_t)bytes); at += 8;
    for (int i = 0; i < frames * channels; i++) {
        double x = samples ? samples[i] : .25;
        uint8_t *p = w + at + (size_t)i * (bits / 8);
        if (format == 1) {
            int64_t v = (int64_t)llround(x * (double)(INT64_C(1) << (bits - 1)));
            int64_t lo = -(INT64_C(1) << (bits - 1)), hi = -lo - 1;
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            if (bits == 8) v += 128;
            uint64_t u = (uint64_t)v;
            for (int b = 0; b < bits / 8; b++) p[b] = (uint8_t)(u >> (8 * b));
        } else if (bits == 32) {
            float f = (float)x; uint32_t u; memcpy(&u, &f, 4); put32(p, u);
        } else {
            uint64_t u; memcpy(&u, &x, 8); put32(p, (uint32_t)u); put32(p + 4, (uint32_t)(u >> 32));
        }
    }
    return w;
}
static size_t chunk_at(const uint8_t *w, size_t len, const char *name) {
    for (size_t at = 12; at + 8 <= len; at += 8 + get32(w + at + 4) + (get32(w + at + 4) & 1))
        if (!memcmp(w + at, name, 4)) return at;
    assert(!"chunk absent"); return 0;
}
static void rejected(const uint8_t *w, size_t len, float *out) {
    out[0] = out[E2_AUDIO_MAX_SAMPLES - 1] = 12345;
    assert(e2_audio_preprocess_wav(w, len, out) == -1 && e2_last_error()[0]);
    assert(out[0] == 12345 && out[E2_AUDIO_MAX_SAMPLES - 1] == 12345);
    assert(e2_audio_preprocess(w, len, out) == -1 && e2_last_error()[0]);
    assert(out[0] == 12345 && out[E2_AUDIO_MAX_SAMPLES - 1] == 12345);
}
static void formats(float *out, float *ref) {
    const int bits[] = {8,16,24,32,32,64};
    double samples[512]; const double pattern[] = {-1,-.75,-.5,-.25,0,.25,.5,.75};
    for (int i = 0; i < 512; i++) samples[i] = pattern[i % 8];
    for (int kind = 0; kind < 6; kind++) for (int ext = 0; ext < 2; ext++) {
        size_t len; int format = kind < 4 ? 1 : 3;
        uint8_t *w = make_wav(format, bits[kind], 16000, 2, 256, ext, samples, &len);
        assert(e2_audio_preprocess_wav(w, len, ref) == 256 && !e2_last_error()[0]);
        assert(e2_audio_preprocess(w, len, out) == 256 && !e2_last_error()[0]);
        for (int i = 0; i < 256; i++) {
            float expected = (float)((samples[2*i] + samples[2*i+1]) * .5);
            assert(ref[i] == expected && out[i] == expected);
        }
        if (ext && bits[kind] == 24) {
            put16(w + 38, 20); /* valid bits left-aligned within a 24-bit container */
            assert(e2_audio_preprocess_wav(w, len, ref) == 256);
            assert(e2_audio_preprocess(w, len, out) == 256);
            assert(!memcmp(ref, out, 256 * sizeof *out));
        }
        free(w);
    }
    for (int bits64 = 0; bits64 < 2; bits64++) {
        for (int i = 0; i < 512; i++) samples[i] = (i & 1) ? -2.0 : 2.0;
        size_t len; uint8_t *w = make_wav(3, bits64 ? 64 : 32, 16000, 1, 512, 0, samples, &len);
        assert(e2_audio_preprocess(w, len, out) == 512);
        for (int i = 0; i < 512; i++) assert(out[i] == ((i & 1) ? -1.f : 1.f));
        free(w);
        const double nonfinite[] = {NAN, INFINITY, -INFINITY};
        for (int bad = 0; bad < 3; bad++) {
            samples[511] = nonfinite[bad];
            w = make_wav(3, bits64 ? 64 : 32, 16000, 1, 512, 0, samples, &len);
            for (int i = 0; i < E2_AUDIO_MAX_SAMPLES; i++) out[i] = 12345;
            rejected(w, len, out);
            for (int i = 0; i < E2_AUDIO_MAX_SAMPLES; i++) assert(out[i] == 12345);
            free(w);
        }
    }
    puts("PCM 8/16/24/32, float 32/64, extensible, downmix, clipping and NaN refusal OK");
}
static void rates(float *out, float *ref) {
    const int rates[] = {8000,16000,22050,44100,47999,48000,96000,192000};
    for (size_t r = 0; r < sizeof rates / sizeof rates[0]; r++) {
        int rate = rates[r], frames = rate / 5 + 1;
        double *samples = malloc(sizeof *samples * frames); assert(samples);
        for (int i = 0; i < frames; i++) samples[i] = .5 * sin(2 * E2_TEST_PI * 1000 * i / rate);
        size_t len; uint8_t *w = make_wav(3, 64, rate, 1, frames, 0, samples, &len);
        int n = (int)(((int64_t)frames * 16000 + rate - 1) / rate);
        assert(e2_audio_preprocess_wav(w, len, ref) == n);
        assert(e2_audio_preprocess(w, len, out) == n);
        assert(!memcmp(out, ref, (size_t)n * sizeof *out));
        double err = 0;
        for (int i = 64; i < n - 64; i++) {
            double d = out[i] - .5 * sin(2 * E2_TEST_PI * 1000 * i / 16000);
            err += d*d;
        }
        double rms = sqrt(err / (n - 128));
        printf("%d -> 16000 Hz: %d samples, 1 kHz sine RMS error %.3g\n", rate, n, rms);
        assert(rms < 1e-4);
        free(w);
        if (rate > 16000) {
            for (int i = 0; i < frames; i++) samples[i] = .5 * sin(2 * E2_TEST_PI * 12000 * i / rate);
            w = make_wav(3, 64, rate, 1, frames, 0, samples, &len);
            assert(e2_audio_preprocess(w, len, out) == n);
            double power = 0;
            for (int i = 64; i < n - 64; i++) power += out[i]*out[i];
            assert(sqrt(power / (n - 128)) < 1e-3); /* reject energy above target Nyquist */
            free(w);
        }
        for (int i = 0; i < frames; i++) samples[i] = .25;
        w = make_wav(3, 64, rate, 1, frames, 0, samples, &len);
        assert(e2_audio_preprocess(w, len, out) == n);
        for (int i = 0; i < n; i++) assert(fabsf(out[i] - .25f) < 1e-6f);
        free(w); free(samples);
    }
    size_t len; uint8_t *w = make_wav(1, 16, 16000, 1, 481000, 0, NULL, &len);
    out[-1] = out[E2_AUDIO_MAX_SAMPLES] = 12345;
    assert(e2_audio_preprocess(w, len, out) == E2_AUDIO_MAX_SAMPLES);
    assert(out[-1] == 12345 && out[E2_AUDIO_MAX_SAMPLES] == 12345);
    for (int i = 0; i < E2_AUDIO_MAX_SAMPLES; i++) assert(out[i] == .25f);
    free(w);
    w = make_wav(1, 16, 16000, 1, 160, 0, NULL, &len); rejected(w, len, out); free(w);
    puts("resampling rates, analytical sine, stopband rejection, 30 s cap and output bounds OK");
}
static void malformed(float *out) {
    size_t len; uint8_t *w = make_wav(1, 16, 16000, 1, 256, 0, NULL, &len);
    for (size_t cut = 0; cut < len; cut++) rejected(w, cut, out);
    uint8_t *copy = malloc(len); assert(copy);
    const int fields[] = {22,24,32,34};
    const unsigned values[] = {0,0,1,12};
    for (int i = 0; i < 4; i++) {
        memcpy(copy, w, len); put16(copy + fields[i], values[i]); rejected(copy, len, out);
    }
    memcpy(copy, w, len); put32(copy + 4, UINT32_MAX); rejected(copy, len, out);
    memcpy(copy, w, len); put16(copy + 22, 33); rejected(copy, len, out);
    memcpy(copy, w, len); put32(copy + 24, 192001); rejected(copy, len, out);
    memcpy(copy, w, len); put32(copy + 28, 0); rejected(copy, len, out);
    size_t data = chunk_at(w, len, "data");
    memcpy(copy, w, len); put32(copy + data + 4, 511); rejected(copy, len, out);
    memcpy(copy, w, len); memcpy(copy + data, "JUNK", 4); rejected(copy, len, out);
    memcpy(copy, w, len); memcpy(copy + 12, "JUNK", 4); rejected(copy, len, out);
    assert(e2_audio_preprocess(NULL, len, out) == -1 && e2_last_error()[0]);
    assert(e2_audio_preprocess(w, len, NULL) == -1 && e2_last_error()[0]);
    assert(e2_audio_preprocess(w, SIZE_MAX, out) == -1 && e2_last_error()[0]);
    uint32_t seed = 1;
    for (int it = 0; it < 256; it++) {
        memcpy(copy, w, len);
        for (int k = 0; k < 3; k++) {
            seed = seed * 1664525 + 1013904223;
            copy[(seed >> 8) % len] ^= (uint8_t)seed;
        }
        int n = e2_audio_preprocess(copy, len, out);
        assert(n == -1 || (n >= 161 && n <= E2_AUDIO_MAX_SAMPLES));
        if (n < 0) assert(e2_last_error()[0]);
        else for (int i = 0; i < n; i++) assert(isfinite(out[i]) && fabsf(out[i]) <= 1);
    }
    assert(e2_audio_preprocess(w, len, out) == 256 && !e2_last_error()[0]);
    free(copy); free(w);
    puts("truncated/malformed WAVs, decoder mutations and error recovery OK");
}
#if defined(E2_HAVE_LIBSNDFILE)
static void containers(float *out) {
    const int formats[] = {SF_FORMAT_WAV|SF_FORMAT_PCM_16, SF_FORMAT_FLAC|SF_FORMAT_PCM_16,
        SF_FORMAT_AIFF|SF_FORMAT_PCM_16, SF_FORMAT_AU|SF_FORMAT_PCM_16,
        SF_FORMAT_OGG|SF_FORMAT_VORBIS, 0x230000|0x82, SF_FORMAT_RF64|SF_FORMAT_PCM_16,
        SF_FORMAT_OGG|0x64}; /* MPEG layer III (older headers supported) */
    const int flags[] = {E2_AUDIO_WAV,E2_AUDIO_FLAC,E2_AUDIO_AIFF,E2_AUDIO_AU,E2_AUDIO_OGG,E2_AUDIO_MP3,E2_AUDIO_WAV,E2_AUDIO_OGG};
    const char *names[] = {"WAV","FLAC","AIFF","AU","Ogg/Vorbis","MP3","RF64","Ogg/Opus"};
    double samples[16000];
    for (int i = 0; i < 16000; i++) samples[i] = .2 * sin(2 * E2_TEST_PI * 1000 * i / 16000);
    for (size_t c = 0; c < sizeof formats / sizeof formats[0]; c++) {
        SF_INFO info = {0}; info.samplerate = 16000; info.channels = 1; info.format = formats[c];
        if (!sf_format_check(&info)) { printf("SKIP writing %s: not in this libsndfile\n", names[c]); continue; }
        assert(e2_audio_supported_formats() & flags[c]);
        char path[512];
        snprintf(path, sizeof path, "%s/e2-audio-format-XXXXXX", e2_test_tmpdir());
        int fd = mkstemp(path); assert(fd >= 0); close(fd);
        SNDFILE *sf = sf_open(path, SFM_WRITE, &info); assert(sf);
        assert(sf_writef_double(sf, samples, 16000) == 16000 && !sf_close(sf));
        FILE *f = fopen(path, "rb"); assert(f); assert(!fseek(f, 0, SEEK_END));
        long len = ftell(f); assert(len > 0); rewind(f);
        uint8_t *data = malloc((size_t)len); assert(data);
        assert(fread(data, 1, (size_t)len, f) == (size_t)len); fclose(f); unlink(path);
        int n = e2_audio_preprocess(data, (size_t)len, out);
        if (n < 0) fprintf(stderr, "%s: %s\n", names[c], e2_last_error());
        assert(n >= 14000 && n <= 18048);
        double re = 0, im = 0;
        for (int i = 1000; i < n - 1000; i++) {
            assert(isfinite(out[i]) && fabsf(out[i]) <= 1);
            re += out[i] * cos(2 * E2_TEST_PI * 1000 * i / 16000);
            im += out[i] * sin(2 * E2_TEST_PI * 1000 * i / 16000);
        }
        double amp = 2 * sqrt(re*re + im*im) / (n - 2000);
        assert(amp > .15 && amp < .25);
        free(data); printf("%s via libsndfile: %d samples OK\n", names[c], n);
    }
}
#endif
int main(void) {
    float *buf = malloc(sizeof(float) * (E2_AUDIO_MAX_SAMPLES + 2));
    float *ref = malloc(sizeof(float) * E2_AUDIO_MAX_SAMPLES);
    assert(buf && ref); float *out = buf + 1;
    assert(e2_audio_supported_formats() & E2_AUDIO_WAV);
#if !defined(E2_HAVE_LIBSNDFILE)
    assert(e2_audio_supported_formats() == E2_AUDIO_WAV);
#endif
    formats(out, ref); rates(out, ref); malformed(out);
#if defined(E2_HAVE_LIBSNDFILE)
    containers(out);
#endif
    free(buf); free(ref);
    return 0;
}
