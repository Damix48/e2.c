/* Audio containers -> bounded 16 kHz mono PCM. No model or worker pool is needed. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/e2.h"
#if defined(E2_HAVE_LIBSNDFILE)
#include <sndfile.h>
#endif

void e2_set_error(const char *fmt, ...);

#define E2_AUDIO_MIN_RATE 8000
#define E2_AUDIO_MAX_RATE 192000
#define E2_AUDIO_MAX_CHANNELS 32
#define E2_AUDIO_PI 3.14159265358979323846

typedef struct {
    const uint8_t *data;
    size_t bytes;
    int rate, channels, bits, format, align;
} wav_info;

static unsigned le16(const uint8_t *p) { return (unsigned)p[0] | (unsigned)p[1] << 8; }
static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int shape_ok(int rate, int channels) {
    if (rate < E2_AUDIO_MIN_RATE || rate > E2_AUDIO_MAX_RATE || channels < 1 || channels > E2_AUDIO_MAX_CHANNELS) {
        e2_set_error("audio: require 8000..192000 Hz and 1..32 channels (got %d Hz, %d channels)", rate, channels);
        return -1;
    }
    return 0;
}
/* Validate chunk bounds even with libsndfile: it otherwise accepts some truncated PCM WAVs. */
static int wav_header(const uint8_t *data, size_t len, wav_info *w) {
    memset(w, 0, sizeof *w);
    if (len < 12 || memcmp(data, "RIFF", 4) || memcmp(data + 8, "WAVE", 4)) {
        e2_set_error("wav: not a RIFF/WAVE file"); return -1;
    }
    const uint64_t end64 = (uint64_t)le32(data + 4) + 8;
    if (end64 < 12 || end64 > len) goto bad;
    const size_t end = (size_t)end64;
    const uint8_t *fmt = NULL;
    size_t fmt_size = 0;
    for (size_t at = 12; at < end;) {
        if (end - at < 8) goto bad;
        const uint8_t *chunk = data + at;
        const size_t n = le32(chunk + 4);
        at += 8;
        if (n > end - at || (n & 1) > end - at - n) goto bad;
        if (!memcmp(chunk, "fmt ", 4)) {
            if (fmt || n < 16) goto bad;
            fmt = data + at; fmt_size = n;
        } else if (!memcmp(chunk, "data", 4)) {
            if (w->data) goto bad;
            w->data = data + at; w->bytes = n;
        }
        at += n + (n & 1);
    }
    if (!fmt || !w->data || !w->bytes) goto bad;
    w->format = (int)le16(fmt);
    w->channels = (int)le16(fmt + 2);
    uint32_t rate = le32(fmt + 4);
    if (rate > E2_AUDIO_MAX_RATE) goto bad;
    w->rate = (int)rate;
    w->align = (int)le16(fmt + 12);
    w->bits = (int)le16(fmt + 14);
    if (shape_ok(w->rate, w->channels)) return -1;
    if (fmt_size > 16 && (fmt_size < 18 || le16(fmt + 16) > fmt_size - 18)) goto bad;
    if (w->format == 0xfffe) {
        static const uint8_t guid_tail[12] = {0,0,16,0,128,0,0,170,0,56,155,113};
        if (fmt_size < 40 || le16(fmt + 16) < 22 || le16(fmt + 18) > (unsigned)w->bits) goto bad;
        /* Other GUIDs (e.g. ambisonics) belong to libsndfile, not the PCM fallback. */
        if (!memcmp(fmt + 28, guid_tail, 12) && le32(fmt + 24) <= 0xffff)
            w->format = (int)le32(fmt + 24);
    }
    if (w->align <= 0) goto bad;
    if (w->format == 1 || w->format == 3) {
        if ((w->format == 1 && w->bits != 8 && w->bits != 16 && w->bits != 24 && w->bits != 32) ||
            (w->format == 3 && w->bits != 32 && w->bits != 64) ||
            w->align != w->channels * (w->bits / 8) || w->bytes % (size_t)w->align ||
            le32(fmt + 8) != (uint32_t)w->rate * (unsigned)w->align) goto bad;
    }
    return 0;
bad:
    e2_set_error("wav: malformed or truncated header/data");
    return -1;
}
static double clip(double x) { return x < -1 ? -1 : x > 1 ? 1 : x; }
static int mix_sample(double x, double *sum) {
    if (!isfinite(x)) { e2_set_error("audio: non-finite sample"); return -1; }
    *sum += clip(x); /* bound float files and lossy-decoder overshoot before downmixing */
    return 0;
}
static float *wav_decode(const wav_info *w, int *frames) {
    if (w->format != 1 && w->format != 3) {
        e2_set_error("wav: in-tree reader supports PCM and IEEE float only"); return NULL;
    }
    size_t n = w->bytes / (size_t)w->align;
    if (n > (size_t)w->rate * 30) n = (size_t)w->rate * 30;
    float *mono = malloc(n * sizeof *mono);
    if (!mono) { e2_set_error("oom: WAV samples"); return NULL; }
    const int bytes = w->bits / 8;
    const uint8_t *p = w->data;
    for (size_t i = 0; i < n; i++) {
        double sum = 0;
        for (int c = 0; c < w->channels; c++, p += bytes) {
            double x;
            if (w->format == 1) {
                if (bytes == 1) x = ((int)p[0] - 128) / 128.0;
                else {
                    uint32_t v = 0;
                    for (int b = 0; b < bytes; b++) v |= (uint32_t)p[b] << (8 * b);
                    int64_t s = v;
                    if (v & (UINT32_C(1) << (w->bits - 1))) s -= INT64_C(1) << w->bits;
                    x = (double)s / (double)(INT64_C(1) << (w->bits - 1));
                }
            } else if (bytes == 4) {
                uint32_t v = le32(p); float f; memcpy(&f, &v, sizeof f); x = f;
            } else {
                uint64_t v = (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
                memcpy(&x, &v, sizeof x);
            }
            if (mix_sample(x, &sum)) { free(mono); return NULL; }
        }
        mono[i] = (float)(sum / w->channels);
    }
    *frames = (int)n;
    return mono;
}

#if defined(E2_HAVE_LIBSNDFILE)
typedef struct { const uint8_t *data; sf_count_t len, pos; } audio_io;
static sf_count_t vio_len(void *ctx) { return ((audio_io *)ctx)->len; }
static sf_count_t vio_tell(void *ctx) { return ((audio_io *)ctx)->pos; }
static sf_count_t vio_seek(sf_count_t offset, int whence, void *ctx) {
    audio_io *io = ctx;
    sf_count_t base;
    if (whence == SEEK_SET) base = 0;
    else if (whence == SEEK_CUR) base = io->pos;
    else if (whence == SEEK_END) base = io->len;
    else return -1;
    if (offset < -base || offset > io->len - base) return -1;
    io->pos = base + offset;
    return io->pos;
}
static sf_count_t vio_read(void *dst, sf_count_t count, void *ctx) {
    audio_io *io = ctx;
    if (count <= 0) return 0;
    if (count > io->len - io->pos) count = io->len - io->pos;
    memcpy(dst, io->data + (size_t)io->pos, (size_t)count);
    io->pos += count;
    return count;
}
static sf_count_t vio_write(const void *src, sf_count_t count, void *ctx) {
    (void)src; (void)count; (void)ctx; return 0;
}
static float *sndfile_decode(const uint8_t *data, size_t len, int *rate, int *frames) {
    SF_VIRTUAL_IO vio = {vio_len, vio_seek, vio_read, vio_write, vio_tell};
    audio_io io = {data, (sf_count_t)len, 0};
    SF_INFO info = {0};
    SNDFILE *sf = sf_open_virtual(&vio, SFM_READ, &info, &io);
    if (!sf) { e2_set_error("audio: %s", sf_strerror(NULL)); return NULL; }
    float *mono = NULL;
    double *block = NULL;
    if (shape_ok(info.samplerate, info.channels)) goto fail;
    if (info.frames <= 0) { e2_set_error("audio: empty or unknown-length recording"); goto fail; }
    sf_count_t n = info.frames;
    if (n > (sf_count_t)info.samplerate * 30) n = (sf_count_t)info.samplerate * 30;
    mono = malloc((size_t)n * sizeof *mono);
    block = malloc((size_t)1024 * info.channels * sizeof *block);
    if (!mono || !block) { e2_set_error("oom: audio decode scratch"); goto fail; }
    for (sf_count_t at = 0; at < n;) {
        sf_count_t want = n - at < 1024 ? n - at : 1024;
        if (sf_readf_double(sf, block, want) != want || sf_error(sf)) {
            e2_set_error("audio: truncated or corrupt sample data"); goto fail;
        }
        for (sf_count_t i = 0; i < want; i++) {
            double sum = 0;
            for (int c = 0; c < info.channels; c++)
                if (mix_sample(block[i * info.channels + c], &sum)) goto fail;
            mono[at + i] = (float)(sum / info.channels);
        }
        at += want;
    }
    free(block);
    if (sf_close(sf)) { free(mono); e2_set_error("audio: decoder close failed"); return NULL; }
    *rate = info.samplerate; *frames = (int)n;
    return mono;
fail:
    free(block); free(mono); sf_close(sf);
    return NULL;
}
#endif

/* Exact rational phases, Blackman-windowed sinc, with a low-pass before decimation.
 * Edge samples are extended constantly; rows are normalized to preserve DC. */
static int resample(const float *mono, int frames, int rate, float *pcm) {
    const int count = (int)(((int64_t)frames * E2_AUDIO_SAMPLE_RATE + rate - 1) / rate);
    if (count < 161) { e2_set_error("audio: fewer than 161 samples after resampling"); return -1; }
    if (rate == E2_AUDIO_SAMPLE_RATE) { memcpy(pcm, mono, (size_t)count * sizeof *pcm); return count; }
    int a = rate, gcd = E2_AUDIO_SAMPLE_RATE;
    while (a) { int rem = gcd % a; gcd = a; a = rem; }
    const int phases = E2_AUDIO_SAMPLE_RATE / gcd;
    const double cutoff = .95 * (rate > E2_AUDIO_SAMPLE_RATE ? (double)E2_AUDIO_SAMPLE_RATE / rate : 1.0);
    const int radius = (int)ceil(16 / cutoff), taps = 2 * radius + 1;
    float *weights = malloc((size_t)phases * taps * sizeof *weights);
    if (!weights) { e2_set_error("oom: audio resampling filter"); return -1; }
    for (int p = 0; p < phases; p++) {
        double sum = 0;
        for (int k = -radius; k <= radius; k++) {
            double x = k - (double)p / phases;
            double t = E2_AUDIO_PI * cutoff * x;
            double w = fabs(x) > radius ? 0 : (fabs(t) < 1e-12 ? cutoff : cutoff * sin(t) / t) *
                (.42 + .5 * cos(E2_AUDIO_PI * x / radius) + .08 * cos(2 * E2_AUDIO_PI * x / radius));
            weights[(size_t)p * taps + (k + radius)] = (float)w;
            sum += w;
        }
        for (int k = 0; k < taps; k++) weights[(size_t)p * taps + k] /= (float)sum;
    }
    for (int i = 0; i < count; i++) {
        const int64_t pos = (int64_t)i * rate;
        int base = (int)(pos / E2_AUDIO_SAMPLE_RATE);
        int phase = (int)(pos % E2_AUDIO_SAMPLE_RATE) / gcd;
        const float *w = weights + (size_t)phase * taps;
        double sum = 0;
        for (int k = -radius; k <= radius; k++) {
            int j = base + k;
            if (j < 0) j = 0;
            if (j >= frames) j = frames - 1;
            sum += mono[j] * (double)w[k + radius];
        }
        pcm[i] = (float)clip(sum);
    }
    free(weights);
    return count;
}
static int preprocess(const uint8_t *data, size_t len, float *pcm, int wav_only) {
    e2_set_error("");
    if (!data || !pcm || len < 12 || len > INT64_MAX) {
        e2_set_error("audio: invalid data/output buffer or file size"); return -1;
    }
    wav_info wav;
    int is_wav = !memcmp(data, "RIFF", 4) && !memcmp(data + 8, "WAVE", 4);
    if ((is_wav || wav_only) && wav_header(data, len, &wav)) return -1;
    float *mono;
    int frames = 0, rate;
    if (wav_only) { rate = wav.rate; mono = wav_decode(&wav, &frames); }
    else {
#if defined(E2_HAVE_LIBSNDFILE)
        mono = sndfile_decode(data, len, &rate, &frames);
#else
        if (!is_wav) { e2_set_error("audio: this build supports RIFF WAV only (enable E2_WITH_SNDFILE)"); return -1; }
        rate = wav.rate; mono = wav_decode(&wav, &frames);
#endif
    }
    if (!mono) return -1;
    int n = resample(mono, frames, rate, pcm);
    free(mono);
    return n;
}
int e2_audio_preprocess(const uint8_t *data, size_t len, float *pcm) {
    return preprocess(data, len, pcm, 0);
}
int e2_audio_preprocess_wav(const uint8_t *data, size_t len, float *pcm) {
    return preprocess(data, len, pcm, 1);
}
int e2_audio_supported_formats(void) {
    int formats = E2_AUDIO_WAV;
#if defined(E2_HAVE_LIBSNDFILE)
    int count = 0;
    sf_command(NULL, SFC_GET_FORMAT_MAJOR_COUNT, &count, sizeof count);
    for (int i = 0; i < count; i++) {
        SF_FORMAT_INFO info = {0}; info.format = i;
        if (sf_command(NULL, SFC_GET_FORMAT_MAJOR, &info, sizeof info)) continue;
        switch (info.format) {
            case SF_FORMAT_FLAC: formats |= E2_AUDIO_FLAC; break;
            case SF_FORMAT_OGG: formats |= E2_AUDIO_OGG; break;
            case SF_FORMAT_AIFF: formats |= E2_AUDIO_AIFF; break;
            case SF_FORMAT_AU: formats |= E2_AUDIO_AU; break;
            case 0x230000: formats |= E2_AUDIO_MP3; break; /* MPEG; also builds with older headers */
        }
    }
#endif
    return formats;
}
