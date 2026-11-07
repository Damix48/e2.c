/* e2_audio_file.c - audio file loading via libsndfile (optional). */
#include "e2_internal.h"

#include <math.h>
#include <stdlib.h>

#define E2_MAX_AUDIO 480000

#ifdef E2_USE_SNDFILE
#include <sndfile.h>

static int resample_mono(const float *in, int nin, int sr, float **out, int *nout) {
    if (sr == 16000) {
        int n = nin > E2_MAX_AUDIO ? E2_MAX_AUDIO : nin;
        float *o = malloc((size_t)(n ? n : 1) * sizeof(float));
        if (!o) return e2_fail("oom");
        for (int i = 0; i < n; i++) o[i] = in[i];
        *out = o; *nout = n;
        return 0;
    }
    const double pi = 3.14159265358979323846;
    double ratio = 16000.0 / (double)sr;
    long n = (long)((double)nin * ratio + 0.5);
    if (n > E2_MAX_AUDIO) n = E2_MAX_AUDIO;
    if (n < 1) { *out = NULL; *nout = 0; return 0; }
    float *o = malloc((size_t)n * sizeof(float));
    if (!o) return e2_fail("oom");
    double cutoff = ratio < 1.0 ? ratio * 0.95 : 1.0;
    const int half = 16;
    for (long j = 0; j < n; j++) {
        double t = (double)j / ratio;
        long i0 = (long)floor(t) - half + 1;
        double sum = 0.0, ws = 0.0;
        for (int k = 0; k < 2 * half; k++) {
            long i = i0 + k;
            if (i < 0 || i >= nin) continue;
            double d = (double)i - t;
            double x = d * cutoff;
            double sinc = fabs(x) < 1e-9 ? 1.0 : sin(pi * x) / (pi * x);
            double win = 0.5 - 0.5 * cos(2.0 * pi * (k + 0.5) / (2.0 * half));
            double w = sinc * win;
            sum += (double)in[i] * w;
            ws += w;
        }
        o[j] = ws != 0.0 ? (float)(sum / ws) : 0.0f;
    }
    *out = o; *nout = (int)n;
    return 0;
}

int e2_audio_read(const char *path, float **samples, int *nsamples) {
    SF_INFO info;
    memset(&info, 0, sizeof(info));
    SNDFILE *f = sf_open(path, SFM_READ, &info);
    if (!f) return e2_fail("audio: %s", sf_strerror(NULL));
    if (info.channels < 1 || info.frames < 1) {
        sf_close(f);
        return e2_fail("audio: empty file %s", path);
    }
    long frames = (long)info.frames;
    /* Do not read more than what could survive truncation at 16 kHz. */
    long max_in = (long)((double)E2_MAX_AUDIO * (double)info.samplerate / 16000.0) + 64;
    if (frames > max_in) frames = max_in;
    float *raw = malloc((size_t)frames * (size_t)info.channels * sizeof(float));
    float *mono = malloc((size_t)frames * sizeof(float));
    if (!raw || !mono) { free(raw); free(mono); sf_close(f); return e2_fail("oom"); }
    sf_count_t got = sf_readf_float(f, raw, (sf_count_t)frames);
    sf_close(f);
    if (got <= 0) { free(raw); free(mono); return e2_fail("audio: read failed %s", path); }
    for (sf_count_t i = 0; i < got; i++) {
        float s = 0.0f;
        for (int c = 0; c < info.channels; c++) s += raw[(size_t)i * info.channels + c];
        mono[i] = s / (float)info.channels;
    }
    free(raw);
    int n = 0;
    int rc = resample_mono(mono, (int)got, info.samplerate, samples, &n);
    free(mono);
    if (rc != 0) return rc;
    *nsamples = n;
    return 0;
}

#else

int e2_audio_read(const char *path, float **samples, int *nsamples) {
    (void)path; (void)samples; (void)nsamples;
    return e2_fail("built without libsndfile");
}

#endif
