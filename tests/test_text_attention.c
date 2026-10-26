/* Private text scheduling checks, without exporting test-only engine APIs.
 * Compile this TU (which includes text.c) and link libe2 normally; do not also link text.o. */
#include "../src/text.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static float *floats(size_t n) {
    float *p = calloc(n, sizeof(float));
    CHECK(p);
    return p;
}

static void check_attention(int S, int sliding) {
    const int D = sliding ? 256 : 512, GD = E2_NUM_HEADS * D, KD = 512;
    e2_layer L = { .head_dim = D, .q_dim = GD, .kv_dim = KD,
                   .n_kv_heads = sliding ? 2 : 1, .sliding = sliding, .window = 512 };
    float *q = floats((size_t)S * GD), *k = floats((size_t)S * KD), *v = floats((size_t)S * KD);
    float *ref = floats((size_t)S * GD), *got = floats((size_t)S * GD);
    float *kt = floats((size_t)S * GD);
    size_t per = E2_ATT_BQ * (size_t)S;
    if (per < E2_MAX_SEQ) per = E2_MAX_SEQ;
    float *scores = floats((size_t)e2_nthreads() * per);
    uint8_t *valid = malloc((size_t)S);
    CHECK(valid);
    memset(valid, 1, (size_t)S);
    for (size_t i = 0; i < (size_t)S * GD; i++) q[i] = sinf((float)i * .031f) * .12f;
    for (size_t i = 0; i < (size_t)S * KD; i++) {
        k[i] = cosf((float)i * .017f) * .12f;
        v[i] = sinf((float)i * .007f);
    }
    struct attn_task t = { NULL, &L, S, valid, q, k, v, ref, scores, 0, 0 };
    atomic_init(&t.next, 0);
    e2_parallel(attn_worker, &t);
    attention_blocked(&L, q, k, v, got, kt, scores, S);
    double md = 0;
    size_t mdi = 0;
    for (size_t i = 0; i < (size_t)S * GD; i++) {
        CHECK(isfinite(got[i]));
        double d = fabs((double)ref[i] - got[i]);
        if (d > md) { md = d; mdi = i; }
    }
    printf("attention S=%d sliding=%d maxdiff=%.9g idx=%zu got=%g ref=%g\n", S, sliding, md, mdi, got[mdi], ref[mdi]);
    CHECK(md < 2e-5);
    /* Uniform probabilities expose inclusive window endpoints without a dot-product oracle. */
    if (sliding && S == 2048) {
        memset(q, 0, sizeof(float) * (size_t)S * GD);
        memset(v, 0, sizeof(float) * (size_t)S * KD);
        for (int j = 0; j < S; j++) v[(size_t)j * KD] = (float)(j * j);
        attention_blocked(&L, q, k, v, got, kt, scores, S);
        const int probes[] = { 0, 511, 512, 513, 1024, 2047 };
        for (int p = 0; p < 6; p++) {
            int i = probes[p], lo = i > 512 ? i - 512 : 0, hi = i + 513 < S ? i + 513 : S;
            double sum = 0;
            for (int j = lo; j < hi; j++) sum += j * j;
            CHECK(fabs(got[(size_t)i * GD] - sum / (hi - lo)) < 2.0);
        }
    }
    free(q); free(k); free(v); free(ref); free(got); free(kt); free(scores); free(valid);
}

static void check_gate_quant(int rows, int D, int stride) {
    int padded = pad_tokens(rows), nb = D / E2_I8_BLOCK;
    e2_model m = { .use_int8 = 1 };
    m.ws.qa = malloc((size_t)padded * D);
    m.ws.qa_scale = floats((size_t)padded * nb);
    CHECK(m.ws.qa);
    memset(m.ws.qa, 19, (size_t)padded * D);
    float *gate = floats((size_t)rows * D), *ref = floats((size_t)rows * D);
    float *up = floats((size_t)rows * stride);
    int8_t *qa = calloc((size_t)padded * D, 1);
    float *sc = floats((size_t)padded * nb);
    CHECK(qa);
    for (int i = 0; i < rows * D; i++) gate[i] = ref[i] = sinf((float)i * .07f) * 3;
    for (int i = 0; i < rows * stride; i++) up[i] = cosf((float)i * .03f);
    e2_k()->gelu_tanh(ref, (size_t)rows * D);
    e2_k()->mul_rows(ref, up, rows, D, stride);
    e2_i8_quantize(qa, sc, ref, rows, D, -INFINITY, INFINITY);
    gate_quant(&m, gate, up, rows, D, stride);
    CHECK(!memcmp(ref, gate, sizeof(float) * (size_t)rows * D));
    CHECK(!memcmp(qa, m.ws.qa, (size_t)padded * D));
    CHECK(!memcmp(sc, m.ws.qa_scale, sizeof(float) * (size_t)padded * nb));
    /* The unfused projection quantizer must also clear, not read, padded rows. */
    memset(m.ws.qa, 19, (size_t)padded * D);
    struct gemm_task t = { .m = &m, .x = ref, .M = rows, .Mp = padded, .K = D, .nb = nb };
    atomic_init(&t.qnext, 0);
    gemm_quant_phase(&t);
    CHECK(!memcmp(qa, m.ws.qa, (size_t)padded * D));
    CHECK(!memcmp(sc, m.ws.qa_scale, sizeof(float) * (size_t)padded * nb));
    float *weights = floats((size_t)16 * D), *y = floats((size_t)padded * 16);
    for (int i = 0; i < 16 * D; i++) weights[i] = sinf((float)i * .01f);
    e2_q8w w;
    CHECK(!e2_q8w_from_f32(&w, weights, 16, D));
    gemm(&m, y, ref, &w, NULL, rows, 16, D, 1);
    for (int i = rows * 16; i < padded * 16; i++) CHECK(y[i] == 0);
    e2_q8w_free(&w);
    free(weights); free(y);
    m.use_int8 = 0;
    for (int i = 0; i < rows * D; i++) gate[i] = sinf((float)i * .07f) * 3;
    gate_quant(&m, gate, up, rows, D, stride);
    CHECK(!memcmp(ref, gate, sizeof(float) * (size_t)rows * D));
    free(m.ws.qa); free(m.ws.qa_scale); free(gate); free(ref); free(up); free(qa); free(sc);
}

int main(void) {
    e2_cpu_init();
    e2_pool_global_init(4);
    const int sizes[] = { 31, 32, 63, 64, 513, 514, 2048 };
    for (int i = 0; i < 7; i++) {
        check_attention(sizes[i], 1);
        check_attention(sizes[i], 0);
    }
    const int rows[] = { 1, 7, 8, 9, 15, 16, 17, 31 };
    for (int i = 0; i < 8; i++) {
        check_gate_quant(rows[i], 2048, 2048);
        check_gate_quant(rows[i], 512, 24 * 512);
    }
    e2_pool_global_free();
    puts("text attention/fusion OK");
    return 0;
}
