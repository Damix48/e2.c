/* Private text scheduling checks, without exporting test-only engine APIs.
 * Compile this TU (which includes text.c) and link libe2 normally; do not also link text.o. */
#include "../src/text.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static float *floats(size_t n) {
    float *p = calloc(n, sizeof(float));
    CHECK(p);
    return p;
}

/* attention_blocked now reads its scratch from the workspace, so the test builds the same slots. */
static e2_ws make_ws(int S) {
    e2_ws w = { 0 };
    /* asblk also backs the row-wise oracle below, which uses one E2_MAX_SEQ row per worker */
    size_t per = E2_ATT_BQ * (size_t)S;
    if (per < E2_MAX_SEQ) per = E2_MAX_SEQ;
    /* kt is also V^T for the int8 path, whose rows are padded up to a block multiple */
    w.kt = floats((size_t)E2_NUM_HEADS * 512 * ((S + E2_I8_BLOCK - 1) & ~(E2_I8_BLOCK - 1)));
    w.asblk = floats((size_t)e2_nthreads() * per);
    w.att_q8_words = floats((size_t)E2_MAX_THREADS * E2_ATT_BQ * 512 / 4 + 4);
    w.att_qs = floats((size_t)E2_MAX_THREADS * E2_ATT_BQ * 4);
    w.att_p8_words = floats((size_t)E2_MAX_THREADS * E2_ATT_BQ *
                                ((S + E2_I8_BLOCK - 1) & ~(E2_I8_BLOCK - 1)) / 4 + 4);
    w.att_ps = floats((size_t)E2_MAX_THREADS * E2_ATT_BQ * 32);
    w.att_ctx = floats((size_t)E2_MAX_THREADS * E2_ATT_BQ * 512);
    w.att_q8 = (uint8_t *)w.att_q8_words;
    w.att_p8 = (uint8_t *)w.att_p8_words;
    return w;
}

static void free_ws(e2_ws *w) {
    free(w->kt); free(w->asblk); free(w->att_q8_words); free(w->att_qs);
    free(w->att_p8_words); free(w->att_ps); free(w->att_ctx);
}

/* i8 selects the int8 attention path; the fp32 path is the exact oracle for both. */
static void check_attention(int S, int sliding, int i8) {
    const int D = sliding ? 256 : 512, GD = E2_NUM_HEADS * D, KD = 512;
    e2_layer L = { .head_dim = D, .q_dim = GD, .kv_dim = KD,
                   .n_kv_heads = sliding ? 2 : 1, .sliding = sliding, .window = 512 };
    float *q = floats((size_t)S * GD), *k = floats((size_t)S * KD), *v = floats((size_t)S * KD);
    float *ref = floats((size_t)S * GD), *got = floats((size_t)S * GD);
    e2_ws ws = make_ws(S);
    float *scores = ws.asblk;
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
    e2_attn_override(E2_TOWER_TEXT, i8 ? E2_ATTN_I8 : E2_ATTN_F32);
    attention_blocked(&L, q, k, v, got, &ws, S);
    double md = 0;
    size_t mdi = 0;
    for (size_t i = 0; i < (size_t)S * GD; i++) {
        CHECK(isfinite(got[i]));
        double d = fabs((double)ref[i] - got[i]);
        if (d > md) { md = d; mdi = i; }
    }
    printf("attention S=%d sliding=%d %s maxdiff=%.9g idx=%zu got=%g ref=%g\n", S, sliding,
           i8 ? "int8" : "fp32", md, mdi, got[mdi], ref[mdi]);
    /* The int8 path quantizes Q, P, K and V^T, so it is a 1/127-accurate approximation, not the
     * same arithmetic; the fp32 path must stay an exact match of the row-wise oracle. */
    CHECK(md < (i8 ? 2e-2 : 2e-5));
    /* Uniform probabilities expose inclusive window endpoints without a dot-product oracle. */
    if (!i8 && sliding && S == 2048) {
        memset(q, 0, sizeof(float) * (size_t)S * GD);
        memset(v, 0, sizeof(float) * (size_t)S * KD);
        for (int j = 0; j < S; j++) v[(size_t)j * KD] = (float)(j * j);
        attention_blocked(&L, q, k, v, got, &ws, S);
        const int probes[] = { 0, 511, 512, 513, 1024, 2047 };
        for (int p = 0; p < 6; p++) {
            int i = probes[p], lo = i > 512 ? i - 512 : 0, hi = i + 513 < S ? i + 513 : S;
            double sum = 0;
            for (int j = lo; j < hi; j++) sum += j * j;
            CHECK(fabs(got[(size_t)i * GD] - sum / (hi - lo)) < 2.0);
        }
    }
    free(q); free(k); free(v); free(ref); free(got); free(valid); free_ws(&ws);
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
        check_attention(sizes[i], 1, 0);
        check_attention(sizes[i], 0, 0);
    }
    /* The int8 path, against the same row-wise fp32 oracle. */
    for (int i = 0; e2_i8_gemm && i < 7; i++) {
        check_attention(sizes[i], 1, 1);
        check_attention(sizes[i], 0, 1);
    }
    const int rows[] = { 1, 7, 8, 9, 15, 16, 17, 31 };
    for (int i = 0; e2_i8_quantize && i < 8; i++) {
        check_gate_quant(rows[i], 2048, 2048);
        check_gate_quant(rows[i], 512, 24 * 512);
    }
    e2_attn_override(E2_TOWER_TEXT, -1);
    e2_pool_global_free();
    puts("text attention/fusion OK");
    return 0;
}
