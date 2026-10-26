/* Exercise private loader helpers without a checkpoint or test-only public API. */
#include "../src/audio.c"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

int main(void) {
    CHECK(acopy(NULL, 128) == NULL);
    const uint16_t data[128] = { 0x3fc0, 0xc000 }; /* 1.5, -2.0 */
    float *copy = acopy(data, 2);
    CHECK(copy && copy[0] == 1.5f && copy[1] == -2.0f);
    free(copy);

    e2_model m = { 0 };
    e2_alin lin = { 0 };
    CHECK(alat_init(&m, &lin, "test", 1, 128, "projection", 0) == -1);
    st_tensor tensor = { .ndim = 2, .shape = { 1, 128 }, .begin = 0, .end = sizeof data };
    snprintf(tensor.name, sizeof tensor.name, "test.projection.linear.weight");
    snprintf(tensor.dtype, sizeof tensor.dtype, "BF16");
    m.st.tensors = &tensor;
    m.st.count = 1;
    m.st.data = (const uint8_t *)data;
    CHECK(alat_init(&m, &lin, "test", 1, 128, "projection", 0) == 0);
    CHECK(lin.w == data && lin.in_min < -1e30f && lin.out_max > 1e30f);
    CHECK(alat_init(&m, &lin, "test", 1, 128, "projection", 1) == -1); /* missing clamps */
    CHECK(alat_init(&m, &lin, "test", 2, 128, "projection", 0) == -1);
    puts("audio loader OK");
    return 0;
}
