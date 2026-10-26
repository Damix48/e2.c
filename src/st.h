/* st.h — safetensors reader (see st.c). */
#ifndef E2_ST_H
#define E2_ST_H

#include <stddef.h>
#include <stdint.h>

#define ST_MAX_NAME 192

typedef struct {
    char name[ST_MAX_NAME];
    char dtype[16];
    int64_t shape[8];
    int ndim;
    int64_t begin, end; /* byte range inside the data section */
} st_tensor;

typedef struct {
    int fd;
    const uint8_t *base;
    const uint8_t *data; /* start of tensor data */
    size_t size;
    st_tensor *tensors;
    size_t count;
} st_file;

int st_open(st_file *f, const char *path);
void st_close(st_file *f);
const st_tensor *st_find(const st_file *f, const char *name);
size_t st_numel(const st_tensor *t);
const uint16_t *st_bf16(const st_file *f, const st_tensor *t);
const float *st_f32(const st_file *f, const st_tensor *t);
int st_is_bf16(const st_tensor *t);
const char *st_error(void);

#endif /* E2_ST_H */
