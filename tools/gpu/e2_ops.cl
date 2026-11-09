/* e2_ops.cl - elementwise, norm, rope and attention kernels for the device
 * pipeline.  These need only portable OpenCL collectives, so clang's SPIR-V
 * frontend compiles them; ocloc is only required by e2_dpas.cl, which uses the
 * Intel sub-group matrix builtin.
 *
 * Conventions:
 *  - every f32 buffer is row-major, and every row-parallel kernel maps one
 *    sub-group to one row with a lane-strided inner loop, so the loads stay
 *    coalesced (adjacent lanes read adjacent elements).
 *  - the sub-group width is taken from the device (16 on this Xe part), never
 *    hardcoded, and reductions use the OpenCL collectives.
 *  - the maths mirrors src/e2_ops.c exactly; that file stays the reference.
 */

#define GELU_K 0.7978845608028654f /* sqrt(2/pi) */
#define GELU_C 0.044715f

/* gelu(x) = x * sigmoid(2u), u = GELU_K*(x + GELU_C*x^3).  The exponential is
 * evaluated with a NONPOSITIVE argument (|u|) and the sign is folded back, so
 * the hardware native_exp cannot overflow, and native_exp is what the ISA's
 * transcendental unit is for.  The general tanh form is >1000 ISA lines. */
static inline float gelu1(float x) {
    float u = GELU_K * (x + GELU_C * x * x * x);
    if (u > 15.0f) return x;
    float e = native_exp(-2.0f * fabs(u));
    float s = (1.0f - e) / (1.0f + e);          /* tanh(|u|) */
    return 0.5f * x * (1.0f + (u < 0.0f ? -s : s));
}

static inline float rms_scale(float sumsq, int dim, float eps) {
    return rsqrt(sumsq / (float)dim + eps);
}

/* ------------------------------------------------------------------ */
/* rmsnorm family                                                      */
/*   mode 0: out = norm(x) * w                                         */
/*   mode 1: out = norm(x*pre) * w        (pre-scale, PLE projection)   */
/*   mode 2: out = x + norm(src) * w      (residual add)                */
/* ------------------------------------------------------------------ */
__kernel void k_rmsnorm(__global const float *x, __global const float *w,
                        __global const float *src, __global float *out,
                        const int rows, const int dim, const float eps,
                        const float pre, const int mode)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rows) return;
    int acc_row = (mode == 2);
    const __global float *sp = acc_row ? src + (size_t)row * dim
                                       : x + (size_t)row * dim;
    float p = (mode == 1) ? pre : 1.0f;
    float sum = 0.0f;
    for (int i = lane; i < dim; i += gs) {
        float v = sp[i] * p;
        sum += v * v;
    }
    sum = sub_group_reduce_add(sum);
    float s = rms_scale(sum, dim, eps);
    __global float *op = out + (size_t)row * dim;
    for (int i = lane; i < dim; i += gs) {
        float v = sp[i] * p * s;
        float r = w ? v * w[i] : v;
        op[i] = acc_row ? op[i] + r : r;
    }
}

/* Same as k_rmsnorm (modes 0/1/2) and it also writes the int8 activation the
 * next GEMM would have produced anyway, so the separate k_quant_s8 pass and its
 * read+write of the f32 row never happen.  `blk` matches k_quant_s8. */
__kernel void k_rmsnorm_q(__global const float *x, __global const float *w,
                          __global const float *src, __global float *out,
                          __global char *q, __global float *sc,
                          const int rows, const int dim, const float eps,
                          const float pre, const int mode, const int blk)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rows) return;
    int acc_row = (mode == 2);
    const __global float *sp = acc_row ? src + (size_t)row * dim
                                       : x + (size_t)row * dim;
    float p = (mode == 1) ? pre : 1.0f;
    /* float4 per lane: with a runtime sub-group stride the compiler cannot
     * vectorise the scalar form, and 16 lanes x 4 B per load is a quarter of
     * the issue budget this kernel has.  dim is a multiple of 4 everywhere. */
    const int gs4 = gs * 4;
    float4 sum4 = (float4)(0.0f);
    for (int i = lane * 4; i < dim; i += gs4) {
        float4 v = *(const __global float4 *)(sp + i) * p;
        sum4 += v * v;
    }
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    sum = sub_group_reduce_add(sum);
    float s = rms_scale(sum, dim, eps);
    __global float *op = out + (size_t)row * dim;
    for (int i = lane * 4; i < dim; i += gs4) {
        float4 v = *(const __global float4 *)(sp + i) * (p * s);
        if (w) v *= *(const __global float4 *)(w + i);
        if (acc_row) v += *(const __global float4 *)(op + i);
        *(__global float4 *)(op + i) = v;
    }
    /* quantise what was just written, one block at a time */
    int nblk = dim / blk;
    __global char *qp = q + (size_t)row * dim;
    for (int b = 0; b < nblk; b++) {
        __global const float *bp = op + b * blk;
        float mx = 0.0f;
        for (int i = lane * 4; i < blk; i += gs4) {
            float4 v = fabs(*(const __global float4 *)(bp + i));
            mx = fmax(mx, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
        }
        mx = sub_group_reduce_max(mx);
        float qs = mx / 127.0f;
        if (qs < 1e-12f) qs = 1e-12f;
        float inv = 1.0f / qs;
        for (int i = lane * 4; i < blk; i += gs4) {
            float4 v = *(const __global float4 *)(bp + i) * inv;
            int4 r = clamp(convert_int4_rte(v), (int4)(-127), (int4)(127));
            *(__global char4 *)(qp + b * blk + i) = convert_char4(r);
        }
        if (lane == 0) sc[(size_t)row * nblk + b] = qs;
    }
}

/* rows are (token, head) pairs: position = row/per_pos - seq_off[seq]. */
static inline int rope_pos(int row, int per_pos, __global const int *seq_off,
                           int nseq)
{
    int gt = row / per_pos;
    int s = nseq - 1;
    while (s > 0 && gt < seq_off[s]) s--;
    return gt - seq_off[s];
}

/* rmsnorm followed by the interleaved-pair rotation, one pass over the row. */
__kernel void k_norm_rope(__global float *x, __global const float *w,
                          __global const float *cs, __global const float *sn,
                          const int rows, const int dim, const float eps,
                          const int per_pos, __global const int *seq_off,
                          const int nseq)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rows) return;
    __global float *xp = x + (size_t)row * dim;
    /* float4 per lane, as in k_rmsnorm_q: a runtime sub-group stride stops the
     * compiler from vectorising the scalar form */
    const int gs4 = gs * 4;
    float4 sum4 = (float4)(0.0f);
    for (int i = lane * 4; i < dim; i += gs4) {
        float4 v = *(const __global float4 *)(xp + i);
        sum4 += v * v;
    }
    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    sum = sub_group_reduce_add(sum);
    float s = rms_scale(sum, dim, eps);
    for (int i = lane * 4; i < dim; i += gs4) {
        float4 v = *(const __global float4 *)(xp + i) * s;
        if (w) v *= *(const __global float4 *)(w + i);
        *(__global float4 *)(xp + i) = v;
    }
    /* hf of the row is one rotation pair; the two halves are `dim/2` apart */
    int hf = dim / 2;
    int pos = rope_pos(row, per_pos, seq_off, nseq);
    __global const float *c = cs + (size_t)pos * hf;
    __global const float *sn2 = sn + (size_t)pos * hf;
    for (int d = lane * 4; d < hf; d += gs4) {
        float4 x1 = *(const __global float4 *)(xp + d);
        float4 x2 = *(const __global float4 *)(xp + d + hf);
        float4 cc = *(const __global float4 *)(c + d);
        float4 ss = *(const __global float4 *)(sn2 + d);
        *(__global float4 *)(xp + d) = x1 * cc - x2 * ss;
        *(__global float4 *)(xp + d + hf) = x2 * cc + x1 * ss;
    }
}

/* one work-item per rotation pair; consecutive items are consecutive d. */
__kernel void k_rope_text(__global float *q, __global float *k,
                          __global const float *cs, __global const float *sn,
                          const int t, const int heads, const int kv_heads,
                          const int head_dim, const int pos_offset)
{
    int hf = head_dim / 2;
    int gid = get_global_id(0);
    int tot = t * (heads + kv_heads);
    if (gid >= tot * hf) return;
    int d = gid % hf;
    int r = gid / hf;
    int pos = r / (heads + kv_heads) + pos_offset;
    int h = r % (heads + kv_heads);
    __global float *base = (h < heads) ? q : k;
    int nh = (h < heads) ? heads : kv_heads;
    int hh = (h < heads) ? h : h - heads;
    __global float *v = base + ((size_t)pos * nh + hh) * head_dim;
    float x1 = v[d], x2 = v[d + hf];
    float c = cs[(size_t)pos * hf + d], s = sn[(size_t)pos * hf + d];
    v[d] = x1 * c - x2 * s;
    v[d + hf] = x2 * c + x1 * s;
}

/* vision: axial rope, sin/cos tables per axis, two rotations per head. */
__kernel void k_rope_vision(__global float *q, __global float *k,
                            __global const float *cx, __global const float *sx,
                            __global const float *cy, __global const float *sy,
                            __global const int *xy, const int t, const int heads,
                            const int head_dim)
{
    int spatial = head_dim / 2, hf = spatial / 2;
    int gid = get_global_id(0);
    if (gid >= t * heads * hf) return;
    int d = gid % hf;
    int r = gid / hf;
    int pos = r / heads, h = r % heads;
    int px = xy[2 * pos], py = xy[2 * pos + 1];
    for (int which = 0; which < 2; which++) {
        __global float *base = which ? k : q;
        if (!base) continue;
        if (which && k == q) continue;
        __global float *v = base + ((size_t)pos * heads + h) * head_dim;
        float c1 = cx[(size_t)px * hf + d], s1 = sx[(size_t)px * hf + d];
        float a = v[d], b = v[d + hf];
        v[d] = a * c1 - b * s1;
        v[d + hf] = b * c1 + a * s1;
        float c2 = cy[(size_t)py * hf + d], s2 = sy[(size_t)py * hf + d];
        __global float *w = v + spatial;
        a = w[d]; b = w[d + hf];
        w[d] = a * c2 - b * s2;
        w[d + hf] = b * c2 + a * s2;
    }
}

/* ------------------------------------------------------------------ */
/* bidirectional GQA attention                                         */
/*                                                                    */
/* Two kernels: the band is split NB ways so the grid has enough work- */
/* groups to fill the GPU (one per 4 query rows was 25% occupancy and  */
/* 90% of the whole forward), then a cheap combine pass.               */
/*   grid = (sequence, head, block of 4 query rows, band chunk)        */
/* The 4 query rows of a block share the key band, so the K/V lines    */
/* they read together are reused out of L1/L2.                         */
/* The output row lives in local memory: keeping head_dim/subgroup     */
/* floats per lane in an array made the compiler spill 2-4 KB/thread   */
/* to scratch, which is 27x slower than the local-memory version.      */
/* ------------------------------------------------------------------ */
#define ATT_ROWS 4            /* query rows per work-group */
#define ATT_DMAX 512          /* ------------------------------------------------------------------ */
/* int8 attention                                                      */
/*                                                                    */
/* scores = Q x K^T and context = P x V^T are plain matrix-engine      */
/* products (k_gemm_xmx, activations centred int8).  These two kernels  */
/* do the preprocessing: pack K / V^T into the A operand layout with a  */
/* per-row scale, and turn the scores into an int8 P with the softmax   */
/* folded into its per-128-block scales.                               */
/* ------------------------------------------------------------------ */

/* Pack one f32 matrix into the A operand layout: pq[(fb*nkc + kstep)*256 +
 * lane*32 + r*4 + c] holds k = 4*lane+c of row fb*8+r, centred int8.  Source
 * element (row, k) is src[row*rstride + k*kstride], so the same kernel packs K
 * (row = key, k = dim, contiguous) and V^T (row = dim, k = key, strided).  Rows
 * past rvalid and k past kvalid are zero, which is what makes them contribute
 * nothing to either product.  One sub-group per row, lane l one k-quad. */
__kernel void k_attn_pack(__global const float *src, __global char *pq,
                          __global float *sc, const int rpad, const int rvalid,
                          const int kpad, const int kvalid, const int rstride,
                          const int kstride, const long zsrc, const long zpq,
                          const long zsc)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rpad) return;
    /* gridCountZ = heads: the per-head operands are at a fixed stride */
    if (get_group_id(2)) {
        src += (size_t)get_group_id(2) * zsrc;
        pq += (size_t)get_group_id(2) * zpq;
        sc += (size_t)get_group_id(2) * zsc;
    }
    int nkc = kpad >> 5;
    __global char *base = pq + (size_t)(row >> 3) * nkc * 256 + (row & 7) * 4;
    if (row >= rvalid) {
        for (int kq = lane; kq < (kpad >> 2); kq += gs)
            *(__global char4 *)(base + (size_t)(kq >> 3) * 256 + (kq & 7) * 32) =
                (char4)(0);
        if (lane == 0) sc[row] = 0.0f;
        return;
    }
    __global const float *rp = src + (size_t)row * rstride;
    float amax = 0.0f;
    for (int i = lane; i < kvalid; i += gs)
        amax = fmax(amax, fabs(rp[(size_t)i * kstride]));
    amax = sub_group_reduce_max(amax);
    float s = amax > 0.0f ? amax / 127.0f : 1e-12f;
    if (lane == 0) sc[row] = s;
    float inv = 1.0f / s;
    for (int kq = lane; kq < (kpad >> 2); kq += gs) {
        int k0 = kq * 4;
        int4 r = (int4)(0);
        if (kstride == 1 && k0 + 3 < kvalid) {
            float4 v = *(const __global float4 *)(rp + k0);
            r = clamp(convert_int4_rte(v * inv), (int4)(-127), (int4)(127));
        } else {
            for (int j = 0; j < 4; j++) {
                int kk = k0 + j;
                if (kk < kvalid) {
                    float v = clamp(rp[(size_t)kk * kstride] * inv, -127.0f, 127.0f);
                    ((int *)&r)[j] = (int)rint(v);
                }
            }
        }
        *(__global char4 *)(base + (size_t)(kq >> 3) * 256 + (kq & 7) * 32) =
            convert_char4(r);
    }
}

/* scores -> P.  One sub-group per query row.  P is int8 in [0,127] with one
 * scale per 128-wide key block, and the 1/sum of the row is folded into those
 * scales, so the context product needs no normalisation pass.  Keys outside
 * [row-window, row+window] get p = 0, and so does every key of another
 * sequence (seq_off[s] <= row < seq_off[s+1]), which is what lets a batch share
 * one score product. */
__kernel void k_attn_softmax(__global const float *S, __global char *P,
                             __global float *psc, __global const int *seq_off,
                             const int nseq, const int rows, const int kpad,
                             const int kvalid, const int window,
                             const long zS, const long zP, const long zsc)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rows) return;
    if (get_group_id(2)) {
        S += (size_t)get_group_id(2) * zS;
        P += (size_t)get_group_id(2) * zP;
        psc += (size_t)get_group_id(2) * zsc;
    }
    int lo = 0, hi = kvalid;
    if (window > 0) {
        lo = row - window; if (lo < 0) lo = 0;
        hi = row + window + 1; if (hi > kvalid) hi = kvalid;
    }
    if (nseq > 1) {
        int sq = 0;
        while (sq + 1 < nseq && row >= seq_off[sq + 1]) sq++;
        if (seq_off[sq] > lo) lo = seq_off[sq];
        int se = seq_off[sq + 1];
        if (se > kvalid) se = kvalid;
        if (se < hi) hi = se;
        if (hi <= lo) hi = lo + 1;
    }
    __global const float *sp = S + (size_t)row * kpad;
    float mx = -3.0e38f;
    for (int i = lo + lane; i < hi; i += gs) mx = fmax(mx, sp[i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (int i = lo + lane; i < hi; i += gs) sum += exp(sp[i] - mx);
    sum = sub_group_reduce_add(sum);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    int nblk = kpad >> 7;
    __global char *pp = P + (size_t)row * kpad;
    for (int b = 0; b < nblk; b++) {
        int b0 = b << 7, b1 = b0 + 128;
        float bm = 0.0f;
        for (int i = b0 + lane; i < b1; i += gs)
            if (i >= lo && i < hi) {
                float v = exp(sp[i] - mx);
                if (v > bm) bm = v;
            }
        bm = sub_group_reduce_max(bm);
        /* Quantise against the block max (q in [0,127]), and fold the row
         * normalisation into the *dequantisation* scale only.  Folding 1/sum
         * into the quantiser step shrinks it by `sum`, so every component above
         * bm/sum saturates at 127 and the row collapses to a uniform mask. */
        float s0 = bm / 127.0f;
        float s = s0 * inv;
        if (s < 1e-20f) s = 1e-20f;
        if (lane == 0) psc[(size_t)row * nblk + b] = s;
        float is = s0 > 0.0f ? 1.0f / s0 : 0.0f;
        for (int i = b0 + lane; i < b1; i += gs) {
            float v = (i >= lo && i < hi) ? exp(sp[i] - mx) : 0.0f;
            int q = (int)rint(v * is);
            pp[i] = (char)(q < 0 ? 0 : (q > 127 ? 127 : q));
        }
    }
}

/* ------------------------------------------------------------------ */
/* quantisation: f32 rows -> centred int8 rows, one scale per `blk`-wide */
/* block.  Same rounding as e2_quant_rows_blk; centred (not biased by   */
/* 128) because the matrix engine's signed/signed DPAS then needs no    */
/* zero-point term in the epilogue, which is the difference between     */
/* 4 and 9 TMAC/s.                                                     */
/* ------------------------------------------------------------------ */
__kernel void k_quant_s8(__global const float *x, __global char *q,
                         __global float *sc, const int rows, const int k,
                         const int blk)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int nblk = k / blk;
    long gid = (long)get_global_id(0) / gs;
    if (gid >= (long)rows * nblk) return;
    int b = (int)(gid % nblk);
    int row = (int)(gid / nblk);
    __global const float *xp = x + (size_t)row * k + (size_t)b * blk;
    const int gs4 = gs * 4;
    float mx = 0.0f;
    for (int i = lane * 4; i < blk; i += gs4) {
        float4 v = fabs(*(const __global float4 *)(xp + i));
        mx = fmax(mx, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
    }
    mx = sub_group_reduce_max(mx);
    float s = mx / 127.0f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.0f / s;
    __global char *qp = q + (size_t)row * k + (size_t)b * blk;
    for (int i = lane * 4; i < blk; i += gs4) {
        float4 v = *(const __global float4 *)(xp + i) * inv;
        int4 r = clamp(convert_int4_rte(v), (int4)(-127), (int4)(127));
        *(__global char4 *)(qp + i) = convert_char4(r);
    }
    if (lane == 0) sc[(size_t)row * nblk + b] = s;
}

/* gelu(gate) * up, quantised in the same pass (the MLP down-projection reads
 * the int8 form directly, so the f32 product never reaches memory). */
__kernel void k_gelu_mul_q(__global const float *gate, __global const float *up,
                           __global char *q, __global float *sc,
                           const int rows, const int k, const int blk)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int nblk = k / blk;
    long gid = (long)get_global_id(0) / gs;
    if (gid >= (long)rows * nblk) return;
    int b = (int)(gid % nblk);
    int row = (int)(gid / nblk);
    size_t base = (size_t)row * k + (size_t)b * blk;
    __global const float *gp = gate + base, *up2 = up + base;
    float mx = 0.0f;
    for (int i = lane; i < blk; i += gs) {
        float v = gelu1(gp[i]) * up2[i];
        mx = fmax(mx, fabs(v));
    }
    mx = sub_group_reduce_max(mx);
    float s = mx / 127.0f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.0f / s;
    __global char *qp = q + base;
    for (int i = lane; i < blk; i += gs) {
        float v = gelu1(gp[i]) * up2[i] * inv;
        int r = (int)rint(v);
        qp[i] = (char)(r < -127 ? -127 : (r > 127 ? 127 : r));
    }
    if (lane == 0) sc[(size_t)row * nblk + b] = s;
}

/* ------------------------------------------------------------------ */
/* small elementwise helpers                                           */
/* ------------------------------------------------------------------ */
__kernel void k_gelu_tanh(__global float *x, const int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;
    x[gid] = gelu1(x[gid]);
}

__kernel void k_scale(__global float *x, const int n, const float k)
{
    int gid = get_global_id(0);
    if (gid >= n) return;
    x[gid] *= k;
}

__kernel void k_mul_slice(__global float *h, __global const float *ple,
                          const int n, const int L, const int P, const int li)
{
    long gid = (long)get_global_id(0);
    if (gid >= n) return;
    long t = gid / P;
    int i = (int)(gid % P);
    h[gid] *= ple[(t * L + li) * P + i];
}

/* 2-D position embeddings: x[p][i] += tab[px][i] + tab[stride + py][i] */
__kernel void k_pos_add(__global float *x, __global const int *xy,
                        __global const float *tab, const int P, const int H,
                        const long stride)
{
    long gid = (long)get_global_id(0);
    if (gid >= (long)P * H) return;
    int p = (int)(gid / H), i = (int)(gid % H);
    x[gid] += tab[(size_t)xy[2 * p] * H + i] + tab[stride + (size_t)xy[2 * p + 1] * H + i];
}

/* softmax along the last axis of [rows][dim] (audio attention). */
__kernel void k_softmax(__global float *x, const int rows, const int dim)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    int row = get_global_id(0) / gs;
    if (row >= rows) return;
    __global float *p = x + (size_t)row * dim;
    float mx = -INFINITY;
    for (int i = lane; i < dim; i += gs) mx = fmax(mx, p[i]);
    mx = sub_group_reduce_max(mx);
    float sum = 0.0f;
    for (int i = lane; i < dim; i += gs) {
        float e = exp(p[i] - mx);
        p[i] = e;
        sum += e;
    }
    sum = sub_group_reduce_add(sum);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = lane; i < dim; i += gs) p[i] *= inv;
}
