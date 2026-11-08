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

static inline float gelu1(float x) {
    float t = GELU_K * (x + GELU_C * x * x * x);
    if (t > 15.0f) return x;
    return x / (1.0f + exp(-2.0f * t));
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
    float sum = 0.0f;
    for (int i = lane; i < dim; i += gs) sum += xp[i] * xp[i];
    sum = sub_group_reduce_add(sum);
    float s = rms_scale(sum, dim, eps);
    for (int i = lane; i < dim; i += gs) xp[i] = w ? xp[i] * s * w[i] : xp[i] * s;
    /* hf of the row is one rotation pair; the two halves are `dim/2` apart */
    int hf = dim / 2;
    int pos = rope_pos(row, per_pos, seq_off, nseq);
    __global const float *c = cs + (size_t)pos * hf;
    __global const float *sn2 = sn + (size_t)pos * hf;
    for (int d = lane; d < hf; d += gs) {
        float x1 = xp[d], x2 = xp[d + hf];
        xp[d] = x1 * c[d] - x2 * sn2[d];
        xp[d + hf] = x2 * c[d] + x1 * sn2[d];
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
#define ATT_DMAX 512          /* widest head in the model */

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void k_attn_part(__global const float *q, __global const float *k,
                          __global const float *v, __global float *part,
                          __global float *stats, __global const int *seq_off,
                          const int nseq, const int t, const int heads,
                          const int kv_heads, const int head_dim, const int window,
                          const int nchunk)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    /* one work-group = ATT_ROWS (4) sub-groups = 4 query rows of one block,
     * one band chunk; the work-group id walks (seq, head, row block, chunk) */
    long sg = (long)get_global_id(0) / gs;   /* sub-group id, independent of the
                                              * work-group size the driver picks */
    long wg = sg / ATT_ROWS;
    int widx = (int)(sg % ATT_ROWS);
    int npos = (t + ATT_ROWS - 1) / ATT_ROWS;
    int chunk = (int)(wg % nchunk);
    long rest = wg / nchunk;
    int pb = (int)(rest % npos);
    rest /= npos;
    int head = (int)(rest % heads);
    int seq = (int)(rest / heads);
    int pos = pb * ATT_ROWS + widx;
    int groups = heads / kv_heads;
    int kvh = head / groups;
    __local float att_acc[ATT_ROWS][ATT_DMAX];
    size_t qs = (size_t)heads * head_dim, ks = (size_t)kv_heads * head_dim;
    size_t off = (size_t)seq_off[seq];
    size_t psz = (size_t)nseq * t * heads * head_dim;   /* one chunk of partials */

    if (pos >= t || head_dim > ATT_DMAX) return;
    int lo = 0, hi = t;
    if (window > 0) {
        lo = pos - window; if (lo < 0) lo = 0;
        hi = pos + window + 1; if (hi > t) hi = t;
    }
    int l0 = lo, band = hi - lo, per = (band + nchunk - 1) / nchunk;
    lo = l0 + chunk * per;
    hi = lo + per; if (hi > l0 + band) hi = l0 + band;
    __global const float *qp = q + (off + pos) * qs + (size_t)head * head_dim;
    __global const float *kp = k + off * ks + (size_t)kvh * head_dim;
    __global const float *vp = v + off * ks + (size_t)kvh * head_dim;
    int per_lane = head_dim / gs;
    __local float *acc = att_acc[widx];
    for (int i = 0; i < per_lane; i++) acc[lane + i * gs] = 0.0f;

    /* Two keys per trip: both score dots are issued before either is reduced,
     * which is what gives the memory pipeline something to overlap.  The key
     * loop is serial per sub-group, and at 2048 tokens this kernel is limited
     * by memory-level parallelism (~10 GB/s of an 81 GB/s part), not by ALU. */
    float m = -INFINITY, l = 0.0f;
    int j = lo;
    for (; j + 2 <= hi; j += 2) {
        __global const float *k0 = kp + (size_t)j * ks;
        __global const float *k1 = k0 + ks;
        float s0 = 0.0f, s1 = 0.0f;
        for (int i = 0; i < per_lane; i++) {
            float qv = qp[lane + i * gs];
            int d = lane + i * gs;
            s0 += qv * k0[d];
            s1 += qv * k1[d];
        }
        s0 = sub_group_reduce_add(s0);
        s1 = sub_group_reduce_add(s1);
        __global const float *v0 = vp + (size_t)j * ks;
        __global const float *v1 = v0 + ks;
        float m0 = fmax(m, s0);
        float e0 = exp(s0 - m0);
        float a0 = exp(m - m0);
        float m1 = fmax(m0, s1);
        float e1 = exp(s1 - m1);
        float a1 = exp(m0 - m1);
        l = (l * a0 + e0) * a1 + e1;
        for (int i = 0; i < per_lane; i++) {
            int d = lane + i * gs;
            acc[d] = (acc[d] * a0 + e0 * v0[d]) * a1 + e1 * v1[d];
        }
        m = m1;
    }
    for (; j < hi; j++) {
        __global const float *kj = kp + (size_t)j * ks;
        float s = 0.0f;
        for (int i = 0; i < per_lane; i++) s += qp[lane + i * gs] * kj[lane + i * gs];
        s = sub_group_reduce_add(s);
        float mn = fmax(m, s);
        float e = exp(s - mn);
        float alpha = exp(m - mn);
        l = l * alpha + e;
        __global const float *vj = vp + (size_t)j * ks;
        for (int i = 0; i < per_lane; i++) {
            int d = lane + i * gs;
            acc[d] = acc[d] * alpha + e * vj[d];
        }
        m = mn;
    }
    if (nchunk == 1) {
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        __global float *op = part + (off + pos) * qs + (size_t)head * head_dim;
        for (int i = 0; i < per_lane; i++) {
            int d = lane + i * gs;
            op[d] = acc[d] * inv;
        }
        return;
    }
    __global float *pp = part + (size_t)chunk * psz +
                         (off + pos) * qs + (size_t)head * head_dim;
    for (int i = 0; i < per_lane; i++) {
        int d = lane + i * gs;
        pp[d] = acc[d];
    }
    if (lane == 0) {
        stats[((size_t)chunk * nseq * t + off + pos) * heads + head] = m;
        stats[((size_t)nchunk * nseq * t + off + pos) * heads + head] = l;
    }
}

/* Merge the band chunks of each (sequence, row, head). */
__kernel void k_attn_comb(__global const float *part, __global const float *stats,
                          __global float *out, const int nseq, const int t,
                          const int heads, const int head_dim, const int nchunk)
{
    int gs = get_sub_group_size();
    int lane = get_sub_group_local_id();
    long row = (long)get_global_id(0) / gs;
    long rows = (long)nseq * t * heads;
    if (row >= rows) return;
    size_t psz = rows * head_dim;
    int head = (int)(row % heads);
    long tr = row / heads;                     /* (seq, pos) flattened */
    float m = -INFINITY;
    for (int c = 0; c < nchunk; c++) {
        float mc = stats[((size_t)c * nseq * t + tr) * heads + head];
        if (mc > m) m = mc;
    }
    float lsum = 0.0f;
    for (int c = 0; c < nchunk; c++) {
        float mc = stats[((size_t)c * nseq * t + tr) * heads + head];
        float lc = stats[((size_t)nchunk * nseq * t + tr) * heads + head];
        lsum += lc * exp(mc - m);
    }
    float inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    int per_lane = head_dim / gs;
    __global float *op = out + row * head_dim;
    for (int i = 0; i < per_lane; i++) {
        int d = lane + i * gs;
        float v = 0.0f;
        for (int c = 0; c < nchunk; c++) {
            float mc = stats[((size_t)c * nseq * t + tr) * heads + head];
            v += part[(size_t)c * psz + row * head_dim + d] * exp(mc - m);
        }
        op[d] = v * inv;
    }
}

/* ------------------------------------------------------------------ */
/* quantisation: f32 rows -> u8 rows (zero point 128), one scale per    */
/* `blk`-wide block.  Same rounding as e2_quant_rows_blk.               */
/* ------------------------------------------------------------------ */
__kernel void k_quant_u8(__global const float *x, __global uchar *q,
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
    float mx = 0.0f;
    for (int i = lane; i < blk; i += gs) mx = fmax(mx, fabs(xp[i]));
    mx = sub_group_reduce_max(mx);
    float s = mx / 127.0f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.0f / s;
    __global uchar *qp = q + (size_t)row * k + (size_t)b * blk;
    for (int i = lane; i < blk; i += gs) {
        int v = (int)rint(xp[i] * inv) + 128;
        qp[i] = (uchar)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    if (lane == 0) sc[(size_t)row * nblk + b] = s;
}

/* gelu(gate) * up, quantised in the same pass (the MLP down-projection reads
 * the u8 form directly, so the f32 product never reaches memory). */
__kernel void k_gelu_mul_q(__global const float *gate, __global const float *up,
                           __global uchar *q, __global float *sc,
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
    __global uchar *qp = q + base;
    for (int i = lane; i < blk; i += gs) {
        float v = gelu1(gp[i]) * up2[i] * inv;
        int r = (int)rint(v) + 128;
        qp[i] = (uchar)(r < 0 ? 0 : (r > 255 ? 255 : r));
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
