// Metal kernels for Qwen2.5-Coder-0.5B, compiled at runtime (no offline metal compiler without Xcode).
// Every shape is a compile-time constant. FMT selects the weight format: 0 = f16, 2 = q4 (affine,
// group 64, f16 scale+bias). One decode step = commit/embed + 24 x 5 layer kernels + lm_head.

let shaderSource = #"""
#include <metal_stdlib>
using namespace metal;

#define H 896
#define FF 4864
#define NH 14
#define NKV 2
#define HD 64
#define QKV 1152
#define VOCAB 151936
#define EPS 1e-6f
#define LMH_ROWS_PER_TG 32
#define LMH_TGS (VOCAB / LMH_ROWS_PER_TG)

struct State {
    int pos;       // position of the first token of the current pass
    int n;         // tokens in the current pass
    int step;      // generated tokens committed so far
    int done;      // stop flag: every kernel returns immediately once set
    int maxctx;
    int maxsteps;
    int eos0;
    int eos1;
};

// ---------------------------------------------------------------------------------------------
// Weight access. W points at the first row of the matrix; SB at its scales (q4 only).

#if FMT == 0
typedef half wtype;
// R rows starting at row r0 (row-major, K columns) dotted with xs; lanes stride K in 8-wide chunks.
template<int K, int R>
inline void mv_rows(device const half* W, device const half2* SB, uint r0,
                    threadgroup const float* xs, uint lane, thread float* acc) {
    constexpr int NC = K / 8;
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int c = lane; c < NC; c += 32) {
        const float4 x0 = *((threadgroup const float4*)(xs + c * 8));
        const float4 x1 = *((threadgroup const float4*)(xs + c * 8 + 4));
        for (int r = 0; r < R; ++r) {
            const uint4 w = *((device const uint4*)(W + (ulong)(r0 + r) * K + c * 8));
            acc[r] += dot(float4(as_type<half4>(w.xy)), x0) + dot(float4(as_type<half4>(w.zw)), x1);
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}
inline float embed_at(device const half* E, device const half2* SB, int row, int i) {
    return float(E[(ulong)row * H + i]);
}
#elif FMT == 2
typedef uchar wtype;
// q4: row = K/2 bytes of nibbles (weight 2j in the low nibble of byte j), SB = [rows][K/64] (scale, bias).
template<int K, int R>
inline void mv_rows(device const uchar* W, device const half2* SB, uint r0,
                    threadgroup const float* xs, uint lane, thread float* acc) {
    constexpr int NC = K / 32;
    constexpr int NG = K / 64;
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int c = lane; c < NC; c += 32) {
        float xv[32];
        float xsum = 0.f;
        for (int j = 0; j < 32; j += 4) {
            const float4 t = *((threadgroup const float4*)(xs + c * 32 + j));
            xv[j] = t.x; xv[j + 1] = t.y; xv[j + 2] = t.z; xv[j + 3] = t.w;
            xsum += t.x + t.y + t.z + t.w;
        }
        for (int r = 0; r < R; ++r) {
            const uint4 w = *((device const uint4*)(W + (ulong)(r0 + r) * (K / 2) + c * 16));
            const float2 sb = float2(SB[(r0 + r) * NG + (c >> 1)]);
            float d = 0.f;
            for (int u = 0; u < 4; ++u) {
                const uint v = w[u];
                for (int k = 0; k < 8; ++k) d += float((v >> (4 * k)) & 0xFu) * xv[u * 8 + k];
            }
            acc[r] += sb.x * d + sb.y * xsum;
        }
    }
    for (int r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}
inline float embed_at(device const uchar* E, device const half2* SB, int row, int i) {
    const uchar b = E[(ulong)row * (H / 2) + (i >> 1)];
    const float q = float((i & 1) ? (b >> 4) : (b & 0xF));
    const float2 sb = float2(SB[row * (H / 64) + (i >> 6)]);
    return sb.x * q + sb.y;
}
#endif

// ---------------------------------------------------------------------------------------------
// RMSNorm of one 896-vector into threadgroup memory (256 threads).
inline void stage_rmsnorm(device const float* x, device const float* w, threadgroup float* xs,
                          threadgroup float* red, uint tid, uint lane, uint sg) {
    float ss = 0.f;
    for (uint i = tid; i < H; i += 256) { const float v = x[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; ++s) tot += red[s];
    const float scale = rsqrt(tot / float(H) + EPS);
    for (uint i = tid; i < H; i += 256) xs[i] = x[i] * scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void stage_plain(device const float* x, threadgroup float* xs, int n, uint tid) {
    for (int i = tid; i < n; i += 256) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------------------------
// Prompt embedding: grid (n), 256 threads.
kernel void k_embed_prompt(device const wtype* E [[buffer(0)]],
                           device const half2* ESB [[buffer(1)]],
                           device const int* toks [[buffer(2)]],
                           device float* x [[buffer(3)]],
                           device const State* st [[buffer(4)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    if (st->done || (int)t >= st->n) return;
    const int tok = toks[t];
    for (uint i = tid; i < H; i += 256) x[t * H + i] = embed_at(E, ESB, tok, i);
}

// Greedy commit + embed: one threadgroup of 1024 threads. Reduces the lm_head partial maxima to the
// next token, records it, advances the position, and embeds it for the following decode step.
kernel void k_commit(device const float* pval [[buffer(0)]],
                     device const int* pidx [[buffer(1)]],
                     device State* st [[buffer(2)]],
                     device int* out_tokens [[buffer(3)]],
                     device const wtype* E [[buffer(4)]],
                     device const half2* ESB [[buffer(5)]],
                     device float* x [[buffer(6)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = tid; i < LMH_TGS; i += 1024) {
        const float v = pval[i];
        const int id = pidx[i];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_down(best, off);
        const int oi = simd_shuffle_down(bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    threadgroup float rv[32];
    threadgroup int ri[32];
    threadgroup int tok_s;
    if (lane == 0) { rv[sg] = best; ri[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = rv[lane]; bi = ri[lane];
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) {
            const int tok = bi;
            out_tokens[st->step] = tok;
            st->step += 1;
            st->pos += st->n;
            st->n = 1;
            const bool stop = tok == st->eos0 || tok == st->eos1 || st->step >= st->maxsteps;
            if (stop) st->done = 1;
            tok_s = stop ? -1 : tok;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tok = tok_s;
    if (tok < 0) return;
    for (uint i = tid; i < H; i += 1024) x[i] = embed_at(E, ESB, tok, i);
}

// ---------------------------------------------------------------------------------------------
// RMSNorm + QKV projection + bias + RoPE + KV-cache write. Grid (36, n), 256 threads.
// Rows are stored in blocks of 32: block b = 2*head + half holds dims [16*half, 16*half+16) and
// their rotary partners [32+16*half, 32+16*half+16) of head `head` (heads 0-13 q, 14-15 k, 16-17 v).
kernel void k_attn_in(device const wtype* W [[buffer(0)]],
                      device const half2* SB [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device const float* wnorm [[buffer(3)]],
                      device const float* x [[buffer(4)]],
                      device float* q [[buffer(5)]],
                      device half* kc [[buffer(6)]],
                      device half* vc [[buffer(7)]],
                      device const float2* rope [[buffer(8)]],
                      device const State* st [[buffer(9)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    const int pos = st->pos + t;
    threadgroup float xs[H];
    threadgroup float red[8];
    threadgroup float outv[32];
    stage_rmsnorm(x + t * H, wnorm, xs, red, tid, lane, sg);
    const uint r0 = tgp.x * 32 + sg * 4;
    float acc[4];
    mv_rows<H, 4>(W, SB, r0, xs, lane, acc);
    if (lane == 0) for (int r = 0; r < 4; ++r) outv[sg * 4 + r] = acc[r] + bias[r0 + r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const int head = tgp.x >> 1;
        const int d = (tgp.x & 1) * 16 + tid;
        float a = outv[tid], b = outv[tid + 16];
        if (head < NH + NKV) {
            const float2 cs = rope[pos * 32 + d];
            const float a2 = a * cs.x - b * cs.y;
            const float b2 = b * cs.x + a * cs.y;
            a = a2; b = b2;
        }
        if (head < NH) {
            q[t * H + head * HD + d] = a;
            q[t * H + head * HD + d + 32] = b;
        } else if (head < NH + NKV) {
            device half* k = kc + ((ulong)(head - NH) * st->maxctx + pos) * HD;
            k[d] = half(a); k[d + 32] = half(b);
        } else {
            device half* v = vc + ((ulong)(head - NH - NKV) * st->maxctx + pos) * HD;
            v[d] = half(a); v[d + 32] = half(b);
        }
    }
}

// Causal attention for one (query head, token). Grid (14, n), 256 threads.
#define MAXCTX_TG 2048
kernel void k_attention(device const float* q [[buffer(0)]],
                        device const half* kc [[buffer(1)]],
                        device const half* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int h = tgp.x;
    const int t = tgp.y;
    if (t >= st->n) return;
    const int L = st->pos + t + 1;
    const int g = h / (NH / NKV);
    threadgroup float qs[HD];
    threadgroup float sc[MAXCTX_TG];
    threadgroup float red[8];
    threadgroup float part[256];
    if (tid < HD) qs[tid] = q[t * H + h * HD + tid] * 0.125f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half* kb = kc + (ulong)g * st->maxctx * HD;
    device const half* vb = vc + (ulong)g * st->maxctx * HD;
    float m = -INFINITY;
    for (int s = tid; s < L; s += 256) {
        device const half4* kr = (device const half4*)(kb + (ulong)s * HD);
        float a = 0.f;
        for (int j = 0; j < 16; ++j) a += dot(float4(kr[j]), *((threadgroup const float4*)(qs + 4 * j)));
        sc[s] = a;
        m = max(m, a);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int i = 1; i < 8; ++i) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.f;
    for (int s = tid; s < L; s += 256) { const float e = exp(sc[s] - m); sc[s] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.f;
    for (int i = 0; i < 8; ++i) sum += red[i];
    const int d = tid & 63;
    const int p = tid >> 6;
    float o = 0.f;
    for (int s = p; s < L; s += 4) o += sc[s] * float(vb[(ulong)s * HD + d]);
    part[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HD) out[t * H + h * HD + tid] = (part[tid] + part[tid + 64] + part[tid + 128] + part[tid + 192]) / sum;
}

// Output projection + residual. Grid (28, n), 256 threads, 32 rows per threadgroup.
kernel void k_o_proj(device const wtype* W [[buffer(0)]],
                     device const half2* SB [[buffer(1)]],
                     device const float* a [[buffer(2)]],
                     device float* x [[buffer(3)]],
                     device const State* st [[buffer(4)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    threadgroup float xs[H];
    stage_plain(a + t * H, xs, H, tid);
    const uint r0 = tgp.x * 32 + sg * 4;
    float acc[4];
    mv_rows<H, 4>(W, SB, r0, xs, lane, acc);
    if (lane < 4) x[t * H + r0 + lane] += acc[lane];
}

// RMSNorm + gate/up + SiLU*up. Rows stored in groups of 4: gate[2b], gate[2b+1], up[2b], up[2b+1].
// Grid (304, n), 256 threads: each simdgroup produces 2 of the 4864 hidden units.
kernel void k_mlp_in(device const wtype* W [[buffer(0)]],
                     device const half2* SB [[buffer(1)]],
                     device const float* wnorm [[buffer(2)]],
                     device const float* x [[buffer(3)]],
                     device float* hbuf [[buffer(4)]],
                     device const State* st [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    threadgroup float xs[H];
    threadgroup float red[8];
    stage_rmsnorm(x + t * H, wnorm, xs, red, tid, lane, sg);
    const uint b = tgp.x * 8 + sg;
    float acc[4];
    mv_rows<H, 4>(W, SB, b * 4, xs, lane, acc);
    if (lane < 2) {
        const float g = acc[lane];
        const float u = acc[lane + 2];
        hbuf[t * FF + b * 2 + lane] = g / (1.f + exp(-g)) * u;
    }
}

// Down projection + residual. Grid (28, n), 256 threads, 32 rows per threadgroup.
kernel void k_mlp_out(device const wtype* W [[buffer(0)]],
                      device const half2* SB [[buffer(1)]],
                      device const float* hbuf [[buffer(2)]],
                      device float* x [[buffer(3)]],
                      device const State* st [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    threadgroup float xs[FF];
    stage_plain(hbuf + t * FF, xs, FF, tid);
    const uint r0 = tgp.x * 32 + sg * 4;
    float acc[4];
    mv_rows<FF, 4>(W, SB, r0, xs, lane, acc);
    if (lane < 4) x[t * H + r0 + lane] += acc[lane];
}

// Final RMSNorm + tied lm_head on the last token of the pass + per-threadgroup argmax.
// Grid (4748), 256 threads, 32 vocabulary rows per threadgroup.
kernel void k_lm_head(device const wtype* E [[buffer(0)]],
                      device const half2* ESB [[buffer(1)]],
                      device const float* wnorm [[buffer(2)]],
                      device const float* x [[buffer(3)]],
                      device float* pval [[buffer(4)]],
                      device int* pidx [[buffer(5)]],
                      device const State* st [[buffer(6)]],
                      device float* logits [[buffer(7)]],
                      constant int& write_logits [[buffer(8)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float xs[H];
    threadgroup float red[8];
    threadgroup float bv[8];
    threadgroup int bix[8];
    stage_rmsnorm(x + (st->n - 1) * H, wnorm, xs, red, tid, lane, sg);
    const uint r0 = tg * LMH_ROWS_PER_TG + sg * 4;
    float acc[4];
    mv_rows<H, 4>(E, ESB, r0, xs, lane, acc);
    if (lane == 0) {
        float best = acc[0];
        int bi = 0;
        for (int r = 1; r < 4; ++r) if (acc[r] > best) { best = acc[r]; bi = r; }
        bv[sg] = best;
        bix[sg] = r0 + bi;
        if (write_logits) for (int r = 0; r < 4; ++r) logits[r0 + r] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = bv[0];
        int bi = bix[0];
        for (int s = 1; s < 8; ++s) if (bv[s] > best) { best = bv[s]; bi = bix[s]; }
        pval[tg] = best;
        pidx[tg] = bi;
    }
}

// ---------------------------------------------------------------------------------------------
// Prefill path: batched GEMMs on simdgroup matrices. A = activations [n][K] (f32), W = [R][K].
// Each threadgroup (4 simdgroups) computes RT rows x 32 tokens; per 32-wide K step it dequantizes
// the weight tile once into threadgroup memory (half) and shares it across all tokens.

kernel void k_rmsnorm_rows(device const float* x [[buffer(0)]],
                           device const float* w [[buffer(1)]],
                           device float* xn [[buffer(2)]],
                           device const State* st [[buffer(3)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done || (int)t >= st->n) return;
    threadgroup float red[8];
    device const float* xr = x + t * H;
    float ss = 0.f;
    for (uint i = tid; i < H; i += 256) { const float v = xr[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; ++s) tot += red[s];
    const float scale = rsqrt(tot / float(H) + EPS);
    for (uint i = tid; i < H; i += 256) xn[t * H + i] = xr[i] * scale * w[i];
}

#define GT 32   // tokens per threadgroup
#define GK 32   // K step

// Dequantize rows [r0, r0+RT) x k [k0, k0+GK) of W into Ws[RT][GK] (128 threads).
template<int K, int RT>
inline void load_w_tile(device const wtype* W, device const half2* SB, uint r0, uint k0,
                        threadgroup half* Ws, uint tid) {
    constexpr int PER = RT * GK / 128;     // weights per thread: 16 (RT 64) or 8 (RT 32)
    constexpr int TPR = GK / PER;          // threads per row
    const uint r = tid / TPR;
    const uint c = (tid % TPR) * PER;
#if FMT == 0
    device const half* src = W + (ulong)(r0 + r) * K + k0 + c;
    for (int j = 0; j < PER; ++j) Ws[r * GK + c + j] = src[j];
#elif FMT == 2
    device const uchar* src = W + (ulong)(r0 + r) * (K / 2) + (k0 + c) / 2;
    const float2 sb = float2(SB[(r0 + r) * (K / 64) + (k0 + c) / 64]);
    for (int j = 0; j < PER / 2; ++j) {
        const uchar b = src[j];
        Ws[r * GK + c + 2 * j] = half(sb.x * float(b & 0xF) + sb.y);
        Ws[r * GK + c + 2 * j + 1] = half(sb.x * float(b >> 4) + sb.y);
    }
#endif
}

// MODE 0: QKV (+bias, RoPE, write q / K / V cache); 1: residual add into x; 2: SiLU(gate)*up -> h.
template<int K, int RT, int MODE>
inline void gemm_body(device const float* A, device const wtype* W, device const half2* SB,
                      device const float* bias, device float* out, device half* kc, device half* vc,
                      device const float2* rope, device const State* st,
                      threadgroup half* Ws, threadgroup half* Xs, threadgroup float* Os,
                      uint2 tgp, uint tid, uint sg) {
    const int n = st->n;
    const int t0 = tgp.y * GT;
    const uint r0 = tgp.x * RT;
    constexpr int RF = RT / 32;            // 8-row fragments per simdgroup (1 or 2)
    simdgroup_float8x8 acc[4][RF];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < RF; ++j) acc[i][j] = simdgroup_float8x8(0.f);
    for (uint k0 = 0; k0 < (uint)K; k0 += GK) {
        {   // activations: 32 tokens x 32 k, 8 per thread
            const int t = tid / 4;
            const int c = (tid % 4) * 8;
            if (t0 + t < n) {
                device const float4* src = (device const float4*)(A + (ulong)(t0 + t) * K + k0 + c);
                const float4 a = src[0], b = src[1];
                Xs[t * GK + c + 0] = half(a.x); Xs[t * GK + c + 1] = half(a.y);
                Xs[t * GK + c + 2] = half(a.z); Xs[t * GK + c + 3] = half(a.w);
                Xs[t * GK + c + 4] = half(b.x); Xs[t * GK + c + 5] = half(b.y);
                Xs[t * GK + c + 6] = half(b.z); Xs[t * GK + c + 7] = half(b.w);
            } else {
                for (int j = 0; j < 8; ++j) Xs[t * GK + c + j] = 0.h;
            }
        }
        load_w_tile<K, RT>(W, SB, r0, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GK; kk += 8) {
            simdgroup_half8x8 a[4], b[RF];
            for (int i = 0; i < 4; ++i) simdgroup_load(a[i], Xs + (i * 8) * GK + kk, GK);
            for (int j = 0; j < RF; ++j) simdgroup_load(b[j], Ws + (sg * RF * 8 + j * 8) * GK + kk, GK, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i) for (int j = 0; j < RF; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < RF; ++j)
        simdgroup_store(acc[i][j], Os + (i * 8) * RT + sg * RF * 8 + j * 8, RT);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (MODE == 0) {
        // RT == 32: one rotary block: rows i and i+16 hold dims d and d+32.
        const int head = tgp.x >> 1;
        for (int w = tid; w < GT * 16; w += 128) {
            const int t = w / 16, i = w % 16;
            if (t0 + t >= n) continue;
            const int pos = st->pos + t0 + t;
            const int d = (tgp.x & 1) * 16 + i;
            float a = Os[t * RT + i] + bias[r0 + i];
            float b = Os[t * RT + i + 16] + bias[r0 + i + 16];
            if (head < NH + NKV) {
                const float2 cs = rope[pos * 32 + d];
                const float a2 = a * cs.x - b * cs.y;
                const float b2 = b * cs.x + a * cs.y;
                a = a2; b = b2;
            }
            if (head < NH) {
                out[(t0 + t) * H + head * HD + d] = a;
                out[(t0 + t) * H + head * HD + d + 32] = b;
            } else if (head < NH + NKV) {
                device half* k = kc + ((ulong)(head - NH) * st->maxctx + pos) * HD;
                k[d] = half(a); k[d + 32] = half(b);
            } else {
                device half* v = vc + ((ulong)(head - NH - NKV) * st->maxctx + pos) * HD;
                v[d] = half(a); v[d + 32] = half(b);
            }
        }
    } else if (MODE == 1) {
        for (int w = tid; w < GT * RT; w += 128) {
            const int t = w / RT, r = w % RT;
            if (t0 + t < n) out[(t0 + t) * H + r0 + r] += Os[t * RT + r];
        }
    } else {
        for (int w = tid; w < GT * RT / 4; w += 128) {
            const int t = w / (RT / 4), b = w % (RT / 4);
            if (t0 + t >= n) continue;
            const int u = (r0 / 4 + b) * 2;
            const float g0 = Os[t * RT + 4 * b], g1 = Os[t * RT + 4 * b + 1];
            const float u0 = Os[t * RT + 4 * b + 2], u1 = Os[t * RT + 4 * b + 3];
            out[(t0 + t) * FF + u] = g0 / (1.f + exp(-g0)) * u0;
            out[(t0 + t) * FF + u + 1] = g1 / (1.f + exp(-g1)) * u1;
        }
    }
}

#define GEMM_KERNEL(NAME, K, RT, MODE) \
kernel void NAME(device const float* A [[buffer(0)]], device const wtype* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* out [[buffer(4)]], device half* kc [[buffer(5)]], device half* vc [[buffer(6)]], \
                 device const float2* rope [[buffer(7)]], device const State* st [[buffer(8)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * GT) >= st->n) return; \
    threadgroup half Ws[RT * GK]; threadgroup half Xs[GT * GK]; threadgroup float Os[GT * RT]; \
    gemm_body<K, RT, MODE>(A, W, SB, bias, out, kc, vc, rope, st, Ws, Xs, Os, tgp, tid, sg); \
}

GEMM_KERNEL(k_gemm_qkv, H, 32, 0)
GEMM_KERNEL(k_gemm_o, H, 32, 1)
GEMM_KERNEL(k_gemm_gu, H, 64, 2)
GEMM_KERNEL(k_gemm_down, FF, 32, 1)

// ---------------------------------------------------------------------------------------------
// Prefill GEMM v2: activations X are half [n][K]; each threadgroup computes 32 rows x TF*8 tokens
// with NS simdgroups splitting K (intra-threadgroup split-K), then reduces through threadgroup
// memory and applies the epilogue. Per 32-wide K step, lane l dequantizes row l's 32 weights.

kernel void k_rmsnorm_rows_h(device const float* x [[buffer(0)]],
                             device const float* w [[buffer(1)]],
                             device half* xn [[buffer(2)]],
                             device const State* st [[buffer(3)]],
                             uint t [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done || (int)t >= st->n) return;
    threadgroup float red[8];
    device const float* xr = x + t * H;
    float ss = 0.f;
    for (uint i = tid; i < H; i += 256) { const float v = xr[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; ++s) tot += red[s];
    const float scale = rsqrt(tot / float(H) + EPS);
    for (uint i = tid; i < H; i += 256) xn[t * H + i] = half(xr[i] * scale * w[i]);
}

kernel void k_f32_to_h(device const float* a [[buffer(0)]], device half* b [[buffer(1)]],
                       device const State* st [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (st->done || (int)i >= st->n * H) return;
    b[i] = half(a[i]);
}

template<int K, int MODE, int TF, int NS>
inline void gemm2_body(device const half* X, device const wtype* W, device const half2* SB,
                       device const float* bias, device float* outf, device half* outh,
                       device half* kc, device half* vc, device const float2* rope, device const State* st,
                       threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    constexpr int KS = K / NS;
    constexpr int NSTEP = KS / 32;
    constexpr int TT = TF * 8;
    const int n = st->n;
    const uint r0 = tgp.x * 32;
    const int t0 = tgp.y * TT;
    threadgroup half* ws = Wsg + sg * 32 * 32;
    simdgroup_float8x8 acc[TF][4];
    for (int i = 0; i < TF; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.f);
    const uint kbase = sg * KS;
    for (int s = 0; s < NSTEP; ++s) {
        const uint k0 = kbase + s * 32;
#if FMT == 2
        const uint4 wv = *((device const uint4*)(W + (ulong)(r0 + lane) * (K / 2) + k0 / 2));
        const float2 sb = float2(SB[(r0 + lane) * (K / 64) + k0 / 64]);
        threadgroup half4* dst = (threadgroup half4*)(ws + lane * 32);
        for (int u = 0; u < 4; ++u) {
            const uint v = wv[u];
            dst[2 * u] = half4(sb.x * float4(float(v & 0xF), float((v >> 4) & 0xF), float((v >> 8) & 0xF), float((v >> 12) & 0xF)) + sb.y);
            dst[2 * u + 1] = half4(sb.x * float4(float((v >> 16) & 0xF), float((v >> 20) & 0xF), float((v >> 24) & 0xF), float(v >> 28)) + sb.y);
        }
#else
        device const uint4* src = (device const uint4*)(W + (ulong)(r0 + lane) * K + k0);
        threadgroup uint4* dst = (threadgroup uint4*)(ws + lane * 32);
        for (int u = 0; u < 4; ++u) dst[u] = src[u];
#endif
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a[TF], b[4];
            for (int i = 0; i < TF; ++i) simdgroup_load(a[i], X + (ulong)(t0 + 8 * i) * K + k0 + kk, K);
            for (int j = 0; j < 4; ++j) simdgroup_load(b[j], ws + (8 * j) * 32 + kk, 32, ulong2(0, 0), true);
            for (int i = 0; i < TF; ++i) for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* os = Os + sg * TT * 32;
    for (int i = 0; i < TF; ++i) for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], os + (8 * i) * 32 + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int NT = NS * 32;
    if (MODE == 0) {
        const int head = tgp.x >> 1;
        for (int w = tid; w < TT * 16; w += NT) {
            const int t = w / 16, i = w % 16;
            if (t0 + t >= n) continue;
            float a = bias[r0 + i], b = bias[r0 + i + 16];
            for (int q = 0; q < NS; ++q) { a += Os[q * TT * 32 + t * 32 + i]; b += Os[q * TT * 32 + t * 32 + i + 16]; }
            const int pos = st->pos + t0 + t;
            const int d = (tgp.x & 1) * 16 + i;
            if (head < NH + NKV) {
                const float2 cs = rope[pos * 32 + d];
                const float a2 = a * cs.x - b * cs.y;
                const float b2 = b * cs.x + a * cs.y;
                a = a2; b = b2;
            }
            if (head < NH) {
                outf[(t0 + t) * H + head * HD + d] = a;
                outf[(t0 + t) * H + head * HD + d + 32] = b;
            } else if (head < NH + NKV) {
                device half* k = kc + ((ulong)(head - NH) * st->maxctx + pos) * HD;
                k[d] = half(a); k[d + 32] = half(b);
            } else {
                device half* v = vc + ((ulong)(head - NH - NKV) * st->maxctx + pos) * HD;
                v[d] = half(a); v[d + 32] = half(b);
            }
        }
    } else if (MODE == 1) {
        for (int w = tid; w < TT * 32; w += NT) {
            const int t = w / 32, r = w % 32;
            if (t0 + t >= n) continue;
            float v = 0.f;
            for (int q = 0; q < NS; ++q) v += Os[q * TT * 32 + w];
            outf[(t0 + t) * H + r0 + r] += v;
        }
    } else {
        for (int w = tid; w < TT * 8; w += NT) {
            const int t = w / 8, b = w % 8;
            if (t0 + t >= n) continue;
            float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
            for (int q = 0; q < NS; ++q) {
                threadgroup const float* o = Os + q * TT * 32 + t * 32 + 4 * b;
                g0 += o[0]; g1 += o[1]; u0 += o[2]; u1 += o[3];
            }
            const int u = (r0 / 4 + b) * 2;
            outh[(t0 + t) * FF + u] = half(g0 / (1.f + exp(-g0)) * u0);
            outh[(t0 + t) * FF + u + 1] = half(g1 / (1.f + exp(-g1)) * u1);
        }
    }
}

#define GEMM2(NAME, K, MODE, TF, NS) \
kernel void NAME(device const half* X [[buffer(0)]], device const wtype* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TF * 8) >= st->n) return; \
    constexpr int WB = NS * 32 * 32 * 2, OB = NS * TF * 8 * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm2_body<K, MODE, TF, NS>(X, W, SB, bias, outf, outh, kc, vc, rope, st, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
#define GEMM2_ALL(TF) \
GEMM2(k_g2_qkv_##TF, H, 0, TF, 4) \
GEMM2(k_g2_o_##TF, H, 1, TF, 4) \
GEMM2(k_g2_gu_##TF, H, 2, TF, 4) \
GEMM2(k_g2_down_##TF, FF, 1, TF, 8)
GEMM2_ALL(1)
GEMM2_ALL(2)
GEMM2_ALL(3)
GEMM2_ALL(4)
"""#
