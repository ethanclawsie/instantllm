// Metal kernels for Qwen2.5-Coder-0.5B, compiled at runtime (no offline metal compiler without Xcode).
// Every shape is a compile-time constant. FMT selects the weight format: 0 = f16, 2 = q4 (affine,
// group 64, f16 scale+bias). One decode step = commit/embed + 24 x 5 layer kernels + lm_head.

let shaderSource = #"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

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
    int nver;      // rows whose argmax the next commit reads (1 after prefill, T after a verify pass)
    int tspec;     // rows per speculative pass (1 = no speculation)
    int pad0;
    int pad1;
    int tlmh;      // rows of the lm_head pass in flight (for the threshold kernel)
    int pad2;
    int pad3;
    int pad4;
};

// n-gram draft table: open addressing, NG_SLOTS entries of {key, next token, count, total}.
#define NG_SLOTS (1 << 20)
#define NG_MAXN 6
struct NGEntry {
    ulong key;     // 0 = empty
    int next;
    ushort count;
    ushort total;
};

inline ulong ng_key(thread const int* ctx, int n) {   // ctx[0..n-1] = the last n tokens, oldest first
    ulong h = 0xcbf29ce484222325ul ^ ulong(n);
    for (int i = 0; i < n; i++) { h ^= ulong(uint(ctx[i])) + 0x9e3779b97f4a7c15ul; h *= 0x100000001b3ul; h ^= h >> 29; }
    return h | 1ul;
}

// Most frequent continuation of the longest known context (orders NG_MAXN..1); -1 if none is confident.
inline int ng_next(device const NGEntry* tab, thread const int* hist, int len, int minc) {
    for (int n = min(NG_MAXN, len); n >= 1; n--) {
        const ulong k = ng_key(hist + len - n, n);
        uint slot = uint(k >> 12) & (NG_SLOTS - 1);
        for (int p = 0; p < 16; p++) {
            const NGEntry e = tab[slot];
            if (e.key == 0) break;
            if (e.key == k) {
                if (int(e.count) >= minc && 2 * int(e.count) >= int(e.total)) return e.next;
                return -1;   // known context, no confident continuation: stop drafting
            }
            slot = (slot + 1) & (NG_SLOTS - 1);
        }
    }
    return -1;
}

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
        // x pre-scaled by 16: a nibble n unpacks to fp16 (1 + n/16) with bit ops only, so the dot
        // product is sum((16 + n) x) = sum(n x) + 16 * sum(x).
        float xv[32];
        float xsum = 0.f;
        for (int j = 0; j < 32; j += 4) {
            const float4 t = *((threadgroup const float4*)(xs + c * 32 + j));
            xv[j] = 16.f * t.x; xv[j + 1] = 16.f * t.y; xv[j + 2] = 16.f * t.z; xv[j + 3] = 16.f * t.w;
            xsum += t.x + t.y + t.z + t.w;
        }
        for (int r = 0; r < R; ++r) {
            const uint4 w = *((device const uint4*)(W + (ulong)(r0 + r) * (K / 2) + c * 16));
            const float2 sb = float2(SB[(r0 + r) * NG + (c >> 1)]);
            float d = 0.f;
            for (int u = 0; u < 4; ++u) {
                const uint v = w[u];
                const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
                thread const float* xu = xv + u * 8;
                d = fma(float(p04.x), xu[0], d); d = fma(float(p15.x), xu[1], d);
                d = fma(float(p26.x), xu[2], d); d = fma(float(p37.x), xu[3], d);
                d = fma(float(p04.y), xu[4], d); d = fma(float(p15.y), xu[5], d);
                d = fma(float(p26.y), xu[6], d); d = fma(float(p37.y), xu[7], d);
            }
            acc[r] += sb.x * (d - 16.f * xsum) + sb.y * xsum;
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
#elif FMT == 3
typedef uchar wtype;
#define Q4A_B896 8064     // bytes per 16-row block, K = 896 (14 nibble chunks + scales + zero points)
#define Q4A_B4864 10624   // bytes per (16-row block, K quarter), K = 4864 (19 nibble chunks + ...)
inline float embed_at(device const uchar* E, device const half2* SB, int row, int i) {
    const int b = row >> 4, r = row & 15;
    const int l = i / 28, j = i - l * 28;
    device const uchar* pb = E + (ulong)b * Q4A_B896;
    const uchar byte = pb[((j >> 1) * 32 + l) * 16 + r];
    const float n = float((j & 1) ? (byte >> 4) : (byte & 0xF));
    const float s = float(as_type<half>(ushort(ushort(pb[(14 * 32 + l) * 16 + r]) << 7)));
    const uint2 zl = ((device const uint2*)(pb + 15 * 512))[l];
    const uint zh = ((device const uint*)(pb + 15 * 512 + 256))[l];
    const uint zq = (((r < 8 ? zl.x : zl.y) >> (4 * (r & 7))) & 0xFu) | (((zh >> (2 * r)) & 0x3u) << 4);
    return s * (n - float(zq) * 0.25f);
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

// Wakes the GPU ahead of a request (an empty command buffer does not).
kernel void k_noop(device int* sink [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xFFFFFFFF) sink[0] = 1;
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

// Speculative commit: one threadgroup of 1024 threads. Rows 0..nver-1 of the pass that just ran were
// [last token, draft 0, ..., draft nver-2]; their argmaxes g_t come from the lm_head partials
// [t][LMH_TGS]. Accepts the longest prefix with draft t == g_t, commits g_0..g_m (stopping at EOS),
// drafts tspec-1 new tokens from the n-gram table and embeds [g_m, drafts...] for the next pass.
kernel void k_commit_spec(device const float* pval [[buffer(0)]],
                          device const int* pidx [[buffer(1)]],
                          device State* st [[buffer(2)]],
                          device int* out_tokens [[buffer(3)]],
                          device const wtype* E [[buffer(4)]],
                          device const half2* ESB [[buffer(5)]],
                          device float* x [[buffer(6)]],
                          device int* drafts [[buffer(7)]],
                          device const NGEntry* tab [[buffer(8)]],
                          device const int* prompt_tail [[buffer(9)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float rv[32];
    threadgroup int ri[32];
    threadgroup int g[8];
    threadgroup int next_rows[8];
    threadgroup int nrows_s;
    const int nv = st->nver;
    for (int t = 0; t < nv; t++) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < LMH_TGS; i += 1024) {
            const float v = pval[t * LMH_TGS + i];
            const int id = pidx[t * LMH_TGS + i];
            if (v > best || (v == best && id < bi)) { best = v; bi = id; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(best, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { rv[sg] = best; ri[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            best = rv[lane]; bi = ri[lane];
            for (ushort off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_down(best, off);
                const int oi = simd_shuffle_down(bi, off);
                if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
            }
            if (lane == 0) g[t] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        st->pad0 += 1;                       // passes committed (diagnostics)
        int m = 0;
        while (m < nv - 1 && drafts[m] == g[m]) m++;
        if (m > 0) st->pad1 += m;            // accepted drafts (diagnostics)
        bool stop = false;
        int last = g[0];
        for (int i = 0; i <= m && !stop; i++) {
            const int tok = g[i];
            out_tokens[st->step] = tok;
            st->step += 1;
            last = tok;
            if (tok == st->eos0 || tok == st->eos1 || st->step >= st->maxsteps) stop = true;
        }
        // the pass covered positions pos .. pos+nver-1; rows 0..m are now real context
        st->pos += (nv == 1 ? st->n : m + 1);
        if (stop) {
            st->done = 1;
            nrows_s = 0;
        } else {
            // history for drafting: last committed tokens (generated, then prompt tail before them)
            int hist[NG_MAXN + 8];
            const int have = st->step;
            int len = 0;
            for (int i = NG_MAXN; i >= 1; i--) {
                const int j = have - i;
                hist[len++] = j >= 0 ? out_tokens[j] : -1;   // -1 marks the start of the output
            }
            const int T = st->tspec;
            int nd = 0;
            next_rows[0] = last;
            for (; nd < T - 1; nd++) {
                const int d = ng_next(tab, hist, len, 1);
                if (d < 0) break;
                drafts[nd] = d;
                next_rows[nd + 1] = d;
                hist[len++] = d;
            }
            for (int i = nd; i < T - 1; i++) { drafts[i] = -2; next_rows[i + 1] = 0; }
            st->nver = T;
            st->n = T;
            nrows_s = T;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nr = nrows_s;
    for (int r = 0; r < nr; r++) {
        const int tok = next_rows[r];
        for (uint i = tid; i < H; i += 1024) x[r * H + i] = embed_at(E, ESB, tok, i);
    }
}

#if FMT == 0
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

#endif

// Causal attention for one (query head, token). Grid (14, n), 256 threads.
#define MAXCTX_TG 2048
kernel void k_attention(device const float* q [[buffer(0)]],
                        device const half* kc [[buffer(1)]],
                        device const half* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        constant int& toff [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int h = tgp.x;
    const int t = tgp.y + toff;
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

#if FMT == 0
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

#endif


#if FMT == 2
// ---------------------------------------------------------------------------------------------
// q4 (group-64 affine, row-major) decode kernels, templated on T = tokens per pass (T = 1 is plain
// decode, T > 1 verifies speculative drafts; the per-token arithmetic is identical, so speculation
// reproduces plain greedy decoding bit for bit). Every simdgroup normalizes its input chunk itself
// (one simd_sum) and computes 8 rows; for K = 896 lane l < 28 owns columns [32l, 32l + 32). Inputs
// are held as fp16 in registers (with fp32 inputs, T = 2 costs 1.5x a step; with fp16, 1.1x). A
// nibble n unpacks to fp16 (1 + n/16) with bit operations; sum(n x) = 16 (sum((1 + n/16) x) - X).
// After the butterfly reduction row r of the simdgroup lives in lanes 4r..4r+3.

static inline float reduce8(thread float* v, ushort lane) {
    float t4[4], t2[2];
    bool b = (lane & 16) != 0;
    for (int i = 0; i < 4; i++) { const float keep = b ? v[4 + i] : v[i], send = b ? v[i] : v[4 + i]; t4[i] = keep + simd_shuffle_xor(send, 16); }
    b = (lane & 8) != 0;
    for (int i = 0; i < 2; i++) { const float keep = b ? t4[2 + i] : t4[i], send = b ? t4[i] : t4[2 + i]; t2[i] = keep + simd_shuffle_xor(send, 8); }
    b = (lane & 4) != 0;
    const float keep = b ? t2[1] : t2[0], send = b ? t2[0] : t2[1];
    float t1 = keep + simd_shuffle_xor(send, 4);
    t1 += simd_shuffle_xor(t1, 2);
    return t1 + simd_shuffle_xor(t1, 1);
}

// 4 rows: row r ends up in lanes 8r..8r+7.
static inline float reduce4(thread float* v, ushort lane) {
    float t2[2];
    bool b = (lane & 16) != 0;
    for (int i = 0; i < 2; i++) { const float keep = b ? v[2 + i] : v[i], send = b ? v[i] : v[2 + i]; t2[i] = keep + simd_shuffle_xor(send, 16); }
    b = (lane & 8) != 0;
    const float keep = b ? t2[1] : t2[0], send = b ? t2[0] : t2[1];
    float t1 = keep + simd_shuffle_xor(send, 8);
    t1 += simd_shuffle_xor(t1, 4);
    t1 += simd_shuffle_xor(t1, 2);
    return t1 + simd_shuffle_xor(t1, 1);
}

template<int R>
static inline float reduceR(thread float* v, ushort lane) { return R == 8 ? reduce8(v, lane) : reduce4(v, lane); }

// 32 nibbles unpacked once, T dot products with the fp16 inputs (in-order fp32 accumulation).
template<int T>
static inline void q4_dot32(uint4 w, thread const uint (&xh)[T][16], thread float* d) {
    for (int t = 0; t < T; t++) d[t] = 0.f;
    for (int u = 0; u < 4; ++u) {
        const uint v = w[u];
        const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
        const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
        const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
        const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
        const float f[8] = {float(p04.x), float(p15.x), float(p26.x), float(p37.x), float(p04.y), float(p15.y), float(p26.y), float(p37.y)};
        for (int t = 0; t < T; t++) {
            float a = d[t];
            for (int k = 0; k < 4; k++) { const float2 xx = float2(as_type<half2>(xh[t][4 * u + k])); a = fma(f[2 * k], xx.x, a); a = fma(f[2 * k + 1], xx.y, a); }
            d[t] = a;
        }
    }
}

// Lane's 32-column chunk of RMSNorm(x) * w as fp16; X = sum of the rounded values. Lanes >= 28: zeros.
static inline void q4_chunk_norm(device const float* x, device const float* w, ushort lane, thread uint* xh, thread float& X) {
    float xv[32];
    float ss = 0.f;
    if (lane < 28) {
        device const float4* xs = (device const float4*)(x + lane * 32);
        for (int i = 0; i < 8; i++) { const float4 t = xs[i]; xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w; ss += dot(t, t); }
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(H) + EPS);
    X = 0.f;
    if (lane < 28) {
        device const float4* ws = (device const float4*)(w + lane * 32);
        for (int i = 0; i < 8; i++) {
            const float4 t = float4(xv[4*i], xv[4*i+1], xv[4*i+2], xv[4*i+3]) * scale * ws[i];
            const half4 h = half4(t);
            xh[2*i] = as_type<uint>(h.xy); xh[2*i+1] = as_type<uint>(h.zw);
            X += float(h.x) + float(h.y) + float(h.z) + float(h.w);
        }
    } else {
        for (int i = 0; i < 16; i++) xh[i] = 0u;
    }
}

static inline void q4_chunk_plain(device const float* x, ushort lane, thread uint* xh, thread float& X) {
    X = 0.f;
    if (lane < 28) {
        device const float4* xs = (device const float4*)(x + lane * 32);
        for (int i = 0; i < 8; i++) {
            const half4 h = half4(xs[i]);
            xh[2*i] = as_type<uint>(h.xy); xh[2*i+1] = as_type<uint>(h.zw);
            X += float(h.x) + float(h.y) + float(h.z) + float(h.w);
        }
    } else {
        for (int i = 0; i < 16; i++) xh[i] = 0u;
    }
}

// R rows (from r0) of a K = 896 matrix for T tokens: lane-local partials v[t][r] before the reduction.
template<int T, int R>
static inline void q4_rows(device const uchar* W, device const half2* SB, uint r0, thread const uint (&xh)[T][16],
                           thread const float* X, ushort lane, thread float (&v)[T][R]) {
    if (lane >= 28) { for (int t = 0; t < T; t++) for (int r = 0; r < R; r++) v[t][r] = 0.f; return; }
    for (int rb = 0; rb < R; rb += 4) {
        uint4 w[4];
        float2 sb[4];
        for (int b = 0; b < 4; b++) {
            w[b] = *((device const uint4*)(W + (ulong)(r0 + rb + b) * (H / 2) + lane * 16));
            sb[b] = float2(SB[(r0 + rb + b) * (H / 64) + (lane >> 1)]);
        }
        for (int b = 0; b < 4; b++) {
            float d[T];
            q4_dot32<T>(w[b], xh, d);
            for (int t = 0; t < T; t++) v[t][rb + b] = sb[b].x * (16.f * (d[t] - X[t])) + sb[b].y * X[t];
        }
    }
}

// RMSNorm + QKV + bias + RoPE + KV write (32-row rotary blocks: rows i, i+16 = dims d, d+32).
// Grid (36), 128 threads: 4 simdgroups x 8 rows. Rows t = 0..T-1 of x at positions pos+t.
template<int T, int R>
kernel void k_attn_in_T(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                        device const float* bias [[buffer(2)]], device const float* wnorm [[buffer(3)]],
                        device const float* x [[buffer(4)]], device float* q [[buffer(5)]],
                        device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                        device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]],
                        uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    constexpr int LPR = 32 / R;
    threadgroup float outv[T][32];
    uint xh[T][16];
    float X[T], v[T][R];
    for (int t = 0; t < T; t++) q4_chunk_norm(x + t * H, wnorm, lane, xh[t], X[t]);
    const uint r0 = tgp.x * 32 + sg * R;
    q4_rows<T, R>(W, SB, r0, xh, X, lane, v);
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(v[t], lane);
        if (lane % LPR == 0) outv[t][sg * R + lane / LPR] = y + bias[r0 + lane / LPR];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16 * T) {
        const int t = tid / 16, i = tid % 16;
        const int pos = st->pos + t;
        const int head = tgp.x >> 1;
        const int d = (tgp.x & 1) * 16 + i;
        float a = outv[t][i], b = outv[t][i + 16];
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
            device half* vv = vc + ((ulong)(head - NH - NKV) * st->maxctx + pos) * HD;
            vv[d] = half(a); vv[d + 32] = half(b);
        }
    }
}

// Output projection + residual. Grid (56), 64 threads: 2 simdgroups x 8 rows.
template<int T, int R>
kernel void k_o_proj_T(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                       device const float* a [[buffer(2)]], device float* x [[buffer(3)]],
                       device const State* st [[buffer(4)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    constexpr int LPR = 32 / R;
    uint xh[T][16];
    float X[T], v[T][R];
    for (int t = 0; t < T; t++) q4_chunk_plain(a + t * H, lane, xh[t], X[t]);
    const uint r0 = (tgp.x * 2 + sg) * R;
    q4_rows<T, R>(W, SB, r0, xh, X, lane, v);
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(v[t], lane);
        if (lane % LPR == 0) x[t * H + r0 + lane / LPR] += y;
    }
}

// RMSNorm + gate/up + SiLU*up; stored rows in groups of 4 = gate[2q], gate[2q+1], up[2q], up[2q+1].
// Grid (608), 64 threads: 2 simdgroups x 8 rows (= 4 hidden units per simdgroup).
template<int T, int R>
kernel void k_mlp_in_T(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                       device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                       device float* hbuf [[buffer(4)]], device const State* st [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    constexpr int LPR = 32 / R;
    uint xh[T][16];
    float X[T], v[T][R];
    for (int t = 0; t < T; t++) q4_chunk_norm(x + t * H, wnorm, lane, xh[t], X[t]);
    const uint r0 = (tgp.x * 2 + sg) * R;
    q4_rows<T, R>(W, SB, r0, xh, X, lane, v);
    const int row = lane / LPR;
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(v[t], lane);
        const float u = simd_shuffle_down(y, 2 * LPR);   // row + 2 = the matching up row
        if (lane % LPR == 0 && (row & 2) == 0) {
            const int unit = 2 * (int(r0 / 4) + (row >> 2)) + (row & 1);
            hbuf[t * FF + unit] = y / (1.f + exp(-y)) * u;
        }
    }
}

// Down projection + residual (K = 4864): lanes stride 152 chunks of 32 columns; input chunks are read
// from device memory (L2) per step and rounded to fp16. Grid (56), 64 threads: 2 simdgroups x 8 rows.
template<int T, int R>
kernel void k_mlp_out_T(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                        device const float* hbuf [[buffer(2)]], device float* x [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    constexpr int LPR = 32 / R;
    const uint r0 = (tgp.x * 2 + sg) * R;
    float acc[T][R];
    for (int t = 0; t < T; t++) for (int r = 0; r < R; r++) acc[t][r] = 0.f;
    for (int c = lane; c < FF / 32; c += 32) {
        uint xh[T][16];
        float X[T];
        for (int t = 0; t < T; t++) {
            X[t] = 0.f;
            device const float4* xs = (device const float4*)(hbuf + t * FF + c * 32);
            for (int i = 0; i < 8; i++) {
                const half4 h = half4(xs[i]);
                xh[t][2*i] = as_type<uint>(h.xy); xh[t][2*i+1] = as_type<uint>(h.zw);
                X[t] += float(h.x) + float(h.y) + float(h.z) + float(h.w);
            }
        }
        for (int rb = 0; rb < R; rb += 4) {
            uint4 w[4];
            float2 sb[4];
            for (int b = 0; b < 4; b++) {
                w[b] = *((device const uint4*)(W + (ulong)(r0 + rb + b) * (FF / 2) + c * 16));
                sb[b] = float2(SB[(r0 + rb + b) * (FF / 64) + (c >> 1)]);
            }
            for (int b = 0; b < 4; b++) {
                float d[T];
                q4_dot32<T>(w[b], xh, d);
                for (int t = 0; t < T; t++) acc[t][rb + b] += sb[b].x * (16.f * (d[t] - X[t])) + sb[b].y * X[t];
            }
        }
    }
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(acc[t], lane);
        if (lane % LPR == 0) x[t * H + r0 + lane / LPR] += y;
    }
}

// Final RMSNorm + tied lm_head + per-threadgroup argmax for rows (st->n - T) .. (st->n - 1) of x;
// partials laid out [t][LMH_TGS]. Grid (4748), 128 threads: 4 simdgroups x 8 vocabulary rows.
template<int T, int R>
kernel void k_lm_head_T(device const uchar* E [[buffer(0)]], device const half2* ESB [[buffer(1)]],
                        device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                        device float* pval [[buffer(4)]], device int* pidx [[buffer(5)]],
                        device const State* st [[buffer(6)]], device float* logits [[buffer(7)]],
                        constant int& write_logits [[buffer(8)]],
                        uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    constexpr int LPR = 32 / R, NSG = 32 / R;
    threadgroup float bv[T][NSG];
    threadgroup int bix[T][NSG];
    uint xh[T][16];
    float X[T], v[T][R];
    device const float* xr = x + (st->n - T) * H;
    for (int t = 0; t < T; t++) q4_chunk_norm(xr + t * H, wnorm, lane, xh[t], X[t]);
    const uint r0 = tg * 32 + sg * R;
    q4_rows<T, R>(E, ESB, r0, xh, X, lane, v);
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(v[t], lane);
        const int row = int(r0) + lane / LPR;
        if (T == 1 && write_logits && lane % LPR == 0) logits[row] = y;
        const float m = simd_max(y);
        const int idx = simd_min(y == m ? row : 0x7fffffff);
        if (lane == 0) { bv[t][sg] = m; bix[t][sg] = idx; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < T) {
        const int t = lane;
        float best = bv[t][0];
        int bi = bix[t][0];
        for (int s2 = 1; s2 < NSG; ++s2) if (bv[t][s2] > best) { best = bv[t][s2]; bi = bix[t][s2]; }
        pval[t * LMH_TGS + tg] = best;
        pidx[t * LMH_TGS + tg] = bi;
    }
}

#define INST_T(T, R, SUF) \
template [[host_name("k_attn_in" SUF)]] kernel void k_attn_in_T<T, R>(device const uchar*, device const half2*, device const float*, device const float*, device const float*, device float*, device half*, device half*, device const float2*, device const State*, uint2, uint, ushort, ushort); \
template [[host_name("k_o_proj" SUF)]] kernel void k_o_proj_T<T, R>(device const uchar*, device const half2*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_mlp_in" SUF)]] kernel void k_mlp_in_T<T, R>(device const uchar*, device const half2*, device const float*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_mlp_out" SUF)]] kernel void k_mlp_out_T<T, R>(device const uchar*, device const half2*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_lm_head" SUF)]] kernel void k_lm_head_T<T, R>(device const uchar*, device const half2*, device const float*, device const float*, device float*, device int*, device const State*, device float*, constant int&, uint, ushort, ushort);
INST_T(1, 8, "")
INST_T(2, 8, "_2")
INST_T(3, 8, "_3")
INST_T(4, 8, "_4")

// ---------------------------------------------------------------------------------------------
// Two-stage lm_head. Stage 1 scores every vocabulary row with a 2-bit sketch of its q4 codes
// (the high bits: w ~ s (4 hi + 1.5) + b) and a per-row uncertainty sigma^2 = 1.25 sum_g s_g^2 |x_g|^2
// (the dropped low bits are ~uniform in 0..3); each 32-row group records the max of A - K sigma and
// A + K sigma. A tiny kernel reduces the global threshold tau = max(A - K sigma); stage 2 recomputes
// exact q4 logits only in groups whose upper bound reaches tau (a handful), so the argmax equals the
// full lm_head's unless a residual exceeds K = 10 sigma.

template<int T>
kernel void k_lmh_sketch_T(device const uchar* HI [[buffer(0)]], device const half2* ESB [[buffer(1)]],
                           device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                           device float* bounds [[buffer(4)]], device const State* st [[buffer(5)]],
                           constant float& LMH_K [[buffer(6)]],
                           uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float tlb[T][4], tub[T][4];
    // inputs: lane's 32 columns, as 2 uints x 4 masks of half4 = (x[16u+m], x[16u+4+m], x[16u+8+m], x[16u+12+m]) / 4^m
    half4 xm[T][2][4];
    float X[T], XX[T];
    device const float* xr = x + (st->n - T) * H;
    for (int t = 0; t < T; t++) {
        uint xh[16];
        q4_chunk_norm(xr + t * H, wnorm, lane, xh, X[t]);
        float xf[32];
        for (int i = 0; i < 16; i++) { const float2 f2 = float2(as_type<half2>(xh[i])); xf[2 * i] = f2.x; xf[2 * i + 1] = f2.y; }
        float xx = 0.f;
        for (int i = 0; i < 32; i++) xx += xf[i] * xf[i];
        XX[t] = xx;
        for (int u = 0; u < 2; u++)
            for (int m = 0; m < 4; m++) {
                const float sc = 1.f / float(1 << (2 * m));
                xm[t][u][m] = half4(float4(xf[16 * u + m], xf[16 * u + 4 + m], xf[16 * u + 8 + m], xf[16 * u + 12 + m]) * sc);
            }
    }
    const uint r0 = tg * 32 + sg * 8;
    float va[T][8], vv[T][8];
    if (lane < 28) {
        uint2 hv[8];
        float2 sb[8];
        for (int r = 0; r < 8; r++) {
            hv[r] = *((device const uint2*)(HI + (ulong)(r0 + r) * (H / 4) + lane * 8));
            sb[r] = float2(ESB[(r0 + r) * (H / 64) + (lane >> 1)]);
        }
        for (int r = 0; r < 8; r++) {
            for (int t = 0; t < T; t++) {
                half acc = 0.h;
                for (int u = 0; u < 2; u++) {
                    const uint v = hv[r][u];
                    acc += dot(half4(as_type<uchar4>(v & 0x03030303u)), xm[t][u][0]);
                    acc += dot(half4(as_type<uchar4>(v & 0x0C0C0C0Cu)), xm[t][u][1]);
                    acc += dot(half4(as_type<uchar4>(v & 0x30303030u)), xm[t][u][2]);
                    acc += dot(half4(as_type<uchar4>(v & 0xC0C0C0C0u)), xm[t][u][3]);
                }
                va[t][r] = sb[r].x * (4.f * float(acc) + 1.5f * X[t]) + sb[r].y * X[t];
                vv[t][r] = sb[r].x * sb[r].x * XX[t];
            }
        }
    } else {
        for (int t = 0; t < T; t++) for (int r = 0; r < 8; r++) { va[t][r] = 0.f; vv[t][r] = 0.f; }
    }
    for (int t = 0; t < T; t++) {
        const float a = reduce8(va[t], lane);
        const float sigma = sqrt(1.25f * reduce8(vv[t], lane));
        const float lb = simd_max(a - LMH_K * sigma), ub = simd_max(a + LMH_K * sigma);
        if (lane == 0) { tlb[t][sg] = lb; tub[t][sg] = ub; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < T) {
        const int t = lane;
        bounds[(2 * t) * LMH_TGS + tg] = max(max(tlb[t][0], tlb[t][1]), max(tlb[t][2], tlb[t][3]));
        bounds[(2 * t + 1) * LMH_TGS + tg] = max(max(tub[t][0], tub[t][1]), max(tub[t][2], tub[t][3]));
    }
}

// tau[t] = max over groups of the lower bound. One threadgroup of 1024 threads.
kernel void k_lmh_tau(device const float* bounds [[buffer(0)]], device float* tau [[buffer(1)]],
                      device const State* st [[buffer(2)]], constant int& T [[buffer(3)]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float red[32];
    for (int t = 0; t < T; t++) {
        float m = -INFINITY;
        for (int i = tid; i < LMH_TGS; i += 1024) m = max(m, bounds[(2 * t) * LMH_TGS + i]);
        m = simd_max(m);
        if (lane == 0) red[sg] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            m = simd_max(red[lane]);
            if (lane == 0) tau[t] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Exact q4 logits for the candidate groups only; other groups report -inf.
template<int T>
kernel void k_lmh_exact_T(device const uchar* E [[buffer(0)]], device const half2* ESB [[buffer(1)]],
                          device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                          device float* pval [[buffer(4)]], device int* pidx [[buffer(5)]],
                          device const State* st [[buffer(6)]], device const float* bounds [[buffer(7)]],
                          device const float* tau [[buffer(8)]],
                          uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    bool any = false;
    bool cand[T];
    for (int t = 0; t < T; t++) { cand[t] = bounds[(2 * t + 1) * LMH_TGS + tg] >= tau[t]; any = any || cand[t]; }
    if (!any) {
        if (sg == 0 && lane < T) { pval[lane * LMH_TGS + tg] = -INFINITY; pidx[lane * LMH_TGS + tg] = 0x7fffffff; }
        return;
    }
    threadgroup float bv[T][4];
    threadgroup int bix[T][4];
    uint xh[T][16];
    float X[T], v[T][8];
    device const float* xr = x + (st->n - T) * H;
    for (int t = 0; t < T; t++) q4_chunk_norm(xr + t * H, wnorm, lane, xh[t], X[t]);
    const uint r0 = tg * 32 + sg * 8;
    q4_rows<T, 8>(E, ESB, r0, xh, X, lane, v);
    for (int t = 0; t < T; t++) {
        const float y = reduce8(v[t], lane);
        const int row = int(r0) + (lane >> 2);
        const float m = simd_max(y);
        const int idx = simd_min(y == m ? row : 0x7fffffff);
        if (lane == 0) { bv[t][sg] = m; bix[t][sg] = idx; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < T) {
        const int t = lane;
        float best = bv[t][0];
        int bi = bix[t][0];
        for (int s2 = 1; s2 < 4; ++s2) if (bv[t][s2] > best) { best = bv[t][s2]; bi = bix[t][s2]; }
        pval[t * LMH_TGS + tg] = cand[t] ? best : -INFINITY;
        pidx[t * LMH_TGS + tg] = cand[t] ? bi : 0x7fffffff;
    }
}

#define INST_LMH(T) \
template [[host_name("k_lmh_sketch_" #T)]] kernel void k_lmh_sketch_T<T>(device const uchar*, device const half2*, device const float*, device const float*, device float*, device const State*, constant float&, uint, ushort, ushort); \
template [[host_name("k_lmh_exact_" #T)]] kernel void k_lmh_exact_T<T>(device const uchar*, device const half2*, device const float*, device const float*, device float*, device int*, device const State*, device const float*, device const float*, uint, ushort, ushort);
INST_LMH(1)
INST_LMH(2)
INST_LMH(3)
INST_LMH(4)



#endif

#if FMT == 3
// ---------------------------------------------------------------------------------------------
// q4a decode kernels (lane-sliced K): a simdgroup owns a 16-row block; lane l holds input columns
// [28l, 28l+28) in registers (pre-scaled by 16 for the fp16 nibble unpack); after the butterfly
// reduction row r of the block lives in lanes 2r and 2r+1.

// Transposed reduction: each lane holds 16 row partials; returns the full sum of row (lane >> 1) & 15.
static inline float reduce16(thread float* v, ushort lane) {
    float t8[8], t4[4], t2[2];
    bool b = (lane & 16) != 0;
    for (int i = 0; i < 8; i++) { const float keep = b ? v[8 + i] : v[i], send = b ? v[i] : v[8 + i]; t8[i] = keep + simd_shuffle_xor(send, 16); }
    b = (lane & 8) != 0;
    for (int i = 0; i < 4; i++) { const float keep = b ? t8[4 + i] : t8[i], send = b ? t8[i] : t8[4 + i]; t4[i] = keep + simd_shuffle_xor(send, 8); }
    b = (lane & 4) != 0;
    for (int i = 0; i < 2; i++) { const float keep = b ? t4[2 + i] : t4[i], send = b ? t4[i] : t4[2 + i]; t2[i] = keep + simd_shuffle_xor(send, 4); }
    b = (lane & 2) != 0;
    const float keep = b ? t2[1] : t2[0], send = b ? t2[0] : t2[1];
    const float t1 = keep + simd_shuffle_xor(send, 2);
    return t1 + simd_shuffle_xor(t1, 1);
}

static inline void e5m3x4(uint w, thread float* s) {
    const half2 a = as_type<half2>((w << 7) & 0x7F807F80u);
    const half2 b = as_type<half2>((w >> 1) & 0x7F807F80u);
    s[0] = a.x; s[1] = b.x; s[2] = a.y; s[3] = b.y;
}

// 16 rows x 2 columns of nibbles -> fp16 (1 + n/16) without int->float conversions.
static inline void q4a_chunk(uint4 w, float x0, float x1, thread float4* acc) {
    for (int j = 0; j < 4; j++) {
        const uint ww = w[j];
        const half2 a = as_type<half2>(((ww << 6) & 0x03C003C0u) | 0x3C003C00u);
        const half2 c = as_type<half2>(((ww << 2) & 0x03C003C0u) | 0x3C003C00u);
        const half2 b = as_type<half2>(((ww >> 2) & 0x03C003C0u) | 0x3C003C00u);
        const half2 d = as_type<half2>(((ww >> 6) & 0x03C003C0u) | 0x3C003C00u);
        acc[j].x = fma(float(a.x), x0, acc[j].x); acc[j].x = fma(float(c.x), x1, acc[j].x);
        acc[j].y = fma(float(b.x), x0, acc[j].y); acc[j].y = fma(float(d.x), x1, acc[j].y);
        acc[j].z = fma(float(a.y), x0, acc[j].z); acc[j].z = fma(float(c.y), x1, acc[j].z);
        acc[j].w = fma(float(b.y), x0, acc[j].w); acc[j].w = fma(float(d.y), x1, acc[j].w);
    }
}

// Per-row partials (lane-local, before the reduction) of one block. NN nibble chunks, loads in batches of BT.
template<int NN, int BT>
static inline void q4a_dot(device const uchar* pb, thread const float* xv, float X, ushort lane, thread float* v) {
    device const uint4* p = (device const uint4*)pb + lane;
    float4 acc[4] = {float4(0.f), float4(0.f), float4(0.f), float4(0.f)};
    for (int c0 = 0; c0 < NN; c0 += BT) {
        uint4 w[BT];
        for (int c = 0; c < BT; c++) if (c0 + c < NN) w[c] = p[(c0 + c) * 32];
        for (int c = 0; c < BT; c++) if (c0 + c < NN) q4a_chunk(w[c], xv[2 * (c0 + c)], xv[2 * (c0 + c) + 1], acc);
    }
    const uint4 sc = p[NN * 32];
    const uint2 zl = ((device const uint2*)(pb + (NN + 1) * 512))[lane];
    const uint zh = ((device const uint*)(pb + (NN + 1) * 512 + 256))[lane];
    float sv[16];
    for (int j = 0; j < 4; j++) e5m3x4(sc[j], sv + 4 * j);
    for (int r = 0; r < 16; r++) {
        const uint lo = ((r < 8 ? zl.x : zl.y) >> (4 * (r & 7))) & 0xFu;
        const uint hi = (zh >> (2 * r)) & 0x3u;
        const float z = float(lo | (hi << 4)) * 0.25f + 16.0f;
        v[r] = sv[r] * fma(-z, X, acc[r >> 2][r & 3]);
    }
}

// Lane's 28-column slice of RMSNorm(x) * w (the whole 896-vector norm via one simd_sum), x16.
static inline void slice_norm(device const float* x, device const float* w, ushort lane, thread float* xv, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 28);
    float ss = 0.f;
    for (int i = 0; i < 7; i++) { const float4 t = xs[i]; xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w; ss += dot(t, t); }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(H) + EPS);
    device const float4* ws = (device const float4*)(w + lane * 28);
    float sum = 0.f;
    for (int i = 0; i < 7; i++) {
        const float4 t = float4(xv[4*i], xv[4*i+1], xv[4*i+2], xv[4*i+3]) * scale * ws[i];
        sum += t.x + t.y + t.z + t.w;
        xv[4*i] = 16.f * t.x; xv[4*i+1] = 16.f * t.y; xv[4*i+2] = 16.f * t.z; xv[4*i+3] = 16.f * t.w;
    }
    X = sum;
}

static inline void slice_plain(device const float* x, ushort lane, thread float* xv, thread float& X) {
    device const float4* xs = (device const float4*)(x + lane * 28);
    float sum = 0.f;
    for (int i = 0; i < 7; i++) {
        const float4 t = xs[i];
        sum += t.x + t.y + t.z + t.w;
        xv[4*i] = 16.f * t.x; xv[4*i+1] = 16.f * t.y; xv[4*i+2] = 16.f * t.z; xv[4*i+3] = 16.f * t.w;
    }
    X = sum;
}

// RMSNorm + QKV + bias + RoPE + KV write. Grid (36, n), 64 threads (2 blocks per threadgroup).
kernel void k_attn_in(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                      device const float* bias [[buffer(2)]], device const float* wnorm [[buffer(3)]],
                      device const float* x [[buffer(4)]], device float* q [[buffer(5)]],
                      device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                      device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    const int pos = st->pos + t;
    const uint b = tgp.x * 2 + sg;
    float xv[28], X, v[16];
    slice_norm(x + t * H, wnorm, lane, xv, X);
    q4a_dot<14, 4>(W + (ulong)b * Q4A_B896, xv, X, lane, v);
    const int rr = (lane >> 1) & 15;
    const float y = reduce16(v, lane) + bias[b * 16 + rr];
    if (b < 64) {
        const float partner = simd_shuffle_down(y, 16);
        if ((lane & 1) == 0 && lane < 16) {
            const int head = b >> 2, d = 8 * (b & 3) + rr;
            const float2 cs = rope[pos * 32 + d];
            const float a2 = y * cs.x - partner * cs.y;
            const float b2 = partner * cs.x + y * cs.y;
            if (head < NH) {
                q[t * H + head * HD + d] = a2;
                q[t * H + head * HD + d + 32] = b2;
            } else {
                device half* k = kc + ((ulong)(head - NH) * st->maxctx + pos) * HD;
                k[d] = half(a2); k[d + 32] = half(b2);
            }
        }
    } else if ((lane & 1) == 0) {
        const int vb = b - 64;
        device half* vv = vc + ((ulong)(vb >> 2) * st->maxctx + pos) * HD;
        vv[16 * (vb & 3) + rr] = half(y);
    }
}

// Output projection + residual. Grid (28, n), 64 threads.
kernel void k_o_proj(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                     device const float* a [[buffer(2)]], device float* x [[buffer(3)]],
                     device const State* st [[buffer(4)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    const uint b = tgp.x * 2 + sg;
    float xv[28], X, v[16];
    slice_plain(a + t * H, lane, xv, X);
    q4a_dot<14, 4>(W + (ulong)b * Q4A_B896, xv, X, lane, v);
    const float y = reduce16(v, lane);
    if ((lane & 1) == 0) x[t * H + b * 16 + (lane >> 1)] += y;
}

// RMSNorm + gate/up + SiLU(gate)*up. Block b = gate rows 8b..8b+7, then up rows. Grid (304, n), 64 threads.
kernel void k_mlp_in(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                     device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                     device float* hbuf [[buffer(4)]], device const State* st [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    const uint b = tgp.x * 2 + sg;
    float xv[28], X, v[16];
    slice_norm(x + t * H, wnorm, lane, xv, X);
    q4a_dot<14, 4>(W + (ulong)b * Q4A_B896, xv, X, lane, v);
    const float y = reduce16(v, lane);
    const float u = simd_shuffle_down(y, 16);
    if ((lane & 1) == 0 && lane < 16) hbuf[t * FF + 8 * b + (lane >> 1)] = y / (1.f + exp(-y)) * u;
}

// Down projection + residual: 4 simdgroups split K into quarters of 1216 (lane slices of 38).
// Grid (56, n), 128 threads.
kernel void k_mlp_out(device const uchar* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
                      device const float* hbuf [[buffer(2)]], device float* x [[buffer(3)]],
                      device const State* st [[buffer(4)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int t = tgp.y;
    if (t >= st->n) return;
    threadgroup float red[4][16];
    const uint b = tgp.x;
    device const float2* hs = (device const float2*)(hbuf + t * FF + sg * 1216 + lane * 38);
    float xv[38], X = 0.f, v[16];
    for (int i = 0; i < 19; i++) { const float2 h2 = hs[i]; X += h2.x + h2.y; xv[2 * i] = 16.f * h2.x; xv[2 * i + 1] = 16.f * h2.y; }
    q4a_dot<19, 3>(W + ((ulong)b * 4 + sg) * Q4A_B4864, xv, X, lane, v);
    const float y = reduce16(v, lane);
    if ((lane & 1) == 0) red[sg][lane >> 1] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 16) x[t * H + b * 16 + lane] += red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
}

// Final RMSNorm + tied lm_head on the last token of the pass + per-threadgroup argmax.
// Grid (4748), 64 threads: 2 blocks of 16 vocabulary rows per threadgroup.
kernel void k_lm_head(device const uchar* E [[buffer(0)]], device const half2* ESB [[buffer(1)]],
                      device const float* wnorm [[buffer(2)]], device const float* x [[buffer(3)]],
                      device float* pval [[buffer(4)]], device int* pidx [[buffer(5)]],
                      device const State* st [[buffer(6)]], device float* logits [[buffer(7)]],
                      constant int& write_logits [[buffer(8)]],
                      uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float bv[2];
    threadgroup int bix[2];
    const uint b = tg * 2 + sg;
    float xv[28], X, v[16];
    slice_norm(x + (st->n - 1) * H, wnorm, lane, xv, X);
    q4a_dot<14, 4>(E + (ulong)b * Q4A_B896, xv, X, lane, v);
    const float y = reduce16(v, lane);
    const int row = b * 16 + ((lane >> 1) & 15);
    if (write_logits && (lane & 1) == 0) logits[row] = y;
    const float m = simd_max(y);
    const int idx = simd_min(y == m ? row : 0x7fffffff);
    if (lane == 0) { bv[sg] = m; bix[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        const bool second = bv[1] > bv[0];
        pval[tg] = second ? bv[1] : bv[0];
        pidx[tg] = second ? bix[1] : bix[0];
    }
}
#endif

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

#if FMT != 3
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

#endif
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

#if FMT != 3
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
#endif

// ---------------------------------------------------------------------------------------------
// Prefill GEMM v3: Neural Accelerators (MPP matmul2d). Each threadgroup computes 32 rows x TT
// tokens; its NS simdgroups split K, each dequantizing its own 32 x 64 weight tile into private
// threadgroup memory per step and running matmul2d with activations read from device memory.
// Partial tiles are reduced through threadgroup memory (deterministic), then the epilogue runs.
#if FMT == 2
template<int K, int MODE, int TT, int NS>
inline void gemm3_body(device half* X, device const uchar* W, device const half2* SB,
                       device const float* bias, device float* outf, device half* outh,
                       device half* kc, device half* vc, device const float2* rope, device const State* st,
                       threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    // Each simdgroup takes a contiguous K range (whole cache lines of its 32 rows): K = 896 -> 7
    // steps of 32 columns, K = 4864 -> 19 steps of 64. Lane = row; w = (16 s)(1 + n/16) + (b - 16 s).
    constexpr int BK = K == H ? 32 : 64;
    constexpr int NSTEP = K / BK / NS;
    const int n = st->n;
    const uint r0 = tgp.x * 32;
    const int t0 = tgp.y * TT;
    threadgroup half* ws = Wsg + sg * 32 * BK;
    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    auto tW = tensor(ws, dextents<int, 2>(BK, 32), array<int, 2>({1, BK}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;
    device const uchar* wrow = W + (ulong)(r0 + lane) * (K / 2);
    device const half2* sbrow = SB + (r0 + lane) * (K / 64);
    threadgroup half2* dst = (threadgroup half2*)(ws + lane * BK);
    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = (sg * NSTEP + i) * BK;
        const half2 sb = sbrow[k0 / 64];
        const half2 mul = half2(half(16.f * float(sb.x)));
        const half2 add = half2(half(float(sb.y) - 16.f * float(sb.x)));
        device const uint4* src = (device const uint4*)(wrow + k0 / 2);
        for (int q = 0; q < BK / 32; ++q) {
            const uint4 wv = src[q];
            for (int u = 0; u < 4; ++u) {
                const uint v = wv[u];   // nibbles k = 0..7 of this word: pairs (0,4) (1,5) (2,6) (3,7)
                const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
                const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
                const int o = q * 16 + u * 4;   // half2 index of weights 8(4q+u) .. +7
                dst[o + 0] = fma(half2(p04.x, p15.x), mul, add);
                dst[o + 1] = fma(half2(p26.x, p37.x), mul, add);
                dst[o + 2] = fma(half2(p04.y, p15.y), mul, add);
                dst[o + 3] = fma(half2(p26.y, p37.y), mul, add);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tX.template slice<BK, TT>(k0, 0);
        op.run(a, tW, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // Os aliases the weight tiles
    auto tO = tensor(Os + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cT.store(tO);
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

#define GEMM3(NAME, K, MODE, TT, NS) \
kernel void NAME(device half* X [[buffer(0)]], device const uchar* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TT) >= st->n) return; \
    constexpr int WB = NS * 32 * 64 * 2, OB = NS * TT * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm3_body<K, MODE, TT, NS>(X, W, SB, bias, outf, outh, kc, vc, rope, st, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
#define GEMM3_ALL(TT) \
GEMM3(k_g3_qkv_##TT, H, 0, TT, 4) \
GEMM3(k_g3_o_##TT, H, 1, TT, 4) \
GEMM3(k_g3_gu_##TT, H, 2, TT, 4) \
GEMM3(k_g3_down_##TT, FF, 1, TT, 4)
GEMM3_ALL(8)
GEMM3_ALL(16)
GEMM3_ALL(24)
GEMM3_ALL(32)
#endif

// ---------------------------------------------------------------------------------------------
// Prefill GEMM on the Neural Accelerators for q4a weights. Threadgroup = 4 simdgroups, tile =
// 32 rows (two 16-row blocks) x TT tokens. Per K step a simdgroup dequantizes its own 32 x BK tile
// (BK = 4 lane slices of 28 for K = 896, 2 slices of 38 for K = 4864) straight from the interleaved
// layout: lane i takes block/slice group i / LPG and every LPG-th nibble chunk of it.
#if FMT == 3
template<int K>
inline void q4a_tile(device const uchar* W, uint blk0, uint step, threadgroup half* ws, ushort lane) {
    constexpr int COLS = K == H ? 28 : 38;          // lane-slice width
    constexpr int NN = COLS / 2;                     // nibble chunks per slice
    constexpr int SL = K == H ? 4 : 2;               // slices per step
    constexpr int BK = SL * COLS;
    constexpr int LPG = 32 / (2 * SL);               // lanes per (block, slice) group
    constexpr uint BB = K == H ? Q4A_B896 : Q4A_B4864;
    const int grp = lane / LPG, sub = lane % LPG;
    const int bb = grp / SL, ls = grp % SL;
    uint unit, l;
    if (K == H) { unit = blk0 + bb; l = step * SL + ls; }
    else { const uint q = step / 16; unit = (blk0 + bb) * 4 + q; l = (step % 16) * SL + ls; }
    device const uchar* pb = W + (ulong)unit * BB;
    const uint4 sc = ((device const uint4*)(pb + NN * 512))[l];
    const uint2 zl = ((device const uint2*)(pb + (NN + 1) * 512))[l];
    const uint zh = ((device const uint*)(pb + (NN + 1) * 512 + 256))[l];
    // w = s (n - z) = (16 s) (1 + n/16) - s (16 + z): one half2 FMA per two weights on the fp16-magic unpack.
    float sv[16];
    half2 mul[16], add[16];
    for (int j = 0; j < 4; j++) e5m3x4(sc[j], sv + 4 * j);
    for (int r = 0; r < 16; r++) {
        const uint lo = ((r < 8 ? zl.x : zl.y) >> (4 * (r & 7))) & 0xFu;
        const float z = float(lo | (((zh >> (2 * r)) & 0x3u) << 4)) * 0.25f;
        mul[r] = half2(half(16.f * sv[r]));
        add[r] = half2(half(-sv[r] * (16.f + z)));
    }
    threadgroup half2* dst = (threadgroup half2*)(ws + (16 * bb) * BK + COLS * ls);
    for (int c = sub; c < NN; c += LPG) {
        const uint4 w = ((device const uint4*)(pb + c * 512))[l];
        for (int j = 0; j < 4; j++) {
            const uint ww = w[j];
            const half2 a = as_type<half2>(((ww << 6) & 0x03C003C0u) | 0x3C003C00u);   // rows 0,2 col 0
            const half2 cc = as_type<half2>(((ww << 2) & 0x03C003C0u) | 0x3C003C00u);  // rows 0,2 col 1
            const half2 b = as_type<half2>(((ww >> 2) & 0x03C003C0u) | 0x3C003C00u);   // rows 1,3 col 0
            const half2 d = as_type<half2>(((ww >> 6) & 0x03C003C0u) | 0x3C003C00u);   // rows 1,3 col 1
            const int r = 4 * j;
            dst[(r + 0) * (BK / 2) + c] = fma(half2(a.x, cc.x), mul[r + 0], add[r + 0]);
            dst[(r + 1) * (BK / 2) + c] = fma(half2(b.x, d.x), mul[r + 1], add[r + 1]);
            dst[(r + 2) * (BK / 2) + c] = fma(half2(a.y, cc.y), mul[r + 2], add[r + 2]);
            dst[(r + 3) * (BK / 2) + c] = fma(half2(b.y, d.y), mul[r + 3], add[r + 3]);
        }
    }
}

template<int K, int MODE, int TT>
inline void gemm3a_body(device half* X, device const uchar* W, device const float* bias, device float* outf,
                        device half* outh, device half* kc, device half* vc, device const float2* rope,
                        device const State* st, threadgroup half* Wsg, threadgroup float* Os,
                        uint2 tgp, uint tid, ushort lane, ushort sg) {
    constexpr int NS = 4;
    constexpr int SL = K == H ? 4 : 2;
    constexpr int BK = SL * (K == H ? 28 : 38);
    constexpr int NSTEP = K / BK;
    const int n = st->n;
    const uint blk0 = tgp.x * 2;
    const uint r0 = tgp.x * 32;
    const int t0 = tgp.y * TT;
    threadgroup half* ws = Wsg + sg * 32 * BK;
    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    auto tW = tensor(ws, dextents<int, 2>(BK, 32), array<int, 2>({1, BK}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;
    for (uint step = sg; step < (uint)NSTEP; step += NS) {
        q4a_tile<K>(W, blk0, step, ws, lane);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint kcol;
        if (K == H) kcol = step * BK;
        else kcol = (step / 16) * 1216 + (step % 16) * BK;
        auto a = tX.template slice<BK, TT>(kcol, 0);
        op.run(a, tW, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tO = tensor(Os + sg * TT * 32, dextents<int, 2>(32, TT), array<int, 2>({1, 32}));
    cT.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int NT = NS * 32;
    if (MODE == 0) {
        // two 16-row blocks: q/k blocks pair row i with row i + 8 (dims 8m+i, 32+8m+i); v blocks plain
        for (int w = tid; w < TT * 16; w += NT) {
            const int t = w / 16, bb = (w % 16) / 8, i = w % 8;
            if (t0 + t >= n) continue;
            const int b = blk0 + bb;
            const int ra = 16 * bb + i, rb = ra + 8;
            float a = bias[r0 + ra], c = bias[r0 + rb];
            for (int q = 0; q < NS; ++q) { a += Os[q * TT * 32 + t * 32 + ra]; c += Os[q * TT * 32 + t * 32 + rb]; }
            const int pos = st->pos + t0 + t;
            if (b < 64) {
                const int head = b >> 2, d = 8 * (b & 3) + i;
                const float2 cs = rope[pos * 32 + d];
                const float a2 = a * cs.x - c * cs.y;
                const float c2 = c * cs.x + a * cs.y;
                if (head < NH) {
                    outf[(t0 + t) * H + head * HD + d] = a2;
                    outf[(t0 + t) * H + head * HD + d + 32] = c2;
                } else {
                    device half* k = kc + ((ulong)(head - NH) * st->maxctx + pos) * HD;
                    k[d] = half(a2); k[d + 32] = half(c2);
                }
            } else {
                const int vb = b - 64;
                device half* v = vc + ((ulong)(vb >> 2) * st->maxctx + pos) * HD + 16 * (vb & 3);
                v[i] = half(a); v[i + 8] = half(c);
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
        // block b: rows 0-7 = gate 8b+i, rows 8-15 = up 8b+i
        for (int w = tid; w < TT * 16; w += NT) {
            const int t = w / 16, bb = (w % 16) / 8, i = w % 8;
            if (t0 + t >= n) continue;
            float g = 0.f, u = 0.f;
            for (int q = 0; q < NS; ++q) { g += Os[q * TT * 32 + t * 32 + 16 * bb + i]; u += Os[q * TT * 32 + t * 32 + 16 * bb + 8 + i]; }
            outh[(t0 + t) * FF + 8 * (blk0 + bb) + i] = half(g / (1.f + exp(-g)) * u);
        }
    }
}

#define GEMM3A(NAME, K, MODE, TT) \
kernel void NAME(device half* X [[buffer(0)]], device const uchar* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TT) >= st->n) return; \
    constexpr int BKX = K == H ? 112 : 76; \
    constexpr int WB = 4 * 32 * BKX * 2, OB = 4 * TT * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm3a_body<K, MODE, TT>(X, W, bias, outf, outh, kc, vc, rope, st, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
#define GEMM3A_ALL(TT) \
GEMM3A(k_g3_qkv_##TT, H, 0, TT) \
GEMM3A(k_g3_o_##TT, H, 1, TT) \
GEMM3A(k_g3_gu_##TT, H, 2, TT) \
GEMM3A(k_g3_down_##TT, FF, 1, TT)
GEMM3A_ALL(8)
GEMM3A_ALL(16)
GEMM3A_ALL(24)
GEMM3A_ALL(32)
#endif
"""#
