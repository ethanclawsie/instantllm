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

// Draft trees (wide verify passes with st->pad4 set): ints [n, depth[16], parent[16], tok[16], anc[16]];
// row 0 is the last committed token, row i a draft whose parent row is parent[i]; anc[i] has bit j set
// for every row j on its path (itself included). Row i sits at KV slot pos + i, rotated as pos + depth.
#define TR_DEPTH 1
#define TR_PARENT 17
#define TR_TOK 33
#define TR_ANC 49
#define TR_INTS 65

// Decode/verify code is compiled with strict fp semantics: under fast math the compiler may
// reassociate differently in each T instantiation, and verify passes must reproduce the T = 1
// numerics bit for bit (speculation never changes the output). Prefill code below switches back.
#pragma METAL fp math_mode(safe)

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

// Most frequent continuation of the longest known context (orders NG_MAXN..1); -1 if none is known.
// Confidence is not checked: a pass costs the same with or without a draft in a slot.
inline int ng_next(device const NGEntry* tab, thread const int* hist, int len) {
    for (int n = min(NG_MAXN, len); n >= 1; n--) {
        const ulong k = ng_key(hist + len - n, n);
        uint slot = uint(k >> 12) & (NG_SLOTS - 1);
        for (int p = 0; p < 16; p++) {
            const NGEntry e = tab[slot];
            if (e.key == 0) break;
            if (e.key == k) return e.next;
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
#elif FMT == 1
typedef char wtype;
// q8: row = K int8 codes, SB (as half) = [rows][K/32] scales: w = s * q.
inline float embed_at(device const char* E, device const half2* SB, int row, int i) {
    return float(((device const half*)SB)[row * (H / 32) + (i >> 5)]) * float(E[(ulong)row * H + i]);
}
#elif FMT == 2
typedef uchar wtype;
// q4: row = K/2 bytes of nibbles (weight 2j in the low nibble of byte j), SB = [rows][K/64] (scale, bias).
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
// Keeps one SIMD lane group busy for `iters` dependent FMAs (GPU clock experiments).
kernel void k_spin(device float* sink [[buffer(0)]], constant int& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float a = float(i), b = 1.0001f;
    for (int k = 0; k < iters; ++k) { a = fma(a, b, 0.5f); b = fma(b, 0.9999f, 0.0001f); }
    if (a == 12345.f) sink[0] = a;
}

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
// drafts tspec-1 new tokens and embeds [g_m, drafts...] for the next pass. Each draft comes from the
// first source with a guess: (1) self-repetition: the most recent earlier occurrence of the longest
// output suffix of 3..8 tokens, continued; (2) the n-gram table; (3) prompt lookup: the token after
// the latest longest (1..4 token) match of the output suffix in the prompt.
#define OB_MAX 256
kernel void k_commit_spec(device const float* pval [[buffer(0)]],
                          device const int* pidx [[buffer(1)]],
                          device State* st [[buffer(2)]],
                          device int* out_tokens [[buffer(3)]],
                          device const wtype* E [[buffer(4)]],
                          device const half2* ESB [[buffer(5)]],
                          device float* x [[buffer(6)]],
                          device int* drafts [[buffer(7)]],
                          device const NGEntry* tab [[buffer(8)]],
                          device const int* prompt [[buffer(9)]],   // [count, -, -, -, last <= 256 prompt tokens]
                          device const int* tree [[buffer(10)]],    // draft tree of the pass (st->pad4)
                          device half* kcache [[buffer(11)]],       // whole KV caches (tree: accepted rows move)
                          device half* vcache [[buffer(12)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float rv[32];
    threadgroup int ri[32];
    threadgroup int g[16];
    threadgroup int next_rows[16];
    threadgroup int nrows_s;
    threadgroup int have_s;
    threadgroup int ob[OB_MAX + 16];  // recent output tokens, then the drafts of this commit
    threadgroup int path_s[16];
    threadgroup int npath_s, pos0_s;
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
        const bool treeMode = st->pad4 != 0;
        if (treeMode) {
            // longest path from the root whose every draft equals its parent row's argmax
            int cur = 0;
            for (;;) {
                int c = -1;
                for (int i = 1; i < nv; ++i) if (tree[TR_PARENT + i] == cur && tree[TR_TOK + i] == g[cur]) { c = i; break; }
                if (c < 0) break;
                path_s[m++] = c;
                cur = c;
            }
        } else {
            while (m < nv - 1 && drafts[m] == g[m]) m++;
        }
        if (m > 0) st->pad1 += m;            // accepted drafts (diagnostics)
        bool stop = false;
        int last = g[0];
        for (int i = 0; i <= m && !stop; i++) {
            const int tok = g[(treeMode && i > 0) ? path_s[i - 1] : i];
            out_tokens[st->step] = tok;
            st->step += 1;
            last = tok;
            if (tok == st->eos0 || tok == st->eos1 || st->step >= st->maxsteps) stop = true;
        }
        // the pass covered positions pos .. pos+nver-1; rows 0..m are now real context
        pos0_s = st->pos;
        npath_s = treeMode ? m : 0;
        st->pos += (nv == 1 ? st->n : m + 1);
        st->pad4 = 0;
        if (stop) {
            st->done = 1;
            nrows_s = 0;
        } else if (st->pad3) {
            nrows_s = 0;                         // k_ildraft drafts and embeds the next rows
        } else {
            nrows_s = -1;
            have_s = st->step;
            next_rows[0] = last;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // tree: the accepted row path[k-1] (KV slot pos0 + path[k-1]) becomes position pos0 + k; it was
    // already rotated as pos0 + depth = pos0 + k. Sources and destinations never overlap.
    if (npath_s > 0) {
        const int p0 = pos0_s, maxctx = st->maxctx;
        for (int k = 1; k <= npath_s; ++k) {
            const int src = p0 + path_s[k - 1], dst = p0 + k;
            if (src == dst) continue;
            for (int j = int(tid); j < 24 * NKV * HD; j += 1024) {
                const int l = j / (NKV * HD), hh = (j / HD) % NKV, d = j % HD;
                const ulong base = (ulong)(l * NKV + hh) * maxctx * HD + d;
                kcache[base + (ulong)dst * HD] = kcache[base + (ulong)src * HD];
                vcache[base + (ulong)dst * HD] = vcache[base + (ulong)src * HD];
            }
        }
    }
    if (nrows_s < 0) {
        const int have = have_s;
        const int w0 = max(0, have - OB_MAX);
        for (int i = w0 + int(tid); i < have; i += 1024) ob[i - w0] = out_tokens[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int L = have - w0;
            int hist[NG_MAXN + 16];
            int len = 0;
            for (int i = NG_MAXN; i >= 1; i--) hist[len++] = have - i >= 0 ? ob[have - i - w0] : -1;  // -1: before the output
            const int T = st->tspec;
            const int maxd = st->pad2 ? 0 : T - 1;   // pad2: no drafting (reference runs with the same kernels)
            const int np = prompt[0];
            device const int* P = prompt + 4;
            int nd = 0;
            while (nd < maxd) {
                int d = -1;
                int bn = 2;
                for (int j = L - 1; j >= 1 && bn < 8; j--) {
                    int m = 0;
                    while (m < 8 && j - 1 - m >= 0 && ob[j - 1 - m] == ob[L - 1 - m]) m++;
                    if (m > bn) { bn = m; d = ob[j]; }
                }
                if (d < 0) d = ng_next(tab, hist, len);
                if (d < 0) {
                    int pm = 0;
                    for (int j = np - 1; j >= 1 && pm < 4; j--) {
                        int m = 0;
                        while (m < 4 && m < L && j - 1 - m >= 0 && P[j - 1 - m] == ob[L - 1 - m]) m++;
                        if (m > pm) { pm = m; d = P[j]; }
                    }
                }
                if (d < 0) break;
                drafts[nd] = d;
                next_rows[nd + 1] = d;
                nd++;
                hist[len++] = d;
                ob[L++] = d;
                if (d == st->eos0 || d == st->eos1) break;
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


// ---------------------------------------------------------------------------------------------
// Learned drafter (reference: tools/draft/ildraft.py). Per drafted token: candidates from the n-gram
// table (top-M continuations of every known context order of TAIL + output), self-repetition,
// prompt lookup, the request's TF-IDF neighbours' outputs and the copy candidates; 53 features per
// candidate, an MLP scorer, argmax (smallest token on ties). Runs after k_commit_spec (which skips
// its own drafting when st->pad3 is set). One threadgroup of 1024 threads.

struct DraftMeta {
    uint ngky_off, ngky_n, ngvl_off, ngvl_n, ngnx_off, ngnx_n, cptk_off, cptk_n;
    uint cpc1_off, cpc1_n, cpc2_off, cpc2_n, mlp_off, mlp_n, pad0, pad1;
    uint cfgN, cfgM, cfgMAXM, cfgH;
};
#define DR_MAXP 256
#define DR_MAXCOPY 256
#define DR_MAXNB 16
#define DR_NBSEQ 80
#define DR_MAXAS 1024
struct DraftReq {
    int np, ncopy, nnb, nas;
    int P[DR_MAXP];
    int copyTok[DR_MAXCOPY];
    int copyInP[DR_MAXCOPY];
    float nbScore[DR_MAXNB];
    int nbOff[DR_MAXNB + 1];
    int nbTok[DR_MAXNB * DR_NBSEQ];
    int asTok[DR_MAXAS];
    float asMax[DR_MAXAS];
    int asCnt[DR_MAXAS];
    int treeB;          // > 0: best-first draft tree with treeB nodes (8 children per expansion)
};
#define DR_F 53
#define DR_RAW (85 + DR_MAXCOPY)

inline int dr_tail(int i) { return i == 0 ? 151645 : (i == 1 ? 198 : (i == 2 ? 151644 : (i == 3 ? 77091 : 198))); }

inline int dr_find_u32(device const uint2* a, int n, uint key) {   // sorted {key, value} pairs
    int lo = 0, hi = n;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (a[mid].x < key) lo = mid + 1; else hi = mid; }
    return (lo < n && a[lo].x == key) ? lo : -1;
}
inline int dr_find_int(device const int* a, int n, int key) {
    int lo = 0, hi = n;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return (lo < n && a[lo] == key) ? lo : -1;
}

kernel void k_ildraft(device const uchar* tab [[buffer(0)]], constant DraftMeta& dm [[buffer(1)]],
                      device const uchar* tflg [[buffer(2)]], device const DraftReq& rq [[buffer(3)]],
                      device State* st [[buffer(4)]], device const int* out_tokens [[buffer(5)]],
                      device int* drafts [[buffer(6)]], device const wtype* E [[buffer(7)]],
                      device const half2* ESB [[buffer(8)]], device float* x [[buffer(9)]],
                      device int* tree [[buffer(10)]],
                      constant int2& shape [[buffer(11)]],     // this pass: rows, tree drafts (0: chain)
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup int ob[OB_MAX + 16];
    threadgroup int s_depth, s_go, s_erow, s_ticks0, s_rows;
    threadgroup float s_best_s;
    threadgroup int s_best_t;
    threadgroup int tr_par[16], tr_tok[16], tr_dep[16];
    threadgroup uint tr_anc[16];
    threadgroup float tr_cum[16];
    threadgroup float h_cum[128];
    threadgroup int h_par[128], h_tok[128], h_ord[128];
    threadgroup int h_n, h_ctr;
    threadgroup int tk_tok[8];
    threadgroup float tk_e[8];
    threadgroup float zpart[32];
    threadgroup float csc[DR_RAW];
    threadgroup int ctk[DR_RAW];
    threadgroup int dist_tot[7], dist_fn[7];
    threadgroup int s_long, s_supp3, s_supp10, s_toplong, s_entlow;
    threadgroup int self_tok, self_n, pr_n, pr_cnt, pr_e[32];
    threadgroup int pm[DR_MAXP];
    threadgroup int vote_tok[DR_MAXNB], vote_n[DR_MAXNB];
    threadgroup float s_c1, s_c2;
    threadgroup int s_firstun, s_ticks;
    threadgroup int raw[DR_RAW];
    threadgroup int uniq[DR_RAW];
    threadgroup atomic_int nuniq;
    threadgroup float bsc[32];
    threadgroup int btk[32];
    threadgroup int s_draft, s_stop, s_nd;
    threadgroup int next_rows[16];
    device const ulong* ngky = (device const ulong*)(tab + dm.ngky_off);
    device const uint2* ngvl = (device const uint2*)(tab + dm.ngvl_off);
    device const uint2* ngnx = (device const uint2*)(tab + dm.ngnx_off);
    device const uint2* cptk = (device const uint2*)(tab + dm.cptk_off);
    device const uint2* cpc1 = (device const uint2*)(tab + dm.cpc1_off);
    device const uchar* cpc2 = tab + dm.cpc2_off;
    device const float* mlp = (device const float*)(tab + dm.mlp_off);
    const int HN = int(dm.cfgH);
    const int have = st->step;
    const int w0 = max(0, have - OB_MAX);
    for (int i = w0 + int(tid); i < have; i += 1024) ob[i - w0] = out_tokens[i];
    const int first_out = out_tokens[0];
    const int T = shape.x;
    const int treeB = shape.y;
    const int K = st->pad2 ? 0 : (treeB > 0 ? min(treeB, T - 1) : T - 1);
    const int base = have - w0;                // committed tokens in the window
    if (tid == 0) {
        next_rows[0] = out_tokens[have - 1];
        s_nd = 0; s_stop = 0;
        int tk = 0;
        for (int i = 0; i < have; ++i) tk += tflg[out_tokens[i]] & 7;
        s_ticks = tk; s_ticks0 = tk;
        s_depth = 0; s_go = K > 0 ? 1 : 0;
        s_rows = 1; s_erow = 0; h_n = 0; h_ctr = 0;
        tr_par[0] = -1; tr_tok[0] = out_tokens[have - 1]; tr_dep[0] = 0; tr_anc[0] = 1u; tr_cum[0] = 1.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One context evaluation per iteration: the chain appends its argmax; the tree pushes the top-8
    // children of the expanded row and pops the next row to expand (best cumulative probability).
    for (int it = 0; it < 64; ++it) {
        if (!s_go) break;
        const int Lo = base + s_depth;         // tokens of the context's output part in ob
        const int Lfull = have + s_depth;      // full output length
        const int L = 5 + Lo;                  // ctx = TAIL + out
        // ---- phase A: sources ----
        if (sg == 0) {
            if (lane < 6) {
                const int n = int(lane) + 1;
                int tot = -1, fn = 0;
                if (n <= int(dm.cfgN) && n <= L) {
                    ulong h = 0xcbf29ce484222325ul;
                    h ^= ulong(n); h *= 0x100000001b3ul;
                    for (int i = L - n; i < L; ++i) {
                        const uint t = uint(i < 5 ? dr_tail(i) : ob[i - 5]);
                        for (int b = 0; b < 4; ++b) { h ^= ulong((t >> (8 * b)) & 0xffu); h *= 0x100000001b3ul; }
                    }
                    int lo = 0, hi = int(dm.ngky_n);
                    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ngky[mid] < h) lo = mid + 1; else hi = mid; }
                    if (lo < int(dm.ngky_n) && ngky[lo] == h) { tot = int(ngvl[lo].x); fn = int(ngvl[lo].y); }
                }
                dist_tot[n] = tot; dist_fn[n] = fn;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                int ln = 0, s3 = 0, s10 = 0;
                for (int n = 1; n <= 6; ++n) if (dist_tot[n] >= 0) { ln = n; if (dist_tot[n] >= 3) s3 = n; if (dist_tot[n] >= 10) s10 = n; }
                s_long = ln; s_supp3 = s3; s_supp10 = s10;
                s_toplong = -1; s_entlow = 0;
                if (ln > 0) {
                    const int first = dist_fn[ln] >> 3, cnt = dist_fn[ln] & 7;
                    const float tot = float(dist_tot[ln]);
                    s_toplong = int(ngnx[first].x);
                    float ent = 0.f;
                    for (int r = 0; r < cnt; ++r) { const float pr = float(ngnx[first + r].y) / tot; ent -= pr * log(pr); }
                    s_entlow = ent < 0.3f ? 1 : 0;
                }
            }
        } else if (sg == 1) {
            // self: longest suffix (<= MAXM) of the output seen earlier, most recent occurrence
            int bestm = 0, beste = -1;
            for (int e = int(lane) + 1; e <= Lo - 1; e += 32) {
                int m = 0;
                const int lim = min(min(int(dm.cfgMAXM), Lo - 1), e);
                while (m < lim && ob[e - 1 - m] == ob[Lo - 1 - m]) m++;
                if (m > bestm || (m == bestm && m > 0 && e > beste)) { bestm = m; beste = e; }
            }
            const int mx = simd_max(bestm);
            const int ex = simd_max(bestm == mx && mx > 0 ? beste : -1);
            if (lane == 0) { self_n = mx; self_tok = (mx > 0 && ex >= 0) ? ob[ex] : -1; }
        } else if (sg == 2) {
            // prompt lookup: all occurrences of the longest suffix (<= MAXM) of the output in P
            const int np = rq.np;
            int bestm = 0;
            for (int e = int(lane) + 1; e <= np - 1; e += 32) {
                int m = 0;
                const int lim = min(min(int(dm.cfgMAXM), Lo), e);
                while (m < lim && rq.P[e - 1 - m] == ob[Lo - 1 - m]) m++;
                pm[e] = m;
                bestm = max(bestm, m);
            }
            const int mx = simd_max(bestm);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                int c = 0;
                if (mx > 0) for (int e = 1; e <= np - 1 && c < 32; ++e) if (pm[e] == mx) pr_e[c++] = e;
                pr_n = mx; pr_cnt = c;
            }
        } else if (sg >= 3 && sg < 3 + DR_MAXNB) {
            // neighbour k: longest suffix (<= MAXM) of ctx in its sequence, first occurrence
            const int k = int(sg) - 3;
            int vt = -1, vn = 0;
            if (k < rq.nnb) {
                const int a = rq.nbOff[k], ls = rq.nbOff[k + 1] - a;
                int bestm = 0, beste = 1 << 20;
                for (int e = int(lane) + 1; e <= ls - 1; e += 32) {
                    int m = 0;
                    const int lim = min(min(int(dm.cfgMAXM), L), e);
                    while (m < lim) {
                        const int ci = L - 1 - m;
                        const int c = ci < 5 ? dr_tail(ci) : ob[ci - 5];
                        if (rq.nbTok[a + e - 1 - m] != c) break;
                        m++;
                    }
                    if (m > bestm || (m == bestm && m > 0 && e < beste)) { bestm = m; beste = e; }
                }
                const int mx = simd_max(bestm);
                const int ex = simd_min(bestm == mx && mx > 0 ? beste : (1 << 20));
                if (mx > 0 && ex < (1 << 20)) { vt = rq.nbTok[a + ex]; vn = mx; }
            }
            if (lane == 0) { vote_tok[k] = vt; vote_n[k] = vn; }
        } else if (sg == 3 + DR_MAXNB) {
            // copy context: c1, c2, first unused prompt token
            if (lane == 0) {
                const int c_1 = L - 1 < 5 ? dr_tail(L - 1) : ob[L - 1 - 5];
                const int c_2 = L - 2 < 5 ? dr_tail(L - 2) : ob[L - 2 - 5];
                const int i1 = dr_find_u32(cpc1, int(dm.cpc1_n), uint(c_1));
                s_c1 = i1 >= 0 ? as_type<float>(cpc1[i1].y) : 0.1f;
                const ulong key = (ulong(uint(c_2)) << 32) | ulong(uint(c_1));
                int lo = 0, hi = int(dm.cpc2_n);
                while (lo < hi) { const int mid = (lo + hi) >> 1; if (*((device const ulong*)(cpc2 + 16 * mid)) < key) lo = mid + 1; else hi = mid; }
                s_c2 = (lo < int(dm.cpc2_n) && *((device const ulong*)(cpc2 + 16 * lo)) == key) ? *((device const float*)(cpc2 + 16 * lo + 8)) : 0.1f;
            }
            int fu = 1 << 20;
            for (int i = int(lane); i < rq.np; i += 32) {
                const int xt = rq.P[i];
                bool used = false;
                for (int j = 0; j < Lo && !used; ++j) used = ob[j] == xt;
                if (!used) {
                    const int ci = dr_find_u32(cptk, int(dm.cptk_n), uint(xt));
                    const float pr = ci >= 0 ? as_type<float>(cptk[ci].y) : 0.2f;
                    if (pr > 0.3f) fu = min(fu, i);
                }
            }
            fu = simd_min(fu);
            if (lane == 0) s_firstun = fu < (1 << 20) ? rq.P[fu] : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- phase B: raw candidate list, then unique tokens ----
        const int ncopy = rq.ncopy;
        const int nraw = 85 + ncopy;
        for (int i = int(tid); i < nraw; i += 1024) {
            int t = -1;
            if (i < 36) {
                const int n = i / 6 + 1, r = i % 6;
                if (dist_tot[n] >= 0 && r < (dist_fn[n] & 7)) t = int(ngnx[(dist_fn[n] >> 3) + r].x);
            } else if (i == 36) {
                t = self_n > 0 ? self_tok : -1;
            } else if (i < 53) {
                t = vote_tok[i - 37];
            } else if (i < 85) {
                const int q = i - 53;
                t = q < pr_cnt ? rq.P[pr_e[q]] : -1;
            } else {
                t = rq.copyTok[i - 85];
            }
            raw[i] = t;
        }
        if (tid == 0) atomic_store_explicit(&nuniq, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < nraw; i += 1024) {
            const int t = raw[i];
            bool keep = t >= 0;
            for (int j = 0; j < i && keep; ++j) keep = raw[j] != t;
            if (keep) uniq[atomic_fetch_add_explicit(&nuniq, 1, memory_order_relaxed)] = t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nu = atomic_load_explicit(&nuniq, memory_order_relaxed);
        // ---- phase C: features + MLP score per candidate ----
        float score = -INFINITY;
        int tok = 0x7fffffff;
        if (int(tid) < nu) {
            tok = uniq[tid];
            float f[DR_F];
            for (int i = 0; i < DR_F; ++i) f[i] = 0.f;
            f[0] = 1.f;
            const int ln = s_long;
            for (int n = 1; n <= 6; ++n) {
                if (dist_tot[n] < 0) continue;
                const int first = dist_fn[n] >> 3, cnt = dist_fn[n] & 7;
                int c = 0, rank = -1;
                for (int r = 0; r < cnt; ++r) if (int(ngnx[first + r].x) == tok) { c = int(ngnx[first + r].y); rank = r; break; }
                if (n == ln && rank == 1) f[13] = 1.f;
                if (n == ln && rank >= 2) f[14] = 1.f;
                if (c > 0) {
                    const float p = float(c) / float(dist_tot[n]);
                    f[6 + n] = p;
                    if (n == ln) { f[1] = p; f[3] = float(n) / 6.f * p; f[4] = log(1.f + float(c)); }
                    if (n == s_supp3) f[5] = p;
                    if (n == s_supp10) f[6] = p;
                }
            }
            if (tok == s_toplong) { f[2] = 1.f; if (s_entlow) f[15] = 1.f; }
            if (self_n > 0 && tok == self_tok) { f[16] = float(self_n) / 4.f; f[17] = 1.f; f[18] = self_n >= 3 ? 1.f : 0.f; }
            for (int q = 0; q < pr_cnt; ++q) {
                if (rq.P[pr_e[q]] == tok) {
                    f[19] = float(pr_n) / 4.f; f[20] = 1.f; f[21] = pr_n >= 2 ? 1.f : 0.f;
                    if (q == 0) f[22] = 1.f;
                }
            }
            const int nnb = max(1, rq.nnb);
            float nbs = 0.f, nbm = 0.f, nbx = 0.f;
            int nbc = 0;
            for (int k = 0; k < rq.nnb; ++k) {
                if (vote_tok[k] != tok) continue;
                const float sk = rq.nbScore[k], mk = float(min(vote_n[k], 8)) / 8.f;
                nbs += sk; nbm += sk * mk; nbc++; nbx = max(nbx, mk);
                if (k == 0) { f[27] = 1.f; f[28] = mk; }
            }
            f[23] = nbs; f[24] = nbm; f[25] = float(nbc) / float(nnb); f[26] = nbx;
            const int ci = dr_find_int(rq.copyTok, ncopy, tok);
            bool usedt = false;
            for (int j = 0; j < Lo && !usedt; ++j) usedt = ob[j] == tok;
            const bool inp = ci >= 0 && rq.copyInP[ci] != 0;
            if (ci >= 0) {
                float pt = 0.3f;
                if (inp) { const int pi = dr_find_u32(cptk, int(dm.cptk_n), uint(tok)); pt = pi >= 0 ? as_type<float>(cptk[pi].y) : 0.2f; }
                f[33] = log(pt / (1.f - pt)) / 4.f;
                f[34] = s_c1; f[35] = s_c2;
                f[36] = usedt ? 0.f : 1.f;
                f[37] = tok == s_firstun ? 1.f : 0.f;
                f[38] = inp ? 0.f : 1.f;
            }
            f[29] = (tok == 151645 || tok == 151643) ? 1.f : 0.f;
            f[30] = inp ? 1.f : 0.f;
            f[31] = f[2] * min(1.f, 2.f * f[25]);
            f[32] = f[17] * min(1.f, 2.f * f[25]);
            f[39] = (Lfull > 0 && (first_out == 73594 || first_out == 13874)) ? 1.f : 0.f;
            f[40] = (Lfull > 0 && first_out == 63) ? 1.f : 0.f;
            const uint tf = tok < VOCAB ? uint(tflg[tok]) : 0u;
            f[41] = (tf & 7u) > 0 ? 1.f : 0.f;
            f[42] = (tf & 8u) != 0 ? 1.f : 0.f;
            f[43] = float(s_ticks & 1);
            f[44] = float(min(Lfull, 32)) / 32.f;
            f[45] = usedt ? 1.f : 0.f;
            const int ai = dr_find_int(rq.asTok, rq.nas, tok);
            if (ai >= 0) { f[46] = rq.asMax[ai] / 4.f; f[47] = float(min(rq.asCnt[ai], 4)) / 4.f; }
            // MLP: W2 . relu(W1 x + b1) + b2 + WL . x
            device const float* W1 = mlp;
            device const float* b1 = mlp + HN * DR_F;
            device const float* W2 = b1 + HN;
            const float b2 = W2[HN];
            device const float* WL = W2 + HN + 1;
            float sc = b2;
            for (int i = 0; i < DR_F; ++i) sc += WL[i] * f[i];
            for (int h = 0; h < HN; ++h) {
                float a = b1[h];
                for (int i = 0; i < DR_F; ++i) a += W1[h * DR_F + i] * f[i];
                sc += W2[h] * max(a, 0.f);
            }
            score = sc;
        }
        // ---- phase D: best candidate; for trees also the top 8 with probabilities ----
        if (int(tid) < nu) { csc[tid] = score; ctk[tid] = tok; }
        float rs = score;
        int rt = tok;
        for (ushort off = 16; off > 0; off >>= 1) {
            const float os = simd_shuffle_down(rs, off);
            const int ot = simd_shuffle_down(rt, off);
            if (os > rs || (os == rs && ot < rt)) { rs = os; rt = ot; }
        }
        if (lane == 0) { bsc[sg] = rs; btk[sg] = rt; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            rs = bsc[lane]; rt = btk[lane];
            for (ushort off = 16; off > 0; off >>= 1) {
                const float os = simd_shuffle_down(rs, off);
                const int ot = simd_shuffle_down(rt, off);
                if (os > rs || (os == rs && ot < rt)) { rs = os; rt = ot; }
            }
            if (lane == 0) { s_best_s = rs; s_best_t = rt; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (treeB > 0) {
            const float mx = max(s_best_s, 0.f);
            float ez = 0.f;
            if (int(tid) < nu) {
                ez = exp(score - mx);
                int rank = 0;
                for (int j = 0; j < nu; ++j) { const float sj = csc[j]; rank += (sj > score || (sj == score && ctk[j] < tok)) ? 1 : 0; }
                if (rank < 8) { tk_tok[rank] = tok; tk_e[rank] = ez; }
            }
            ez = simd_sum(ez);
            if (lane == 0) zpart[sg] = ez;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            if (treeB == 0) {
                const int bt = s_best_t;
                if (nu == 0 || bt == 0x7fffffff) {
                    s_go = 0;
                } else {
                    const int d = s_nd;
                    drafts[d] = bt;
                    next_rows[d + 1] = bt;
                    ob[Lo] = bt;
                    s_nd = d + 1;
                    s_depth = d + 1;
                    s_ticks += tflg[bt] & 7;
                    s_go = (s_nd < K && bt != st->eos0 && bt != st->eos1) ? 1 : 0;
                }
            } else {
                float Z = exp(-max(s_best_s, 0.f));
                for (int q = 0; q < 32; ++q) Z += zpart[q];
                const int ntop = min(8, nu);
                const float pc = tr_cum[s_erow];
                for (int k = 0; k < ntop && h_n < 128; ++k) {
                    h_cum[h_n] = pc * tk_e[k] / Z; h_par[h_n] = s_erow; h_tok[h_n] = tk_tok[k]; h_ord[h_n] = h_ctr++;
                    h_n++;
                }
                s_go = 0;
                while (s_rows < K + 1 && h_n > 0) {
                    int b = 0;
                    for (int q = 1; q < h_n; ++q) if (h_cum[q] > h_cum[b] || (h_cum[q] == h_cum[b] && h_ord[q] < h_ord[b])) b = q;
                    const float cum = h_cum[b];
                    const int par = h_par[b], tkn = h_tok[b];
                    h_n--;
                    h_cum[b] = h_cum[h_n]; h_par[b] = h_par[h_n]; h_tok[b] = h_tok[h_n]; h_ord[b] = h_ord[h_n];
                    const int i = s_rows++;
                    tr_par[i] = par; tr_tok[i] = tkn; tr_dep[i] = tr_dep[par] + 1;
                    tr_anc[i] = tr_anc[par] | (1u << i); tr_cum[i] = cum;
                    if (tkn == st->eos0 || tkn == st->eos1 || s_rows == K + 1) continue;   // not expanded
                    int r = i, dpos = tr_dep[i], tk2 = s_ticks0;
                    while (r > 0) { ob[base + dpos - 1] = tr_tok[r]; tk2 += tflg[tr_tok[r]] & 7; dpos--; r = tr_par[r]; }
                    s_depth = tr_dep[i]; s_erow = i; s_ticks = tk2; s_go = 1;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int nd = s_nd;
    if (tid == 0) {
        if (treeB > 0) {
            const int rows = s_rows;
            tree[0] = rows;
            for (int i = 0; i < T; ++i) {
                int dep = 1, par = -1, tk = 0;
                uint an = 1u | (1u << i);              // padding rows: unmatched children of nothing
                if (i == 0) { dep = 0; tk = next_rows[0]; an = 1u; }
                else if (i < rows) { dep = tr_dep[i]; par = tr_par[i]; tk = tr_tok[i]; an = tr_anc[i]; }
                tree[TR_DEPTH + i] = dep; tree[TR_PARENT + i] = par; tree[TR_TOK + i] = tk; tree[TR_ANC + i] = int(an);
                next_rows[i] = tk;
            }
            st->pad4 = 1;
        } else {
            for (int i = nd; i < T - 1; i++) { drafts[i] = -2; next_rows[i + 1] = 0; }
        }
        st->tspec = T;
        st->nver = T;
        st->n = T;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < T; r++) {
        const int t = next_rows[r];
        for (uint i = tid; i < H; i += 1024) x[r * H + i] = embed_at(E, ESB, t, i);
    }
}


// Causal attention for one (query head, token). Grid (14, n), 256 threads.
#define MAXCTX_TG 2048
kernel void k_attention(device const float* q [[buffer(0)]],
                        device const half* kc [[buffer(1)]],
                        device const half* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        constant int& toff [[buffer(5)]],
                        device half* outh [[buffer(6)]],      // the same output as fp16 (prefill GEMM input)
                        device const int* tree [[buffer(7)]], // draft tree (st->pad4): attend to ancestors only
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    const int h = tgp.x;
    const int t = tgp.y + toff;
    if (t >= st->n) return;
    // Tree rows attend to the context and their ancestors only. They iterate over virtual positions
    // (context, then the path in depth order, each mapped to its KV slot), so every sum runs in the
    // same order as for the same token decoded on its own: bit-identical results.
    const bool tm = st->pad4 != 0;
    const int p0 = st->pos;
    const int depth = tm ? tree[TR_DEPTH + t] : t;
    const int L = p0 + depth + 1;
    const int g = h / (NH / NKV);
    threadgroup float qs[HD];
    threadgroup float sc[MAXCTX_TG];
    threadgroup float red[8];
    threadgroup float part[256];
    threadgroup int prow[16];
    if (tm && tid == 0) {
        int r = t;
        for (int d = depth; d >= 0; --d) { prow[d] = r; r = r > 0 ? tree[TR_PARENT + r] : 0; }
    }
    if (tid < HD) qs[tid] = q[t * H + h * HD + tid] * 0.125f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half* kb = kc + (ulong)g * st->maxctx * HD;
    device const half* vb = vc + (ulong)g * st->maxctx * HD;
    float m = -INFINITY;
    if (tm) {
        for (int s = tid; s < L; s += 256) {
            const int slot = s >= p0 ? p0 + prow[s - p0] : s;
            device const half4* kr = (device const half4*)(kb + (ulong)slot * HD);
            float a = 0.f;
            for (int j = 0; j < 16; ++j) a += dot(float4(kr[j]), *((threadgroup const float4*)(qs + 4 * j)));
            sc[s] = a;
            m = max(m, a);
        }
    } else {
        for (int s = tid; s < L; s += 256) {
            device const half4* kr = (device const half4*)(kb + (ulong)s * HD);
            float a = 0.f;
            for (int j = 0; j < 16; ++j) a += dot(float4(kr[j]), *((threadgroup const float4*)(qs + 4 * j)));
            sc[s] = a;
            m = max(m, a);
        }
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
    if (tm) { for (int s = p; s < L; s += 4) o += sc[s] * float(vb[(ulong)(s >= p0 ? p0 + prow[s - p0] : s) * HD + d]); }
    else { for (int s = p; s < L; s += 4) o += sc[s] * float(vb[(ulong)s * HD + d]); }
    part[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HD) {
        const float o = (part[tid] + part[tid + 64] + part[tid + 128] + part[tid + 192]) / sum;
        out[t * H + h * HD + tid] = o;
        outh[t * H + h * HD + tid] = half(o);
    }
}



// Causal attention for up to 8 rows of one query head per threadgroup: the rows share the K/V
// rows they read; each score is one thread's 64-term dot product, each row's softmax one
// simdgroup's, each output element one thread's sum over positions in order, so a row's result
// does not depend on the rows it is grouped with. Contexts up to ATT_ROWS_MAXL; tree rows use
// virtual positions like k_attention. Grid (NH, ceil(rows / 8)), 256 threads.
#define ATT_ROWS_MAXL 512
kernel void k_attn_rows(device const float* q [[buffer(0)]],
                        device const half* kc [[buffer(1)]],
                        device const half* vc [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        device const State* st [[buffer(4)]],
                        constant int& toff [[buffer(5)]],
                        device half* outh [[buffer(6)]],
                        device const int* tree [[buffer(7)]],
                        uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (st->done) return;
    threadgroup float qs[8][HD];
    threadgroup float S[8][ATT_ROWS_MAXL];
    threadgroup int prow[8][16];
    threadgroup int Lr[8];
    threadgroup float inv[8];
    const int h = int(tgp.x);
    const int r0 = toff + int(tgp.y) * 8;
    const int nr = min(8, st->n - r0);
    if (nr <= 0) return;
    const bool tm = st->pad4 != 0;
    const int p0 = st->pos;
    const int g = h / (NH / NKV);
    if (tid < 8) {
        const int r = int(tid);
        if (r < nr) {
            const int t = r0 + r;
            const int depth = tm ? tree[TR_DEPTH + t] : t;
            Lr[r] = p0 + depth + 1;
            if (tm) { int rr = t; for (int d = depth; d >= 0; --d) { prow[r][d] = rr; rr = rr > 0 ? tree[TR_PARENT + rr] : 0; } }
        } else {
            Lr[r] = 0;
        }
    }
    for (int i = int(tid); i < 8 * HD; i += 256) {
        const int r = i / HD, d = i % HD;
        qs[r][d] = r < nr ? q[(r0 + r) * H + h * HD + d] * 0.125f : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half* kb = kc + (ulong)g * st->maxctx * HD;
    device const half* vb = vc + (ulong)g * st->maxctx * HD;
    for (int r = 0; r < nr; ++r) {
        const int L = Lr[r];
        threadgroup const float4* q4 = (threadgroup const float4*)qs[r];
        for (int s = int(tid); s < L; s += 256) {
            const int slot = (tm && s >= p0) ? p0 + prow[r][s - p0] : s;
            device const half4* kr = (device const half4*)(kb + (ulong)slot * HD);
            float a = 0.f;
            for (int j = 0; j < 16; ++j) a += dot(float4(kr[j]), q4[j]);
            S[r][s] = a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(sg) < nr) {
        const int r = int(sg), L = Lr[r];
        float m = -INFINITY;
        for (int s = int(lane); s < L; s += 32) m = max(m, S[r][s]);
        m = simd_max(m);
        float sum = 0.f;
        for (int s = int(lane); s < L; s += 32) { const float e = exp(S[r][s] - m); S[r][s] = e; sum += e; }
        sum = simd_sum(sum);
        if (lane == 0) inv[r] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int r = int(tid) / 32, d = 2 * (int(tid) % 32);
    if (r < nr) {
        const int L = Lr[r];
        float o0 = 0.f, o1 = 0.f;
        for (int s = 0; s < L; ++s) {
            const int slot = (tm && s >= p0) ? p0 + prow[r][s - p0] : s;
            const float e = S[r][s];
            const float2 v = float2(*((device const half2*)(vb + (ulong)slot * HD + d)));
            o0 = fma(e, v.x, o0);
            o1 = fma(e, v.y, o1);
        }
        const int o = (r0 + r) * H + h * HD + d;
        out[o] = o0 / inv[r]; out[o + 1] = o1 / inv[r];
        outh[o] = half(o0 / inv[r]); outh[o + 1] = half(o1 / inv[r]);
    }
}


#if FMT == 0 || FMT == 1 || FMT == 2
// ---------------------------------------------------------------------------------------------
// q4 (group-64 affine, row-major) / q8 (group-32 symmetric) / f16 decode kernels, templated on T = tokens per pass (T = 1 is plain
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

// Packs a lane's 32 inputs as 16 fp16 pairs in the order the format's dot product consumes them:
// q4 pairs nibbles k and k+4 of each 8-weight word, q8/f16 take consecutive pairs.
static inline void pack_pairs(thread const float* xv, thread uint* xh, thread float& X) {
    X = 0.f;
#if FMT == 2
    for (int u = 0; u < 4; u++)
        for (int k = 0; k < 4; k++) {
            const half2 h = half2(half(xv[8 * u + k]), half(xv[8 * u + k + 4]));
            xh[4 * u + k] = as_type<uint>(h);
            X += float(h.x) + float(h.y);
        }
#else
    for (int i = 0; i < 16; i++) {
        const half2 h = half2(half(xv[2 * i]), half(xv[2 * i + 1]));
        xh[i] = as_type<uint>(h);
        X += float(h.x) + float(h.y);
    }
#endif
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
    if (lane < 28) {
        device const float4* ws = (device const float4*)(w + lane * 32);
        for (int i = 0; i < 8; i++) {
            const float4 t = float4(xv[4*i], xv[4*i+1], xv[4*i+2], xv[4*i+3]) * scale * ws[i];
            xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w;
        }
        pack_pairs(xv, xh, X);
    } else {
        X = 0.f;
        for (int i = 0; i < 16; i++) xh[i] = 0u;
    }
}

static inline void q4_chunk_plain(device const float* x, ushort lane, thread uint* xh, thread float& X) {
    if (lane < 28) {
        float xv[32];
        device const float4* xs = (device const float4*)(x + lane * 32);
        for (int i = 0; i < 8; i++) { const float4 t = xs[i]; xv[4*i] = t.x; xv[4*i+1] = t.y; xv[4*i+2] = t.z; xv[4*i+3] = t.w; }
        pack_pairs(xv, xh, X);
    } else {
        X = 0.f;
        for (int i = 0; i < 16; i++) xh[i] = 0u;
    }
}

#if FMT == 0
// f16: 32 weights (64 bytes) of one row chunk, T dot products with the fp16 inputs.
template<int T>
static inline void f16_dot32(thread const uint4 (&w)[4], thread const uint (&xh)[T][16], thread float* d) {
    for (int t = 0; t < T; t++) d[t] = 0.f;
    for (int j = 0; j < 16; j++) {
        const float2 wf = float2(as_type<half2>(w[j >> 2][j & 3]));
        for (int t = 0; t < T; t++) {
            const float2 xf = float2(as_type<half2>(xh[t][j]));
            d[t] = fma(wf.x, xf.x, fma(wf.y, xf.y, d[t]));
        }
    }
}

template<int T, int R>
static inline void q_rows(device const half* W, device const half2* SB, uint r0, thread const uint (&xh)[T][16],
                          thread const float* X, ushort lane, thread float (&v)[T][R]) {
    if (lane >= 28) { for (int t = 0; t < T; t++) for (int r = 0; r < R; r++) v[t][r] = 0.f; return; }
    for (int rb = 0; rb < R; rb += 2) {
        uint4 w[2][4];
        for (int b = 0; b < 2; b++) {
            device const uint4* p = (device const uint4*)(W + (ulong)(r0 + rb + b) * H + lane * 32);
            for (int q = 0; q < 4; q++) w[b][q] = p[q];
        }
        for (int b = 0; b < 2; b++) {
            float d[T];
            f16_dot32<T>(w[b], xh, d);
            for (int t = 0; t < T; t++) v[t][rb + b] = d[t];
        }
    }
}

template<int T>
static inline void q_chunk_down(device const half* W, device const half2* SB, uint row, int c,
                                thread const uint (&xh)[T][16], thread const float* X, thread float* out) {
    device const uint4* p = (device const uint4*)(W + (ulong)row * FF + c * 32);
    uint4 w[4];
    for (int q = 0; q < 4; q++) w[q] = p[q];
    f16_dot32<T>(w, xh, out);
}
#elif FMT == 1
// q8: 32 codes (8 words) of one 32-column group, T dot products with the fp16 inputs.
template<int T>
static inline void q8_dot32(uint4 w0, uint4 w1, thread const uint (&xh)[T][16], thread float* d) {
    for (int t = 0; t < T; t++) d[t] = 0.f;
    for (int j = 0; j < 8; j++) {
        const uint word = j < 4 ? w0[j] : w1[j - 4];
        const float4 c = float4(as_type<char4>(word));
        for (int t = 0; t < T; t++) {
            const float2 a = float2(as_type<half2>(xh[t][2 * j])), b = float2(as_type<half2>(xh[t][2 * j + 1]));
            d[t] = fma(c.x, a.x, fma(c.y, a.y, fma(c.z, b.x, fma(c.w, b.y, d[t]))));
        }
    }
}

template<int T, int R>
static inline void q_rows(device const char* W, device const half2* SB, uint r0, thread const uint (&xh)[T][16],
                          thread const float* X, ushort lane, thread float (&v)[T][R]) {
    if (lane >= 28) { for (int t = 0; t < T; t++) for (int r = 0; r < R; r++) v[t][r] = 0.f; return; }
    device const half* S = (device const half*)SB;
    for (int rb = 0; rb < R; rb += 4) {
        uint4 w0[4], w1[4];
        float sc[4];
        for (int b = 0; b < 4; b++) {
            device const uint4* p = (device const uint4*)(W + (ulong)(r0 + rb + b) * H + lane * 32);
            w0[b] = p[0]; w1[b] = p[1];
            sc[b] = float(S[(r0 + rb + b) * (H / 32) + lane]);
        }
        for (int b = 0; b < 4; b++) {
            float d[T];
            q8_dot32<T>(w0[b], w1[b], xh, d);
            for (int t = 0; t < T; t++) v[t][rb + b] = sc[b] * d[t];
        }
    }
}

// one 32-column chunk c of a K = 4864 row (down projection)
template<int T>
static inline void q_chunk_down(device const char* W, device const half2* SB, uint row, int c,
                                thread const uint (&xh)[T][16], thread const float* X, thread float* out) {
    device const uint4* p = (device const uint4*)(W + (ulong)row * FF + c * 32);
    float d[T];
    q8_dot32<T>(p[0], p[1], xh, d);
    const float s = float(((device const half*)SB)[row * (FF / 32) + c]);
    for (int t = 0; t < T; t++) out[t] = s * d[t];
}
#else
// 32 nibbles unpacked once, T dot products with the fp16 inputs (in-order fp32 accumulation).
template<int T>
static inline void q4_dot32(uint4 w, thread const uint (&xh)[T][16], thread float* d) {
    // xh holds the inputs as fp16 pairs in nibble order: pair 4u+k = (x[8u+k], x[8u+k+4]). A nibble
    // pair unpacks exactly to fp16 via (1024 + n) - 1024; per token, 8 weights = 4 half2 FMAs whose
    // partial sum goes to fp32 (8-term fp16 sums keep the rounding small).
    for (int t = 0; t < T; t++) d[t] = 0.f;
    const half2 k1024 = half2(1024.h);
    for (int u = 0; u < 4; ++u) {
        const uint v = w[u];
        const half2 n04 = as_type<half2>((v & 0x000F000Fu) | 0x64006400u) - k1024;
        const half2 n15 = as_type<half2>(((v >> 4) & 0x000F000Fu) | 0x64006400u) - k1024;
        const half2 n26 = as_type<half2>(((v >> 8) & 0x000F000Fu) | 0x64006400u) - k1024;
        const half2 n37 = as_type<half2>(((v >> 12) & 0x000F000Fu) | 0x64006400u) - k1024;
        for (int t = 0; t < T; t++) {
            half2 a = n04 * as_type<half2>(xh[t][4 * u]);
            a = fma(n15, as_type<half2>(xh[t][4 * u + 1]), a);
            a = fma(n26, as_type<half2>(xh[t][4 * u + 2]), a);
            a = fma(n37, as_type<half2>(xh[t][4 * u + 3]), a);
            d[t] += float(a.x) + float(a.y);
        }
    }
}

// R rows (from r0) of a K = 896 matrix for T tokens: lane-local partials v[t][r] before the reduction.
template<int T, int R>
static inline void q_rows(device const uchar* W, device const half2* SB, uint r0, thread const uint (&xh)[T][16],
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
            for (int t = 0; t < T; t++) v[t][rb + b] = sb[b].x * d[t] + sb[b].y * X[t];
        }
    }
}


template<int T>
static inline void q_chunk_down(device const uchar* W, device const half2* SB, uint row, int c,
                                thread const uint (&xh)[T][16], thread const float* X, thread float* out) {
    const uint4 w = *((device const uint4*)(W + (ulong)row * (FF / 2) + c * 16));
    const float2 sb = float2(SB[row * (FF / 64) + (c >> 1)]);
    float d[T];
    q4_dot32<T>(w, xh, d);
    for (int t = 0; t < T; t++) out[t] = sb.x * d[t] + sb.y * X[t];
}
#endif

// RMSNorm + QKV + bias + RoPE + KV write (32-row rotary blocks: rows i, i+16 = dims d, d+32).
// Grid (36), 128 threads: 4 simdgroups x 8 rows. Rows t = 0..T-1 of x at positions pos+t.
template<int T, int R>
kernel void k_attn_in_T(device const wtype* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
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
    q_rows<T, R>(W, SB, r0, xh, X, lane, v);
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
kernel void k_o_proj_T(device const wtype* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
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
    q_rows<T, R>(W, SB, r0, xh, X, lane, v);
    for (int t = 0; t < T; t++) {
        const float y = reduceR<R>(v[t], lane);
        if (lane % LPR == 0) x[t * H + r0 + lane / LPR] += y;
    }
}

// RMSNorm + gate/up + SiLU*up; stored rows in groups of 4 = gate[2q], gate[2q+1], up[2q], up[2q+1].
// Grid (608), 64 threads: 2 simdgroups x 8 rows (= 4 hidden units per simdgroup).
template<int T, int R>
kernel void k_mlp_in_T(device const wtype* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
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
    q_rows<T, R>(W, SB, r0, xh, X, lane, v);
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
kernel void k_mlp_out_T(device const wtype* W [[buffer(0)]], device const half2* SB [[buffer(1)]],
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
            float xv[32];
            device const float4* xs = (device const float4*)(hbuf + t * FF + c * 32);
            for (int i = 0; i < 8; i++) { const float4 v4 = xs[i]; xv[4*i] = v4.x; xv[4*i+1] = v4.y; xv[4*i+2] = v4.z; xv[4*i+3] = v4.w; }
            pack_pairs(xv, xh[t], X[t]);
        }
        for (int b = 0; b < R; b++) {
            float o[T];
            q_chunk_down<T>(W, SB, r0 + b, c, xh, X, o);
            for (int t = 0; t < T; t++) acc[t][b] += o[t];
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
kernel void k_lm_head_T(device const wtype* E [[buffer(0)]], device const half2* ESB [[buffer(1)]],
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
    q_rows<T, R>(E, ESB, r0, xh, X, lane, v);
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
template [[host_name("k_attn_in" SUF)]] kernel void k_attn_in_T<T, R>(device const wtype*, device const half2*, device const float*, device const float*, device const float*, device float*, device half*, device half*, device const float2*, device const State*, uint2, uint, ushort, ushort); \
template [[host_name("k_o_proj" SUF)]] kernel void k_o_proj_T<T, R>(device const wtype*, device const half2*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_mlp_in" SUF)]] kernel void k_mlp_in_T<T, R>(device const wtype*, device const half2*, device const float*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_mlp_out" SUF)]] kernel void k_mlp_out_T<T, R>(device const wtype*, device const half2*, device const float*, device float*, device const State*, uint2, ushort, ushort); \
template [[host_name("k_lm_head" SUF)]] kernel void k_lm_head_T<T, R>(device const wtype*, device const half2*, device const float*, device const float*, device float*, device int*, device const State*, device float*, constant int&, uint, ushort, ushort);
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
        for (int u = 0; u < 4; u++)
            for (int k = 0; k < 4; k++) { const float2 f2 = float2(as_type<half2>(xh[4 * u + k])); xf[8 * u + k] = f2.x; xf[8 * u + k + 4] = f2.y; }
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
kernel void k_lmh_exact_T(device const wtype* E [[buffer(0)]], device const half2* ESB [[buffer(1)]],
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
    q_rows<T, 8>(E, ESB, r0, xh, X, lane, v);
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
template [[host_name("k_lmh_exact_" #T)]] kernel void k_lmh_exact_T<T>(device const wtype*, device const half2*, device const float*, device const float*, device float*, device int*, device const State*, device const float*, device const float*, uint, ushort, ushort);
INST_LMH(1)
INST_LMH(2)
INST_LMH(3)
INST_LMH(4)



#endif

#pragma METAL fp math_mode(fast)

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
#if FMT == 0 || FMT == 1 || FMT == 2
template<int K, int MODE, int TT, int NS, int BKP, typename TX>
inline void gemm3_core(TX tX, device const wtype* W, device const half2* SB,
                       device const float* bias, device float* outf, device half* outh,
                       device half* kc, device half* vc, device const float2* rope, device const State* st,
                       device const int* tree,
                       threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    // Each simdgroup takes a contiguous K range (whole cache lines of its 32 rows): K = 896 -> 7
    // steps of 32 columns, K = 4864 -> 19 steps of 64. Lane = row; w = (16 s)(1 + n/16) + (b - 16 s).
    constexpr int BK = BKP;
    constexpr int NSTEP = K / BK / NS;
    const int n = st->n;
    const uint r0 = tgp.x * 32;
    const int t0 = tgp.y * TT;
    threadgroup half* ws = Wsg + sg * 32 * BK;

    auto tW = tensor(ws, dextents<int, 2>(BK, 32), array<int, 2>({1, BK}));
    constexpr auto desc = matmul2d_descriptor(TT, 32, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;
#if FMT == 0
    // f16 weights are used in place: the right operand is a tensor view of the 32 rows in device memory.
    auto tWd = tensor((device half*)(W + (ulong)r0 * K), dextents<int, 2>(K, 32), array<int, 2>({1, K}));
    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = (sg * NSTEP + i) * BK;
        auto a = tX.template slice<BK, TT>(k0, 0);
        auto b = tWd.template slice<BK, 32>(k0, 0);
        op.run(a, b, cT);
    }
    const uint k0 = 0;   // the shared loop tail below is dead for f16
    if (false) {
#elif FMT == 1
    device const char* wrow = W + (ulong)(r0 + lane) * K;
    device const half* srow = (device const half*)SB + (r0 + lane) * (K / 32);
    threadgroup half4* dst = (threadgroup half4*)(ws + lane * BK);
    for (int i = 0; i < NSTEP; i++) {
        const uint k0 = (sg * NSTEP + i) * BK;
        for (int q = 0; q < BK / 32; ++q) {
            const half s = srow[k0 / 32 + q];
            device const uint4* src = (device const uint4*)(wrow + k0 + q * 32);
            const uint4 a = src[0], b = src[1];
            for (int j = 0; j < 4; ++j) {
                dst[q * 8 + j] = half4(as_type<char4>(a[j])) * s;
                dst[q * 8 + 4 + j] = half4(as_type<char4>(b[j])) * s;
            }
        }
#else
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
#endif
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
            const int rpos = st->pad4 ? st->pos + tree[TR_DEPTH + t0 + t] : pos;   // tree rows: rotate by depth
            const int d = (tgp.x & 1) * 16 + i;
            if (head < NH + NKV) {
                const float2 cs = rope[rpos * 32 + d];
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
    } else if (MODE == 3) {
        // lm_head: per token, the max over this tile's 32 vocabulary rows (smallest id on ties),
        // as the [t][LMH_TGS] partials the commit kernels reduce. outh carries the index buffer.
        device int* pidx = (device int*)outh;
        for (int t = sg; t < TT; t += NS) {
            if (t0 + t >= n) continue;
            float v = 0.f;
            for (int q = 0; q < NS; ++q) v += Os[q * TT * 32 + t * 32 + lane];
            const float m = simd_max(v);
            const int idx = simd_min(v == m ? int(r0 + lane) : 0x7fffffff);
            if (lane == 0) { outf[(t0 + t) * LMH_TGS + tgp.x] = m; pidx[(t0 + t) * LMH_TGS + tgp.x] = idx; }
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

template<int K, int MODE, int TT, int NS, int BKP = (K == H ? 32 : 64)>
inline void gemm3_body(device half* X, device const wtype* W, device const half2* SB,
                       device const float* bias, device float* outf, device half* outh,
                       device half* kc, device half* vc, device const float2* rope, device const State* st,
                       device const int* tree,
                       threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    auto tX = tensor(X + (ulong)(tgp.y * TT) * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    gemm3_core<K, MODE, TT, NS, BKP>(tX, W, SB, bias, outf, outh, kc, vc, rope, st, tree, Wsg, Os, tgp, tid, lane, sg);
}

// K = H GEMMs with the RMSNorm folded in: each threadgroup normalizes its TT rows of x into
// threadgroup memory (simdgroup per row, no separate norm dispatch) and feeds them to matmul2d.
template<int MODE, int TT, int NS>
inline void gemm3f_body(device const float* x, device const float* wn, threadgroup half* Xs,
                        device const wtype* W, device const half2* SB,
                        device const float* bias, device float* outf, device half* outh,
                        device half* kc, device half* vc, device const float2* rope, device const State* st,
                        device const int* tree,
                        threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    const int t0 = tgp.y * TT, n = st->n;
    for (int t = int(sg); t < TT; t += NS) {
        if (t0 + t < n) {
            device const float* xr = x + (t0 + t) * H;
            float ss = 0.f;
            for (int i = int(lane); i < H; i += 32) { const float v = xr[i]; ss += v * v; }
            ss = simd_sum(ss);
            const float scale = rsqrt(ss / float(H) + EPS);
            for (int i = int(lane); i < H; i += 32) Xs[t * H + i] = half(xr[i] * scale * wn[i]);
        } else {
            for (int i = int(lane); i < H; i += 32) Xs[t * H + i] = 0.h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tX = tensor(Xs, dextents<int, 2>(H, TT), array<int, 2>({1, H}));
    gemm3_core<H, MODE, TT, NS, 32>(tX, W, SB, bias, outf, outh, kc, vc, rope, st, tree, Wsg, Os, tgp, tid, lane, sg);
}

#define GEMM3(NAME, K, MODE, TT, NS) \
kernel void NAME(device half* X [[buffer(0)]], device const wtype* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 device const int* tree [[buffer(12)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TT) >= st->n) return; \
    constexpr int WB = NS * 32 * (K == H ? 32 : 64) * 2, OB = NS * TT * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm3_body<K, MODE, TT, NS>(X, W, SB, bias, outf, outh, kc, vc, rope, st, tree, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
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

#define GEMM3F(NAME, MODE, TT, NS) \
kernel void NAME(device half* X [[buffer(0)]], device const wtype* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 device const float* xin [[buffer(10)]], device const float* wn [[buffer(11)]], \
                 device const int* tree [[buffer(12)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TT) >= st->n) return; \
    constexpr int WB = NS * 32 * 32 * 2, OB = NS * TT * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    threadgroup half Xs[TT * H]; \
    gemm3f_body<MODE, TT, NS>(xin, wn, Xs, W, SB, bias, outf, outh, kc, vc, rope, st, tree, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
GEMM3F(k_g3f_qkv_8, 0, 8, 4)
GEMM3F(k_g3f_gu_8, 2, 8, 4)

#if FMT == 2
// Prefill / wide-verify GEMM v4 (q4, experiments): RT-row tiles (16 or 32 rows), LPR = 32 / RT
// lanes per row, each lane dequantizing 32 weights (16 bytes) per step; NS simdgroups split K.
template<int K, int MODE, int TT, int NS, int RT>
inline void gemm4_body(device half* X, device const uchar* W, device const half2* SB,
                       device float* outf, device half* outh, device const State* st,
                       threadgroup half* Wsg, threadgroup float* Os, uint2 tgp, uint tid, uint lane, uint sg) {
    constexpr int LPR = 32 / RT;
    constexpr int BK = 32 * LPR;
    constexpr int NSTEP = K / BK / NS;
    static_assert(NSTEP * BK * NS == K, "K must split evenly");
    const int n = st->n;
    const uint r0 = tgp.x * RT;
    const int t0 = tgp.y * TT;
    threadgroup half* ws = Wsg + sg * RT * BK;
    auto tX = tensor(X + (ulong)t0 * K, dextents<int, 2>(K, TT), array<int, 2>({1, K}));
    auto tW = tensor(ws, dextents<int, 2>(BK, RT), array<int, 2>({1, BK}));
    constexpr auto desc = matmul2d_descriptor(TT, RT, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto cT = op.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    for (int i = 0; i < cT.get_capacity(); i++) if (cT.is_valid_element(i)) cT[i] = 0.f;
    const int row = lane % RT, part = lane / RT;
    device const uchar* wrow = W + (ulong)(r0 + row) * (K / 2);
    device const half2* sbrow = SB + (r0 + row) * (K / 64);
    threadgroup half2* dst = (threadgroup half2*)(ws + row * BK + part * 32);
    for (int i = 0; i < NSTEP; i++) {
        const uint kb = (sg * NSTEP + i) * BK;
        const uint kl = kb + part * 32;
        const half2 sb = sbrow[kl / 64];
        const half2 mul = half2(half(16.f * float(sb.x)));
        const half2 add = half2(half(float(sb.y) - 16.f * float(sb.x)));
        const uint4 wv = *((device const uint4*)(wrow + kl / 2));
        for (int u = 0; u < 4; ++u) {
            const uint v = wv[u];   // nibbles k = 0..7 of this word: pairs (0,4) (1,5) (2,6) (3,7)
            const half2 p04 = as_type<half2>(((v << 6) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p15 = as_type<half2>(((v << 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p26 = as_type<half2>(((v >> 2) & 0x03C003C0u) | 0x3C003C00u);
            const half2 p37 = as_type<half2>(((v >> 6) & 0x03C003C0u) | 0x3C003C00u);
            dst[u * 4 + 0] = fma(half2(p04.x, p15.x), mul, add);
            dst[u * 4 + 1] = fma(half2(p26.x, p37.x), mul, add);
            dst[u * 4 + 2] = fma(half2(p04.y, p15.y), mul, add);
            dst[u * 4 + 3] = fma(half2(p26.y, p37.y), mul, add);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tX.template slice<BK, TT>(kb, 0);
        op.run(a, tW, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // Os aliases the weight tiles
    auto tO = tensor(Os + sg * TT * RT, dextents<int, 2>(RT, TT), array<int, 2>({1, RT}));
    cT.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int NT = NS * 32;
    if (MODE == 1) {
        for (int w = tid; w < TT * RT; w += NT) {
            const int t = w / RT, r = w % RT;
            if (t0 + t >= n) continue;
            float v = 0.f;
            for (int q = 0; q < NS; ++q) v += Os[q * TT * RT + w];
            outf[(t0 + t) * H + r0 + r] += v;
        }
    } else {
        for (int w = tid; w < TT * (RT / 4); w += NT) {
            const int t = w / (RT / 4), b = w % (RT / 4);
            if (t0 + t >= n) continue;
            float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
            for (int q = 0; q < NS; ++q) {
                threadgroup const float* o = Os + q * TT * RT + t * RT + 4 * b;
                g0 += o[0]; g1 += o[1]; u0 += o[2]; u1 += o[3];
            }
            const int u = (r0 / 4 + b) * 2;
            outh[(t0 + t) * FF + u] = half(g0 / (1.f + exp(-g0)) * u0);
            outh[(t0 + t) * FF + u + 1] = half(g1 / (1.f + exp(-g1)) * u1);
        }
    }
}

#define GEMM4(NAME, K, MODE, TT, NS, RT) \
kernel void NAME(device half* X [[buffer(0)]], device const uchar* W [[buffer(1)]], \
                 device const half2* SB [[buffer(2)]], device const float* bias [[buffer(3)]], \
                 device float* outf [[buffer(4)]], device half* outh [[buffer(5)]], \
                 device half* kc [[buffer(6)]], device half* vc [[buffer(7)]], \
                 device const float2* rope [[buffer(8)]], device const State* st [[buffer(9)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done || (int)(tgp.y * TT) >= st->n) return; \
    constexpr int WB = NS * 1024 * 2, OB = NS * TT * RT * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm4_body<K, MODE, TT, NS, RT>(X, W, SB, outf, outh, st, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
#define GEMM4_ALL(TT) \
GEMM4(k_g4_o_##TT##_16_2, H, 1, TT, 2, 16) \
GEMM4(k_g4_o_##TT##_16_7, H, 1, TT, 7, 16) \
GEMM4(k_g4_o_##TT##_32_4, H, 1, TT, 4, 32) \
GEMM4(k_g4_o_##TT##_32_7, H, 1, TT, 7, 32) \
GEMM4(k_g4_gu_##TT##_16_2, H, 2, TT, 2, 16) \
GEMM4(k_g4_gu_##TT##_16_7, H, 2, TT, 7, 16) \
GEMM4(k_g4_gu_##TT##_32_4, H, 2, TT, 4, 32) \
GEMM4(k_g4_gu_##TT##_32_7, H, 2, TT, 7, 32) \
GEMM4(k_g4_down_##TT##_16_4, FF, 1, TT, 4, 16) \
GEMM4(k_g4_down_##TT##_16_2, FF, 1, TT, 2, 16) \
GEMM4(k_g4_down_##TT##_32_4, FF, 1, TT, 4, 32) \
GEMM4(k_g4_down_##TT##_32_8, FF, 1, TT, 8, 32)
GEMM4_ALL(8)
GEMM4_ALL(16)
#endif

// Tied lm_head on the Neural Accelerators for the rows of a wide verify pass: X = half(final
// RMSNorm(x)), grid (VOCAB / 32), per-tile argmax partials like k_lm_head_T.
#define GEMM3L(NAME, TT) \
kernel void NAME(device half* X [[buffer(0)]], device const wtype* E [[buffer(1)]], \
                 device const half2* ESB [[buffer(2)]], device float* pval [[buffer(3)]], \
                 device int* pidx [[buffer(4)]], device const State* st [[buffer(5)]], \
                 uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    if (st->done) return; \
    constexpr int WB = 4 * 32 * 64 * 2, OB = 4 * TT * 32 * 4; \
    threadgroup float smem[(WB > OB ? WB : OB) / 4]; \
    gemm3_body<H, 3, TT, 4>(X, E, ESB, pval, pval, (device half*)pidx, nullptr, nullptr, nullptr, st, nullptr, (threadgroup half*)smem, smem, tgp, tid, lane, sg); \
}
GEMM3L(k_g3_lmh_8, 8)
GEMM3L(k_g3_lmh_16, 16)
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
