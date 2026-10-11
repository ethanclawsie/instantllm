"""GPU-style n-gram drafter: an exact Python model of the engine's NGramTable (Sources/Engine/
NGram.swift) + ng_next/k_commit_spec drafting (Sources/Engine/Shaders.swift), plus GPU-friendly
variants, and a writer for the engine's 16 MB table layout.

Engine semantics (as of 2026-10-11 00:45):
  * contexts: the last n = 1..6 committed output tokens, with -1 for positions before the output
    start; each context stores its most frequent next token (ties -> smaller id), the count and the
    total (u16, saturating)
  * ng_next: for n = 6..1, the first context found decides: if count >= minc and 2*count >= total
    return its next token, otherwise STOP drafting; unknown contexts back off to n-1
  * a pass drafts up to tspec-1 tokens by chaining ng_next
  * table: 2^20 slots x 16 bytes {u64 key, i32 next, u16 count, u16 total}, open addressing from
    slot (key >> 12) & (2^20-1), at most 16 linear probes; key 0 = empty
"""
import struct
from collections import defaultdict

SLOTS = 1 << 20
MAXN = 6
STOPS = (151645, 151643)
M64 = (1 << 64) - 1


def ng_key(ctx):
    h = 0xcbf29ce484222325 ^ len(ctx)
    for t in ctx:
        h ^= ((t & 0xffffffff) + 0x9e3779b97f4a7c15) & M64
        h = (h * 0x100000001b3) & M64
        h ^= h >> 29
    return h | 1


class GPUTable:
    def __init__(self, seqs, maxn=MAXN, weights=None, keep_top=2):
        counts = defaultdict(lambda: defaultdict(int))
        for si, y in enumerate(seqs):
            w = 1 if weights is None else weights[si]
            s = [-1] * maxn + list(y)
            for i in range(maxn, len(s)):
                for n in range(1, maxn + 1):
                    counts[tuple(s[i - n:i])][s[i]] += w
        self.maxn = maxn
        self.tab = {}
        for ctx, d in counts.items():
            tot = sum(d.values())
            top = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:keep_top]
            self.tab[ctx] = (top, tot)

    def entry(self, ctx):
        return self.tab.get(tuple(ctx))

    def write_engine_format(self, path, min_total=1):
        """Engine layout (NGramTable.slots * 16 bytes). Contexts with total < min_total are dropped."""
        buf = bytearray(SLOTS * 16)
        used = dropped = 0
        # deterministic insertion order: by key
        items = sorted(((ng_key(list(c)), c, v) for c, v in self.tab.items() if v[1] >= min_total))
        for k, c, (top, tot) in items:
            tok, cnt = top[0]
            slot = (k >> 12) & (SLOTS - 1)
            for _ in range(16):
                if struct.unpack_from("<Q", buf, slot * 16)[0] == 0:
                    struct.pack_into("<QiHH", buf, slot * 16, k, tok, min(cnt, 65535), min(tot, 65535))
                    used += 1
                    break
                slot = (slot + 1) & (SLOTS - 1)
            else:
                dropped += 1
        with open(path, "wb") as f:
            f.write(bytes(buf))
        return used, dropped


class GPUDrafter:
    """Chained drafting with the engine's rule and optional GPU-friendly extensions.
    thr:        P(best) threshold (engine: 0.5, i.e. 2*count >= total); minc: engine 1
    on_weak:    "stop" (engine) | "backoff" (try shorter contexts when the longest is unconfident)
    self_min:   >0 enables a self-repetition lookup: if the last self_min+ output tokens occurred
                earlier in the output, continue with what followed (checked before the n-gram)
    prompt_min: >0 enables prompt lookup when the n-gram gives nothing: longest match (>= prompt_min)
                of the output suffix inside the prompt tokens, continue with the prompt's next token
    """

    def __init__(self, table, thr=0.5, minc=1, on_weak="stop", self_min=0, prompt_min=0, self_maxn=8):
        self.t, self.thr, self.minc, self.on_weak = table, thr, minc, on_weak
        self.self_min, self.prompt_min, self.self_maxn = self_min, prompt_min, self_maxn

    def begin(self, pids, text=None):
        self.P = list(pids)

    def _ng(self, hist):
        L = len(hist)
        for n in range(min(self.t.maxn, L), 0, -1):
            e = self.t.entry(hist[L - n:])
            if e is None:
                continue
            (tok, cnt), tot = e[0][0], e[1]
            cnt = min(cnt, 65535)
            tot = min(tot, 65535)
            if cnt >= self.minc and cnt >= self.thr * tot:
                return tok
            if self.on_weak == "stop":
                return None
        return None

    def _self(self, o):
        L = len(o)
        for n in range(min(self.self_maxn, L - 1), self.self_min - 1, -1):
            pat = o[L - n:]
            for j in range(L - n - 1, -1, -1):
                if o[j:j + n] == pat:
                    return o[j + n]
        return None

    def _prompt(self, o):
        P = self.P
        L = len(o)
        for n in range(min(4, L), self.prompt_min - 1, -1):
            pat = o[L - n:]
            for j in range(0, len(P) - n):
                if P[j:j + n] == pat:
                    return P[j + n]
        return None

    def propose(self, tail, out, K):
        o = list(out)
        hist = [-1] * self.t.maxn + o
        d = []
        while len(d) < K:
            tok = None
            if self.self_min > 0 and len(o) > self.self_min:
                tok = self._self(o)
            if tok is None:
                tok = self._ng(hist)
            if tok is None and self.prompt_min > 0 and o:
                tok = self._prompt(o)
            if tok is None:
                break
            d.append(tok)
            o.append(tok)
            hist.append(tok)
            if tok in STOPS:
                break
        return d


class RetrievalGPUDrafter(GPUDrafter):
    """GPU-friendly retrieval: at request start the CPU picks the top-k training prompts by TF-IDF
    (ildraft.prompt_words) and uploads their outputs; per drafted token the GPU takes, in order:
      1. self-repetition (>= self_min matched tokens),
      2. the neighbour vote: every neighbour whose output contains the longest suffix of
         (-1 pads + output) of length >= nb_min votes for the token after its first match;
         the token with most votes wins (ties: higher-ranked neighbour),
      3. the n-gram table (engine lookup, threshold thr),
      4. prompt lookup."""

    def __init__(self, table, index_recs, idx=None, k=8, nb_min=2, nb_maxn=12, **kw):
        super().__init__(table, **kw)
        self.recs = index_recs
        self.k, self.nb_min, self.nb_maxn = k, nb_min, nb_maxn
        self.idx = idx

    def begin(self, pids, text=None):
        super().begin(pids, text)
        self.nb = []
        if text is not None and self.idx is not None:
            for j, s in self.idx.query(text, self.k):
                self.nb.append([-1] * self.t.maxn + list(self.recs[j]["y"]))

    def _nb(self, hist):
        L = len(hist)
        votes = {}
        best_n = 0
        for rank, seq in enumerate(self.nb):
            for n in range(min(self.nb_maxn, L), self.nb_min - 1, -1):
                pat = hist[L - n:]
                hit = -1
                for j in range(0, len(seq) - n):
                    if seq[j:j + n] == pat:
                        hit = j + n
                        break
                if hit >= 0:
                    if n > best_n:
                        best_n = n
                        votes = {}
                    if n == best_n:
                        v = votes.setdefault(seq[hit], [0, rank])
                        v[0] += 1
                    break
        if not votes:
            return None
        return min(votes.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[0]

    def propose(self, tail, out, K):
        o = list(out)
        hist = [-1] * self.t.maxn + o
        d = []
        while len(d) < K:
            tok = None
            if self.self_min > 0 and len(o) > self.self_min:
                tok = self._self(o)
            if tok is None:
                tok = self._nb(hist)
            if tok is None:
                tok = self._ng(hist)
            if tok is None and self.prompt_min > 0 and o:
                tok = self._prompt(o)
            if tok is None:
                break
            d.append(tok)
            o.append(tok)
            hist.append(tok)
            if tok in STOPS:
                break
        return d
