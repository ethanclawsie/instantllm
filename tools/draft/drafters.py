"""Drafters for exact speculative decoding (exploration versions, pure Python).

Every drafter implements
    begin(prompt_ids)                 once per request
    propose(tail, out, K) -> [ids]    up to K draft tokens continuing TAIL + out
"""
from collections import defaultdict

from common import STOPS, TAIL


class NGramTable:
    """counts[n][ctx tuple of length n] -> {next: count}, built from TAIL + y for every
    training sequence (so position 0 is predicted from the template tail)."""

    def __init__(self, seqs, max_order=6, weights=None):
        self.N = max_order
        self.counts = [None] + [defaultdict(lambda: defaultdict(int)) for _ in range(max_order)]
        self.uni = defaultdict(int)
        for si, y in enumerate(seqs):
            w = 1 if weights is None else weights[si]
            s = TAIL + list(y)
            for i in range(len(TAIL), len(s)):
                nxt = s[i]
                self.uni[nxt] += w
                for n in range(1, max_order + 1):
                    if i - n < 0:
                        break
                    self.counts[n][tuple(s[i - n:i])][nxt] += w
        self.best = [None] + [{} for _ in range(max_order)]
        for n in range(1, max_order + 1):
            for ctx, d in self.counts[n].items():
                tot = sum(d.values())
                b = min(d.items(), key=lambda kv: (-kv[1], kv[0]))
                self.best[n][ctx] = (b[0], b[1], tot)

    def lookup(self, ctx, min_count=1, max_order=None, min_order=1):
        """Longest-match backoff. Returns (token, count_best, total, order) or None."""
        N = min(self.N, max_order or self.N)
        L = len(ctx)
        for n in range(min(N, L), min_order - 1, -1):
            e = self.best[n].get(tuple(ctx[L - n:]))
            if e is not None and e[2] >= min_count:
                return e[0], e[1], e[2], n
        return None

    def dist(self, ctx, n):
        L = len(ctx)
        if n > L:
            return None
        return self.counts[n].get(tuple(ctx[L - n:]))


class NGramDrafter:
    """Greedy chain over a global n-gram table.
    tau: stop when P(best|ctx) < tau.  cum: stop when the product of confidences < cum.
    min_count: contexts seen fewer times are skipped (backoff to shorter)."""

    def __init__(self, table, tau=0.0, cum=0.0, min_count=1, max_order=None, min_order=1):
        self.t = table
        self.tau, self.cum, self.min_count = tau, cum, min_count
        self.max_order, self.min_order = max_order, min_order

    def begin(self, pids, text=None):
        pass

    def propose(self, tail, out, K):
        ctx = list(tail) + list(out)
        d = []
        pc = 1.0
        while len(d) < K:
            e = self.t.lookup(ctx, self.min_count, self.max_order, self.min_order)
            if e is None:
                break
            tok, cb, tot, n = e
            p = cb / tot
            if p < self.tau:
                break
            pc *= p
            if pc < self.cum and d:
                break
            d.append(tok)
            ctx.append(tok)
            if tok in STOPS:
                break
        return d


def suffix_match(hay, ctx, max_n, min_n):
    """Longest suffix of ctx (length in [min_n, max_n]) that occurs in hay at a position
    followed by at least one token. Returns (start of continuation, matched length) of the
    LAST (most recent) occurrence, or None."""
    L = len(ctx)
    for n in range(min(max_n, L), min_n - 1, -1):
        pat = ctx[L - n:]
        for j in range(len(hay) - n - 1, -1, -1):
            if hay[j:j + n] == pat:
                return j + n, n
    return None


class PromptLookupDrafter:
    """Classic prompt lookup: match the context suffix in prompt + output so far."""

    def __init__(self, max_n=3, min_n=1, use_output=True):
        self.max_n, self.min_n, self.use_output = max_n, min_n, use_output

    def begin(self, pids, text=None):
        self.p = list(pids)

    def propose(self, tail, out, K):
        ctx = list(out)
        if not ctx:
            return []
        hay = self.p + (list(out[:-1]) if self.use_output else [])
        # match against prompt, and against earlier output (excluding the trivial self match)
        m = suffix_match(self.p, ctx, self.max_n, self.min_n)
        if m is None and self.use_output and len(ctx) > 1:
            m2 = suffix_match(ctx[:-1], ctx, self.max_n, self.min_n)
            if m2 is not None:
                st, n = m2
                return ctx[st:st + K]
            return []
        if m is None:
            return []
        st, n = m
        return self.p[st:st + K]


class ComboDrafter:
    """Per position: prefer a confident long n-gram match; otherwise a prompt/self match;
    otherwise the n-gram fallback. Chains token by token so sources can alternate."""

    def __init__(self, table, tau=0.0, min_count=1, pl_max=4, pl_min=1, ng_strong=3,
                 ng_strong_p=0.5, use_prompt=True, use_self=True):
        self.t = table
        self.tau, self.min_count = tau, min_count
        self.pl_max, self.pl_min = pl_max, pl_min
        self.ng_strong, self.ng_strong_p = ng_strong, ng_strong_p
        self.use_prompt, self.use_self = use_prompt, use_self

    def begin(self, pids, text=None):
        self.p = list(pids)

    def _pl(self, ctx_out):
        best = None
        if self.use_prompt and ctx_out:
            m = suffix_match(self.p, ctx_out, self.pl_max, self.pl_min)
            if m is not None:
                best = (self.p[m[0]], m[1])
        if self.use_self and len(ctx_out) > 1:
            m = suffix_match(ctx_out[:-1], ctx_out, self.pl_max, self.pl_min)
            if m is not None and (best is None or m[1] > best[1]):
                best = (ctx_out[m[0]], m[1])
        return best

    def propose(self, tail, out, K):
        ctx = list(tail) + list(out)
        o = list(out)
        d = []
        while len(d) < K:
            e = self.t.lookup(ctx, self.min_count)
            pl = self._pl(o)
            tok = None
            if e is not None and e[3] >= self.ng_strong and e[1] / e[2] >= self.ng_strong_p:
                tok = e[0]
            elif pl is not None:
                tok = pl[0]
            elif e is not None and e[1] / e[2] >= self.tau:
                tok = e[0]
            if tok is None:
                break
            d.append(tok)
            ctx.append(tok)
            o.append(tok)
            if tok in STOPS:
                break
        return d


# ------------------------------------------------------------------------------------------
# prompt-similarity retrieval

import math
import re

_WRE = re.compile(r"[a-z0-9]+")


def prompt_words(text):
    return _WRE.findall(text.lower())


class PromptIndex:
    """TF-IDF (binary tf) cosine over lowercase alphanumeric words of training prompts."""

    def __init__(self, recs):
        self.recs = recs
        df = defaultdict(int)
        self.docs = []
        for r in recs:
            ws = set(prompt_words(r["prompt"]))
            self.docs.append(ws)
            for w in ws:
                df[w] += 1
        N = len(recs)
        self.idf = {w: math.log((N + 1) / (c + 0.5)) for w, c in df.items()}
        self.post = defaultdict(list)
        self.norm = []
        for j, ws in enumerate(self.docs):
            self.norm.append(math.sqrt(sum(self.idf[w] ** 2 for w in ws)) or 1.0)
            for w in ws:
                self.post[w].append(j)

    def query(self, text, k):
        ws = set(prompt_words(text))
        sc = defaultdict(float)
        qn = math.sqrt(sum(self.idf.get(w, 0.0) ** 2 for w in ws)) or 1.0
        for w in ws:
            iw = self.idf.get(w)
            if iw is None:
                continue
            for j in self.post[w]:
                sc[j] += iw * iw
        top = sorted(sc.items(), key=lambda kv: -kv[1])[:k]
        return [(j, s / (qn * self.norm[j])) for j, s in top]


def longest_suffix_match_end(hay, ctx, max_n):
    """Longest n <= max_n such that ctx[-n:] occurs in hay followed by a token; returns
    (n, index of continuation) preferring the earliest occurrence, or (0, -1)."""
    L = len(ctx)
    for n in range(min(max_n, L), 0, -1):
        pat = ctx[L - n:]
        for j in range(0, len(hay) - n):
            if hay[j:j + n] == pat:
                return n, j + n
    return 0, -1


class RetrievalDrafter:
    """Neighbours = top-k training prompts by TF-IDF cosine. Per drafted position, each
    neighbour votes for the token following its longest suffix match of the context
    (weight sim^a * match_len^b, only if match_len >= m_min); the global n-gram adds
    lam * P(t|ctx). Chain while the winner's vote share >= tau."""

    def __init__(self, index, table, k=8, max_n=8, m_min=2, a=1.0, b=1.0, lam=0.5, tau=0.0,
                 ng_min_count=1, use_prompt=True, lam_p=0.5, prompt_text_of=None, lam_s=0.0, s_min=2):
        self.ix, self.t = index, table
        self.k, self.max_n, self.m_min, self.a, self.b = k, max_n, m_min, a, b
        self.lam, self.tau, self.ng_min_count = lam, tau, ng_min_count
        self.use_prompt, self.lam_p = use_prompt, lam_p
        self.prompt_text_of = prompt_text_of or {}
        self.lam_s, self.s_min = lam_s, s_min

    def begin(self, pids, text=None):
        self.p = list(pids)
        self.nb = []
        if text is not None:
            for j, s in self.ix.query(text, self.k):
                self.nb.append((s, TAIL + self.ix.recs[j]["y"]))

    def scores(self, ctx, out):
        sc = defaultdict(float)
        for s, hay in self.nb:
            n, at = longest_suffix_match_end(hay, ctx, self.max_n)
            if n >= self.m_min:
                sc[hay[at]] += (s ** self.a) * (n ** self.b)
        if self.lam > 0:
            e = self.t.lookup(ctx, self.ng_min_count)
            if e is not None:
                dist = self.t.counts[e[3]][tuple(ctx[len(ctx) - e[3]:])]
                tot = e[2]
                for tok, c in dist.items():
                    sc[tok] += self.lam * c / tot
        if self.use_prompt and out:
            m = suffix_match(self.p, out, 4, 1)
            if m is not None:
                sc[self.p[m[0]]] += self.lam_p * m[1]
        if self.lam_s > 0 and len(out) > self.s_min:
            m = suffix_match(out[:-1], out, 16, self.s_min)
            if m is not None:
                sc[out[m[0]]] += self.lam_s * m[1]
        return sc

    def propose(self, tail, out, K):
        ctx = list(tail) + list(out)
        o = list(out)
        d = []
        while len(d) < K:
            sc = self.scores(ctx, o)
            if not sc:
                break
            tok, v = min(sc.items(), key=lambda kv: (-kv[1], kv[0]))
            if v / sum(sc.values()) < self.tau:
                break
            d.append(tok)
            ctx.append(tok)
            o.append(tok)
            if tok in STOPS:
                break
        return d
