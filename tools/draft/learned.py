"""Learned candidate scorer for drafting (exploration version).

At each draft position, candidates come from four cheap sources:
  NG  the global n-gram table (top tokens of every matched order 1..N)
  SF  self lookup: token after the longest earlier match of the output suffix
  PR  prompt lookup: token after the longest match of the output suffix in the prompt
  NB  neighbours: for each of the top-k training prompts (TF-IDF cosine), the token after the
      longest match of the context suffix in that neighbour's output
Each candidate gets a feature vector; a conditional logit (softmax over candidates + a null
option with score 0) is trained on group-disjoint folds of TRAIN. Drafting picks the argmax
candidate and stops when its probability falls below tau.
"""
import math
import re
from collections import defaultdict

import numpy as np

from common import STOPS, TAIL
from drafters import NGramTable, PromptIndex

MAXN_NB = 12     # max suffix length matched in neighbours / self
TOPM = 3         # candidates taken from each n-gram order


class SeqIndex:
    """n-gram -> first continuation index, for one sequence (orders 1..MAXN_NB)."""
    __slots__ = ("seq", "d")

    def __init__(self, seq):
        self.seq = seq
        d = {}
        L = len(seq)
        for n in range(1, MAXN_NB + 1):
            for i in range(0, L - n):
                key = tuple(seq[i:i + n])
                if key not in d:
                    d[key] = i + n
        self.d = d

    def match(self, ctx):
        L = len(ctx)
        for n in range(min(MAXN_NB, L), 0, -1):
            j = self.d.get(tuple(ctx[L - n:]))
            if j is not None:
                return n, self.seq[j]
        return 0, None


FEATS = [
    "bias",
    "ng_p_long", "ng_top_long", "ng_ord_long", "ng_lcnt_long", "ng_p_supp3", "ng_p_supp10",
    "ng_p1", "ng_p2", "ng_p3", "ng_p4", "ng_p5", "ng_p6",
    "sf_m", "sf_any", "sf_m_ge3",
    "pr_m", "pr_any", "pr_m_ge2",
    "nb_sum", "nb_msum", "nb_cnt", "nb_maxm", "nb_top1", "nb_top1_m",
    "is_stop", "in_prompt",
    "agree_ng_nb", "agree_sf_nb", "ng_ent_low",
    "cp_tok", "cp_prop", "cp_ctx1", "cp_ctx2", "cp_unused", "cp_first_unused", "cp_spacevar",
    "ng_rank2", "ng_rank3plus",
]
FI = {f: i for i, f in enumerate(FEATS)}


class CopyStats:
    """From training records: per-token copy propensity P(t in output | t in prompt) and
    per-context copy rate P(next token in prompt | last n tokens), n = 1, 2."""

    def __init__(self, recs, space_first=None):
        inp = defaultdict(int)
        cop = defaultdict(int)
        ctx = [None, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])]
        for r in recs:
            ps = set(r["pids"])
            ys = set(r["y"])
            for t in ps:
                inp[t] += 1
                if t in ys:
                    cop[t] += 1
            s = TAIL + r["y"]
            for i in range(len(TAIL), len(s)):
                hit = s[i] in ps
                for n in (1, 2):
                    c = ctx[n][tuple(s[i - n:i])]
                    c[0] += hit
                    c[1] += 1
        self.prop = {t: (cop[t] + 0.5) / (inp[t] + 2.0) for t in inp}
        self.ctx = ctx

    def p_tok(self, t):
        return self.prop.get(t, 0.2)

    def p_ctx(self, c, n):
        e = self.ctx[n].get(tuple(c[len(c) - n:]))
        if e is None:
            return 0.1
        return (e[0] + 0.2) / (e[1] + 2.0)


class Featurizer:
    def __init__(self, table: NGramTable, index: PromptIndex, k=12, nb_seq_index=None, copy=None,
                 topm=TOPM, tokenizer=None):
        self.t = table
        self.ix = index
        self.k = k
        self.copy = copy
        self.topm = topm
        self.tok = tokenizer
        # per training record SeqIndex over TAIL + y (built lazily)
        self.si = nb_seq_index if nb_seq_index is not None else {}

    def seqindex(self, j):
        s = self.si.get(j)
        if s is None:
            s = SeqIndex(TAIL + self.ix.recs[j]["y"])
            self.si[j] = s
        return s

    def begin(self, pids, text, exclude_group=None):
        self.p = list(pids)
        self.pset = set(pids)
        self.spacevar = {}
        if self.copy is not None and self.tok is not None and text is not None:
            # first token of each prompt word written with a leading space (as it would appear
            # mid-command), when that differs from the prompt's own tokenization
            for m in re.finditer(r"[A-Za-z0-9_.\-/~:@]+", text):
                w = m.group(0)
                ids = self.tok.encode(" " + w, add_special_tokens=False).ids
                if ids and ids[0] not in self.pset:
                    self.spacevar.setdefault(ids[0], m.start())
        self.pidx = SeqIndex(self.p) if self.p else None
        nb = []
        for j, s in self.ix.query(text, self.k + 8):
            if exclude_group is not None and self.ix.recs[j]["group"] == exclude_group:
                continue
            nb.append((s, self.seqindex(j)))
            if len(nb) >= self.k:
                break
        self.nb = nb

    def candidates(self, ctx, out):
        """Returns (list of tokens, feature matrix [n_cand, n_feat])."""
        t = self.t
        L = len(ctx)
        cand = {}

        def row(tok):
            r = cand.get(tok)
            if r is None:
                r = np.zeros(len(FEATS))
                r[0] = 1.0
                cand[tok] = r
            return r

        # n-gram
        long_n = 0
        dists = {}
        for n in range(min(t.N, L), 0, -1):
            d = t.counts[n].get(tuple(ctx[L - n:]))
            if d is not None:
                dists[n] = d
                if long_n == 0:
                    long_n = n
        supp3 = next((n for n in sorted(dists, reverse=True) if sum(dists[n].values()) >= 3), 0)
        supp10 = next((n for n in sorted(dists, reverse=True) if sum(dists[n].values()) >= 10), 0)
        for n, d in dists.items():
            top = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:self.topm]
            for rk, (tok, c) in enumerate(top):
                r = row(tok)
                if n == long_n:
                    if rk == 1:
                        r[FI["ng_rank2"]] = 1.0
                    elif rk >= 2:
                        r[FI["ng_rank3plus"]] = 1.0
        for tok, r in cand.items():
            for n, d in dists.items():
                tot = sum(d.values())
                c = d.get(tok, 0)
                if c:
                    if n <= 6:
                        r[FI[f"ng_p{n}"]] = c / tot
                    if n == long_n:
                        r[FI["ng_p_long"]] = c / tot
                        r[FI["ng_ord_long"]] = n / 6.0 * (c / tot)
                        r[FI["ng_lcnt_long"]] = math.log1p(c)
                    if n == supp3:
                        r[FI["ng_p_supp3"]] = c / tot
                    if n == supp10:
                        r[FI["ng_p_supp10"]] = c / tot
        if long_n:
            d = dists[long_n]
            tot = sum(d.values())
            btok = min(d.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if btok in cand:
                cand[btok][FI["ng_top_long"]] = 1.0
            ent = -sum((c / tot) * math.log(c / tot) for c in d.values())
            if ent < 0.3:
                for r in cand.values():
                    r[FI["ng_ent_low"]] = 1.0 if r[FI["ng_top_long"]] else 0.0
        # self lookup (earlier output only)
        sf_tok = None
        if len(out) > 1:
            o = out
            Lo = len(o)
            for n in range(min(MAXN_NB, Lo - 1), 0, -1):
                pat = o[Lo - n:]
                hit = None
                for j in range(Lo - n - 1, -1, -1):
                    if o[j:j + n] == pat:
                        hit = j + n
                        break
                if hit is not None:
                    sf_tok = o[hit]
                    r = row(sf_tok)
                    r[FI["sf_m"]] = n / 4.0
                    r[FI["sf_any"]] = 1.0
                    r[FI["sf_m_ge3"]] = 1.0 if n >= 3 else 0.0
                    break
        # prompt lookup
        if out and self.pidx is not None:
            n, tok = self.pidx.match(out)
            if n:
                r = row(tok)
                r[FI["pr_m"]] = n / 4.0
                r[FI["pr_any"]] = 1.0
                r[FI["pr_m_ge2"]] = 1.0 if n >= 2 else 0.0
        # neighbours
        nb_votes = defaultdict(list)
        first = True
        for s, si in self.nb:
            n, tok = si.match(ctx)
            if n and tok is not None:
                nb_votes[tok].append((s, n, first))
            first = False
        for tok, vs in nb_votes.items():
            r = row(tok)
            r[FI["nb_sum"]] = sum(s for s, _, _ in vs)
            r[FI["nb_msum"]] = sum(s * min(n, 8) / 8.0 for s, n, _ in vs)
            r[FI["nb_cnt"]] = len(vs) / max(1, len(self.nb))
            r[FI["nb_maxm"]] = max(min(n, 8) for _, n, _ in vs) / 8.0
            for s, n, f in vs:
                if f:
                    r[FI["nb_top1"]] = 1.0
                    r[FI["nb_top1_m"]] = min(n, 8) / 8.0
        if self.copy is not None:
            used = set(out)
            c1 = self.copy.p_ctx(ctx, 1)
            c2 = self.copy.p_ctx(ctx, 2)
            first_unused = None
            for tk in self.p:
                if tk not in used and self.copy.p_tok(tk) > 0.3:
                    first_unused = tk
                    break
            for tk in set(self.p) | set(self.spacevar):
                r = row(tk)
                isp = tk in self.pset
                r[FI["cp_tok"]] = 1.0 if isp else 0.0
                r[FI["cp_spacevar"]] = 0.0 if isp else 1.0
                pt = self.copy.p_tok(tk) if isp else 0.3
                r[FI["cp_prop"]] = math.log(pt / (1 - pt + 1e-9) + 1e-9) / 4.0
                r[FI["cp_ctx1"]] = c1
                r[FI["cp_ctx2"]] = c2
                r[FI["cp_unused"]] = 1.0 if tk not in used else 0.0
                r[FI["cp_first_unused"]] = 1.0 if tk == first_unused else 0.0
        toks = list(cand.keys())
        X = np.stack([cand[tk] for tk in toks]) if toks else np.zeros((0, len(FEATS)))
        for i, tk in enumerate(toks):
            X[i, FI["is_stop"]] = 1.0 if tk in STOPS else 0.0
            X[i, FI["in_prompt"]] = 1.0 if tk in self.pset else 0.0
            X[i, FI["agree_ng_nb"]] = X[i, FI["ng_top_long"]] * min(1.0, X[i, FI["nb_cnt"]] * 2)
            X[i, FI["agree_sf_nb"]] = X[i, FI["sf_any"]] * min(1.0, X[i, FI["nb_cnt"]] * 2)
        return toks, X


def collect(feat: Featurizer, recs, exclude_own_group=False, positions="all"):
    """Teacher-forced candidate sets for every position of every record.
    Returns list of (X, label) with label = candidate index or -1 (null)."""
    data = []
    for r in recs:
        feat.begin(r["pids"], r["prompt"], r["group"] if exclude_own_group else None)
        y = r["y"]
        for i in range(0, len(y)):
            ctx = TAIL + y[:i]
            toks, X = feat.candidates(ctx, y[:i])
            lab = toks.index(y[i]) if y[i] in toks else -1
            data.append((X, lab, i))
    return data


def train_clogit(data, l2=1e-3, iters=300, lr=0.5):
    """Conditional logit with a null option (score 0). Full-batch gradient descent (Adam)."""
    nF = len(FEATS)
    w = np.zeros(nF)
    Xs = [d[0] for d in data if d[0].shape[0] > 0]
    labs = np.array([d[1] for d in data if d[0].shape[0] > 0])
    sizes = np.array([x.shape[0] for x in Xs])
    X = np.concatenate(Xs)
    seg = np.repeat(np.arange(len(Xs)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    lab_rows = np.where(labs >= 0, starts + np.maximum(labs, 0), -1)
    m = np.zeros(nF)
    v = np.zeros(nF)
    b1, b2, eps = 0.9, 0.999, 1e-8
    nseg = len(Xs)
    for it in range(1, iters + 1):
        s = X @ w
        smax = np.maximum(np.maximum.reduceat(s, starts), 0.0)
        e = np.exp(s - smax[seg])
        Z = np.add.reduceat(e, starts) + np.exp(-smax)
        p = e / Z[seg]
        # loss = -sum(log p_label) (null label: log(exp(-smax)/Z))
        g = X.T @ p
        has = lab_rows >= 0
        g -= X[lab_rows[has]].sum(axis=0)
        g = g / nseg + l2 * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it)
        vh = v / (1 - b2 ** it)
        w -= lr * mh / (np.sqrt(vh) + eps) * (0.98 ** (it / 10))
    s = X @ w
    smax = np.maximum(np.maximum.reduceat(s, starts), 0.0)
    e = np.exp(s - smax[seg])
    Z = np.add.reduceat(e, starts) + np.exp(-smax)
    lp = np.where(lab_rows >= 0, np.log(e[np.maximum(lab_rows, 0)] / Z + 1e-30), np.log(np.exp(-smax) / Z))
    # accuracy of argmax
    am = np.array([np.argmax(x @ w) for x in Xs])
    top = np.array([x[i] @ w for x, i in zip(Xs, am)])
    acc = np.mean((am == labs) & (labs >= 0))
    return w, {"nll": -lp.mean(), "acc": acc, "coverage": np.mean(labs >= 0)}


class LearnedDrafter:
    def __init__(self, feat: Featurizer, w, tau=0.0, cum=0.0):
        self.f, self.w, self.tau, self.cum = feat, w, tau, cum

    def begin(self, pids, text=None):
        self.f.begin(pids, text)

    def propose(self, tail, out, K):
        ctx = list(tail) + list(out)
        o = list(out)
        d = []
        pc = 1.0
        while len(d) < K:
            toks, X = self.f.candidates(ctx, o)
            if not toks:
                break
            s = X @ self.w
            i = int(np.argmax(s))
            mx = max(s.max(), 0.0)
            p = math.exp(s[i] - mx) / (np.exp(s - mx).sum() + math.exp(-mx))
            if p < self.tau:
                break
            pc *= p
            if d and pc < self.cum:
                break
            tok = toks[i]
            d.append(tok)
            ctx.append(tok)
            o.append(tok)
            if tok in STOPS:
                break
        return d
