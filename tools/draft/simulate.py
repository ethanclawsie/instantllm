"""Simulate exact speculative decoding with the drafter table on held-out / bench outputs.

Per verify step the drafter proposes up to K tokens (chain) or a B-node tree; the verifier keeps
the longest prefix (path) that matches the model's own greedy tokens plus one verified token.
Output is identical to plain greedy by construction; we only count steps.

Metrics
  tok/step   tokens produced by verify steps / verify steps (the prefill yields the first token)
  hit        fraction of verify steps with >= 1 accepted draft token
  passes     forward passes per request incl. prefill, relative to plain greedy (= len(y))
  +prefill   the same, when drafts are also appended to the prefill pass

usage: .venv/bin/python -I tools/draft/simulate.py [--table models/draft/qwen-q4-ngram.bin]
       [--K 1,2,3,4,6,8] [--tree 4,8,16,32] [--sets heldout,bench,bench_f16] [--baselines]
"""
import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_bench, load_split, tokenizer  # noqa: E402
from ildraft import STOPS, TAIL, Drafter, Tables, accept_tree  # noqa: E402


class Memo:
    """Caches Drafter.probs per request (drafts depend only on the request and `out`)."""

    def __init__(self, d):
        self.d = d
        self.cache = {}

    def begin(self, r):
        self.cache = {}
        self.d.begin(r["pids"], r["prompt"])

    def probs(self, out):
        k = tuple(out)
        v = self.cache.get(k)
        if v is None:
            v = self.d.probs(list(out))
            self.cache[k] = v
        return v

    def chain(self, out, K, tau=0.0):
        o = list(out)
        dr = []
        while len(dr) < K:
            pr = self.probs(o)
            if not pr or pr[0][1] < tau:
                break
            dr.append(pr[0][0])
            o.append(pr[0][0])
            if pr[0][0] in STOPS:
                break
        return dr

    def tprobs(self, out, branch):
        """top-`branch` children with probabilities re-tempered: p^(1/T) / (sum p^(1/T) + null^(1/T))."""
        pr = self.probs(out)
        T = getattr(self, "temp", 1.0)
        if T == 1.0 or not pr:
            return pr[:branch]
        null = max(0.0, 1.0 - sum(p for _, p in pr))
        w = [(t, p ** (1.0 / T)) for t, p in pr]
        Z = sum(x for _, x in w) + null ** (1.0 / T)
        return [(t, x / Z) for t, x in w[:branch]]

    def tree(self, out, B, branch=8):
        import heapq
        parents, tokens = [], []
        heap = []
        tie = 0
        ctxs = {-1: list(out)}
        for tok, p in self.tprobs(out, branch):
            tie += 1
            heapq.heappush(heap, (-p, tie, -1, tok))
        while heap and len(tokens) < B:
            negpp, _, par, tok = heapq.heappop(heap)
            nid = len(tokens)
            parents.append(par)
            tokens.append(tok)
            o = ctxs[par] + [tok]
            ctxs[nid] = o
            if tok in STOPS or len(tokens) >= B:
                continue
            for tok2, p2 in self.tprobs(o, branch):
                tie += 1
                heapq.heappush(heap, (negpp * p2, tie, nid, tok2))
        return parents, tokens


def run_chain_exact(m, recs, K, prefill=False, tau=0.0):
    """Returns tok/step computed exactly as (tokens produced by verify steps) / steps."""
    steps = produced = hit = passes = base = ndraft = 0
    hist = Counter()
    for r in recs:
        m.begin(r)
        y = r["y"]
        n = len(y)
        if n == 0:
            continue
        if prefill:
            d = m.chain([], K, tau)
            ndraft += len(d)
            a = 0
            while a < len(d) and a < n and d[a] == y[a]:
                a += 1
            p = min(n, a + 1)
        else:
            p = 1
        passes += 1
        p0 = p
        while p < n:
            d = m.chain(y[:p], K, tau)
            ndraft += len(d)
            a = 0
            while a < len(d) and p + a < n and d[a] == y[p + a]:
                a += 1
            hist[a] += 1
            hit += a > 0
            steps += 1
            passes += 1
            p = min(n, p + a + 1)
        produced += n - p0
        base += n
    return {"tps": produced / max(1, steps), "steps": steps, "hit": hit / max(1, steps), "passes": passes,
            "base": base, "hist": hist, "draft_per_step": ndraft / max(1, steps + (len(recs) if prefill else 0))}


def run_tree(m, recs, B):
    steps = produced = 0
    for r in recs:
        m.begin(r)
        y = r["y"]
        n = len(y)
        p = 1
        while p < n:
            par, tok = m.tree(y[:p], B)
            a = accept_tree(par, tok, y, p)
            steps += 1
            p = min(n, p + a + 1)
        produced += max(0, n - 1)
    return produced / max(1, steps)


def oracle(recs, K):
    steps = produced = 0
    for r in recs:
        mm = len(r["y"]) - 1
        if mm > 0:
            steps += -(-mm // (K + 1))
            produced += mm
    return produced / max(1, steps)


class GPUStyle:
    """Engine-style n-gram drafting (gpudraft.GPUDrafter) on a table built from the same TRAIN outputs."""

    def __init__(self, gtab, **kw):
        from gpudraft import GPUDrafter
        self.d = GPUDrafter(gtab, **kw)

    def begin(self, r):
        self.d.begin(r["pids"], r["prompt"])

    def chain(self, out, K, tau=0.0):
        return self.d.propose(None, list(out), K)


class PLD:
    """Baseline: prompt lookup (longest suffix match of the output in the prompt, max 3)."""

    def begin(self, r):
        self.P = r["pids"]

    def chain(self, out, K, tau=0.0):
        P = self.P
        for n in range(min(3, len(out)), 0, -1):
            pat = list(out[len(out) - n:])
            for j in range(0, len(P) - n):
                if P[j:j + n] == pat:
                    return P[j + n:j + n + K]
        return []


class NGramOnly:
    """Baseline: the table's n-gram alone (longest context match, most frequent next token)."""

    def __init__(self, tab):
        self.t = tab

    def begin(self, r):
        pass

    def chain(self, out, K, tau=0.0):
        o = TAIL + list(out)
        d = []
        while len(d) < K:
            e = None
            for n in range(min(self.t.cfg.N, len(o)), 0, -1):
                e = self.t.ng_get(tuple(o[len(o) - n:]))
                if e is not None:
                    break
            if e is None:
                break
            tok = e[1][0][0]
            d.append(tok)
            o.append(tok)
            if tok in STOPS:
                break
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(ROOT, "models/draft/qwen-q4-ngram.bin"))
    ap.add_argument("--K", default="1,2,3,4,6,8")
    ap.add_argument("--tree", default="4,8,16,32")
    ap.add_argument("--sets", default="heldout,bench,bench_f16")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--prefill", action="store_true", help="also report prefill drafting")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="q4", help="outputs tag (held-out/bench), e.g. q4n")
    ap.add_argument("--extra", default="x,y", help="TRAIN extra suffixes (for the baselines' tables)")
    ap.add_argument("--f16tag", default="f16")
    ap.add_argument("--tau", default="", help="comma list of probability thresholds for adaptive chains (K = max K)")
    ap.add_argument("--tree-temp", dest="tree_temp", default="1.0", help="comma list of tree temperatures")
    a = ap.parse_args()
    t0 = time.time()
    tab = Tables.load(a.table)
    tk = tokenizer()
    m = Memo(Drafter(tab, tk))
    train, held, bench = load_split(a.tag, extra=tuple(x for x in a.extra.split(",") if x))
    sets = {"heldout": held, "bench": bench, "bench_f16": load_bench(a.f16tag)}
    gtab = None
    if a.baselines:
        from gpudraft import GPUTable
        gtab = GPUTable([r["y"] for r in train])
    Ks = [int(x) for x in a.K.split(",") if x]
    Bs = [int(x) for x in a.tree.split(",") if x]
    print(f"table {a.table}: {os.path.getsize(a.table) / 1e6:.2f} MB, cfg N={tab.cfg.N} M={tab.cfg.M} "
          f"KNB={tab.cfg.KNB} H={tab.cfg.H} minc={tab.cfg.minc_hi}, {len(tab.seqs)} sequences\n")
    for name in a.sets.split(","):
        recs = sets[name][:a.limit] if a.limit else sets[name]
        ntok = sum(len(r["y"]) for r in recs)
        print(f"## {name}: {len(recs)} requests, {ntok} tokens (incl. stop)\n")
        print("| drafter | K | tok/step | hit | drafted/step | passes vs greedy | +prefill draft | accepted-length histogram |")
        print("|---|---|---|---|---|---|---|---|")
        variants = [("ildraft", m)]
        if a.baselines:
            variants += [("prompt-lookup", PLD()), ("ngram-only (ildraft table)", NGramOnly(tab))]
            if gtab is not None:
                variants += [("engine rule (thr .5, stop)", GPUStyle(gtab)),
                             ("engine table, thr 0", GPUStyle(gtab, thr=0.0)),
                             ("engine table, thr 0 + self + prompt", GPUStyle(gtab, thr=0.0, self_min=3, prompt_min=1))]
        for vn, dr in variants:
            for K in Ks:
                res = run_chain_exact(dr, recs, K)
                pf = ""
                if a.prefill and vn == "ildraft":
                    rp = run_chain_exact(dr, recs, K, prefill=True)
                    pf = f"{rp['base'] / rp['passes']:.2f}x"
                h = res["hist"]
                tot = sum(h.values())
                hs = " ".join(f"{k}:{h[k] / tot:.2f}" for k in sorted(h))
                print(f"| {vn} | {K} | {res['tps']:.3f} | {res['hit']:.2f} | {res['draft_per_step']:.2f} | "
                      f"{res['base'] / res['passes']:.2f}x | {pf} | {hs} |", flush=True)
        print("| oracle | " + " ".join(f"K{K}={oracle(recs, K):.2f}" for K in Ks) + " | | | | | | |")
        if a.tau:
            print("\n| adaptive chain (K=%d): stop when p(best) < tau | tok/step | drafted/step | hit |" % max(Ks))
            print("|---|---|---|---|")
            for tau in [float(x) for x in a.tau.split(",")]:
                res = run_chain_exact(m, recs, max(Ks), tau=tau)
                print(f"| tau={tau} | {res['tps']:.3f} | {res['draft_per_step']:.2f} | {res['hit']:.2f} |", flush=True)
        if Bs:
            print("\n| tree budget (nodes) | " + " | ".join(str(B) for B in Bs) + " |")
            print("|---|" + "---|" * len(Bs))
            for T in [float(x) for x in a.tree_temp.split(",")]:
                m.temp = T
                lab = "ildraft tok/step" + ("" if T == 1.0 else f" (temp {T})")
                print(f"| {lab} | " + " | ".join(f"{run_tree(m, recs, B):.3f}" for B in Bs) + " |", flush=True)
            m.temp = 1.0
        print(f"\n({time.time() - t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    main()
