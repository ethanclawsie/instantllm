"""Build the drafter table models/draft/qwen-q4-ngram.bin from TRAIN outputs.

  1. load TRAIN records (scratch/draft/outputs_q4*.jsonl, split == "train")
  2. group 5-fold cross-fitting: for every fold, build the tables from the other folds and
     compute teacher-forced candidate features on the fold (so the scorer learns how the
     sources behave on requests whose family is NOT in the tables, like held-out/bench)
  3. train the MLP candidate scorer (conditional softmax over candidates + a null option),
     PyTorch on the CPU
  4. build the tables from all of TRAIN, attach the scorer, write the binary (format in ildraft.py)

usage: .venv/bin/python -I tools/draft/build.py [--out models/draft/qwen-q4-ngram.bin]
       [--tag q4] [--H 16] [--minc 1] [--epochs 800]
"""
import argparse
import os
import pickle
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DDIR, ROOT, load_split, tokenizer  # noqa: E402
from ildraft import NF, Config, Drafter, Tables  # noqa: E402


def collect(tables, recs, tk):
    d = Drafter(tables, tk, weights=(None,))
    out = []
    for r in recs:
        d.begin(r["pids"], r["prompt"])
        y = r["y"]
        for i in range(len(y)):
            toks, rows = d.features(y[:i])
            lab = toks.index(y[i]) if y[i] in toks else -1
            out.append((np.array(rows, dtype=np.float32).reshape(-1, NF), lab, i))
    return out


def train_scorer(data, H, epochs, lr, wd, seed, threads=3):
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data = [d for d in data if d[0].shape[0] > 0]
    sizes = np.array([d[0].shape[0] for d in data])
    labs = np.array([d[1] for d in data])
    X = torch.tensor(np.concatenate([d[0] for d in data]))
    seg = torch.tensor(np.repeat(np.arange(len(data)), sizes))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    lab_rows = torch.tensor(np.where(labs >= 0, starts + np.maximum(labs, 0), 0))
    has = torch.tensor(labs >= 0)
    nseg = len(data)
    l1 = torch.nn.Linear(NF, H)
    l2 = torch.nn.Linear(H, 1)
    lin = torch.nn.Linear(NF, 1, bias=False)
    params = list(l1.parameters()) + list(l2.parameters()) + list(lin.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad()
        s = (l2(torch.relu(l1(X))) + lin(X)).squeeze(-1)
        m = torch.full((nseg,), -1e9).scatter_reduce(0, seg, s, reduce="amax")
        m = torch.clamp(m, min=0.0).detach()
        e = torch.exp(s - m[seg])
        Z = torch.zeros(nseg).index_add(0, seg, e) + torch.exp(-m)
        num = torch.where(has, s[lab_rows] - m, -m)
        loss = (torch.log(Z) - num).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        s = (l2(torch.relu(l1(X))) + lin(X)).squeeze(-1)
        m = torch.full((nseg,), -1e9).scatter_reduce(0, seg, s, reduce="amax")
        idx = torch.arange(len(s))
        ismax = s >= m[seg]
        first = torch.full((nseg,), len(s)).scatter_reduce(0, seg[ismax], idx[ismax], reduce="amin")
        acc = ((first == lab_rows) & has).float().mean().item()
    W = {"W1": l1.weight.detach().double().tolist(), "b1": l1.bias.detach().double().tolist(),
         "W2": l2.weight.detach().double().view(-1).tolist(), "b2": float(l2.bias.item()),
         "WL": lin.weight.detach().double().view(-1).tolist()}
    return W, {"loss": loss.item(), "acc": acc, "coverage": float(np.mean(labs >= 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models/draft/qwen-q4-ngram.bin"))
    ap.add_argument("--tag", default="q4")
    ap.add_argument("--extra", default="x,y", help="suffixes of extra TRAIN output files")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--knb", type=int, default=16)
    ap.add_argument("--minc", type=int, default=1, help="min count for contexts of order >= 3")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", default="", help="pickle of the cross-fitted features (reused if present)")
    ap.add_argument("--flags", type=int, default=1,
                    help="bit 1 = association features, bit 2 = association candidates, bit 4 = edited neighbours")
    ap.add_argument("--gpu-out", dest="gpu_out", default="", help="also write the engine's GPU table layout")
    ap.add_argument("--exclude-near-bench", dest="exclude_near_bench", type=float, default=0.0,
                    help="drop TRAIN prompts this similar to a bench prompt (contamination check)")
    ap.add_argument("--jsonl-out", dest="jsonl_out", default="", help="also write the TRAIN outputs as JSONL")
    ap.add_argument("--as-top", dest="as_top", type=int, default=4)
    ap.add_argument("--edit-E", dest="edit_E", type=int, default=4, help="neighbours to edit (flags & 4)")
    ap.add_argument("--subsample", type=float, default=1.0, help="use this fraction of TRAIN (learning curve)")
    ap.add_argument("--max-df", dest="max_df", type=float, default=1.0, help="drop prompt words in more than this doc fraction")
    ap.add_argument("--cc2-min", dest="cc2_min", type=int, default=1, help="min observations for an order-2 copy-rate context")
    ap.add_argument("--as-min", dest="as_min", type=int, default=3, help="min co-occurrences for an association pair")
    a = ap.parse_args()
    t0 = time.time()
    extra = tuple(x for x in a.extra.split(",") if x)
    train, _, _ = load_split(a.tag, extra=extra)
    if a.subsample < 1.0:
        import hashlib
        train = [r for r in train
                 if int(hashlib.md5(r["prompt"].encode()).hexdigest()[:8], 16) / 0xffffffff < a.subsample]
        print(f"subsample {a.subsample}: {len(train)} TRAIN prompts", flush=True)
    if a.exclude_near_bench > 0:
        # contamination check: drop TRAIN prompts whose word-level similarity to any bench prompt
        # (max of difflib ratio and content-word Jaccard, as in gen_prompts.BenchFilter) >= threshold
        import gen_prompts as G
        bench_prompts = [l for l in open(os.path.join(ROOT, "bench/prompts.txt")).read().split("\n") if l]
        bfil = G.BenchFilter(bench_prompts)
        keep = []
        for r in train:
            rr, jj, _ = bfil.score(r["prompt"])
            if max(rr, jj) < a.exclude_near_bench:
                keep.append(r)
        print(f"exclude-near-bench {a.exclude_near_bench}: kept {len(keep)} of {len(train)} TRAIN prompts", flush=True)
        train = keep
    tk = tokenizer()
    cfg = Config(N=a.N, M=a.M, KNB=a.knb, H=a.H, minc_hi=a.minc, flags=a.flags, as_top=a.as_top, edit_E=a.edit_E,
                 max_df=a.max_df, cc2_min=a.cc2_min, as_min=a.as_min)
    cache = a.cache or os.path.join(DDIR, f"cvfeat_{a.tag}{''.join(extra)}_N{a.N}M{a.M}k{a.knb}c{a.minc}"
                                          f"f{a.flags}t{a.as_top}e{a.edit_E}_F{NF}_x{a.exclude_near_bench}"
                                          f"_df{a.max_df}_c2{a.cc2_min}_am{a.as_min}"
                                          f"_s{a.subsample}_H{a.H}.pkl".replace(f"_H{a.H}.pkl", ".pkl"))
    if os.path.exists(cache):
        data = pickle.load(open(cache, "rb"))
        print(f"loaded cross-fitted features from {cache}", flush=True)
    else:
        groups = sorted({r["group"] for r in train})
        random.Random(a.seed).shuffle(groups)
        fold = {g: i % a.folds for i, g in enumerate(groups)}
        data = []
        for f in range(a.folds):
            tr = [r for r in train if fold[r["group"]] != f]
            dv = [r for r in train if fold[r["group"]] == f]
            tab = Tables.build(tr, cfg)
            tab.set_tokenizer_flags(tk)
            data += [d + (f,) for d in collect(tab, dv, tk)]
            print(f"fold {f}: {len(dv)} records, {len(data)} positions ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump(data, open(cache, "wb"))
    W, st = train_scorer(data, a.H, a.epochs, a.lr, a.wd, a.seed)
    print(f"scorer: train loss {st['loss']:.4f} acc {st['acc']:.4f} coverage {st['coverage']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    tab = Tables.build(train, cfg)
    tab.set_tokenizer_flags(tk)
    tab.W = W
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = tab.save(a.out)
    print(f"wrote {a.out}: {n / 1e6:.2f} MB, {len(tab.ng)} contexts, {len(tab.vocab)} words, "
          f"{len(tab.seqs)} sequences ({time.time() - t0:.0f}s)")
    if a.gpu_out:
        # the engine's own GPU table layout (Sources/Engine/NGram.swift), from the same TRAIN outputs
        from gpudraft import GPUTable
        used, dropped = GPUTable([r["y"] for r in train]).write_engine_format(a.gpu_out)
        print(f"wrote {a.gpu_out}: engine NGramTable layout, {used} contexts ({dropped} dropped)")
    if a.jsonl_out:
        import json
        with open(a.jsonl_out, "w") as f:
            for r in train:
                f.write(json.dumps({"prompt": r["prompt"], "ids": r["ids"], "stop": r["stop"]}) + "\n")
        print(f"wrote {a.jsonl_out}: {len(train)} TRAIN outputs (input for `instantllm ngram`)")


if __name__ == "__main__":
    main()
