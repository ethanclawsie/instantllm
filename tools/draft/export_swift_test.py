"""Export test vectors for the Swift port: per request the prompt text/ids, the space-variant
tokens, the target sequence, and the Python drafter's teacher-forced top-3 probabilities and K=8
chains at every position (plus one 16-node tree per request).

usage: .venv/bin/python -I tools/draft/export_swift_test.py --table T.bin --out vectors.json [--n 200]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_split, tokenizer  # noqa: E402
from ildraft import _SPACEVAR_RE, Drafter, Tables  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--tag", default="q4")
a = ap.parse_args()
tab = Tables.load(a.table)
tk = tokenizer()


class Rec:
    """Tokenizer proxy recording decode/encode calls (replayed by the Swift test's mock tokenizer)."""

    def __init__(self, tk):
        self.tk = tk
        self.dec = {}
        self.enc = {}

    def decode(self, ids):
        t = self.tk.decode(ids)
        self.dec[",".join(map(str, ids))] = t
        return t

    def encode(self, text, add_special_tokens=False):
        e = self.tk.encode(text, add_special_tokens=add_special_tokens)
        self.enc[text] = e.ids
        return e


rec = Rec(tk)
d = Drafter(tab, rec)
_, held, bench = load_split(a.tag, extra=())
recs = bench + held[:a.n]
out = []
for r in recs:
    rec.dec, rec.enc = {}, {}
    d.begin(r["pids"], r["prompt"])
    io = {"dec": dict(rec.dec), "enc": dict(rec.enc)}
    sv = []
    for m in _SPACEVAR_RE.finditer(r["prompt"]):
        ids = tk.encode(" " + m.group(0), add_special_tokens=False).ids
        if ids:
            sv.append(ids[0])
    y = r["y"]
    pos = []
    for i in range(len(y)):
        pr = d.probs(y[:i])
        pos.append({"top": [[t, p] for t, p in pr[:3]], "chain": d.chain(y[:i], 8)})
    par, toks = d.tree(y[:1], 16) if len(y) > 1 else ([], [])
    out.append({"prompt": r["prompt"], "pids": r["pids"], "spacevar": sv, "y": y, "pos": pos,
                "tree": {"parents": par, "tokens": toks}, "io": io})
json.dump(out, open(a.out, "w"))
print(f"wrote {len(out)} requests, {sum(len(x['y']) for x in out)} positions -> {a.out}")
