"""Compact head-to-head table from bench/results/<config>-<mode>.tsv (written by bench/run_all.sh).

  .venv/bin/python -I bench/table.py [results_dir]

Per config: median TTFT (prefix mode), median decode tok/s (fixed mode: 32 tokens for everyone),
median end-to-end latency for natural stop (prefix), fixed 32 tokens (fixed), whole prompt (full)
and cold (fresh process per request).
"""
import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench/results")
NAMES = {"iq4": "**instantllm q4**", "iq8": "**instantllm q8**", "if16": "**instantllm f16**",
         "lq40": "llama.cpp Q4_0", "lq4km": "llama.cpp Q4_K_M", "lq80": "llama.cpp Q8_0", "lf16": "llama.cpp F16",
         "m4": "mlx-lm 4bit", "m8": "mlx-lm 8bit", "mbf16": "mlx-lm bf16"}


def load(cfg, mode):
    path = os.path.join(RES, f"{cfg}-{mode}.tsv")
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path), delimiter="\t"))


def med(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return statistics.median(vals) if vals else float("nan")


def fmt(v, unit="ms", prec=1):
    return "-" if v != v else f"{v:.{prec}f} {unit}"


cfgs = [c for c in NAMES if any(os.path.exists(os.path.join(RES, f"{c}-{m}.tsv")) for m in ("prefix", "fixed", "full", "cold"))]
print("| median | " + " | ".join(NAMES[c] for c in cfgs) + " |")
print("|---|" + "---:|" * len(cfgs))
rows = [
    ("time to first token (prompt cached)", lambda c: fmt(med(load(c, "prefix"), "ttft_ms"))),
    ("decode (32 tokens)", lambda c: fmt(med(load(c, "fixed"), "decode_tok_s"), "tok/s", 0)),
    ("end to end, exactly 32 tokens", lambda c: fmt(med(load(c, "fixed"), "e2e_ms"))),
    ("end to end, natural stop", lambda c: fmt(med(load(c, "prefix"), "e2e_ms"))),
    ("end to end, whole prompt (no cache)", lambda c: fmt(med(load(c, "full"), "e2e_ms"))),
    ("cold: fresh process per request", lambda c: fmt(med(load(c, "cold"), "e2e_ms"), "ms", 0)),
]
for name, f in rows:
    print(f"| {name} | " + " | ".join(f(c) for c in cfgs) + " |")
print()
print("samples: " + ", ".join(f"{c} {len(load(c, 'prefix'))}/{len(load(c, 'fixed'))}/{len(load(c, 'full'))}/{len(load(c, 'cold'))}" for c in cfgs)
      + " (prefix/fixed/full/cold)")
