"""Summarize benchmark TSVs (competitor harnesses or any engine using the same schema).

usage: .venv/bin/python -I bench/summarize.py [TSV ...] [--golden DIR] [--max-rows 15] [--sort e2e|ttft|name]
       (default inputs: bench/results/*.tsv)

TSV schema: tab-separated, one header row, one row per request:
  engine  variant  mode  prompt_idx  rep  e2e_ms  ttft_ms  prefill_tokens  gen_tokens  decode_tok_s  text
  - mode: full (whole templated prompt prefilled), prefix (system prefix KV cached, only the rest
    prefilled) or cold (fresh process per request, spawn -> exit).
  - gen_tokens counts sampled tokens including the stop token (<= 64); decode_tok_s =
    (gen_tokens - 1) / (t_last_token - t_first_token). Missing numbers are "nan".
  - text is escaped: backslash -> \\\\, tab -> \\t, newline -> \\n, CR -> \\r.

Prints one markdown row per (engine, variant, mode): requests, median and min e2e, median TTFT,
median decode tok/s, mean generated tokens (e2e scales with it, and it differs between variants
whose outputs differ) and, when the fp32 reference exists (tools/reference/golden/index.json, records
{"index", "prompt", "text"}), the fraction of prompts whose generated text exactly equals the
reference text (the most common text over that prompt's reps is used; prompts are matched by
prompt string, falling back to prompt_idx). At most --max-rows (15) rows: the fastest variant of
every (engine, mode) is always listed, remaining slots go to the next fastest by median e2e; the
table is then sorted by --sort.
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["engine", "variant", "mode", "prompt_idx", "rep", "e2e_ms", "ttft_ms",
          "prefill_tokens", "gen_tokens", "decode_tok_s", "text"]


def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            out.append({"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}.get(n, "\\" + n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return math.nan
    return v


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = None
            for rec in reader:
                if not rec or (len(rec) == 1 and not rec[0].strip()):
                    continue
                if rec[0] == "engine":  # header (possibly repeated when files were concatenated)
                    header = rec
                    continue
                if header is None:
                    header = FIELDS
                if len(rec) < len(header):
                    rec += [""] * (len(header) - len(rec))
                r = dict(zip(header, rec))
                r["text"] = unescape(r.get("text", ""))
                rows.append(r)
    return rows


def load_prompts():
    try:
        with open(os.path.join(REPO, "bench", "prompts.txt"), encoding="utf-8") as f:
            return [ln for ln in f.read().split("\n") if ln.strip()]
    except OSError:
        return []


def decode_ids(ids):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(REPO, "models", "hf", "tokenizer.json"))
    return tok.decode(list(ids), skip_special_tokens=False)


def load_golden(path):
    """Returns ({prompt_idx: text}, {prompt string: text}) or None when there is no reference."""
    if not os.path.exists(path):
        return None
    records = []
    if os.path.isdir(path):
        index = os.path.join(path, "index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as f:
                records = json.load(f)
        else:  # per-prompt pNN.json files
            for p in sorted(glob.glob(os.path.join(path, "p[0-9]*.json"))):
                with open(p, encoding="utf-8") as f:
                    rec = json.load(f)
                rec.setdefault("index", int("".join(ch for ch in os.path.basename(p).split(".")[0] if ch.isdigit())))
                records.append(rec)
    elif path.endswith(".tsv"):  # a reference written in the benchmark TSV schema
        records = [{"index": int(r["prompt_idx"]), "text": r["text"]} for r in read_rows([path])]
    else:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    by_idx, by_prompt = {}, {}
    for rec in records:
        text = rec.get("text")
        if text is None and "output_ids" in rec:
            text = decode_ids(rec["output_ids"])
        if text is None:
            continue
        by_idx.setdefault(int(rec["index"]), text)
        if rec.get("prompt"):
            by_prompt.setdefault(rec["prompt"], text)
    return (by_idx, by_prompt) if by_idx else None


def median(vals):
    v = [x for x in vals if not math.isnan(x)]
    return statistics.median(v) if v else math.nan


def fmt(v, prec=1):
    return "-" if math.isnan(v) else f"{v:.{prec}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tsv", nargs="*", help="input TSV files (default bench/results/*.tsv)")
    ap.add_argument("--golden", default=os.path.join(REPO, "tools", "reference", "golden"),
                    help="fp32 reference: directory with index.json / pNN.json, or a TSV")
    ap.add_argument("--max-rows", type=int, default=15)
    ap.add_argument("--sort", choices=["e2e", "ttft", "name"], default="e2e")
    args = ap.parse_args()

    paths = args.tsv or sorted(glob.glob(os.path.join(REPO, "bench", "results", "*.tsv")))
    if not paths:
        sys.exit("summarize.py: no TSV files (run bench/run_competitors.sh first)")
    rows = read_rows(paths)
    if not rows:
        sys.exit("summarize.py: no rows")

    golden = load_golden(args.golden)
    prompts = load_prompts()

    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["engine"], r["variant"], r["mode"]), []).append(r)

    table = []
    for (engine, variant, mode), rs in groups.items():
        e2e = [num(r["e2e_ms"]) for r in rs]
        gens = [x for x in (num(r["gen_tokens"]) for r in rs) if not math.isnan(x)]
        entry = {
            "engine": engine, "variant": variant, "mode": mode, "n": len(rs),
            "e2e_med": median(e2e), "e2e_min": min((x for x in e2e if not math.isnan(x)), default=math.nan),
            "ttft_med": median([num(r["ttft_ms"]) for r in rs]),
            "dec_med": median([num(r["decode_tok_s"]) for r in rs]),
            "gen_mean": statistics.fmean(gens) if gens else math.nan,
            "match": None,
        }
        if golden:
            by_idx, by_prompt = golden
            texts = collections.defaultdict(list)
            for r in rs:
                texts[int(r["prompt_idx"])].append(r["text"])
            hit = tot = 0
            for i, ts in texts.items():
                prompt = prompts[i] if i < len(prompts) else None
                ref = by_prompt.get(prompt) if prompt in by_prompt else by_idx.get(i)
                if ref is None:
                    continue
                tot += 1
                hit += collections.Counter(ts).most_common(1)[0][0] == ref
            entry["match"] = (hit, tot) if tot else None
        table.append(entry)

    by_e2e = lambda e: (math.isnan(e["e2e_med"]), e["e2e_med"])  # noqa: E731
    key = {"e2e": by_e2e,
           "ttft": lambda e: (math.isnan(e["ttft_med"]), e["ttft_med"]),
           "name": lambda e: (e["engine"], e["variant"], e["mode"])}[args.sort]
    # Row cap: always keep the fastest variant of every (engine, mode), then fill by median e2e.
    ranked = sorted(table, key=by_e2e)
    shown = list({(e["engine"], e["mode"]): None for e in ranked})
    best = {}
    for e in ranked:
        best.setdefault((e["engine"], e["mode"]), e)
    chosen = [best[k] for k in shown][: args.max_rows]
    for e in ranked:
        if len(chosen) >= args.max_rows:
            break
        if all(e is not c for c in chosen):
            chosen.append(e)
    n_hidden = len(table) - len(chosen)
    table = sorted(chosen, key=key)

    cols = ["engine", "variant", "mode", "n", "e2e med ms", "e2e min ms", "TTFT med ms", "decode tok/s", "gen tok"]
    if golden:
        cols.append("exact vs fp32")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * 3 + ["---:"] * (len(cols) - 3)) + "|")
    for e in table:
        cells = [e["engine"], e["variant"], e["mode"], str(e["n"]), fmt(e["e2e_med"]), fmt(e["e2e_min"]),
                 fmt(e["ttft_med"]), fmt(e["dec_med"]), fmt(e["gen_mean"])]
        if golden:
            m = e["match"]
            cells.append("-" if m is None else f"{m[0] / m[1]:.2f} ({m[0]}/{m[1]})")
        print("| " + " | ".join(cells) + " |")
    if n_hidden:
        print(f"\n({n_hidden} rows not shown: the fastest variant of each engine/mode is always listed, "
              f"the rest by median e2e; --max-rows to change)")
    if not golden:
        print(f"\n(no fp32 reference at {args.golden}; exact-match column omitted)")


if __name__ == "__main__":
    main()
