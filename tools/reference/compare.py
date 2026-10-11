#!/usr/bin/env python3
"""Compare an engine's output directory with the golden reference data.

The engine directory uses the golden layout (see README.md): per prompt NN,
  pNN.json        {"output_ids": [...], "stop_token": 151645|151643|null, ...}
  pNN.logits.bin  float32 little-endian [151936], logits at the last prompt position
Optional pNN.json fields: "input_ids" (tokenizer check), "text", "forced_argmax"
(teacher-forced argmax per step), "step_top5" (the engine's own top-5 per step).
Optional debug_pNN/<name>.bin files are compared with the golden dumps of the same name.

Usage:
  .venv/bin/python -I tools/reference/compare.py ENGINE_DIR [--golden DIR] [--json OUT]
      [--prompts 0,3] [--max-abs-tol 0.5] [--require-top1] [--require-greedy] [--quiet]

Exit status: 0 = ran (and gates passed), 1 = a requested gate failed, 2 = nothing to compare.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcommon as rc  # noqa: E402


def top_ids(logits: np.ndarray, k: int) -> list[int]:
    return [t for t, _ in rc.topk(logits, k)]


def golden_gap(step_top5: list, step: int, token: int):
    """(top1 - top2 margin, top1 - logit(token)) of the golden distribution at `step`;
    the second value is None when `token` is not in the golden top-5."""
    if step >= len(step_top5):
        return None, None
    st = step_top5[step]
    margin = st[0][1] - st[1][1]
    for t, l in st:
        if t == token:
            return margin, st[0][1] - l
    return margin, None


def compare_prompt(i: int, gdir: Path, edir: Path, near_tie: float) -> dict | None:
    g = rc.read_json(gdir / f"{rc.pname(i)}.json")
    ejson, ebin = edir / f"{rc.pname(i)}.json", edir / f"{rc.pname(i)}.logits.bin"
    if not ejson.exists() and not ebin.exists():
        return None
    e = rc.read_json(ejson) if ejson.exists() else {}
    r: dict = {"index": i}

    if "input_ids" in e:
        r["input_ids_match"] = list(e["input_ids"]) == list(g["input_ids"])

    if ebin.exists():
        gl = rc.read_f32(gdir / f"{rc.pname(i)}.logits.bin").astype(np.float64)
        el = rc.read_f32(ebin).astype(np.float64)
        if el.shape != gl.shape:
            raise SystemExit(f"{ebin}: {el.size} floats, expected {gl.size}")
        d = np.abs(el - gl)
        g5, e5 = top_ids(gl, 5), top_ids(el, 5)
        r.update(
            max_abs=float(d.max()), mean_abs=float(d.mean()),
            argmax_golden=g5[0], argmax_engine=e5[0],
            top1=g5[0] == e5[0], top5_overlap=len(set(g5) & set(e5)),
            nonfinite=int((~np.isfinite(el)).sum()),
        )

    if "output_ids" in e:
        has_stop = "stop_token" in e
        gseq = rc.greedy_seq(g) if has_stop else list(g["output_ids"])
        eseq = rc.greedy_seq(e) if has_stop else list(e["output_ids"])
        n = 0
        while n < min(len(gseq), len(eseq)) and gseq[n] == eseq[n]:
            n += 1
        agree = sum(1 for a, b in zip(gseq, eseq) if a == b)
        r.update(seq_exact=gseq == eseq, prefix_len=n, gold_len=len(gseq), eng_len=len(eseq),
                 token_agree=agree, includes_stop=has_stop)
        if "text" in e:
            r["text_match"] = e["text"] == g["text"]
        if gseq != eseq:
            div = {"step": n,
                   "golden_token": gseq[n] if n < len(gseq) else None,
                   "engine_token": eseq[n] if n < len(eseq) else None}
            if div["engine_token"] is not None and div["golden_token"] is not None:
                margin, gap = golden_gap(g["step_top5"], n, div["engine_token"])
                div.update(golden_margin=margin, golden_gap=gap,
                           near_tie=gap is not None and gap <= near_tie)
                est = e.get("step_top5")
                if est and n < len(est):
                    eg = {t: l for t, l in est[n]}
                    if div["golden_token"] in eg:
                        div["engine_gap"] = est[n][0][1] - eg[div["golden_token"]]
            r["divergence"] = div
        if "forced_argmax" in e:
            gfull = rc.greedy_seq(g)
            fa = list(e["forced_argmax"])[: len(gfull)]
            bad = [s for s, (a, b) in enumerate(zip(gfull, fa)) if a != b]
            bad += list(range(len(fa), len(gfull)))
            r["forced"] = {"steps": len(gfull), "agree": len(gfull) - len(bad), "mismatch": []}
            for s in bad:
                tokn = fa[s] if s < len(fa) else None
                margin, gap = golden_gap(g["step_top5"], s, tokn) if tokn is not None else (None, None)
                r["forced"]["mismatch"].append({"step": s, "golden_token": gfull[s], "engine_token": tokn,
                                                "golden_margin": margin, "golden_gap": gap})
    return r


def compare_debug(gdir: Path, edir: Path) -> list[dict]:
    rows = []
    for gd in sorted(gdir.glob("debug_p*")):
        ed = edir / gd.name
        if not ed.is_dir():
            continue
        index = rc.read_json(gd / "index.json")
        for name, info in index["tensors"].items():
            ef = ed / info["file"]
            if not ef.exists():
                continue
            ga = rc.read_f32(gd / info["file"]).astype(np.float64)
            ea = rc.read_f32(ef).astype(np.float64)
            row = {"dir": gd.name, "tensor": name, "shape": info["shape"]}
            if ea.size != ga.size:
                row["error"] = f"size {ea.size} != {ga.size}"
            else:
                d = np.abs(ea - ga)
                row.update(max_abs=float(d.max()), mean_abs=float(d.mean()),
                           rel=float(d.max() / max(np.abs(ga).max(), 1e-30)))
            rows.append(row)
    return rows


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def num(v, width=9):
    """Fixed-point for ordinary values, scientific for tiny ones (fp32-vs-fp64 level)."""
    if v is None:
        return "-".rjust(width)
    return (f"{v:.4f}" if v == 0 or v >= 0.01 else f"{v:.2e}").rjust(width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine_dir", type=Path)
    ap.add_argument("--golden", type=Path, default=rc.GOLDEN_DIR)
    ap.add_argument("--prompts", default="all", help="comma-separated indices or 'all'")
    ap.add_argument("--near-tie", type=float, default=0.5,
                    help="divergence counts as a near-tie when the golden logit of the engine's token is within this of the golden top-1 (default 0.5)")
    ap.add_argument("--json", type=Path, help="also write the full results as JSON")
    ap.add_argument("--max-abs-tol", type=float, help="gate: fail if any last-position max |dlogit| exceeds this")
    ap.add_argument("--require-top1", action="store_true", help="gate: fail on any last-position top-1 mismatch")
    ap.add_argument("--require-greedy", action="store_true", help="gate: fail on any greedy sequence mismatch")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    gdir, edir = args.golden, args.engine_dir
    index = rc.read_json(gdir / "index.json")
    sel = [r["index"] for r in index] if args.prompts == "all" else [int(s) for s in args.prompts.split(",")]
    gtext = {r["index"]: r["text"] for r in index}

    results = [r for i in sel if (r := compare_prompt(i, gdir, edir, args.near_tie)) is not None]
    if not results:
        print(f"no engine outputs found in {edir}", file=sys.stderr)
        return 2
    debug_rows = compare_debug(gdir, edir)

    if not args.quiet:
        print(f"golden: {gdir}\nengine: {edir}\n")
        print(f"{'p':>3} {'tok':>4} {'max|d|':>9} {'mean|d|':>9} {'top1':>4} {'top5':>4}  "
              f"{'seq':>5} {'prefix':>7} {'agree':>7}  divergence")
        for r in results:
            tokc = "-" if "input_ids_match" not in r else ("ok" if r["input_ids_match"] else "DIFF")
            top1 = "-" if "top1" not in r else ("ok" if r["top1"] else "NO")
            top5 = "-" if "top5_overlap" not in r else f"{r['top5_overlap']}/5"
            seq = "-" if "seq_exact" not in r else ("ok" if r["seq_exact"] else "DIFF")
            pre = "-" if "prefix_len" not in r else f"{r['prefix_len']}/{r['gold_len']}"
            agr = "-" if "token_agree" not in r else f"{r['token_agree']}/{r['gold_len']}"
            div = ""
            if "divergence" in r:
                dv = r["divergence"]
                div = f"step {dv['step']}: gold {dv['golden_token']} vs eng {dv['engine_token']}"
                if dv.get("golden_margin") is not None:
                    gap = "not in gold top5" if dv.get("golden_gap") is None else f"gap {dv['golden_gap']:.3f}"
                    div += f" (gold margin {dv['golden_margin']:.3f}, {gap}"
                    if dv.get("engine_gap") is not None:
                        div += f", eng gap {dv['engine_gap']:.3f}"
                    div += ", near-tie)" if dv.get("near_tie") else ")"
            print(f"{r['index']:>3} {tokc:>4} {num(r.get('max_abs'))} {num(r.get('mean_abs'))} "
                  f"{top1:>4} {top5:>4}  {seq:>5} {pre:>7} {agr:>7}  {div}")
        if any("forced" in r for r in results):
            print("\nteacher-forced argmax (engine fed the golden tokens):")
            for r in results:
                if "forced" in r:
                    f = r["forced"]
                    ms = ", ".join(f"step {m['step']} gold {m['golden_token']} eng {m['engine_token']}"
                                   f" (gap {fmt(m['golden_gap'], '.3f')})" for m in f["mismatch"])
                    print(f"{r['index']:>3} {f['agree']}/{f['steps']}  {ms}")
        mism = [r for r in results if r.get("seq_exact") is False]
        if mism:
            print("\ngreedy text of mismatching prompts:")
            for r in mism:
                e = rc.read_json(edir / f"{rc.pname(r['index'])}.json")
                print(f"{r['index']:>3} gold: {gtext[r['index']]!r}")
                if "text" in e:
                    print(f"    eng:  {e['text']!r}")
        if debug_rows:
            print("\ndebug tensors:")
            for d in debug_rows:
                if "error" in d:
                    print(f"  {d['dir']}/{d['tensor']:<16} {d['error']}")
                else:
                    print(f"  {d['dir']}/{d['tensor']:<16} {str(d['shape']):<14} max|d| {d['max_abs']:.3e}  "
                          f"mean|d| {d['mean_abs']:.3e}  max|d|/max|gold| {d['rel']:.3e}")
        print()

    # ---- summary ----
    summ: dict = {"prompts": len(results)}
    lg = [r for r in results if "max_abs" in r]
    if lg:
        summ.update(
            logits_compared=len(lg),
            max_abs_max=max(r["max_abs"] for r in lg),
            max_abs_mean=float(np.mean([r["max_abs"] for r in lg])),
            mean_abs_mean=float(np.mean([r["mean_abs"] for r in lg])),
            top1_agree=sum(r["top1"] for r in lg),
            top5_overlap_mean=float(np.mean([r["top5_overlap"] for r in lg])) / 5,
            top5_exact_sets=sum(r["top5_overlap"] == 5 for r in lg),
            nonfinite=sum(r["nonfinite"] for r in lg),
        )
    gr = [r for r in results if "seq_exact" in r]
    if gr:
        summ.update(
            greedy_compared=len(gr),
            seq_exact=sum(r["seq_exact"] for r in gr),
            token_agree=sum(r["token_agree"] for r in gr),
            gold_tokens=sum(r["gold_len"] for r in gr),
            prefix_tokens=sum(r["prefix_len"] for r in gr),
            divergences_near_tie=sum(bool(r.get("divergence", {}).get("near_tie")) for r in gr),
        )
    tk = [r for r in results if "input_ids_match" in r]
    if tk:
        summ["input_ids_match"] = sum(r["input_ids_match"] for r in tk)
        summ["input_ids_compared"] = len(tk)
    fr = [r for r in results if "forced" in r]
    if fr:
        summ["forced_agree"] = sum(r["forced"]["agree"] for r in fr)
        summ["forced_steps"] = sum(r["forced"]["steps"] for r in fr)

    print("summary:")
    if lg:
        print(f"  last-position logits ({summ['logits_compared']} prompts): max|d| max {num(summ['max_abs_max'], 0)}, "
              f"mean of max|d| {num(summ['max_abs_mean'], 0)}, mean|d| {num(summ['mean_abs_mean'], 0)}; "
              f"top-1 {summ['top1_agree']}/{len(lg)}; top-5 overlap {100 * summ['top5_overlap_mean']:.1f}% "
              f"(identical sets {summ['top5_exact_sets']}/{len(lg)})"
              + (f"; NON-FINITE {summ['nonfinite']}" if summ["nonfinite"] else ""))
    if gr:
        print(f"  greedy: exact sequences {summ['seq_exact']}/{len(gr)}; per-token (position-wise) "
              f"{summ['token_agree']}/{summ['gold_tokens']} = {100 * summ['token_agree'] / max(summ['gold_tokens'], 1):.1f}%; "
              f"common prefix {summ['prefix_tokens']}/{summ['gold_tokens']}; "
              f"near-tie divergences {summ['divergences_near_tie']}/{len(gr) - summ['seq_exact']}")
    if fr:
        print(f"  teacher-forced per-token agreement: {summ['forced_agree']}/{summ['forced_steps']} "
              f"= {100 * summ['forced_agree'] / max(summ['forced_steps'], 1):.1f}%")
    if tk:
        print(f"  input_ids identical: {summ['input_ids_match']}/{summ['input_ids_compared']}")

    if args.json:
        rc.write_json(args.json, {"summary": summ, "prompts": results, "debug": debug_rows})

    failed = []
    if args.max_abs_tol is not None and lg and summ["max_abs_max"] > args.max_abs_tol:
        failed.append(f"max|d| {summ['max_abs_max']:.4f} > {args.max_abs_tol}")
    if args.require_top1 and lg and summ["top1_agree"] < len(lg):
        failed.append("top-1 mismatch")
    if args.require_greedy and gr and summ["seq_exact"] < len(gr):
        failed.append("greedy sequence mismatch")
    if lg and summ["nonfinite"]:
        failed.append("non-finite logits")
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
