#!/usr/bin/env python3
"""Run mlx-lm greedily on the golden prompts and write the engine-output layout, for
validating the numpy reference against an independent implementation.

The prompt ids are the golden input_ids (the mlx tokenizer's chat-template ids are checked
against them). Tokens come from mlx_lm.generate.generate_step with an argmax sampler, the
same path as `mlx_lm generate --temp 0`. The last-prompt-position logits are recomputed the
way generate_step computes them (prefill of n-1 tokens, then one step for the last token).
forced_argmax comes from one batched forward over prompt + golden tokens.

GPU work: run under the shared lock, e.g.
  lockf -k /tmp/instantllm-gpu.lock .venv/bin/python -I tools/reference/run_mlx.py
  lockf -k /tmp/instantllm-gpu.lock .venv/bin/python -I tools/reference/run_mlx.py --dtype float32
  .venv/bin/python -I tools/reference/run_mlx.py --dtype float32 --device cpu   # CPU, no lock
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcommon as rc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=rc.REPO / "models" / "mlx" / "bf16")
    ap.add_argument("--dtype", choices=["native", "float32"], default="native",
                    help="float32 casts all weights to f32 (an independent fp32 implementation)")
    ap.add_argument("--golden", type=Path, default=rc.GOLDEN_DIR)
    ap.add_argument("--out", type=Path, help="default scratch/reference/validation/mlx_<model>[_f32][_cpu]")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                    help="cpu: full-precision fp32 matmuls (mlx's M5 GPU fp32 GEMM is TF32-like)")
    ap.add_argument("--prompts", default="all")
    args = ap.parse_args()

    import mlx.core as mx
    # must precede importing mlx_lm.generate, which creates its stream on the default device
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    out = args.out or rc.SCRATCH_DIR / "validation" / (
        f"mlx_{args.model.name}" + ("_f32" if args.dtype == "float32" else "")
        + ("_cpu" if args.device == "cpu" else ""))
    model, tokenizer = load(str(args.model))
    if args.dtype == "float32":
        model.set_dtype(mx.float32)
    system = rc.load_system()
    hf_tok = rc.load_tokenizer()  # decode text exactly as the golden data does
    index = rc.read_json(args.golden / "index.json")
    sel = [r["index"] for r in index] if args.prompts == "all" else [int(s) for s in args.prompts.split(",")]

    def argmax_sampler(logprobs):
        return mx.argmax(logprobs, axis=-1)

    for i in sel:
        g = rc.read_json(args.golden / f"{rc.pname(i)}.json")
        ids = list(g["input_ids"])
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": g["prompt"]}]
        mids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        if not isinstance(mids, list):
            mids = list(mids["input_ids"])
        t0 = time.perf_counter()

        # greedy tokens through the stock generation loop
        out_ids, stop, step_top5 = [], None, []
        for tok, logprobs in generate_step(mx.array(ids), model, max_tokens=rc.MAX_NEW, sampler=argmax_sampler):
            lp = np.array(logprobs.astype(mx.float32))
            step_top5.append(rc.topk(lp, 5))
            if tok in rc.STOP_IDS:
                stop = int(tok)
                break
            out_ids.append(int(tok))

        # raw logits at the last prompt position, computed as generate_step does
        cache = make_prompt_cache(model)
        model(mx.array(ids[:-1])[None], cache=cache)
        last = model(mx.array(ids[-1:])[None], cache=cache)[0, -1].astype(mx.float32)
        last = np.array(last)

        # teacher-forced argmax over prompt + golden tokens (one batched forward)
        gseq = rc.greedy_seq(g)
        full = model(mx.array(ids + gseq[:-1])[None])[0, len(ids) - 1:]
        forced = [int(t) for t in np.array(mx.argmax(full, axis=-1))]
        dt = time.perf_counter() - t0

        rec = {
            "index": i, "engine": f"mlx-lm {args.model.name} ({args.dtype}, {args.device})",
            "input_ids": ids, "tokenizer_input_ids_match": mids == ids,
            "output_ids": out_ids, "stop_token": stop,
            "finish_reason": "stop" if stop is not None else "length",
            "text": rc.decode(hf_tok, out_ids),
            "top10": rc.topk(last, 10),
            "step_top5": step_top5, "step_top5_kind": "log-probabilities (differences equal logit differences)",
            "forced_argmax": forced,
            "first_token_matches_last_logits": bool(step_top5) and step_top5[0][0][0] == int(np.argmax(last)),
        }
        rc.write_record(out, i, rec, last)
        print(f"[{i:2d}] {dt:5.2f}s tok_match={mids == ids} out={len(out_ids):2d} {rec['text']!r}", flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
