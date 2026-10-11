"""mlx_lm competitor harness for instantllm (mlx 0.32.3 + mlx-lm 0.32.0).

Run from the repo root with the project venv:
    .venv/bin/python -I bench/competitors/mlx/bench.py --variant 4bit --mode full

Modes (one TSV row per request on stdout, readable progress on stderr):
  full    A fresh KV cache (make_prompt_cache) per request; the whole templated prompt is prefilled.
  prefix  "<|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\n" is prefilled once into a
          prompt cache that is kept. Each request trims the cache back to that prefix
          (trim_prompt_cache) and prefills only "{PROMPT}<|im_end|>\\n<|im_start|>assistant\\n".
  cold    A fresh `python -I -m mlx_lm generate` process per request (greedy, --temp 0, mlx_lm's own
          chat template with --system-prompt, verified identical to the protocol template), timed
          spawn -> exit. --verbose True is used so the CLI reports token counts; the harness keeps
          only the generated text between the CLI's "==========" lines.

Generation uses mlx_lm.generate.generate_step (mlx_lm's pipelined decode loop, default greedy
argmax sampler), stopping at 151645/151643, at most --n-predict (64) sampled tokens. The weights
stay wired (mx.set_wired_limit once at startup, as stream_generate does per call). After each
request the harness calls mx.synchronize() outside the timed region, so the step that
generate_step speculatively queued past the stop token cannot leak into the next request.

Metrics (full/prefix):
  e2e_ms        prompt string -> final text: template + tokenize + cache reset + prefill + greedy
                decode + detokenize (tokenizer.decode of the generated ids).
  ttft_ms       prompt string -> first generated token id on the host.
  gen_tokens    sampled tokens including the stop token when one is emitted (<= n_predict).
  decode_tok_s  (gen_tokens - 1) / (t_last_token - t_first_token); nan when gen_tokens == 1.
cold: e2e_ms = spawn -> exit; ttft_ms = spawn -> first stdout bytes (the CLI flushes when it prints
  the first token); prefill/gen tokens from the CLI's "Prompt:"/"Generation:" lines; decode_tok_s
  converted from the CLI's generation tps to the (gen_tokens - 1) convention above.
"""

import argparse
import os
import re
import selectors
import statistics
import subprocess
import sys
import time

STOP = (151645, 151643)  # <|im_end|>, <|endoftext|>
PREFIX_HEAD = "<|im_start|>system\n"
PREFIX_TAIL = "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
HEADER = "engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext"
ENGINE = "mlx_lm"


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def fnum(v, prec):
    return "nan" if v is None or v != v else f"{v:.{prec}f}"


def fint(v):
    return "nan" if v is None or v < 0 else str(v)


def emit(row):
    print("\t".join([
        ENGINE, row["variant"], row["mode"], str(row["prompt_idx"]), str(row["rep"]),
        fnum(row["e2e"], 3), fnum(row["ttft"], 3), fint(row["prefill"]), fint(row["gen"]),
        fnum(row["decode"], 2), esc(row["text"]),
    ]), flush=True)
    print(f"[{row['variant']} {row['mode']} p{row['prompt_idx']} r{row['rep']}] e2e {row['e2e']:8.2f} ms  "
          f"ttft {fnum(row['ttft'], 2):>8} ms  prefill {fint(row['prefill']):>3}  gen {fint(row['gen']):>3}  "
          f"decode {fnum(row['decode'], 1):>7} tok/s | {esc(row['text'])}", file=sys.stderr, flush=True)


def summary(label, rows):
    def med(key):
        v = [r[key] for r in rows if r[key] is not None and r[key] == r[key]]
        return statistics.median(v) if v else float("nan")
    e2e = [r["e2e"] for r in rows]
    print(f"# {label}: n={len(rows)}  e2e median {med('e2e'):.2f} ms (min {min(e2e):.2f})  "
          f"ttft median {med('ttft'):.2f} ms  decode median {med('decode'):.1f} tok/s", file=sys.stderr, flush=True)


def load_inputs(args):
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [ln.rstrip("\r") for ln in f.read().split("\n") if ln.rstrip("\r")]
    with open(args.system, encoding="utf-8") as f:
        system = f.read().rstrip("\r\n")
    idx = [int(x) for x in args.prompt_idx.split(",") if x != ""] if args.prompt_idx else list(range(len(prompts)))
    for i in idx:
        if not 0 <= i < len(prompts):
            sys.exit(f"bench.py: prompt index {i} out of range")
    return prompts, system, idx


# ---------------------------------------------------------------------------------------------
# in-process (full / prefix)

def run_warm(args, prompts, system, idx):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    model, tokenizer = load(args.model_dir)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    encode = tokenizer.encode
    decode = tokenizer.decode
    prefix = PREFIX_HEAD + system + PREFIX_TAIL
    prefix_mode = args.mode == "prefix"
    n_predict = args.n_predict

    pcache = None
    n_prefix = 0
    if prefix_mode:
        pids = encode(prefix, add_special_tokens=False)
        for i, p in enumerate(prompts):  # untimed: the split must not change the model input
            if encode(prefix + p + SUFFIX, add_special_tokens=False) != pids + encode(p + SUFFIX, add_special_tokens=False):
                print(f"bench.py: WARNING prompt {i}: prefix+suffix tokens differ from full tokens", file=sys.stderr)
        pcache = make_prompt_cache(model)
        model(mx.array(pids)[None], cache=pcache)
        mx.eval([c.state for c in pcache])
        n_prefix = len(pids)
        print(f"# prefix: {n_prefix} tokens cached", file=sys.stderr)

    def run(prompt):
        t0 = time.perf_counter()
        if prefix_mode:
            ids = encode(prompt + SUFFIX, add_special_tokens=False)
            trim_prompt_cache(pcache, pcache[0].offset - n_prefix)
            cache = pcache
        else:
            ids = encode(prefix + prompt + SUFFIX, add_special_tokens=False)
            cache = make_prompt_cache(model)
        out = []
        n_gen = 0
        t1 = t_last = None
        for tok, _ in generate_step(mx.array(ids), model, max_tokens=n_predict, prompt_cache=cache):
            t_last = time.perf_counter()
            n_gen += 1
            if n_gen == 1:
                t1 = t_last
            if tok in STOP and not args.ignore_eos:
                break
            out.append(tok)
        text = decode(out, skip_special_tokens=False)  # same rendering as the fp32 reference
        t2 = time.perf_counter()
        mx.synchronize()  # untimed: drain the step generate_step queued past the last token
        return {
            "e2e": (t2 - t0) * 1e3, "ttft": (t1 - t0) * 1e3, "prefill": len(ids), "gen": n_gen,
            "decode": (n_gen - 1) / (t_last - t1) if n_gen > 1 else float("nan"), "text": text,
        }

    n_warmup = len(idx) if args.warmup < 0 else args.warmup
    print(f"# mlx_lm {args.variant} {args.mode}: {args.model_dir}", file=sys.stderr)
    for w in range(n_warmup):
        run(prompts[idx[w % len(idx)]])
    rows = []
    for rep in range(args.reps):
        for i in idx:
            r = run(prompts[i])
            r.update(variant=args.variant, mode=args.mode, prompt_idx=i, rep=args.rep_start + rep)
            emit(r)
            rows.append(r)
    summary(f"mlx_lm {args.variant} {args.mode}", rows)


# ---------------------------------------------------------------------------------------------
# cold: fresh `python -m mlx_lm generate` processes

SEP = "=" * 10


def spawn_capture(cmd):
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ, "out")
    sel.register(p.stderr, selectors.EVENT_READ, "err")
    out, err, t_first = bytearray(), bytearray(), None
    while sel.get_map():
        for key, _ in sel.select():
            data = os.read(key.fileobj.fileno(), 1 << 16)
            if not data:
                sel.unregister(key.fileobj)
                continue
            if key.data == "out":
                if t_first is None:
                    t_first = time.perf_counter()
                out += data
            else:
                err += data
    rc = p.wait()
    t_exit = time.perf_counter()
    return rc, (t_exit - t0) * 1e3, ((t_first - t0) * 1e3 if t_first else float("nan")), out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def run_cold(args, prompts, system, idx):
    def cmd(prompt):
        return [sys.executable, "-I", "-m", "mlx_lm", "generate", "--model", args.model_dir,
                "--system-prompt", system, "--prompt", prompt, "--max-tokens", str(args.n_predict),
                "--temp", "0", "--extra-eos-token", "<|endoftext|>", "--verbose", "True"]

    def one(prompt):
        rc, e2e, ttft, out, err = spawn_capture(cmd(prompt))
        if rc != 0:
            sys.stderr.write(err)
            sys.exit(f"bench.py: mlx_lm generate exited with {rc}")
        head = SEP + "\n"
        i = out.find(head)
        j = out.rfind("\n" + SEP + "\n")
        if i < 0:
            sys.exit(f"bench.py: unexpected mlx_lm output: {out[:200]!r}")
        text = out[i + len(head):j] if j > i else ""
        mp = re.search(r"^Prompt: (\d+) tokens", out, re.M)
        mg = re.search(r"^Generation: (\d+) tokens, ([\d.]+) tokens-per-sec", out, re.M)
        gen = int(mg.group(1)) if mg else -1
        tps = float(mg.group(2)) if mg else float("nan")
        return {
            "e2e": e2e, "ttft": ttft, "prefill": int(mp.group(1)) if mp else -1, "gen": gen,
            "decode": tps * (gen - 1) / gen if gen > 1 else float("nan"), "text": text,
        }

    print(f"# mlx_lm {args.variant} cold: {' '.join(cmd('<prompt>'))}", file=sys.stderr)
    if args.warmup != 0:
        one(prompts[idx[0]])  # untimed: warm the file cache and the Metal shader cache
    rows = []
    for rep in range(args.reps):
        for i in idx:
            r = one(prompts[i])
            r.update(variant=args.variant, mode="cold", prompt_idx=i, rep=args.rep_start + rep)
            emit(r)
            rows.append(r)
    summary(f"mlx_lm {args.variant} cold", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="4bit", help="models/mlx/<variant> (4bit, 8bit, bf16)")
    ap.add_argument("--model-dir", default=None, help="model directory (default models/mlx/<variant>)")
    ap.add_argument("--mode", choices=["full", "prefix", "cold"], default="full")
    ap.add_argument("--prompts", default="bench/prompts.txt")
    ap.add_argument("--system", default="bench/system.txt")
    ap.add_argument("--prompt-idx", default="", help="comma-separated 0-based subset (default all)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-start", type=int, default=0, help="label of the first rep")
    ap.add_argument("--warmup", type=int, default=-1, help="untimed requests first (-1 = one pass over the prompts)")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--ignore-eos", action="store_true", help="always generate exactly n-predict tokens")
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()
    args.model_dir = args.model_dir or os.path.join("models", "mlx", args.variant)
    prompts, system, idx = load_inputs(args)
    if not args.no_header:
        print(HEADER, flush=True)
    if args.mode == "cold":
        run_cold(args, prompts, system, idx)
    else:
        run_warm(args, prompts, system, idx)


if __name__ == "__main__":
    main()
