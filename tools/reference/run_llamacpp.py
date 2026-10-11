#!/usr/bin/env python3
"""Run llama.cpp (libllama C API via llama_dump.c) greedily on the golden prompts and write
the engine-output layout, for validating the numpy reference.

Builds llama_dump.c with clang into scratch/reference/bin/ (when missing or stale), feeds
it the golden input ids, then converts its output. Context defaults are kept (Metal, flash
attention auto, f16 KV cache); prompt ids are the golden ones, and the llama.cpp tokenizer's
ids for the same text are recorded for a separate check.

The whole script uses the GPU; run it under the shared lock (it does not lock by itself):
  lockf -k /tmp/instantllm-gpu.lock .venv/bin/python -I tools/reference/run_llamacpp.py
  lockf -k /tmp/instantllm-gpu.lock .venv/bin/python -I tools/reference/run_llamacpp.py \
      --gguf models/gguf/qwen2.5-coder-0.5b-instruct-q8_0.gguf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refcommon as rc  # noqa: E402

LLAMA = Path("/opt/homebrew/opt/llama.cpp")
GGML = Path("/opt/homebrew/opt/ggml")  # llama.h includes ggml.h from the separate ggml formula


def build(src: Path, exe: Path) -> None:
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["clang", "-O2", "-std=c11", "-Wall", f"-I{LLAMA / 'include'}", f"-I{GGML / 'include'}", str(src),
           f"-L{LLAMA / 'lib'}", "-lllama", f"-Wl,-rpath,{LLAMA / 'lib'}", "-o", str(exe)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=rc.REPO / "models" / "gguf" / "qwen2.5-coder-0.5b-instruct-fp16.gguf")
    ap.add_argument("--golden", type=Path, default=rc.GOLDEN_DIR)
    ap.add_argument("--out", type=Path, help="default scratch/reference/validation/llamacpp_<gguf name suffix>")
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers (0 = CPU only)")
    ap.add_argument("--prompts", default="all")
    args = ap.parse_args()

    quant = args.gguf.stem.replace("qwen2.5-coder-0.5b-instruct-", "")  # fp16, q8_0, hf-f16, ...
    out = args.out or rc.SCRATCH_DIR / "validation" / (f"llamacpp_{quant}" + ("_cpu" if args.ngl == 0 else ""))
    out.mkdir(parents=True, exist_ok=True)
    exe = rc.SCRATCH_DIR / "bin" / "llama_dump"
    build(rc.HERE / "llama_dump.c", exe)

    system = rc.load_system()
    hf_tok = rc.load_tokenizer()
    index = rc.read_json(args.golden / "index.json")
    sel = [r["index"] for r in index] if args.prompts == "all" else [int(s) for s in args.prompts.split(",")]
    goldens = {i: rc.read_json(args.golden / f"{rc.pname(i)}.json") for i in sel}
    jobs = out / "jobs.txt"
    with open(jobs, "w") as f:
        for i, g in goldens.items():
            gseq = rc.greedy_seq(g)
            f.write(" ".join(map(str, [i, len(g["input_ids"]), *g["input_ids"], len(gseq), *gseq])) + "\n")
            (out / f"{rc.pname(i)}.prompt.txt").write_bytes(rc.chat_text(system, g["prompt"]).encode("utf-8"))

    t0 = time.perf_counter()
    subprocess.run([str(exe), str(args.gguf), str(jobs), str(out), str(args.ngl), str(rc.MAX_NEW)], check=True)
    print(f"llama_dump: {time.perf_counter() - t0:.1f}s", flush=True)

    for i, g in goldens.items():
        fields = {}
        top5 = []
        for line in (out / f"{rc.pname(i)}.txt").read_text().splitlines():
            key, _, rest = line.partition(" ")
            if key == "top5":
                top5.append([[int(a), float(b)] for a, b in (p.split(":") for p in rest.split())])
            else:
                fields[key] = [int(x) for x in rest.split()]
        n_vocab = fields["n_vocab"][0]
        if n_vocab != rc.VOCAB:
            raise SystemExit(f"llama.cpp n_vocab {n_vocab} != {rc.VOCAB}")
        stop = fields["stop"][0]
        out_ids = fields.get("out", [])
        logits = rc.read_f32(out / f"{rc.pname(i)}.logits.bin")
        rec = {
            "index": i, "engine": f"llama.cpp {args.gguf.name} (ngl={args.ngl})",
            "input_ids": g["input_ids"],
            "llama_tokenizer_ids": fields.get("tok"),
            "tokenizer_input_ids_match": fields.get("tok") == g["input_ids"],
            "output_ids": out_ids, "stop_token": None if stop < 0 else stop,
            "finish_reason": "stop" if stop >= 0 else "length",
            "text": rc.decode(hf_tok, out_ids),
            "top10": rc.topk(logits, 10),
            "step_top5": top5, "step_top5_kind": "logits",
            "forced_argmax": fields["forced"],
        }
        rc.write_record(out, i, rec, logits)
        print(f"[{i:2d}] tok_match={rec['tokenizer_input_ids_match']} out={len(out_ids):2d} {rec['text']!r}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
