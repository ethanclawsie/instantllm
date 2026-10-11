"""Cold-start benchmark: one fresh process per request, timed from spawn to exit, interleaved per
prompt across engines so drift affects all of them equally.

  .venv/bin/python -I bench/cold.py --configs iq4 lq40 m4 --rounds 2 --out bench/results

instantllm runs its one-shot CLI (`instantllm run`); llama.cpp and mlx-lm run their harnesses' cold
modes (which spawn llama-completion / `python -m mlx_lm generate` and time them the same way).
Writes <out>/<config>-cold.tsv in the shared schema.
"""
import argparse, fcntl, os, selectors, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = "engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext"
LOCK = []  # the GPU lock is held by this script around each request (clock starts after acquiring it)
LOCKFILE = "/tmp/instantllm-gpu.lock"


class gpu_lock:
    def __enter__(self):
        self.fd = os.open(LOCKFILE, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def __exit__(self, *a):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
PY = [os.path.join(ROOT, ".venv/bin/python"), "-I"]


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def run_instantllm(variant, prompt):
    cmd = [os.path.join(ROOT, ".build/release/instantllm"), "run", "--model", f"models/ill/qwen-{variant}.ill",
           "--system-file", "bench/system.txt", "--stats", prompt]
    t0 = time.perf_counter()
    p = subprocess.Popen(LOCK + cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    first, out = None, b""
    while True:
        if sel.select(timeout=30):
            chunk = os.read(p.stdout.fileno(), 65536)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter()
            out += chunk
    p.wait()
    t1 = time.perf_counter()
    err = p.stderr.read().decode()
    stats = dict(kv.split("=") for kv in err.split() if "=" in kv)
    text = out.decode().rstrip("\n")
    gen = int(stats.get("gen_tokens", "-1"))
    dec = stats.get("decode_tok_s", "nan")
    return f"instantllm\t{variant}\tcold\t{{i}}\t{{r}}\t{(t1 - t0) * 1e3:.3f}\t{((first or t1) - t0) * 1e3:.3f}\t{stats.get('prefill', 'nan')}\t{gen}\t{dec}\t{esc(text)}"


def run_harness(engine, variant, i):
    if engine == "llamacpp":
        cmd = ["bench/competitors/llamacpp/bench", "--model", f"models/gguf/qwen2.5-coder-0.5b-instruct-{variant}.gguf",
               "--mode", "cold", "--reps", "1", "--prompt-idx", str(i), "--no-header"]
    else:
        cmd = PY + ["bench/competitors/mlx/bench.py", "--variant", variant, "--mode", "cold", "--reps", "1",
                    "--prompt-idx", str(i), "--no-header"]
    out = subprocess.run(LOCK + cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    return out[-1] if out else None


CONFIGS = {"iq4": ("instantllm", "q4"), "iq8": ("instantllm", "q8"), "if16": ("instantllm", "f16"),
           "lq40": ("llamacpp", "q4_0"), "lq4km": ("llamacpp", "q4_k_m"), "lq80": ("llamacpp", "q8_0"),
           "lf16": ("llamacpp", "fp16"), "m4": ("mlx", "4bit"), "m8": ("mlx", "8bit"), "mbf16": ("mlx", "bf16")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["iq4", "lq40", "m4"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--prompts", default="bench/prompts.txt")
    a = ap.parse_args()
    prompts = [l for l in open(os.path.join(ROOT, a.prompts)).read().split("\n") if l]
    files = {}
    for c in a.configs:
        path = os.path.join(ROOT, a.out, f"{c}-cold.tsv")
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        files[c] = open(path, "a")
        if new:
            files[c].write(HEADER + "\n")
    for r in range(a.rounds):
        for i, prompt in enumerate(prompts):
            order = a.configs[(i + r) % len(a.configs):] + a.configs[:(i + r) % len(a.configs)]
            for c in order:
                engine, variant = CONFIGS[c]
                with gpu_lock():
                    if engine == "instantllm":
                        row = run_instantllm(variant, prompt).replace("{i}", str(i)).replace("{r}", str(r))
                    else:
                        row = run_harness(engine, variant, i)
                    if row:
                        f = row.split("\t"); f[4] = str(r); row = "\t".join(f)
                if row:
                    files[c].write(row + "\n"); files[c].flush()
                print(f"cold r{r} p{i} {c}: {row.split(chr(9))[5] if row else 'FAILED'} ms", file=sys.stderr)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
