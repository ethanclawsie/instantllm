"""Generate the model's greedy outputs for scratch/draft/prompts.jsonl with our engine.

Runs `instantllm batch` on chunks of ~300 prompts, each chunk under the shared GPU lock
(so the lock is never held for long), resumable. Writes scratch/draft/outputs_<tag>.jsonl
with {"id", "prompt", "split", "group", "ids", "stop"}.

usage: .venv/bin/python -I tools/draft/run_outputs.py --model models/ill/qwen-q4.ill --tag q4
       .venv/bin/python -I tools/draft/run_outputs.py --model models/ill/qwen-q4.ill --tag q4 --bench
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY = os.path.join(ROOT, ".build/release/instantllm")


def provenance(model):
    """Identity of the model file and engine binary used (outputs depend on both)."""
    import hashlib
    h = hashlib.sha256()
    with open(os.path.join(ROOT, model), "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    hb = hashlib.sha256(open(BINARY, "rb").read()).hexdigest()
    return {"model": model, "model_sha256": h.hexdigest(), "binary": BINARY, "binary_sha256": hb,
            "time": time.time()}


def run_chunk(model, pfile, ofile):
    cmd = ["lockf", "-k", "/tmp/instantllm-gpu.lock", BINARY,
           "batch", "--prompts", pfile, "--out", ofile + ".tmp", "--model", model]
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    os.replace(ofile + ".tmp", ofile)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ill/qwen-q4.ill")
    ap.add_argument("--tag", default="q4")
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--bench", action="store_true", help="run bench/prompts.txt instead")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "scratch/draft/prompts.jsonl"))
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between chunks (lets others take the lock)")
    ap.add_argument("--binary", default=BINARY, help="engine binary (use a snapshot so rebuilds cannot interfere)")
    a = ap.parse_args()
    globals()["BINARY"] = os.path.abspath(a.binary)
    d = os.path.join(ROOT, "scratch/draft")
    if a.bench:
        out = os.path.join(d, f"bench_{a.tag}.jsonl")
        bp = os.path.join(ROOT, "bench/prompts.txt")
        tmp = os.path.join(d, f"bench_{a.tag}.raw.jsonl")
        dt = run_chunk(a.model, bp, tmp)
        prompts = [l for l in open(bp).read().split("\n") if l]
        rows = [json.loads(l) for l in open(tmp)]
        assert len(rows) == len(prompts)
        with open(out, "w") as f:
            for i, (p, r) in enumerate(zip(prompts, rows)):
                assert r["prompt"] == p
                f.write(json.dumps({"id": i, "prompt": p, "split": "bench", "group": f"bench:{i}",
                                    "ids": r["ids"], "stop": r["stop"]}) + "\n")
        json.dump(provenance(a.model), open(out + ".prov.json", "w"), indent=1)
        print(f"bench {a.tag}: {len(rows)} outputs in {dt:.1f}s -> {out}")
        return
    recs = [json.loads(l) for l in open(a.prompts)]
    cdir = os.path.join(d, f"chunks_{a.tag}")
    prov0 = provenance(a.model)
    pf = os.path.join(cdir, "provenance.json")
    if os.path.exists(pf):
        old = json.load(open(pf))
        if old["model_sha256"] != prov0["model_sha256"] or old.get("binary_sha256") != prov0["binary_sha256"]:
            print("model or binary changed since earlier chunks: restarting this tag", flush=True)
            for fn in os.listdir(cdir):
                if fn.endswith(".jsonl"):
                    os.remove(os.path.join(cdir, fn))
    os.makedirs(cdir, exist_ok=True)
    json.dump(prov0, open(pf, "w"), indent=1)
    os.makedirs(cdir, exist_ok=True)
    chunks = [recs[i:i + a.chunk] for i in range(0, len(recs), a.chunk)]
    for ci, ch in enumerate(chunks):
        pfile = os.path.join(cdir, f"c{ci:03d}.txt")
        ofile = os.path.join(cdir, f"c{ci:03d}.jsonl")
        body = "\n".join(r["prompt"] for r in ch) + "\n"
        if not os.path.exists(pfile) or open(pfile).read() != body:
            open(pfile, "w").write(body)
            if os.path.exists(ofile):
                os.remove(ofile)
        if os.path.exists(ofile) and sum(1 for _ in open(ofile)) == len(ch):
            continue
        cur = provenance(a.model) if ci % 10 == 9 else prov0
        if cur["model_sha256"] != prov0["model_sha256"]:
            sys.exit(f"model or binary changed during the run (chunk {ci}); rerun to restart this tag")
        dt = run_chunk(a.model, pfile, ofile)
        print(f"chunk {ci + 1}/{len(chunks)}: {len(ch)} prompts in {dt:.1f}s", flush=True)
        time.sleep(a.pause)
    out = os.path.join(d, f"outputs_{a.tag}.jsonl")
    n = 0
    with open(out, "w") as f:
        for ci, ch in enumerate(chunks):
            rows = [json.loads(l) for l in open(os.path.join(cdir, f"c{ci:03d}.jsonl"))]
            assert len(rows) == len(ch), (ci, len(rows), len(ch))
            for r, row in zip(ch, rows):
                assert row["prompt"] == r["prompt"], (row["prompt"], r["prompt"])
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "split": r["split"], "group": r["group"],
                                    "ids": row["ids"], "stop": row["stop"]}) + "\n")
                n += 1
    json.dump(provenance(a.model), open(out + ".prov.json", "w"), indent=1)
    print(f"wrote {n} outputs -> {out}")


if __name__ == "__main__":
    sys.exit(main())
