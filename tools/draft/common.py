"""Shared helpers for the n-gram drafter tools: data loading, prompt tokenization, and the
exact speculative-decoding simulator.

Conventions
- An output record has "ids" (generated ids, stop excluded) and "stop" (151645/151643, or -1
  when 64 tokens were generated without a stop). The sequence the verifier must reproduce is
  y = ids + [stop] (stop included when present).
- The prefill pass yields y[0]. Every verify step feeds [last token, d1..dk] and returns the
  model's greedy token at each position; we keep the longest matching draft prefix (a tokens)
  plus one verified token, so each step yields a+1 tokens (capped at the end of y).
- "Context" handed to a drafter = TAIL + y[:p], where TAIL is the fixed chat-template tail
  <|im_end|>\\n<|im_start|>assistant\\n that precedes every output (so short n-gram contexts at
  the start of an output are anchored at "beginning of answer").
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DDIR = os.path.join(ROOT, "scratch/draft")
IM_START, IM_END, NL, ASSISTANT = 151644, 151645, 198, 77091
EOS2 = 151643
TAIL = [IM_END, NL, IM_START, ASSISTANT, NL]
PREFIX_LEN = 44
STOPS = (IM_END, EOS2)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(os.path.join(ROOT, "models/hf/tokenizer.json"))
    return _tok


def system_text():
    return open(os.path.join(ROOT, "bench/system.txt")).read()


def chat_ids(prompt, system=None):
    system = system_text() if system is None else system
    s = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return tokenizer().encode(s, add_special_tokens=False).ids


def user_ids(prompt):
    """Token ids of the user prompt inside the exact chat template (between the 44-token
    system prefix and the 5-token template tail)."""
    ids = chat_ids(prompt)
    assert ids[-5:] == TAIL, ids[-5:]
    return ids[PREFIX_LEN:-5]


def attach_prompt_ids(recs, cache_path=None):
    cache = {}
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    changed = False
    for r in recs:
        if r["prompt"] not in cache:
            cache[r["prompt"]] = user_ids(r["prompt"])
            changed = True
        r["pids"] = cache[r["prompt"]]
    if cache_path and changed:
        json.dump(cache, open(cache_path, "w"))
    return recs


def seq(r):
    y = list(r["ids"])
    if r.get("stop", -1) in STOPS:
        y.append(r["stop"])
    return y


def load_split(tag="q4", extra=("x",)):
    """Returns (train, heldout, bench) record lists with "pids" and "y" attached.
    extra: suffixes of additional TRAIN-only output files (outputs_<tag><suffix>.jsonl)."""
    cache = os.path.join(DDIR, "prompt_ids_cache.json")
    out = load_jsonl(os.path.join(DDIR, f"outputs_{tag}.jsonl"))
    for e in extra:
        f = os.path.join(DDIR, f"outputs_{tag}{e}.jsonl")
        if os.path.exists(f):
            xs = load_jsonl(f)
            assert all(r["split"] == "train" for r in xs)
            out += xs
    bench = load_jsonl(os.path.join(DDIR, f"bench_{tag}.jsonl"))
    attach_prompt_ids(out + bench, cache)
    for r in out + bench:
        r["y"] = seq(r)
    train = [r for r in out if r["split"] == "train"]
    held = [r for r in out if r["split"] == "heldout"]
    return train, held, bench


def load_bench(tag):
    cache = os.path.join(DDIR, "prompt_ids_cache.json")
    bench = load_jsonl(os.path.join(DDIR, f"bench_{tag}.jsonl"))
    attach_prompt_ids(bench, cache)
    for r in bench:
        r["y"] = seq(r)
    return bench


# ------------------------------------------------------------------------------------------
# simulator

def simulate_one(drafter, r, K, prefill_draft=False):
    """Returns (steps, produced, accepted_per_step list). produced = tokens produced by verify
    steps (len(y)-1 normally; len(y) with prefill drafting, where the prefill is a step)."""
    y = r["y"]
    n = len(y)
    drafter.begin(r["pids"], r["prompt"])
    acc = []
    if prefill_draft:
        p = 0
        d = drafter.propose(TAIL, [], K)
        a = 0
        while a < len(d) and a < n and d[a] == y[a]:
            a += 1
        acc.append(a)
        p = min(n, a + 1)
    else:
        p = 1
    while p < n:
        d = drafter.propose(TAIL, y[:p], K)
        a = 0
        while a < len(d) and p + a < n and d[a] == y[p + a]:
            a += 1
        acc.append(a)
        p = min(n, p + a + 1)
    return acc


def evaluate(drafter, recs, K, prefill_draft=False):
    steps = 0
    produced = 0
    hit = 0
    hist = {}
    passes = 0
    base_passes = 0
    for r in recs:
        n = len(r["y"])
        if n == 0:
            continue
        acc = simulate_one(drafter, r, K, prefill_draft)
        steps += len(acc)
        produced += n if prefill_draft else n - 1
        hit += sum(1 for a in acc if a > 0)
        for a in acc:
            hist[a] = hist.get(a, 0) + 1
        passes += len(acc) + (0 if prefill_draft else 1)
        base_passes += n
    return {
        "tps": produced / max(1, steps),          # tokens per verify step
        "hit": hit / max(1, steps),               # fraction of steps with >= 1 accepted draft
        "steps": steps,
        "produced": produced,
        "passes": passes,                         # forward passes incl. prefill
        "base_passes": base_passes,               # plain greedy: one pass per token in y
        "speedup_passes": base_passes / max(1, passes),
        "hist": hist,
    }


def oracle(recs, K, prefill_draft=False):
    steps = produced = passes = base = 0
    for r in recs:
        n = len(r["y"])
        if n == 0:
            continue
        m = n if prefill_draft else n - 1
        s = -(-m // (K + 1)) if m > 0 else 0
        steps += s
        produced += m
        passes += s + (0 if prefill_draft else 1)
        base += n
    return {"tps": produced / max(1, steps), "passes": passes, "base_passes": base,
            "speedup_passes": base / max(1, passes)}
