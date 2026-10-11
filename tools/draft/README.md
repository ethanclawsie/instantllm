# Draft table and drafter experiments

`corpus-q4.jsonl` holds 22,820 greedy outputs of instantllm q4 (`{"prompt", "ids", "stop"}`) for
synthetic command requests. These are the TRAIN split only. Requests are split by family into
train / held-out, and none is an exact or near duplicate of a benchmark prompt. The engine's draft
table is built from it:

```sh
.build/release/instantllm ngram tools/draft/corpus-q4.jsonl --out models/ill/ngram.bin
```

Regenerating the corpus (data goes to `scratch/draft/`):

1. `gen_prompts.py`, `gen_prompts2.py`, `gen_prompts3.py`, `gen_prompts4.py` and the hand-written
   bases in `prompts_hand*.py` write the request list `scratch/draft/prompts.jsonl`, deduplicated
   and checked against the benchmark prompts.
2. `run_outputs.py --model models/ill/qwen-q4.ill --tag q4` runs `instantllm batch` over them
   (resumable, under the GPU lock) and writes `scratch/draft/outputs_q4*.jsonl`.
3. The TRAIN records of those files make up the corpus.

Drafter research. None of this is used by the engine yet:

- `simulate.py`: exact pass-count simulation of speculative decoding (chains and trees) on
  held-out and benchmark outputs. Its engine-style drafter reproduces the engine's measured pass
  counts exactly.
- `gpudraft.py`: Python model of the engine's drafter (always draft; self-repetition, n-gram
  table, prompt lookup), the engine table writer, and a retrieval variant (TF-IDF neighbours'
  outputs as a draft source; 2-6% fewer passes in simulation).
- `ildraft.py`, `build.py`, `edit.py`, `learned.py`: a learned candidate scorer (n-gram, neighbour,
  copy, state and association features, small MLP) with its own table format. In simulation it
  gets 8-12% fewer passes at 2 drafts per pass, and 3.2-3.8 tokens per pass with 8-16 node trees,
  against 1.9-2.1 for the engine's chain drafter.
- `export_swift_test.py`: test vectors for a port of `ildraft.py`.
