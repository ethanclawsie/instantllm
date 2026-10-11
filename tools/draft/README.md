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

Drafters:

- `ildraft.py` is the reference for the engine's learned drafter (`k_ildraft` in
  `Sources/Engine/Shaders.swift`, request setup in `Sources/Engine/ILDraft.swift`). Its table
  ships as `ildraft-q4.bin` (install as `models/ill/ildraft.bin`). It was built with `build.py`
  from the TRAIN outputs (`--tag q4n --extra x,y --flags 1 --minc 4 --max-df 0.1 --cc2-min 3
  --as-min 4`; this needs PyTorch and the regenerated outputs). The features cover n-grams,
  neighbours, copying, state and associations, scored by a 16-unit MLP. Chains and best-first trees
  match `simulate.py` pass counts exactly.
- `simulate.py`: exact pass-count simulation of speculative decoding (chains and trees) on held-out
  and benchmark outputs.
- `gpudraft.py`: Python model of the engine's fallback n-gram drafter (always draft;
  self-repetition, n-gram table, prompt lookup), the engine table writer, and a retrieval variant.
- `edit.py`, `learned.py`, `export_swift_test.py`: experiments (edited neighbours, earlier learned
  variants, test vectors).
