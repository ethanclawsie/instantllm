# instantllm

The lowest-latency way to run **Qwen2.5-Coder-0.5B-Instruct** on an **Apple M5** for short,
single-request command generation ("find large files" -> `find ~ -size +100M`).

A single-model, single-machine engine: Swift host code with no dependencies, plus hand-fused Metal
kernels compiled at runtime and hard-coded to this model's exact shapes. It is not a general
runtime; that is the point.

## Results (base M5, same model, same 20 prompts)

All engines run the same protocol (`docs/SPEC.md`): exact chat template, greedy decoding, at most
64 new tokens, timed from the prompt string to the final text. Each engine is at its best
configuration and keeps the system-prompt KV cache between requests ("prompt cached"). Two rounds
over the 20 prompts, all ten configurations interleaved round-robin so drift affects all of them
equally.

**4-bit**

| median | **instantllm q4** | llama.cpp Q4_0 | llama.cpp Q4_K_M | mlx-lm 4bit |
|---|---:|---:|---:|---:|
| time to first token (prompt cached) | **4.0 ms** | 7.5 ms | 8.5 ms | 14.5 ms |
| decode (32 tokens) | **501 tok/s** | 275 tok/s | 248 tok/s | 348 tok/s |
| end to end, exactly 32 tokens | **66.1 ms** | 120.5 ms | 134.1 ms | 103.6 ms |
| end to end, natural stop | **18.7 ms** | 42.6 ms | 58.5 ms | 61.2 ms |
| *(mean)* end to end, natural stop | **18.5 ms** | 46.4 ms | 60.1 ms | 90.2 ms |
| end to end, whole prompt (no cache) | **21.3 ms** | 44.1 ms | 62.8 ms | 64.9 ms |
| cold: fresh process per request | **44 ms** | 236 ms | - | 1872 ms |
| resident daemon + C client (`ill`), spawn to exit | **20 ms** | - | - | - |
| weights read per token | 278 MB | 346 MB | 392 MB | 278 MB |

**8-bit**

| median | **instantllm q8** | llama.cpp Q8_0 | mlx-lm 8bit |
|---|---:|---:|---:|
| time to first token (prompt cached) | **5.8 ms** | 7.9 ms | 17.7 ms |
| decode (32 tokens) | **315 tok/s** | 208 tok/s | 215 tok/s |
| end to end, exactly 32 tokens | **104.6 ms** | 157.1 ms | 162.4 ms |
| end to end, natural stop | **29.6 ms** | 65.5 ms | 59.8 ms |
| *(mean)* end to end, natural stop | **27.0 ms** | 67.4 ms | 73.5 ms |
| end to end, whole prompt (no cache) | **31.8 ms** | 68.7 ms | 62.8 ms |

**16-bit**

| median | **instantllm f16** | llama.cpp F16 | mlx-lm bf16 |
|---|---:|---:|---:|
| time to first token (prompt cached) | **10.1 ms** | 10.3 ms | 24.2 ms |
| decode (32 tokens) | **175 tok/s** | 124 tok/s | 122 tok/s |
| end to end, exactly 32 tokens | **187.5 ms** | 261.8 ms | 279.2 ms |
| end to end, natural stop | **51.7 ms** | 102.8 ms | 99.0 ms |
| *(mean)* end to end, natural stop | **47.1 ms** | 108.1 ms | 121.6 ms |
| end to end, whole prompt (no cache) | **52.7 ms** | 105.5 ms | 101.4 ms |

For real commands (natural stop, 4-bit), that is 2.3x faster than llama.cpp and 3.3x faster than
mlx-lm, or 2.5x and 4.9x by the mean. For a fixed 32-token generation, where answer length cannot
differ, it is 1.8x and 1.6x. With a fresh process per request it is 5x and 43x. Exact speculative
decoding (below) is on for instantllm and never changes the output. The daemon row is the median
over 2 rounds of the 20 prompts with 0, 1 or 3 s of idle time before each request: 19.6 to 20.4 ms
whatever the gap, with a 90th percentile of 26 to 27 ms (see "GPU clock" below). Raw results:
`bench/results/*.tsv`; tables: `bench/table.py`.

### Output quality (agreement with an fp32 reference of the original weights)

| 4-bit variant | teacher-forced next-token agreement | identical greedy outputs (of 20) |
|---|---:|---:|
| **instantllm q4** (GPTQ-style, 4.5 bits/weight) | **96.9%** | **13** |
| llama.cpp Q4_K_M | 95.0% | 6 |
| llama.cpp Q4_0 | 92.7% | 10 |
| mlx-lm 4bit | 87.7% | 3 |
| *(llama.cpp Q8_0, for scale)* | *97.7%* | *14* |

Our 4-bit model is about as accurate as llama.cpp's 8-bit one while reading 20% fewer bytes per
token than its 4-bit one. instantllm q8 agrees on 100% of teacher-forced tokens (19 of 20 identical
outputs), f16 on 100% (20 of 20). (The official Qwen GGUFs are converted from Qwen's AWQ
checkpoint; mlx-lm uses the original weights, as we do.)

## How it gets there

- **~5 fused kernels per layer** (vs ~500 dispatches per token in llama.cpp and MLX): RMSNorm is
  folded into the matrix-vector kernels, bias + RoPE + KV-cache write into the QKV epilogue,
  SiLU(gate)*up into the gate/up kernel, residual adds into O and down.
- **Exact speculative decoding with draft trees on the Neural Accelerators**: a learned drafter
  runs on the GPU between passes (a Metal port of `tools/draft/ildraft.py`). It scores candidates
  from an n-gram table of the model's own outputs (~23K synthetic command requests, disjoint from
  the benchmark prompts), the outputs of the 16 most similar training requests, the prompt and the
  output so far, using 53 features and a small MLP. From these it builds a best-first tree of 15
  drafts. One 16-row pass on the Neural Accelerators verifies the whole tree (rows cost almost
  nothing there: 3.8 ms for 16 rows, against 2.5 ms for a single-token step), and the longest
  accepted path is kept: ~3 tokens per pass on commands. When drafts stop being accepted
  (unpredictable text) the passes shrink to 8-row chains. Tree rows attend to their ancestors in
  the same order as sequential decoding, and every pass shape shares the same kernels, so the
  output is bit-identical to decoding one token at a time with those kernels (checked on 200
  held-out prompts in all three formats).
- **The decode loop never returns to the CPU**: the lm_head reduces argmax candidates on the GPU,
  a commit kernel walks the accepted path and records tokens in shared memory, and the drafter
  embeds the next pass's rows. The CPU only prepares the drafter's per-request context (while the
  GPU prefills), encodes one pass ahead and polls the shared token buffer.
- **4-bit weights, quantized with error compensation** (GPTQ-style, activation-weighted) into an
  affine group-64 format, including the tied lm_head: 278 MB per token, the minimum of the field.
  Multi-token passes unpack nibbles exactly into fp16 and use packed half2 FMAs.
- **Prompt processing on the M5's Neural Accelerators** (Metal Performance Primitives `matmul2d`)
  with deterministic split-K and the same fused epilogues; the system prompt's KV cache is reused
  across requests and saved to disk for one-shot runs.
- **Startup**: zero-copy mmapped weights, `MTLCopyAllDevices` (12-16 ms in a fresh process vs ~40 ms
  for `MTLCreateSystemDefaultDevice`), runtime shader compile served from the system cache
  (~1 ms), a tokenizer that loads in ~10 us and encodes a prompt in ~0.5 us (bit-exact with
  Hugging Face `tokenizers` on 10.5M tokens).
- **GPU clock**: the GPU drops its clock within ~50 ms of idle, and a 20 ms request finishes before
  it ramps back up, so a request after a pause used to take twice as long. The resident daemon
  (`instantllm serve`, C client `ill`) keeps a single 32-thread spin kernel running on a second
  queue for 5 minutes after each request (`--heat-seconds`, measured < 1 W), so requests after a
  pause are as fast as back-to-back ones. It also pins the weights and runs at latency-critical QoS.

## Use

Requires macOS 26 on Apple Silicon (built and tuned on a base M5), Swift 6.2+ command line tools.

```sh
swift build -c release
models/fetch.sh                                   # downloads the HF weights (+ competitor formats)
.build/release/instantllm convert --format q4 --out models/ill/qwen-q4.ill
.build/release/instantllm tokenize --compile models/hf/tokenizer.json --out models/ill/qwen.tkn

.build/release/instantllm run "show which process is listening on port 8080"
# -> sudo lsof -i :8080

.build/release/ill "kill all processes named node"   # starts the daemon on first use
# -> killall node
```

Speculative decoding uses two tables, both made from the shipped corpus of model outputs
(`tools/draft/README.md`). The learned drafter's table ships as `tools/draft/ildraft-q4.bin`: copy
it to `models/ill/ildraft.bin`. The n-gram table, which the drafter falls back to for requests
without a prompt context, is built with
`instantllm ngram tools/draft/corpus-q4.jsonl --out models/ill/ngram.bin`. Without either table,
the engine decodes token by token.

`convert --quant FILE` takes pre-quantized codes from the GPTQ-style pipeline (the published
results use it; its scripts are being cleaned up into `tools/quant/`). Plain `convert --format q4`
uses a per-group MSE-optimal quantizer: same speed, lower quality (92.3% teacher-forced agreement).

## Reproduce the benchmark

```sh
bench/competitors/llamacpp/build.sh                  # Homebrew llama.cpp 0.6.0
uv venv .venv && uv pip install --python .venv/bin/python mlx-lm numpy safetensors tokenizers
ROUNDS=2 bench/run_all.sh                            # interleaved, ~45 min, run on an idle machine
.venv/bin/python -I bench/table.py
.venv/bin/python -I tools/reference/qwen_ref.py      # fp32 golden data (9 s, CPU)
.venv/bin/python -I tools/reference/compare.py scratch/out_q4   # after `instantllm golden`
```

## Layout

```
Sources/Engine/      Metal engine: kernels (Shaders.swift), host loop, model file, converter
Sources/Tokenizer/   Qwen2 byte-level BPE, compiled to an mmappable table
Sources/instantllm/  CLI: run, serve, bench, convert, golden, profile
Sources/ill/         C client for the daemon
bench/               protocol runner, competitor harnesses (llama.cpp C++, mlx-lm Python)
tools/reference/     numpy fp32 reference implementation and comparison tool
tools/draft/         draft-table corpus and drafter experiments
docs/SPEC.md         model facts, protocol, environment
```
