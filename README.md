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
| time to first token (prompt cached) | **4.1 ms** | 7.5 ms | 8.5 ms | 14.4 ms |
| decode (32 tokens) | **503 tok/s** | 275 tok/s | 248 tok/s | 354 tok/s |
| end to end, exactly 32 tokens | **65.7 ms** | 120.2 ms | 133.7 ms | 102.1 ms |
| end to end, natural stop | **20.2 ms** | 42.5 ms | 58.5 ms | 61.0 ms |
| end to end, whole prompt (no cache) | **22.5 ms** | 44.1 ms | 64.0 ms | 63.5 ms |
| cold: fresh process per request | **47 ms** | 235 ms | - | 1807 ms |
| resident daemon + C client (`ill`), spawn to exit | **22 ms** | - | - | - |
| weights read per token | 278 MB | 346 MB | 392 MB | 278 MB |

**8-bit**

| median | **instantllm q8** | llama.cpp Q8_0 | mlx-lm 8bit |
|---|---:|---:|---:|
| time to first token (prompt cached) | **5.7 ms** | 7.9 ms | 17.7 ms |
| decode (32 tokens) | **340 tok/s** | 208 tok/s | 216 tok/s |
| end to end, exactly 32 tokens | **97.0 ms** | 156.7 ms | 161.0 ms |
| end to end, natural stop | **31.1 ms** | 65.5 ms | 59.6 ms |
| end to end, whole prompt (no cache) | **33.3 ms** | 68.5 ms | 62.4 ms |

**16-bit**

| median | **instantllm f16** | llama.cpp F16 | mlx-lm bf16 |
|---|---:|---:|---:|
| time to first token (prompt cached) | **9.9 ms** | 10.3 ms | 25.4 ms |
| decode (32 tokens) | **178 tok/s** | 123 tok/s | 122 tok/s |
| end to end, exactly 32 tokens | **184.4 ms** | 262.2 ms | 277.7 ms |
| end to end, natural stop | **57.8 ms** | 102.3 ms | 99.2 ms |
| end to end, whole prompt (no cache) | **58.3 ms** | 105.4 ms | 101.3 ms |

For real commands (natural stop, 4-bit) that is 2.1x faster than llama.cpp and 3.0x faster than
mlx-lm; 1.8x and 1.6x for a fixed 32-token generation, where answer length cannot differ; 5x and
38x for a fresh process per request. Exact
speculative decoding (below) is on for instantllm: it never changes the output. The daemon row is
the median over the 20 prompts with 0 to 3 s of idle time before each request (see "GPU clock"
below). Raw results: `bench/results/*.tsv`; tables: `bench/table.py`.

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
- **Exact speculative decoding**: a drafter on the GPU proposes the next two tokens from, in order,
  a repetition of the output so far, an n-gram table of the model's own outputs (built from ~23K
  synthetic command requests, disjoint from the benchmark prompts) and the prompt. One pass over
  the weights verifies [last token, two drafts] together (1.2x the cost of a single-token step,
  since the multi-token kernels unpack each weight once) and keeps the longest matching prefix:
  ~1.7 tokens per pass on commands. The verify kernels compile with strict floating point, so the
  3-token and 1-token kernels agree bit for bit and the output is identical to plain greedy
  decoding (checked on 220 prompts, all three formats).
- **The decode loop never returns to the CPU**: the lm_head kernel reduces argmax candidates on the
  GPU, a commit kernel accepts drafts, records tokens in shared memory, drafts the next ones and
  embeds them for the next pass. The CPU only encodes up to two passes ahead and polls the shared
  token buffer.
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

Speculative decoding needs the draft table, built from the shipped corpus of model outputs
(`tools/draft/README.md`): `instantllm ngram tools/draft/corpus-q4.jsonl --out models/ill/ngram.bin`.
Without it the engine decodes token by token.

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
