# instantllm

The lowest-latency way to run **Qwen2.5-Coder-0.5B-Instruct** on an **Apple M5** for short,
single-request command generation ("find large files" -> `find ~ -size +100M`).

A single-model, single-machine engine: Swift host code with no dependencies, plus hand-fused Metal
kernels compiled at runtime and hard-coded to this model's exact shapes. It is not a general
runtime; that is the point.

## Results (base M5, 4-bit, same model, same 20 prompts)

All engines run the same protocol (`docs/SPEC.md`): exact chat template, greedy decoding, at most
64 new tokens, timed from the prompt string to the final text. Each engine is at its best
configuration and keeps the system-prompt KV cache between requests ("prompt cached"). Two rounds
over the 20 prompts, engines interleaved round-robin so drift affects all of them equally.

| median | **instantllm q4** | llama.cpp Q4_0 | llama.cpp Q4_K_M | mlx-lm 4bit |
|---|---:|---:|---:|---:|
| time to first token (prompt cached) | **5.4 ms** | 7.6 ms | 9.3 ms | 14.6 ms |
| decode (32 tokens) | **372 tok/s** | 271 tok/s | 240 tok/s | 338 tok/s |
| end to end, exactly 32 tokens | **88.0 ms** | 123.4 ms | 137.8 ms | 106.3 ms |
| end to end, natural stop | **26.4 ms** | 43.3 ms | 70.6 ms | 61.8 ms |
| end to end, whole prompt (no cache) | **28.8 ms** | 45.8 ms | 64.3 ms | 65.0 ms |
| cold: fresh process per request | **70 ms** | 248 ms | - | 1897 ms |
| resident daemon + 1 ms C client (`ill`) | **~23-30 ms** spawn to exit | - | - | - |
| weights read per token | 278 MB | 346 MB | 392 MB | 278 MB |

For real commands (natural stop) that is 1.6x faster than llama.cpp and 2.3x faster than mlx-lm;
1.4x and 1.2x for a fixed 32-token generation, where answer length cannot differ. Exact speculative
decoding (below) is on: it never changes the output, and helps most on command-shaped text (decoding
past the end of a command, as the fixed-length row does, is mostly text the drafter has not seen).
Raw results: `bench/results/*.tsv`; table: `bench/table.py`.

### Output quality (agreement with an fp32 reference of the original weights)

| 4-bit variant | teacher-forced next-token agreement | identical greedy outputs (of 20) |
|---|---:|---:|
| **instantllm q4** (GPTQ-style, 4.5 bits/weight) | **96.9%** | **13** |
| llama.cpp Q4_K_M | 95.0% | 6 |
| llama.cpp Q4_0 | 92.7% | 10 |
| mlx-lm 4bit | 87.7% | 3 |
| *(llama.cpp Q8_0, for scale)* | *97.7%* | *14* |

Our 4-bit model is about as accurate as llama.cpp's 8-bit one while reading 20% fewer bytes per
token than its 4-bit one. (The official Qwen GGUFs are converted from Qwen's AWQ checkpoint;
mlx-lm uses the original weights, as we do.)

## How it gets there

- **~5 fused kernels per layer** (vs ~500 dispatches per token in llama.cpp and MLX): RMSNorm is
  folded into the matrix-vector kernels, bias + RoPE + KV-cache write into the QKV epilogue,
  SiLU(gate)*up into the gate/up kernel, residual adds into O and down.
- **Exact speculative decoding**: a GPU-resident n-gram table of the model's own outputs (built from
  ~10K synthetic command requests, disjoint from the benchmark prompts) drafts the next token; one
  pass over the weights verifies [last token, draft] together (about 1.1x the cost of a normal step,
  since the multi-token kernels unpack each weight once), and the longest matching prefix is kept,
  so the output is bit-identical to plain greedy decoding. ~1.5 tokens per pass on commands.
- **The decode loop never returns to the CPU**: the lm_head kernel reduces argmax candidates on the
  GPU, a commit kernel picks the token, records it in shared memory and embeds it for the next step.
  The CPU only encodes ahead and polls the shared token buffer (one step per command buffer, so each
  token becomes visible within ~15 us).
- **4-bit weights, quantized with error compensation** (GPTQ-style, activation-weighted) into an
  affine group-64 format, including the tied lm_head: 278 MB per token, the minimum of the field.
- **Prompt processing on the M5's Neural Accelerators** (Metal Performance Primitives `matmul2d`)
  with deterministic split-K and the same fused epilogues; the system prompt's KV cache is reused
  across requests and saved to disk for one-shot runs.
- **Startup**: zero-copy mmapped weights, runtime shader compile served from the system cache
  (~1 ms), a tokenizer that loads in ~10 us and encodes a prompt in ~0.5 us (bit-exact with
  Hugging Face `tokenizers` on 10.5M tokens).
- **Resident daemon** (`instantllm serve`) with a C client (`ill`): pinned weights, latency-critical
  QoS and a GPU keepalive, so a request after minutes of idle is as fast as back-to-back ones.

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

Speculative decoding needs the draft table: `instantllm ngram corpus.jsonl --out models/ill/ngram.bin`
(builds it from model outputs, see `tools/draft/`); without it the engine decodes token by token.

`convert --quant FILE` takes pre-quantized codes from the GPTQ-style pipeline (the published
results use it; its scripts are being cleaned up into `tools/quant/`). Plain `convert --format q4`
uses a per-group MSE-optimal quantizer: same speed, lower quality (92.3% teacher-forced agreement).

## Reproduce the benchmark

```sh
bench/competitors/llamacpp/build.sh                  # Homebrew llama.cpp 0.6.0
uv venv .venv && uv pip install --python .venv/bin/python mlx-lm numpy safetensors tokenizers
bench/run_all.sh                                     # interleaved, ~25 min, run on an idle machine
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
docs/SPEC.md         model facts, protocol, environment
```
