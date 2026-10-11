# instantllm: shared spec

Goal: the lowest end-to-end latency for short, single-batch, greedy command generation with
**Qwen2.5-Coder-0.5B-Instruct** on a **base Apple M5** (10-core GPU with Neural Accelerators,
4P+6E CPU, 24 GB, ~153 GB/s unified memory, macOS 26.3). The runner is single-model and
single-machine on purpose: every kernel may hard-code the model's shapes.

## Environment facts

- Swift 6.3 + Command Line Tools only. **No Xcode, no offline `metal` compiler**: Metal shaders are
  compiled at runtime from source (`MTLDevice.makeLibrary(source:options:)`). Metal 4 / MSL 4 and
  `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` are expected to be usable at runtime.
- Python for tooling only: `~/Development/instantllm/.venv/bin/python` (numpy, safetensors,
  tokenizers, mlx, mlx_lm). Always run Python with `-I` when it reads downloaded files.
- Competitors: llama.cpp 0.6.0 from Homebrew (`/opt/homebrew/bin/llama-*`, headers in
  `/opt/homebrew/opt/llama.cpp/include`, `libllama.dylib` in `/opt/homebrew/opt/llama.cpp/lib`),
  mlx 0.32.3 + mlx-lm 0.32.0 in the venv above.
- Models (all the same model): `models/hf/` (HF bf16 safetensors + tokenizer.json),
  `models/gguf/qwen2.5-coder-0.5b-instruct-{q4_0,q4_k_m,q8_0,fp16}.gguf`,
  `models/mlx/{4bit,8bit,bf16}/`.

## GPU timing etiquette (important)

Several agents may run at once on this one GPU (the user is away from the Mac).
- Wrap **every** GPU timing run in the shared lock so only one runs at a time:
  `lockf -k /tmp/instantllm-gpu.lock <command...>`
- Keep each timed run short (well under a minute). Report min and median of repeated runs.
- Agents: smoke tests (1-2 prompts) only for the competitor protocol; the formal head-to-head is
  run by the lead with nothing else on the GPU.

## Model facts (Qwen2ForCausalLM)

| | |
|---|---|
| layers | 24 |
| hidden | 896 |
| intermediate | 4864 (SwiGLU: down(silu(gate(x)) * up(x))) |
| heads | 14 query heads, 2 KV heads (GQA, 7 q-heads per kv-head), head_dim 64 |
| vocab | 151936 (embedding rows; tokenizer has 151643 BPE tokens + 22 added) |
| embeddings | tied: lm_head = embed_tokens |
| norm | RMSNorm, eps 1e-6, weight multiply (no +1) |
| attention | q/k/v projections have bias; o_proj has no bias |
| RoPE | theta 1e6, non-interleaved (rotate_half: pairs (i, i+32) within each 64-dim head) |
| eos | 151645 `<|im_end|>`, 151643 `<|endoftext|>` |

HF tensor names: `model.embed_tokens.weight [151936,896]`, `model.norm.weight`, and per layer
`model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight [896]`,
`self_attn.{q,k,v}_proj.{weight,bias}` (q 896x896, k/v 128x896), `self_attn.o_proj.weight [896,896]`,
`mlp.{gate,up}_proj.weight [4864,896]`, `mlp.down_proj.weight [896,4864]`. All bf16. Weight shape
is [out, in] (PyTorch Linear).

Per-token weight bytes at decode (everything except the embedding row; lm_head reads all 136M):
494M params -> f16 988 MB, 8.5 bpw 525 MB, 4.5 bpw 278 MB -> at 153 GB/s: 6.5 / 3.4 / 1.8 ms.

## Prompt format and benchmark protocol

Chat template (exact):

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n
```

- `{SYSTEM}` = contents of `bench/system.txt` (no trailing newline).
- Prompts: `bench/prompts.txt`, one per line (20 command-generation requests).
- Greedy decoding (argmax), no repetition penalty, at most 64 new tokens, stop at 151645 or 151643
  (the stop token is not printed).
- Metrics per prompt: end-to-end latency (prompt string in -> full command text out), TTFT,
  decode tokens/s, generated text. Warm = model resident in the process; cold = fresh process.

## Repo layout and ownership

```
Package.swift          SwiftPM (tools 6.2, macOS 26, Swift 5 language mode), no dependencies
Sources/Tokenizer/     Qwen2 byte-level BPE tokenizer library (module Tokenizer)
Sources/tokcheck/      tokenizer test/benchmark executable
Sources/Engine/        Metal inference engine (module Engine)
Sources/instantllm/    CLI: run / serve / convert / bench
tools/reference/       numpy fp32 reference forward pass + golden test data
bench/                 prompts, system prompt, protocol runner, competitors/{llamacpp,mlx}/
scratch/               experiments and probes (gitignored)
```
