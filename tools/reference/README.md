# tools/reference: numerical ground truth

A plain numpy fp32 implementation of Qwen2.5-Coder-0.5B-Instruct (`qwen_ref.py`), the golden test
data it produces (`golden/`), and a checker (`compare.py`) that scores an engine's outputs against
that data. Everything here runs from the repo root with the project venv:

```sh
.venv/bin/python -I tools/reference/qwen_ref.py                 # regenerate golden/ (CPU, ~10 s)
.venv/bin/python -I tools/reference/compare.py ENGINE_OUT_DIR    # score an engine
```

| file | role |
|---|---|
| `qwen_ref.py` | reference model: bf16 weights widened to f32, batched prefill, KV-cached greedy decode; writes `golden/` |
| `compare.py` | compares an engine output directory with `golden/` (logits, top-k, greedy tokens, debug tensors) |
| `refcommon.py` | shared paths, chat template, tokenization, file layout helpers |
| `run_mlx.py` | validation: mlx-lm greedy runs written in the engine layout |
| `run_llamacpp.py`, `llama_dump.c` | validation: llama.cpp (libllama C API) greedy runs written in the engine layout |
| `make_gguf.py` | validation: a GGUF holding exactly the `models/hf` weights (see "llama.cpp" below) |
| `golden/` | the golden data (about 21 MB, 60 files) |

The reference runs on the CPU (numpy with Accelerate BLAS), so it needs no GPU lock. The mlx and
llama.cpp runners use the GPU and must be wrapped in `lockf -k /tmp/instantllm-gpu.lock`.
Their outputs and builds go to `scratch/reference/` (gitignored).

## Conventions

These are the exact semantics the golden data encodes. An engine that wants bit-for-bit greedy
agreement should follow them.

- **Weights**: `models/hf/model.safetensors`, all bf16, widened to f32 exactly (`u32(bits) << 16`).
  The meta.json file records its sha256.
- **Compute**: float32 throughout. RoPE angles are `pos * theta^(-2i/64)`, computed in float64,
  then cos/sin are rounded once to f32. The rotation is rotate_half: pairs (i, i+32) within each
  64-dim head, theta 1e6. RMSNorm is `x * (1/sqrt(mean(x^2) + 1e-6)) * w`. Query head h uses
  KV head h // 7. Attention is `softmax(q k^T * 0.125 + causal mask) v`, and q/k/v have biases.
  The MLP is `down(silu(gate(x)) * up(x))`. The lm_head is the embedding matrix and covers all
  151936 rows, including the 271 padding rows past the tokenizer's 151665 tokens. The padding
  rows are nonzero (norm about 0.32), so they get logits too.
- **Prompt**: `<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n`.
  SYSTEM is `bench/system.txt` verbatim, with no trailing newline. The prompts are the lines of
  `bench/prompts.txt`. Tokenization uses HF `tokenizers` with `models/hf/tokenizer.json` and
  adds no BOS. The ids are identical to transformers' `apply_chat_template`, to the mlx-lm
  tokenizer and to the llama.cpp tokenizer for all 20 prompts.
- **Prefix**: the system prefix, everything through `<|im_start|>user\n`, is 44 tokens. It
  tokenizes identically on its own and inside every prompt, so an engine can prefill it once and
  cache it.
- **Greedy decoding**: argmax over all 151936 f32 logits, with exact ties going to the lowest
  id. At most 64 tokens are *sampled*, and a sampled stop token (151645 or 151643) counts as one
  of them. The stop token ends generation and is not part of `output_ids` or `text`. This
  matches mlx-lm's `max_tokens` semantics.
- **Text**: `tokenizer.decode(output_ids, skip_special_tokens=False)`.
- **Accuracy of the golden data**: the same code in float64 differs by at most 3.4e-5 in any
  last-position logit and gives identical greedy sequences. The smallest top-1/top-2 margin on
  any golden greedy step is 0.047, more than 1000 times that error. So the golden sequences are
  the exact greedy outputs of the bf16 weights, not artifacts of fp32 rounding.

## Golden data layout (`golden/`)

All `.bin` files are raw little-endian float32 in C order (last dimension fastest), with no
header. `NN` is the zero-padded prompt index (`p00` .. `p19`).

| file | contents |
|---|---|
| `meta.json` | model and config, weights sha256, compute dtype, chat template, generation rules, versions, timestamp |
| `prefix.json` | `{"text", "ids" (44 ids), "n": 44, "note"}`: the cacheable system prefix |
| `index.json` | list of `{index, prompt, n_input, output_ids, stop_token, finish_reason, text}` for all prompts |
| `pNN.json` | the full per-prompt record (below) |
| `pNN.logits.bin` | `[151936]` f32 logits at the last prompt position, which pick the first new token |
| `debug_p00/` | per-layer dumps for prompt 0 (below) |

`pNN.json` fields:

| field | meaning |
|---|---|
| `index`, `prompt` | prompt number and text |
| `input_ids`, `n_input` | full templated prompt ids (54 to 66 tokens) |
| `prefix_len` | 44; `input_ids[:44] == prefix.json ids` |
| `output_ids` | greedy tokens, stop token excluded |
| `stop_token` | 151645 / 151643, or `null` if 64 tokens were sampled without a stop |
| `finish_reason` | `"stop"` or `"length"` |
| `text` | decoded `output_ids` |
| `top10` | `[[id, logit], ...]` at the last prompt position, descending (ties by lower id) |
| `step_top5` | one entry per sampled token, including the stop token: `[[id, logit] x5]` of the logits that chose it. Step 0 is the last prompt position, and `step_top5[s][0][0]` is sampled token s. It gives the margin at every step, which is how divergences are classified as near-ties. |
| `last_logits_file` | `"pNN.logits.bin"` |

`debug_p00/index.json` holds `n_input` (63), `decode_tokens`, `decode_positions` and a
`tensors` map from name to `{file, shape, desc}`. The tensors are:

| tensor | shape | contents |
|---|---|---|
| `resid` | [25, 896] | residual stream at the last prompt position. Row 0 is the token embedding (layer-0 input); row i+1 is the output of layer i. |
| `final_norm` | [896] | `model.norm` output at the last position; `final_norm @ embed^T == p00.logits.bin` |
| `l0_attn_norm` | [63, 896] | layer-0 input_layernorm output, all prompt positions |
| `l0_q_pre_rope`, `l0_k_pre_rope` | [63, 14, 64], [63, 2, 64] | q / k projections plus bias, before RoPE |
| `l0_q`, `l0_k` | [63, 14, 64], [63, 2, 64] | after bias and RoPE (`l0_k` is what enters the KV cache) |
| `l0_v` | [63, 2, 64] | v projection plus bias |
| `l0_attn` | [63, 896] | attention context (heads concatenated, head-major), before o_proj |
| `l0_attn_out` | [63, 896] | o_proj output |
| `l0_resid_mid` | [63, 896] | embedding + attn_out |
| `l0_mlp_norm` | [63, 896] | post_attention_layernorm output |
| `l0_mlp_act` | [63, 4864] | `silu(gate) * up` |
| `l0_mlp_out` | [63, 896] | down_proj output |
| `l0_resid_out` | [63, 896] | layer-0 output at all positions (last row == `resid[1]`) |
| `decode_logits` | [8, 151936] | decode step j feeds `output_ids[j]` at position 63+j. Row j holds the resulting logits, whose argmax is sampled token j+1; row 7 picks the stop token. |

Reading a `.bin` file from Swift (both macOS targets are little-endian):

```swift
let logits: [Float] = try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
```

## Engine output layout (input to `compare.py`)

An engine writes one directory in the golden layout. Every file is optional, and `compare.py`
scores whatever is present.

- `pNN.logits.bin`: `[151936]` f32 logits at the last prompt position, before any sampling.
- `pNN.json`:
  - `output_ids` (list of ints): greedy tokens, stop token excluded.
  - `stop_token`: the stop id or `null`. Include it, because the stop decision is then scored
    as one more token. Without the key, only `output_ids` are compared.
  - `text` (optional): checked for exact equality with the golden text.
  - `input_ids` (optional): checked against the golden ids, which tests the engine's tokenizer
    and template.
  - `forced_argmax` (optional, teacher forcing): feed `input_ids + golden_seq[:-1]`, where
    `golden_seq = output_ids + [stop_token]` from the golden record. Report the argmax at
    positions `n_input-1 ...`, giving one prediction per golden step. This scores every step
    independently of earlier divergences.
  - `step_top5` (optional): the engine's own `[[id, value] x5]` per sampled step. Only
    differences are used, so log-probabilities are fine. It adds the engine's own gap at a
    divergence to the report.
- `debug_p00/<tensor>.bin` (optional): any tensor named in the table above, in the same shape.
  Matching names are compared, including `decode_logits.bin`.

Minimal example `p03.json`: `{"output_ids": [73594, 927, 198, ...], "stop_token": 151645}`.

## compare.py

```sh
.venv/bin/python -I tools/reference/compare.py ENGINE_DIR [--golden DIR] [--prompts 0,3]
    [--json results.json] [--near-tie 0.5] [--max-abs-tol X] [--require-top1] [--require-greedy] [--quiet]
```

The report has one row per prompt, then teacher-forced mismatches, the text of mismatching
prompts, debug-tensor errors (max and mean |d|, and max |d| / max |gold|) and a summary.

| column | definition |
|---|---|
| `tok` | engine `input_ids` equal the golden ids |
| `max\|d\|`, `mean\|d\|` | max / mean absolute logit difference over all 151936 last-position logits |
| `top1` | engine argmax == golden argmax at the last prompt position |
| `top5` | size of the overlap of the engine's and golden top-5 id sets (x/5) |
| `seq` | exact greedy sequence match (`output_ids` + stop token) |
| `prefix` | length of the common leading run of tokens / golden length |
| `agree` | per-token agreement, position by position: same token at the same index / golden length. Positions after the first divergence are usually misaligned, so this mostly tracks `prefix`. |
| divergence | first differing step: golden token vs engine token. "gold margin" is golden top-1 minus top-2 at that step. "gap" is golden top-1 minus the golden logit of the engine's token. The divergence is called a **near-tie** when gap <= `--near-tie` (default 0.5). |

The summary also gives the teacher-forced per-token agreement when `forced_argmax` is present.
`--json` writes every number above.

Exit status: 0 means it ran and all requested gates passed. 1 means a gate failed
(`--max-abs-tol`, `--require-top1`, `--require-greedy`, or any non-finite logit). 2 means the
directory held no engine outputs.

## Validation (2026-10-10, this M5)

The reference was checked against independent implementations. All runs use the golden prompt
ids, greedy decoding, at most 64 tokens and the two stop ids. `max|d|` is the worst
last-position logit difference over the 20 prompts.

| implementation | weights | max\|d\| | mean\|d\| | top-1 | exact seq | teacher-forced |
|---|---|---|---|---|---|---|
| this code, float64 | HF bf16 | 3.4e-5 | 4.4e-6 | 20/20 | 20/20 | n/a |
| mlx-lm fp32, CPU (`run_mlx.py --dtype float32 --device cpu`) | HF bf16 | 6.5e-5 | 6.7e-6 | 20/20 | 20/20 | 261/261 |
| mlx-lm fp32, GPU | HF bf16 | 0.030 | 3.3e-3 | 20/20 | 20/20 | 261/261 |
| llama.cpp Metal, `make_gguf.py --f32` GGUF | HF bf16 | 0.041 | 4.7e-3 | 20/20 | 20/20 | 261/261 |
| llama.cpp Metal, `make_gguf.py` f16 GGUF | HF bf16 to f16 | 0.041 | 4.7e-3 | 20/20 | 20/20 | 261/261 |
| **mlx-lm bf16** (`models/mlx/bf16`) | HF bf16 | 0.231 | 0.033 | 18/20 | **17/20** | 258/261 |
| mlx-lm 8bit (`models/mlx/8bit`) | 8-bit affine | 0.280 | 0.038 | 20/20 | 19/20 | 260/261 |
| **llama.cpp Metal, official fp16 GGUF** | AWQ-derived | 1.59 | 0.24 | 16/20 | **14/20** | 255/261 |
| llama.cpp Metal, official q8_0 | AWQ-derived | 1.64 | 0.24 | 16/20 | 14/20 | 255/261 |
| llama.cpp Metal, official q4_0 | AWQ-derived | 3.63 | 0.48 | 14/20 | 10/20 | 242/261 |
| llama.cpp Metal, official q4_k_m | AWQ-derived | 2.71 | 0.43 | 9/20 | 6/20 | 248/261 |
| mlx-lm 4bit (`models/mlx/4bit`) | 4-bit affine | 5.18 | 0.60 | 9/20 | 3/20 | 229/261 |

An independent fp32 implementation (mlx-lm on the CPU) agrees with the reference to 6.5e-5,
which is fp32 rounding level, and gives identical tokens. The stock `llama-completion` CLI gave
the same text as `llama_dump` for both the official and the HF-derived GGUF (spot-checked on
p00 and p03). Per-layer checks against mlx fp32 show RMSNorm agreeing to 1e-7 and RoPE to 3e-7
given the same inputs.

### Per-prompt agreement for the two requested comparisons

"same" means an identical greedy sequence. Otherwise the entry gives the first differing step
and the golden gap (golden top-1 minus the golden logit of the competitor's token).

| p | golden text | mlx-lm bf16 | llama.cpp official fp16 | llama.cpp HF f16 |
|---|---|---|---|---|
| 00 | `` `ls -lh /home/user` `` | same | step 0, gap 0.143 | same |
| 01 | ```` ```sh\nfind . -type f -name "*.py" -mtime -7\n``` ```` | same | same | same |
| 02 | `` `lsof -i :8080` `` | same | same | same |
| 03 | ```` ```sh\nfind . -type f -name "*.swift" -exec cat {} + \| wc -l\n``` ```` | step 0, gap 0.069 | step 15, gap 0.274 | same |
| 04 | `tar -czvf logs.tar.gz logs` | same | same | same |
| 05 | `` `killall node` `` | same | same | same |
| 06 | `` `du -sh *` `` | same | same | same |
| 07 | ```` ```sh\nfind . -type f -name "*.txt" -exec sed -i 's/foo/bar/g' {} \;\n``` ```` | same | step 9, gap 1.646 | same |
| 08 | `tail -n 50 /var/log/system.log` | same | same | same |
| 09 | `` `git checkout -b feature/login` `` | same | step 0, gap 0.058 | same |
| 10 | `` `git reset --hard HEAD` `` | same | same | same |
| 11 | ```` ```sh\ncurl -O https://example.com/file.zip\nunzip file.zip\n``` ```` | same | same | same |
| 12 | `` `find . -type f -name "*.DS_Store" -exec rm {} \;` `` | same | same | same |
| 13 | `ipconfig -a` | same | same | same |
| 14 | `` `ffmpeg -i video.mov -c:v libx264 -c:a aac video.mp4` `` | same | step 0, gap 0.441 | same |
| 15 | `ls -lh \| sort -nr \| head -10` | step 0, gap 0.063 | same | same |
| 16 | `` `git log --oneline` `` | same | same | same |
| 17 | `` `rename *.jpeg *.jpg` `` | same | step 0, gap 0.717 | same |
| 18 | `python3.8` | same | same | same |
| 19 | `` `grep -r 'TODO' *.js` `` | step 4, gap 0.047 | same | same |

(The model often wraps its answer in backticks or a fenced block despite the system prompt.
These are the model's real greedy outputs.)

### Why they differ

- **mlx-lm bf16 (p03, p15, p19): bf16 near-ties.** All three are among the four smallest golden
  margins (0.069, 0.063, 0.047). mlx-lm computes and returns logits in bf16. For |logit| in
  [16, 32), bf16 values sit on a 0.125 grid, so both candidates round to the *same* value:
  p03 has 18.125 vs 18.125 and p15 has 18.375 vs 18.375 at step 0, and p19 ties at step 4. The
  argmax then takes the lower id: backtick (63) over triple backtick (73594) or `"ls"` (4730),
  and `" \""` (330) over `" '"` (364). The same code with fp32 weights matches 20/20. mlx-lm's own gap at
  each of these divergences is exactly 0.
- **llama.cpp official fp16 GGUF (p00, p03, p07, p09, p14, p17): different weights.** Every
  GGUF in `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF` (fp16, q8_0, q4_k_m, q4_0) was converted from
  the **Instruct-AWQ** checkpoint. Its `general.name` is "Qwen2.5 Coder 0.5B Instruct AWQ". The
  AWQ scales are folded into the RMSNorm weights, and the linear weights are dequantized 4-bit
  values. Only `token_embd` equals the f16 of the HF embedding. Hence the logit error of about
  1.3 to 1.6, and two divergences that are not near-ties (p07 gap 1.646, p17 gap 0.717). On a
  GGUF written from `models/hf` (`make_gguf.py`), the same llama.cpp agrees 20/20 with max|d|
  0.041. Speed comparisons with the official GGUFs are still fair (same shapes and sizes), but
  their text is a different model's.

### Near-ties in the golden data

These are the golden greedy steps with a top-1/top-2 margin below 0.3. Engines with logit error
around this size can legitimately flip them.

| prompt | step | margin | golden token | runner-up |
|---|---|---|---|---|
| p19 | 4 | 0.047 | `" '"` (364) | `" \""` (330) |
| p09 | 0 | 0.058 | backtick (63) | triple backtick (73594) |
| p15 | 0 | 0.063 | `"ls"` (4730) | backtick (63) |
| p03 | 0 | 0.069 | triple backtick (73594) | backtick (63) |
| p13 | 2 | 0.128 | `" -"` (481) | `" \|"` (760) |
| p00 | 0 | 0.143 | backtick (63) | triple backtick (73594) |
| p01 | 0 | 0.153 | triple backtick (73594) | backtick (63) |
| p15 | 10 | 0.217 | `"1"` (16) | `"n"` (77) |
| p13 | 0 | 0.251 | `"ip"` (573) | backtick (63) |
| p19 | 0 | 0.264 | backtick (63) | triple backtick (73594) |
| p06 | 5 | 0.266 | backtick (63) | `" \|"` (760) |
| p03 | 15 | 0.274 | `" cat"` (8251) | `" wc"` (26548) |
| p07 | 25 | 0.275 | `" \\"` (1124) | `" +\n"` (3610) |
| p00 | 7 | 0.294 | backtick (63) | `" \|"` (760) |
| p12 | 1 | 0.295 | `"find"` (3903) | `"rm"` (8719) |

Tokens are shown as JSON strings of their decoded text.

### Guidance for engines

- Engines with f16 or f32 activations and f32 accumulation should expect max|d| of about 0.03
  to 0.04 and 20/20 exact sequences. That is what llama.cpp (Metal, f16 KV cache) and mlx fp32
  on the GPU achieve. A reasonable gate is `--max-abs-tol 0.1 --require-greedy`.
- Keep the final logits (or at least the argmax inputs) in f32. bf16 logits create exact ties at
  the near-tie steps above, which then resolve toward the lower id.
- bf16 activation paths should expect max|d| of about 0.2 to 0.3. Judge them by teacher-forced
  agreement (about 99%), with every divergence classified as a near-tie, rather than by exact
  sequences.
- 4-bit weights change this model's outputs substantially (3/20 to 10/20 exact sequences), so
  validate them separately.

### Gotchas found along the way

- On this M5, mlx's GPU fp32 GEMM (M > 1) has a relative error of about 7.5e-4, which looks
  like TF32. Its fp32 matvec (2e-7) and CPU GEMM (1e-6) are full precision. That is why
  "mlx-lm fp32, GPU" is 0.030 away while the CPU run is 6.5e-5 away.
- `llama-completion -f prompt.txt` strips the trailing `\n` of the templated prompt, giving 62
  tokens instead of 63 and a different answer. Pass the prompt with `-e -p '...\n'` (escaped
  newlines) or use the C API.
- `llama.h` includes `ggml.h` from the separate Homebrew `ggml` formula, so building needs
  `-I/opt/homebrew/opt/ggml/include`.

## Re-running the validation

```sh
L="lockf -k /tmp/instantllm-gpu.lock"
P=".venv/bin/python -I tools/reference"
$P/qwen_ref.py --dtype f64 --out scratch/reference/ref_f64 --no-debug   # precision floor
$P/qwen_ref.py --check-cache                                            # KV cache vs re-prefill (3e-5)
$L $P/run_mlx.py                                   # mlx-lm bf16  -> scratch/reference/validation/mlx_bf16
$P/run_mlx.py --dtype float32 --device cpu         # mlx-lm fp32 on the CPU (no lock needed)
$L $P/run_llamacpp.py                              # official fp16 GGUF -> .../llamacpp_fp16
$P/make_gguf.py && $L $P/run_llamacpp.py --gguf scratch/reference/gguf/qwen2.5-coder-0.5b-instruct-hf-f16.gguf
$P/compare.py scratch/reference/validation/mlx_bf16
```
