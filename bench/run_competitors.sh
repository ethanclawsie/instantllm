#!/usr/bin/env bash
# Competitor benchmark protocol (docs/SPEC.md): llama.cpp and mlx_lm on bench/prompts.txt with the
# exact chat template, greedy, <= 64 new tokens, stop at 151645/151643, in three modes:
#   full    in-process, KV cleared per request, whole templated prompt prefilled
#   prefix  in-process, system-prompt prefix KV kept, only the rest prefilled
#   cold    fresh one-shot CLI process per request (llama-completion / python -m mlx_lm generate),
#           timed spawn -> exit
# Writes bench/results/<engine>-<variant>-<mode>.tsv (schema in bench/summarize.py), stderr logs to
# bench/logs/, then prints the summary table. Run it with nothing else on the GPU (~12 min):
#   bench/run_competitors.sh
#   SMOKE=1 bench/run_competitors.sh        # 2 prompts x 1 rep, results in bench/logs/smoke/
# Environment overrides:
#   REPS=5 COLD_REPS=2 COOLDOWN=10 OUT=bench/results PROMPT_IDX=0,1,...
#   ENGINES="llamacpp mlx" LLAMA_VARIANTS="q4_0 q4_k_m q8_0 fp16" MLX_VARIANTS="4bit 8bit bf16"
#   MODES="full prefix cold"
# Every timed run holds the shared GPU lock (lockf -k /tmp/instantllm-gpu.lock) and stays well under
# a minute: cold runs are split per rep, and mlx_lm cold runs (~1 s per process) per 10 prompts.
#
# llama.cpp settings (fastest measured on the base M5; details in competitors/llamacpp/bench.cpp):
# flash attention on, n_batch = n_ubatch = 512, 1 thread, CPU greedy sampler; Metal tensor API on
# (default) except in prefix mode (GGML_METAL_TENSOR_DISABLE=1, ~1 ms faster on 17-22-token
# suffixes); cold CLI adds -fit off --no-warmup (-~200 ms start-up) and greedy via top_k 1.
# mlx_lm: generate_step defaults (prefill_step_size 2048, greedy argmax), weights wired once.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${INSTANTLLM_CAFFEINATED:-}" ]; then
  export INSTANTLLM_CAFFEINATED=1
  exec caffeinate -dims "$0" "$@"
fi

if [ -n "${SMOKE:-}" ]; then
  : "${PROMPT_IDX:=0,1}" "${REPS:=1}" "${COLD_REPS:=1}" "${COOLDOWN:=2}" "${OUT:=bench/logs/smoke}"
fi
REPS=${REPS:-5}
COLD_REPS=${COLD_REPS:-2}
COOLDOWN=${COOLDOWN:-10}
OUT=${OUT:-bench/results}
PROMPT_IDX=${PROMPT_IDX:-}
ENGINES=${ENGINES:-"llamacpp mlx"}
LLAMA_VARIANTS=${LLAMA_VARIANTS:-"q4_0 q4_k_m q8_0 fp16"}
MLX_VARIANTS=${MLX_VARIANTS:-"4bit 8bit bf16"}
MODES=${MODES:-"full prefix cold"}

LOCK=(lockf -k /tmp/instantllm-gpu.lock)
PY=("$ROOT/.venv/bin/python" -I)
LLAMA_BENCH=bench/competitors/llamacpp/bench
HEADER=$'engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext'

mkdir -p "$OUT" bench/logs
LOG="bench/logs/competitors-$(date +%Y%m%d-%H%M%S).log"
echo "log: $LOG"

if [[ " $ENGINES " == *" llamacpp "* ]] && { [ ! -x "$LLAMA_BENCH" ] || [ bench/competitors/llamacpp/bench.cpp -nt "$LLAMA_BENCH" ]; }; then
  bench/competitors/llamacpp/build.sh
fi

# Prompt indices to run, and chunks of at most N of them (comma lists).
if [ -n "$PROMPT_IDX" ]; then
  IDX_LIST=${PROMPT_IDX//,/ }
else
  IDX_LIST=$(seq 0 $(($(grep -c . bench/prompts.txt) - 1)))
fi
chunks() { # N -> one comma list per line
  local n=$1 i=0 cur=""
  for x in $IDX_LIST; do
    cur="${cur:+$cur,}$x"
    i=$((i + 1))
    if [ "$i" -eq "$n" ]; then echo "$cur"; cur=""; i=0; fi
  done
  if [ -n "$cur" ]; then echo "$cur"; fi
}
ALL_IDX=$(chunks 100000)

timed() { # out_file cmd... : one GPU-locked run appending TSV rows, then a cooldown
  local out=$1; shift
  echo "+ $*" | tee -a "$LOG" >&2
  "${LOCK[@]}" "$@" --no-header >> "$out" 2>> "$LOG"
  grep '^# .*: n=' "$LOG" | tail -1 >&2 || true
  sleep "$COOLDOWN"
}

start() { # out_file : truncate and write the header
  printf '%s\n' "$HEADER" > "$1"
}

for engine in $ENGINES; do
  if [ "$engine" = llamacpp ]; then variants=$LLAMA_VARIANTS; else variants=$MLX_VARIANTS; fi
  for v in $variants; do
    for mode in $MODES; do
      out="$OUT/$engine-$v-$mode.tsv"
      start "$out"
      if [ "$engine" = llamacpp ]; then
        base=("$LLAMA_BENCH" --model "models/gguf/qwen2.5-coder-0.5b-instruct-$v.gguf" --variant "$v" --mode "$mode")
        case $mode in
          full)   timed "$out" "${base[@]}" --prompt-idx "$ALL_IDX" --reps "$REPS" ;;
          prefix) timed "$out" env GGML_METAL_TENSOR_DISABLE=1 "${base[@]}" --prompt-idx "$ALL_IDX" --reps "$REPS" ;;
          cold)
            for ((r = 0; r < COLD_REPS; r++)); do
              timed "$out" "${base[@]}" --prompt-idx "$ALL_IDX" --reps 1 --rep-start "$r" --warmup $((r == 0 ? 1 : 0))
            done ;;
        esac
      else
        base=("${PY[@]}" bench/competitors/mlx/bench.py --variant "$v" --mode "$mode")
        case $mode in
          full|prefix) timed "$out" "${base[@]}" --prompt-idx "$ALL_IDX" --reps "$REPS" ;;
          cold)
            first=1
            for ((r = 0; r < COLD_REPS; r++)); do
              while read -r chunk; do
                timed "$out" "${base[@]}" --prompt-idx "$chunk" --reps 1 --rep-start "$r" --warmup "$first"
                first=0
              done < <(chunks 10)
            done ;;
        esac
      fi
    done
  done
done

echo
"${PY[@]}" bench/summarize.py "$OUT"/*.tsv
