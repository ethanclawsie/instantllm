#!/usr/bin/env bash
# Head-to-head benchmark: instantllm vs llama.cpp vs mlx-lm on Qwen2.5-Coder-0.5B-Instruct, same
# protocol for every engine (docs/SPEC.md). Engines are interleaved: every round runs each config
# once over all 20 prompts, in a rotated order, so thermal and clock drift hit everyone equally.
#
#   bench/run_all.sh                 # everything (~25 min); run with nothing else on the machine
#   ROUNDS=2 CONFIGS="iq4 lq40 m4" bench/run_all.sh
#
# Modes per config:
#   prefix   system-prompt KV cached by the engine, natural stop (<= 64 tokens)
#   fixed    same, but exactly 32 generated tokens (--ignore-eos): equal work for every engine
#   full     whole templated prompt prefilled per request, natural stop
#   cold     a fresh process per request (one-shot CLI), spawn -> exit
# Results: bench/results/<config>-<mode>.tsv (schema in bench/summarize.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -z "${INSTANTLLM_CAFFEINATED:-}" ]; then
  export INSTANTLLM_CAFFEINATED=1
  exec caffeinate -dims "$0" "$@"
fi

ROUNDS=${ROUNDS:-3}
COOLDOWN=${COOLDOWN:-5}
OUT=${OUT:-bench/results}
MODES=${MODES:-"prefix fixed full"}
COLD_CONFIGS=${COLD_CONFIGS-"iq4 lq40 m4"}   # set to "" to skip the cold runs
# config name -> "engine variant" (bash 3.2: no associative arrays)
cfg_run() {
  case $1 in
    iq4) echo "instantllm q4" ;; iq8) echo "instantllm q8" ;; if16) echo "instantllm f16" ;;
    lq40) echo "llamacpp q4_0" ;; lq4km) echo "llamacpp q4_k_m" ;; lq80) echo "llamacpp q8_0" ;; lf16) echo "llamacpp fp16" ;;
    m4) echo "mlx 4bit" ;; m8) echo "mlx 8bit" ;; mbf16) echo "mlx bf16" ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}
CONFIGS=${CONFIGS:-"iq4 lq40 lq4km m4 iq8 lq80 m8 if16 lf16 mbf16"}
LOCK=(lockf -k /tmp/instantllm-gpu.lock)
PY=("$ROOT/.venv/bin/python" -I)
mkdir -p "$OUT" bench/logs
LOG="bench/logs/run_all-$(date +%Y%m%d-%H%M%S).log"
echo "log: $LOG"
swift build -c release >/dev/null
[ -x bench/competitors/llamacpp/bench ] || bench/competitors/llamacpp/build.sh

relabel() { awk -v m="$1" 'BEGIN{FS=OFS="\t"} {$3=m; print}'; }  # set the mode column

run_one() { # config mode -> TSV rows on stdout (no header)
  local cfg=$1 mode=$2 engine variant
  read -r engine variant <<<"$(cfg_run "$cfg")"
  local hm=$mode extra=()
  if [ "$mode" = fixed ]; then hm=prefix; extra=(--ignore-eos --n-predict 32); fi
  case $engine in
    instantllm)
      "${LOCK[@]}" .build/release/instantllm bench --model "models/ill/qwen-$variant.ill" --mode "$hm" --reps 1 \
        --variant "$variant" ${extra[@]+"${extra[@]}"} | tail -n +2 | relabel "$mode" ;;
    llamacpp)
      local envs=(INSTANTLLM_BENCH=1)
      [ "$hm" = prefix ] && envs+=(GGML_METAL_TENSOR_DISABLE=1)
      "${LOCK[@]}" env "${envs[@]}" bench/competitors/llamacpp/bench \
        --model "models/gguf/qwen2.5-coder-0.5b-instruct-$variant.gguf" --mode "$hm" --reps 1 --no-header ${extra[@]+"${extra[@]}"} \
        | relabel "$mode" ;;
    mlx)
      "${LOCK[@]}" "${PY[@]}" bench/competitors/mlx/bench.py --variant "$variant" --mode "$hm" --reps 1 --no-header ${extra[@]+"${extra[@]}"} \
        | relabel "$mode" ;;
  esac
}

HEADER=$'engine\tvariant\tmode\tprompt_idx\trep\te2e_ms\tttft_ms\tprefill_tokens\tgen_tokens\tdecode_tok_s\ttext'
read -r -a cfgs <<<"$CONFIGS"
for mode in $MODES; do
  for c in "${cfgs[@]}"; do f="$OUT/$c-$mode.tsv"; [ -s "$f" ] || echo "$HEADER" > "$f"; done
done
n=${#cfgs[@]}
for ((r = 0; r < ROUNDS; r++)); do
  for mode in $MODES; do
    for ((i = 0; i < n; i++)); do
      c=${cfgs[$(((i + r) % n))]}
      echo "$(date +%H:%M:%S) round $r $mode $c" | tee -a "$LOG"
      run_one "$c" "$mode" 2>>"$LOG" | awk -v r="$r" 'BEGIN{FS=OFS="\t"} {$5=r; print}' >> "$OUT/$c-$mode.tsv" || echo "!! failed: $c $mode" | tee -a "$LOG"
      sleep "$COOLDOWN"
    done
  done
done

# Cold: fresh process per request, spawn -> exit, every prompt, interleaved per prompt.
if [ -n "$COLD_CONFIGS" ]; then
  "${PY[@]}" bench/cold.py --configs $COLD_CONFIGS --rounds "$ROUNDS" --out "$OUT" 2>>"$LOG"
fi
"${PY[@]}" bench/summarize.py "$OUT"/*.tsv
