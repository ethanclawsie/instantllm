#!/usr/bin/env bash
# Downloads Qwen2.5-Coder-0.5B-Instruct in every format the benchmark uses (idempotent).
set -uo pipefail
cd "$(dirname "$0")"
HF=https://huggingface.co
get() { # repo file dest
  local repo=$1 file=$2 dest=$3
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then echo "have $dest"; return; fi
  curl -sSfL --retry 5 -o "$dest.part" "$HF/$repo/resolve/main/$file" && mv "$dest.part" "$dest" && echo "got $dest" || echo "FAILED $repo/$file"
}
R=Qwen/Qwen2.5-Coder-0.5B-Instruct
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do get $R $f hf/$f; done
G=Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF
for q in q4_0 q4_k_m q8_0 fp16; do get $G qwen2.5-coder-0.5b-instruct-$q.gguf gguf/qwen2.5-coder-0.5b-instruct-$q.gguf; done
for v in 4bit 8bit bf16; do
  M=mlx-community/Qwen2.5-Coder-0.5B-Instruct-$v
  for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json model.safetensors model.safetensors.index.json; do get $M $f mlx/$v/$f; done
done
echo DONE
