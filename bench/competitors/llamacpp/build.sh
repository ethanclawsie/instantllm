#!/usr/bin/env bash
# Builds the llama.cpp competitor harness against Homebrew's libllama 0.6.0 (ggml 0.26.0).
# Output: bench/competitors/llamacpp/bench (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
LLAMA=/opt/homebrew/opt/llama.cpp
GGML=/opt/homebrew/opt/ggml
clang++ -std=c++17 -O2 -Wall -Wextra -o bench bench.cpp \
  -I"$LLAMA/include" -I"$GGML/include" \
  -L"$LLAMA/lib" -L"$GGML/lib" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/lib" -Wl,-rpath,"$GGML/lib"
echo "built $(pwd)/bench"
