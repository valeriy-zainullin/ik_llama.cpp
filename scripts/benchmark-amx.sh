#!/bin/bash
# Build and run the AMX GEMM correctness test and benchmark.
#
# Usage:  scripts/benchmark-amx.sh [--extra-cmake-flags ...]
# Env:    BUILD_DIR=build-amx           build directory
#         GGML_AMX_DISABLE=1            runtime A/B switch (also honored by llama-cli/llama-bench)
#
set -e

BUILD_DIR="${BUILD_DIR:-build-amx}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cmake -B "$BUILD_DIR" -S "$REPO_DIR" -DGGML_AMX_INT8=ON -DLLAMA_BUILD_TESTS=ON "$@"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target test-amx-gemm

echo
echo "== correctness =="
"$BUILD_DIR/bin/test-amx-gemm"

echo
echo "== benchmark =="
"$BUILD_DIR/bin/test-amx-gemm" bench

echo
echo "AMX runtime switch: GGML_AMX_DISABLE=1 disables the AMX path."
echo "For end-to-end numbers on a Q4_0/Q8_0 model compare:"
echo "  llama-bench -m model.q4_0.gguf -p 512 -n 0 [-t <threads>]"
echo "with and without GGML_AMX_DISABLE=1 in the environment."
