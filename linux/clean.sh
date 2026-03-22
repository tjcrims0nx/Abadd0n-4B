#!/usr/bin/env bash
# Abadd0n — Remove cached/build artifacts (keeps lora_model, outputs, dataset)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Cleaning cache artifacts ..."
rm -rf __pycache__ core/__pycache__ tests/__pycache__ unsloth_compiled_cache
rm -rf abadd0n_merged abadd0n_gguf abadd0n_gguf_gguf
rm -rf tests/_test_tools_run tests/_slash_run
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "Done. lora_model, outputs, dataset.jsonl kept."
