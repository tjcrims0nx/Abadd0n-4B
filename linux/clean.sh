#!/usr/bin/env bash
# Abadd0n — Remove cached/build artifacts (keeps lora_model, outputs, dataset)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Cleaning cache artifacts ..."
rm -rf __pycache__ core/__pycache__ tests/__pycache__ unsloth_compiled_cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "Done. lora_model, outputs, dataset.jsonl kept."
