#!/usr/bin/env bash
# One benchmark tier: import (cached), ingest, run. Resumable: rerun the same command after an
# interruption and it continues the batch (eval run --resume) and skips ingested documents.
#
#   benchmarks/run_tier.sh <dataset> <limit|full> <config> <batch> [eval run options...]
#   benchmarks/run_tier.sh scifact 100 llama3.2-3b scifact-100-a --retrieval-only \
#       --strategy traditional --strategy vectorless
#
# RFB_QUESTIONS caps the question count at import (--questions); RFB_SEED sets the seed (11).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
dataset="$1"; limit="$2"; config="$3"; batch="$4"; shift 4
export RAGFABRIC_CONFIG="$RFB_CONFIG_DIR/$config.yaml"
out="$RFB_BENCH_DIR/$dataset-$limit${RFB_QUESTIONS:+-q$RFB_QUESTIONS}"
collection="$dataset-$limit${RFB_QUESTIONS:+-q$RFB_QUESTIONS}"
if [[ "$config" == *-graph ]]; then collection="$collection-graph"; fi
import_args=(--seed "${RFB_SEED:-11}" --out "$out")
if [[ "$limit" != "full" ]]; then import_args+=(--limit "$limit"); fi
if [[ -n "${RFB_QUESTIONS:-}" ]]; then import_args+=(--questions "$RFB_QUESTIONS"); fi
if [[ ! -f "$out/manifest.json" ]]; then uv run ragfabric eval import "$dataset" "${import_args[@]}"; fi
uv run ragfabric db upgrade >/dev/null
echo "[$(date -u +%FT%TZ)] ingest $collection"
uv run ragfabric eval corpus --path "$out/corpus" --collection "$collection"
echo "[$(date -u +%FT%TZ)] run $batch"
uv run ragfabric eval run --questions "$out/questions.json" --collection "$collection" \
  --batch "$batch" --resume "$@"
echo "[$(date -u +%FT%TZ)] done $batch"
