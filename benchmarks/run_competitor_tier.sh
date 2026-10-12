#!/usr/bin/env bash
# A comparison target on a tier that run_tier.sh already imported and ingested. Resumable.
#
#   benchmarks/run_competitor_tier.sh <llamaindex|langchain> <dataset-dir-name> <config> <batch> \
#       [--retrieval-only] [--judge-metric correctness]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
framework="$1"; tier="$2"; config="$3"; batch="$4"; shift 4
export RAGFABRIC_CONFIG="$RFB_CONFIG_DIR/$config.yaml"
echo "[$(date -u +%FT%TZ)] $framework $batch"
uv run --with-requirements "$here/requirements-competitors.txt" python "$here/run_competitors.py" \
  --framework "$framework" --questions "$RFB_BENCH_DIR/$tier/questions.json" \
  --collection "$tier" --batch "$batch" "$@"
echo "[$(date -u +%FT%TZ)] done $framework $batch"
