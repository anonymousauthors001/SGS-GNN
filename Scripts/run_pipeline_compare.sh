#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

PIPELINES_CSV="${PIPELINES:-two_pass,straight_through,hybrid}"
DATASETS_CSV="${DATASETS_CSV:-}"

IFS=',' read -r -a PIPELINES <<< "${PIPELINES_CSV}"

if [[ -n "${DATASETS_CSV}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
else
  DATASETS=(
    amherst41
    Amazon-ratings
    Tolokers
    johnshopkins55
    cornell5
    arxiv-year
    wiki
    Reddit
  )
fi

COMMON_ARGS=(
  --mode learned
  --runs 1
  --epochs 3
  --save_csv True
  --edge_mlp_type GCN
  --GNN GCN
  --log False
  --sparse_edge_mlp True
  --conditional True
  --reg1 True
  --reg2 True
  --stats True
  --hybrid_checkpoint True
  --gpu_profile True
)

run_pipeline () {
  local dataset="$1"
  local pipeline="$2"
  local log_file="${LOG_DIR}/pipeline_${dataset}_${pipeline}.log"
  echo "=== Running dataset=${dataset} pipeline=${pipeline} ==="
  (cd "${ROOT_DIR}" && python main.py "${COMMON_ARGS[@]}" --dataset "${dataset}" --pipeline "${pipeline}") | tee "${log_file}"
  echo "--- Stats (${dataset} | ${pipeline}) ---"
  grep -n "\\[stats\\]" "${log_file}" || true
  echo ""  
}

for dataset in "${DATASETS[@]}"; do
  for pipeline in "${PIPELINES[@]}"; do
    run_pipeline "${dataset}" "${pipeline}"
  done
done
