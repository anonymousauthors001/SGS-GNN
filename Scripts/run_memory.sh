#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# PIPELINE="${PIPELINE:-hybrid}"
PIPELINE="${PIPELINE:-hybrid}"
DATASETS_CSV="${DATASETS_CSV:-}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"

if [[ -n "${DATASETS_CSV}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
else
  DATASETS=(
    amherst41
    Amazon-ratings
    Tolokers
    johnshopkins55
    cornell5
    wiki
    arxiv-year    
    Reddit
    # SmallCora
  )
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_STR}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_ARGS_STR})
fi

COMMON_ARGS=(
  --mode learned
  --runs 1
  --epochs 10
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
  --metis_threshold 1000000
)

run_memory () {
  local dataset="$1"
  local log_file="${LOG_DIR}/memory_${dataset}_${PIPELINE}.log"
  echo "=== Running memory profile: dataset=${dataset} pipeline=${PIPELINE} ==="
  (cd "${ROOT_DIR}" && python main.py "${COMMON_ARGS[@]}" --dataset "${dataset}" --pipeline "${PIPELINE}" "${EXTRA_ARGS[@]}") | tee "${log_file}"
  echo "--- Stats (${dataset} | ${PIPELINE}) ---"
  grep -n "\\[stats\\]" "${log_file}" || true
  echo ""
}

for dataset in "${DATASETS[@]}"; do
  run_memory "${dataset}"
done
