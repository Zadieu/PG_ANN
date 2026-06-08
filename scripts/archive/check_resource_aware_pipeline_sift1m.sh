#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/dell/data/gorgeous/sift1M/resource_aware_pipeline_check/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${ROOT}"

for threads in 16 32; do
  echo "==== feedback_only T=${threads} ===="
  RESULT_ROOT="${ROOT}/feedback_only_t${threads}" \
    MEM_L=10 \
    THREADS="${threads}" \
    L_LIST="20 30" \
    GORGEOUS_PIPEANN_RESOURCE_AWARE=0 \
    bash "${SCRIPT_DIR}/compare_latest_full_vs_original_sift1m.sh" \
    | tee "${ROOT}/feedback_only_t${threads}.out"

  echo "==== resource_aware T=${threads} ===="
  RESULT_ROOT="${ROOT}/resource_aware_t${threads}" \
    MEM_L=10 \
    THREADS="${threads}" \
    L_LIST="20 30" \
    GORGEOUS_PIPEANN_RESOURCE_AWARE=1 \
    GORGEOUS_PIPEANN_PIPE_DOWN_WASTE_THRESHOLD=0.20 \
    GORGEOUS_PIPEANN_PIPE_FEEDBACK_WINDOW=4 \
    bash "${SCRIPT_DIR}/compare_latest_full_vs_original_sift1m.sh" \
    | tee "${ROOT}/resource_aware_t${threads}.out"
done

python3 - "${ROOT}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fields = [
    "mode",
    "L",
    "orig_QPS",
    "new_QPS",
    "speedup_pct",
    "recall_delta",
    "new_graph_io",
    "new_pipe_use_pct",
    "new_pipe_w",
    "new_pipe_max",
    "new_pipe_adj",
]

print(",".join(fields))
for path in sorted(root.glob("*/run_*/compare.csv")):
    mode = path.parts[-3]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            print(",".join([mode] + [row[field] for field in fields[1:]]))

print(f"RESULT_ROOT={root}")
PY
