#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/dell/data/gorgeous/sift1M/why_pipeline_gain_check/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${ROOT}"

for mem_l in 0 10; do
  for threads in 8 16 32; do
    case_dir="${ROOT}/mem${mem_l}_t${threads}"
    echo "==== MEM_L=${mem_l} T=${threads} L=20 30 ===="
    RESULT_ROOT="${case_dir}" \
      MEM_L="${mem_l}" \
      THREADS="${threads}" \
      L_LIST="20 30" \
      bash "${SCRIPT_DIR}/compare_latest_full_vs_original_sift1m.sh" \
      | tee "${ROOT}/mem${mem_l}_t${threads}.out"
  done
done

python3 - "${ROOT}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
fields = [
    "case",
    "L",
    "orig_QPS",
    "new_QPS",
    "speedup_pct",
    "orig_recall",
    "new_recall",
    "orig_graph_io",
    "new_graph_io",
    "new_pipe_w",
    "new_pipe_use_pct",
]

print(",".join(fields))
for path in sorted(root.glob("mem*_t*/run_*/compare.csv")):
    case_name = path.parts[-3]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            print(",".join([case_name] + [row[field] for field in fields[1:]]))

print(f"RESULT_ROOT={root}")
PY
