#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dell/data/gorgeous/sift1M/latest_full_vs_original_3run}"
RUNS="${RUNS:-3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${ROOT}"

for i in $(seq 1 "${RUNS}"); do
  echo "==== MULTIRUN ${i}/${RUNS} ===="
  RESULT_ROOT="${ROOT}" bash "${SCRIPT_DIR}/compare_latest_full_vs_original_sift1m.sh"
done

python3 - "${ROOT}" <<'PY'
import csv
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
runs = sorted(root.glob("run_*/compare.csv"))
rows_by_l = {}

for path in runs:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_l.setdefault(row["L"], []).append(row)

out = root / "summary_mean.csv"
fields = [
    "L",
    "runs",
    "orig_QPS_mean",
    "new_QPS_mean",
    "speedup_pct_mean",
    "orig_recall_mean",
    "new_recall_mean",
    "recall_delta_mean",
    "orig_graph_io_mean",
    "new_graph_io_mean",
    "graph_io_delta_mean",
    "orig_mean_ltc_mean",
    "new_mean_ltc_mean",
    "mean_ltc_delta_mean",
    "new_pipe_use_pct_mean",
    "new_pipe_w_mean",
    "new_pipe_max_mean",
    "new_pipe_adj_mean",
]

def avg(rows, key):
    return statistics.mean(float(row[key]) for row in rows)

with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for lval in sorted(rows_by_l, key=lambda value: int(float(value))):
        rows = rows_by_l[lval]
        orig_qps = avg(rows, "orig_QPS")
        new_qps = avg(rows, "new_QPS")
        orig_recall = avg(rows, "orig_recall")
        new_recall = avg(rows, "new_recall")
        orig_graph_io = avg(rows, "orig_graph_io")
        new_graph_io = avg(rows, "new_graph_io")
        orig_mean_ltc = avg(rows, "orig_mean_ltc")
        new_mean_ltc = avg(rows, "new_mean_ltc")
        writer.writerow({
            "L": lval,
            "runs": len(rows),
            "orig_QPS_mean": f"{orig_qps:.2f}",
            "new_QPS_mean": f"{new_qps:.2f}",
            "speedup_pct_mean": f"{(new_qps / orig_qps - 1.0) * 100.0:.2f}",
            "orig_recall_mean": f"{orig_recall:.2f}",
            "new_recall_mean": f"{new_recall:.2f}",
            "recall_delta_mean": f"{new_recall - orig_recall:.2f}",
            "orig_graph_io_mean": f"{orig_graph_io:.2f}",
            "new_graph_io_mean": f"{new_graph_io:.2f}",
            "graph_io_delta_mean": f"{new_graph_io - orig_graph_io:.2f}",
            "orig_mean_ltc_mean": f"{orig_mean_ltc:.2f}",
            "new_mean_ltc_mean": f"{new_mean_ltc:.2f}",
            "mean_ltc_delta_mean": f"{new_mean_ltc - orig_mean_ltc:.2f}",
            "new_pipe_use_pct_mean": f"{avg(rows, 'new_pipe_use_pct'):.2f}",
            "new_pipe_w_mean": f"{avg(rows, 'new_pipe_w'):.2f}",
            "new_pipe_max_mean": f"{avg(rows, 'new_pipe_max'):.2f}",
            "new_pipe_adj_mean": f"{avg(rows, 'new_pipe_adj'):.2f}",
        })

print(out)
print(out.read_text())
PY
