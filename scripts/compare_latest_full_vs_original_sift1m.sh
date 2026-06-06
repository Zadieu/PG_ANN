#!/usr/bin/env bash
set -euo pipefail

BASE_EXE="${BASE_EXE:-/home/dell/projects/Gorgeous-original-baseline/build/tests/search_disk_index}"
NEW_EXE="${NEW_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/latest_full_vs_original}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

THREADS="${THREADS:-8}"
BEAM_WIDTH="${BEAM_WIDTH:-8}"
MEM_L="${MEM_L:-10}"
L_LIST="${L_LIST:-18 20 25 30 35 40}"

mkdir -p "${RUN_DIR}"

COMMON_ARGS=(
  --data_type float
  --dist_fn l2
  --index_path_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/
  --pq_path_prefix /home/dell/data/gorgeous/sift1M/PQ/C4/
  --query_file /home/dell/data/sift/sift_query.fbin
  --gt_file /home/dell/data/sift/computed_gt_1000_sift1m.bin
  -K 10
  -T "${THREADS}"
  -W "${BEAM_WIDTH}"
  --mem_L "${MEM_L}"
  --sector_len 4096
  --mem_index_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index
  --mem_sample_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin
  --use_page_search 1
  --use_ratio 0.3
  --pq_ratio 1
  --disk_file_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index
  --graph_rep_index_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/
  --disk_graph_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/
  --deco_impl 1
  --use_graph_rep_index 0
  --mem_graph_use_ratio 0.1
  --mem_emb_use_ratio 0.0
  --emb_search_ratio 0.4
  --result_path "${RUN_DIR}/search_results.tsv"
)

DETAIL_CSV="${RUN_DIR}/detail.csv"
COMPARE_CSV="${RUN_DIR}/compare.csv"

echo "version,L,BW,QPS,MeanLtc,P999Ltc,GraphIO,EmbIO,ExtCmp,PQCmp,Recall10,PipeSub,PipeUsePct,PipeWst,PipeStl,PipeW,PipeMax,PipeAdj" > "${DETAIL_CSV}"

parse_log() {
  local version="$1"
  local lval="$2"
  local log_file="$3"

  awk -v version="${version}" -v lval="${lval}" -v bw="${BEAM_WIDTH}" '
    $1 == lval && $2 == bw {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        version, $1, $2, $3, $4, $5, $6, $7, $8, $9, $18,
        ($19 == "" ? 0 : $19), ($20 == "" ? 0 : $20),
        ($21 == "" ? 0 : $21), ($22 == "" ? 0 : $22),
        ($23 == "" ? 0 : $23), ($24 == "" ? 0 : $24),
        ($25 == "" ? 0 : $25)
    }
  ' "${log_file}" >> "${DETAIL_CSV}"
}

run_original() {
  local lval="$1"
  local log_file="${RUN_DIR}/original_mem_L${lval}.log"
  echo "[RUN] original_mem L=${lval}"
  env \
    GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
    GORGEOUS_PIPEANN_STATE_MACHINE=0 \
    GORGEOUS_PIPEANN_STATE_SCHEDULER=0 \
    GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=0 \
    GORGEOUS_PIPELINED_GRAPH_IO=0 \
    GORGEOUS_PIPELINED_REFINE_IO=0 \
    GORGEOUS_EARLY_REFINE_PREFETCH=0 \
    "${BASE_EXE}" -L "${lval}" "${COMMON_ARGS[@]}" > "${log_file}" 2>&1
  parse_log original_mem "${lval}" "${log_file}"
}

run_latest() {
  local lval="$1"
  local log_file="${RUN_DIR}/latest_full_L${lval}.log"
  echo "[RUN] latest_full L=${lval}"
  env \
    GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto \
    GORGEOUS_PIPEANN_STATE_MACHINE=1 \
    GORGEOUS_PIPEANN_STATE_SCHEDULER=1 \
    GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
    GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=1 \
    GORGEOUS_PIPEANN_PIPE_START=auto \
    GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10 \
    GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5 \
    GORGEOUS_PIPEANN_PIPE_FEEDBACK=1 \
    GORGEOUS_PIPELINED_GRAPH_IO=0 \
    GORGEOUS_PIPELINED_REFINE_IO=0 \
    GORGEOUS_EARLY_REFINE_PREFETCH=0 \
    "${NEW_EXE}" -L "${lval}" "${COMMON_ARGS[@]}" > "${log_file}" 2>&1
  parse_log latest_full "${lval}" "${log_file}"
}

for lval in ${L_LIST}; do
  run_original "${lval}"
  run_latest "${lval}"
done

python3 - "${DETAIL_CSV}" "${COMPARE_CSV}" <<'PY'
import csv
import sys

src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
by_key = {(row["version"], row["L"]): row for row in rows}

fields = [
    "L",
    "orig_QPS",
    "new_QPS",
    "speedup_pct",
    "orig_recall",
    "new_recall",
    "recall_delta",
    "orig_graph_io",
    "new_graph_io",
    "graph_io_delta",
    "orig_mean_ltc",
    "new_mean_ltc",
    "mean_ltc_delta",
    "new_pipe_use_pct",
    "new_pipe_w",
    "new_pipe_max",
    "new_pipe_adj",
]

def f(row, name):
    return float(row[name])

with open(dst, "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for lval in sorted({row["L"] for row in rows}, key=lambda x: int(float(x))):
        original = by_key.get(("original_mem", lval))
        latest = by_key.get(("latest_full", lval))
        if original is None or latest is None:
            continue
        writer.writerow({
            "L": lval,
            "orig_QPS": original["QPS"],
            "new_QPS": latest["QPS"],
            "speedup_pct": f"{(f(latest, 'QPS') / f(original, 'QPS') - 1.0) * 100.0:.2f}",
            "orig_recall": original["Recall10"],
            "new_recall": latest["Recall10"],
            "recall_delta": f"{f(latest, 'Recall10') - f(original, 'Recall10'):.2f}",
            "orig_graph_io": original["GraphIO"],
            "new_graph_io": latest["GraphIO"],
            "graph_io_delta": f"{f(latest, 'GraphIO') - f(original, 'GraphIO'):.2f}",
            "orig_mean_ltc": original["MeanLtc"],
            "new_mean_ltc": latest["MeanLtc"],
            "mean_ltc_delta": f"{f(latest, 'MeanLtc') - f(original, 'MeanLtc'):.2f}",
            "new_pipe_use_pct": latest["PipeUsePct"],
            "new_pipe_w": latest["PipeW"],
            "new_pipe_max": latest["PipeMax"],
            "new_pipe_adj": latest["PipeAdj"],
        })
PY

cat "${COMPARE_CSV}"
echo "RESULT_DIR=${RUN_DIR}"
