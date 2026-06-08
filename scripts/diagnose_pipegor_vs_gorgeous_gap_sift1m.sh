#!/usr/bin/env bash
set -euo pipefail

ORIG_EXE="${ORIG_EXE:-/home/dell/projects/Gorgeous-original-baseline/build/tests/search_disk_index}"
PIPEGOR_EXE="${PIPEGOR_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"

THREADS="${THREADS:-8}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
L_LIST="${L_LIST:-20 30 35}"
K="${K:-10}"

INDEX_PREFIX="${INDEX_PREFIX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/}"
PQ_PREFIX="${PQ_PREFIX:-/home/dell/data/gorgeous/sift1M/PQ/C4/}"
DISK_FILE="${DISK_FILE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index}"
MEM_INDEX="${MEM_INDEX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index}"
MEM_SAMPLE="${MEM_SAMPLE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin}"
DISK_GRAPH="${DISK_GRAPH:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/}"
GRAPH_REP="${GRAPH_REP:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/}"
QUERY_FILE="${QUERY_FILE:-/home/dell/data/sift/sift_query.fbin}"
GT_FILE="${GT_FILE:-/home/dell/data/sift/computed_gt_1000_sift1m.bin}"
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/diagnose_pipegor_gap}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}"

COMMON_ARGS=(
  --data_type float
  --dist_fn l2
  --index_path_prefix "${INDEX_PREFIX}"
  --pq_path_prefix "${PQ_PREFIX}"
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  -K "${K}"
  -L ${L_LIST}
  -T "${THREADS}"
  -W "${WIDTH}"
  --mem_L "${MEM_L}"
  --sector_len 4096
  --mem_index_path "${MEM_INDEX}"
  --mem_sample_path "${MEM_SAMPLE}"
  --use_page_search 1
  --use_ratio 0.3
  --pq_ratio 1.0
  --disk_file_path "${DISK_FILE}"
  --disk_graph_prefix "${DISK_GRAPH}"
  --graph_rep_index_prefix "${GRAPH_REP}"
  --deco_impl 1
  --mem_graph_use_ratio 0.0
  --mem_emb_use_ratio 0.0
)

run_case() {
  local name="$1"
  local exe="$2"
  local graph_rep="$3"
  local emb_ratio="$4"
  local scheduler="$5"
  local refine_pipe="${6:-0}"
  local grouped_refine="${7:-0}"
  local early_refine="${8:-0}"
  local log_file="${RUN_DIR}/${name}.log"

  echo "[RUN] ${name}"
  env \
    GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
    GORGEOUS_PIPEANN_STATE_MACHINE="${scheduler}" \
    GORGEOUS_PIPEANN_STATE_SCHEDULER="${scheduler}" \
    GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
    GORGEOUS_PIPEANN_RECALL_SAFE_WINDOW=1 \
    GORGEOUS_PIPEANN_STALE_OUTSIDE_WINDOW=0 \
    GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH="${scheduler}" \
    GORGEOUS_PIPEANN_PIPE_START=auto \
    GORGEOUS_PIPEANN_PIPE_MIN=auto \
    GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10 \
    GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5 \
    GORGEOUS_PIPEANN_PIPE_FEEDBACK="${scheduler}" \
    GORGEOUS_PIPEANN_RESOURCE_AWARE=0 \
    GORGEOUS_PIPEANN_RANK_READY_EXPAND=0 \
    GORGEOUS_PIPELINED_GRAPH_IO=0 \
    GORGEOUS_PIPELINED_REFINE_IO="${refine_pipe}" \
    GORGEOUS_GRAPH_REP_GROUPED_REFINE="${grouped_refine}" \
    GORGEOUS_EARLY_REFINE_PREFETCH="${early_refine}" \
    GORGEOUS_EARLY_REFINE_FEEDBACK="${early_refine}" \
    "${exe}" \
      "${COMMON_ARGS[@]}" \
      --result_path "${RUN_DIR}/${name}_result" \
      --use_graph_rep_index "${graph_rep}" \
      --emb_search_ratio "${emb_ratio}" > "${log_file}" 2>&1
}

run_case original_nongraph "${ORIG_EXE}" 0 1.0 0 0 0
run_case pipegor_nongraph "${PIPEGOR_EXE}" 0 1.0 0 0 0
run_case pipegor_graphrep_sched_emb100 "${PIPEGOR_EXE}" 1 1.0 1 0 0
run_case pipegor_graphrep_sched_emb100_refinepipe "${PIPEGOR_EXE}" 1 1.0 1 1 0
run_case pipegor_graphrep_sched_emb100_groupref "${PIPEGOR_EXE}" 1 1.0 1 0 1
run_case pipegor_graphrep_sched_emb100_groupref_pipe "${PIPEGOR_EXE}" 1 1.0 1 1 1
run_case pipegor_graphrep_sched_emb100_groupref_pipe_early "${PIPEGOR_EXE}" 1 1.0 1 1 1 1
run_case pipegor_graphrep_sched_emb040 "${PIPEGOR_EXE}" 1 0.4 1 0 0
run_case pipegor_graphrep_nosched_emb100 "${PIPEGOR_EXE}" 1 1.0 0 0 0

python3 - "${RUN_DIR}" <<'PY'
import csv
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])

def parse(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 18 and parts[0].isdigit():
            rows[parts[0]] = {
                "qps": float(parts[2]),
                "mean": float(parts[3]),
                "p999": float(parts[4]),
                "graph": float(parts[5]),
                "emb": float(parts[6]),
                "ext": float(parts[7]),
                "pq": float(parts[8]),
                "pre": float(parts[9]),
                "disp": float(parts[10]),
                "read": float(parts[11]),
                "cache": float(parts[13]),
                "diskn": float(parts[14]),
                "post": float(parts[15]),
                "recall": float(parts[17]),
                "pipe_w": float(parts[22]) if len(parts) > 22 else 0.0,
                "pipe_max": float(parts[23]) if len(parts) > 23 else 0.0,
                "pipe_adj": float(parts[24]) if len(parts) > 24 else 0.0,
                "refine": float(parts[25]) if len(parts) > 25 else 0.0,
                "sort": float(parts[26]) if len(parts) > 26 else 0.0,
                "ref_io": float(parts[27]) if len(parts) > 27 else 0.0,
                "ref_exact": float(parts[28]) if len(parts) > 28 else 0.0,
            }
    return rows

cases = [
    "original_nongraph",
    "pipegor_nongraph",
    "pipegor_graphrep_sched_emb100",
    "pipegor_graphrep_sched_emb100_refinepipe",
    "pipegor_graphrep_sched_emb100_groupref",
    "pipegor_graphrep_sched_emb100_groupref_pipe",
    "pipegor_graphrep_sched_emb100_groupref_pipe_early",
    "pipegor_graphrep_sched_emb040",
    "pipegor_graphrep_nosched_emb100",
]
data = {case: parse(run_dir / f"{case}.log") for case in cases}
fields = [
    "case", "L", "QPS", "Recall", "MeanLtc", "P999",
    "GraphIO", "EmbIO", "ExtCmp", "PQCmp",
    "PreT", "DispT", "ReadT", "CacheT", "DiskNT", "PostT", "RefineT", "SortT",
    "RefIOT", "RefExactT", "PipeW", "PipeMax", "PipeAdj",
]
out_path = run_dir / "diagnose.csv"
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for case in cases:
        for lval in sorted(data[case], key=lambda x: int(x)):
            row = data[case][lval]
            writer.writerow({
                "case": case,
                "L": lval,
                "QPS": f"{row['qps']:.2f}",
                "Recall": f"{row['recall']:.2f}",
                "MeanLtc": f"{row['mean']:.2f}",
                "P999": f"{row['p999']:.2f}",
                "GraphIO": f"{row['graph']:.2f}",
                "EmbIO": f"{row['emb']:.2f}",
                "ExtCmp": f"{row['ext']:.2f}",
                "PQCmp": f"{row['pq']:.2f}",
                "PreT": f"{row['pre']:.2f}",
                "DispT": f"{row['disp']:.2f}",
                "ReadT": f"{row['read']:.2f}",
                "CacheT": f"{row['cache']:.2f}",
                "DiskNT": f"{row['diskn']:.2f}",
                "PostT": f"{row['post']:.2f}",
                "RefineT": f"{row['refine']:.2f}",
                "SortT": f"{row['sort']:.2f}",
                "RefIOT": f"{row['ref_io']:.2f}",
                "RefExactT": f"{row['ref_exact']:.2f}",
                "PipeW": f"{row['pipe_w']:.2f}",
                "PipeMax": f"{row['pipe_max']:.2f}",
                "PipeAdj": f"{row['pipe_adj']:.2f}",
            })
print(out_path.read_text(), end="")
PY

echo "RESULT_DIR=${RUN_DIR}"
