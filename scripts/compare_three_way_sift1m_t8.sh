#!/usr/bin/env bash
set -euo pipefail

PIPEANN_EXE="${PIPEANN_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/search_disk_index}"
PIPEANN_BUILD_MEM_EXE="${PIPEANN_BUILD_MEM_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/build_memory_index}"
GORGEOUS_EXE="${GORGEOUS_EXE:-/home/dell/projects/Gorgeous-original-baseline/build/tests/search_disk_index}"
PIPEGOR_EXE="${PIPEGOR_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"

THREADS="${THREADS:-8}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
L_LIST="${L_LIST:-15 18 20 25 30 35}"
K="${K:-10}"

PIPEANN_INDEX_PREFIX="${PIPEANN_INDEX_PREFIX:-/home/dell/data/pg_ann_from_gorgeous/sift1m/sift1m}"
PIPEANN_DATA_FILE="${PIPEANN_DATA_FILE:-/home/dell/data/sift/sift_base.fbin}"
PIPEANN_TAGS_FILE="${PIPEANN_TAGS_FILE:-${PIPEANN_INDEX_PREFIX}_disk.index.tags}"
PIPEANN_MEM_INDEX="${PIPEANN_MEM_INDEX:-${PIPEANN_INDEX_PREFIX}_mem.index}"
PIPEANN_MEM_R="${PIPEANN_MEM_R:-64}"
PIPEANN_MEM_BUILD_L="${PIPEANN_MEM_BUILD_L:-128}"
PIPEANN_MEM_ALPHA="${PIPEANN_MEM_ALPHA:-1.2}"
PIPEANN_MEM_THREADS="${PIPEANN_MEM_THREADS:-16}"

GORGEOUS_INDEX_PREFIX="${GORGEOUS_INDEX_PREFIX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/}"
GORGEOUS_PQ_PREFIX="${GORGEOUS_PQ_PREFIX:-/home/dell/data/gorgeous/sift1M/PQ/C4/}"
GORGEOUS_DISK_FILE="${GORGEOUS_DISK_FILE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index}"
GORGEOUS_MEM_INDEX="${GORGEOUS_MEM_INDEX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index}"
GORGEOUS_MEM_SAMPLE="${GORGEOUS_MEM_SAMPLE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin}"
GORGEOUS_DISK_GRAPH="${GORGEOUS_DISK_GRAPH:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/}"
GORGEOUS_GRAPH_REP="${GORGEOUS_GRAPH_REP:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/}"
GORGEOUS_USE_RATIO="${GORGEOUS_USE_RATIO:-0.3}"
GORGEOUS_PQ_RATIO="${GORGEOUS_PQ_RATIO:-1.0}"

QUERY_FILE="${QUERY_FILE:-/home/dell/data/sift/sift_query.fbin}"
GT_FILE="${GT_FILE:-/home/dell/data/sift/computed_gt_1000_sift1m.bin}"
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/three_way_t8}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}/original" "${RUN_DIR}/pipegor"

ensure_pipeann_mem_index() {
  if [[ "${MEM_L}" == "0" ]]; then
    return
  fi
  if [[ -f "${PIPEANN_MEM_INDEX}" && -f "${PIPEANN_MEM_INDEX}.tags" ]]; then
    return
  fi
  echo "[INFO] Building PipeANN mem index at ${PIPEANN_MEM_INDEX}"
  "${PIPEANN_BUILD_MEM_EXE}" \
    float \
    "${PIPEANN_DATA_FILE}" \
    "${PIPEANN_TAGS_FILE}" \
    "${PIPEANN_MEM_INDEX}" \
    "${PIPEANN_MEM_R}" \
    "${PIPEANN_MEM_BUILD_L}" \
    "${PIPEANN_MEM_ALPHA}" \
    "${PIPEANN_MEM_THREADS}" \
    l2
}

COMMON_GORGEOUS_ARGS=(
  --data_type float
  --dist_fn l2
  --index_path_prefix "${GORGEOUS_INDEX_PREFIX}"
  --pq_path_prefix "${GORGEOUS_PQ_PREFIX}"
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  -K "${K}"
  -L ${L_LIST}
  -T "${THREADS}"
  -W "${WIDTH}"
  --mem_L "${MEM_L}"
  --sector_len 4096
  --mem_index_path "${GORGEOUS_MEM_INDEX}"
  --mem_sample_path "${GORGEOUS_MEM_SAMPLE}"
  --use_page_search 1
  --use_ratio "${GORGEOUS_USE_RATIO}"
  --pq_ratio "${GORGEOUS_PQ_RATIO}"
  --disk_file_path "${GORGEOUS_DISK_FILE}"
  --disk_graph_prefix "${GORGEOUS_DISK_GRAPH}"
  --graph_rep_index_prefix "${GORGEOUS_GRAPH_REP}"
  --deco_impl 1
  --mem_graph_use_ratio 0.0
  --mem_emb_use_ratio 0.0
  --emb_search_ratio 1.0
)

cat > "${RUN_DIR}/PARAMS.txt" <<EOF
Dataset: SIFT1M
threads = ${THREADS}
width = ${WIDTH}
mem_L = ${MEM_L}
K = ${K}
L values = ${L_LIST}

PipeANN:
  exe = ${PIPEANN_EXE}
  search_mode = 2
  index_prefix = ${PIPEANN_INDEX_PREFIX}

Gorgeous original:
  exe = ${GORGEOUS_EXE}
  dynamic cache = 0
  pipeann scheduler = 0
  graph_rep_index = 0

PipeGor_ANN:
  exe = ${PIPEGOR_EXE}
  dynamic cache = 0
  pipeann scheduler = 1
  dynamic width = 1
  graph_rep_index = 1
  rank_ready_expand = 0
EOF

ensure_pipeann_mem_index

PIPEANN_LOG="${RUN_DIR}/pipeann.log"
GORGEOUS_LOG="${RUN_DIR}/gorgeous_original.log"
PIPEGOR_LOG="${RUN_DIR}/pipegor.log"

echo "[RUN] PipeANN baseline"
"${PIPEANN_EXE}" \
  float \
  "${PIPEANN_INDEX_PREFIX}" \
  "${THREADS}" \
  "${WIDTH}" \
  "${QUERY_FILE}" \
  "${GT_FILE}" \
  "${K}" \
  l2 \
  pq \
  2 \
  "${MEM_L}" \
  ${L_LIST} > "${PIPEANN_LOG}" 2>&1

echo "[RUN] Gorgeous original"
env \
  GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
  GORGEOUS_PIPEANN_STATE_MACHINE=0 \
  GORGEOUS_PIPEANN_STATE_SCHEDULER=0 \
  GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=0 \
  GORGEOUS_PIPELINED_GRAPH_IO=0 \
  GORGEOUS_PIPELINED_REFINE_IO=0 \
  GORGEOUS_EARLY_REFINE_PREFETCH=0 \
  "${GORGEOUS_EXE}" \
    "${COMMON_GORGEOUS_ARGS[@]}" \
    --result_path "${RUN_DIR}/original/result" \
    --use_graph_rep_index 0 > "${GORGEOUS_LOG}" 2>&1

echo "[RUN] PipeGor_ANN"
env \
  GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
  GORGEOUS_PIPEANN_STATE_MACHINE=1 \
  GORGEOUS_PIPEANN_STATE_SCHEDULER=1 \
  GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
  GORGEOUS_PIPEANN_RECALL_SAFE_WINDOW=1 \
  GORGEOUS_PIPEANN_STALE_OUTSIDE_WINDOW=0 \
  GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=1 \
  GORGEOUS_PIPEANN_PIPE_START=auto \
  GORGEOUS_PIPEANN_PIPE_MIN=auto \
  GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10 \
  GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5 \
  GORGEOUS_PIPEANN_PIPE_FEEDBACK=1 \
  GORGEOUS_PIPEANN_RESOURCE_AWARE=0 \
  GORGEOUS_PIPEANN_RANK_READY_EXPAND=0 \
  GORGEOUS_PIPELINED_GRAPH_IO=0 \
  GORGEOUS_PIPELINED_REFINE_IO=0 \
  GORGEOUS_EARLY_REFINE_PREFETCH=0 \
  "${PIPEGOR_EXE}" \
    "${COMMON_GORGEOUS_ARGS[@]}" \
    --result_path "${RUN_DIR}/pipegor/result" \
    --use_graph_rep_index 1 > "${PIPEGOR_LOG}" 2>&1

python3 - "${PIPEANN_LOG}" "${GORGEOUS_LOG}" "${PIPEGOR_LOG}" "${RUN_DIR}/compare.csv" <<'PY'
import csv
import pathlib
import sys

pipeann_log = pathlib.Path(sys.argv[1])
gorgeous_log = pathlib.Path(sys.argv[2])
pipegor_log = pathlib.Path(sys.argv[3])
compare_csv = pathlib.Path(sys.argv[4])

def parse_pipeann(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            rows[parts[0]] = {
                "qps": float(parts[2]),
                "mean_ltc": float(parts[3]),
                "p99": float(parts[4]),
                "graph_io": float(parts[6]),
                "recall": float(parts[7]),
                "pipe_w": 0.0,
                "pipe_max": 0.0,
                "pipe_adj": 0.0,
            }
    return rows

def parse_gorgeous_like(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 18 and parts[0].isdigit():
            rows[parts[0]] = {
                "qps": float(parts[2]),
                "mean_ltc": float(parts[3]),
                "p99": float(parts[4]),
                "graph_io": float(parts[5]),
                "recall": float(parts[17]),
                "pipe_w": float(parts[22]) if len(parts) > 22 else 0.0,
                "pipe_max": float(parts[23]) if len(parts) > 23 else 0.0,
                "pipe_adj": float(parts[24]) if len(parts) > 24 else 0.0,
            }
    return rows

pipeann = parse_pipeann(pipeann_log)
gorgeous = parse_gorgeous_like(gorgeous_log)
pipegor = parse_gorgeous_like(pipegor_log)

fields = [
    "L",
    "PipeANN_QPS",
    "Gorgeous_QPS",
    "PipeGor_QPS",
    "PipeGor_vs_PipeANN_QPS_pct",
    "PipeGor_vs_Gorgeous_QPS_pct",
    "PipeANN_Recall",
    "Gorgeous_Recall",
    "PipeGor_Recall",
    "PipeGor_vs_PipeANN_Recall_delta",
    "PipeGor_vs_Gorgeous_Recall_delta",
    "PipeANN_GraphIO",
    "Gorgeous_GraphIO",
    "PipeGor_GraphIO",
    "PipeGor_PipeW",
    "PipeGor_PipeMax",
    "PipeGor_PipeAdj",
]

with compare_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for lval in sorted(set(pipeann) & set(gorgeous) & set(pipegor), key=lambda x: int(x)):
        p = pipeann[lval]
        g = gorgeous[lval]
        pg = pipegor[lval]
        writer.writerow({
            "L": lval,
            "PipeANN_QPS": f"{p['qps']:.2f}",
            "Gorgeous_QPS": f"{g['qps']:.2f}",
            "PipeGor_QPS": f"{pg['qps']:.2f}",
            "PipeGor_vs_PipeANN_QPS_pct": f"{(pg['qps'] / p['qps'] - 1.0) * 100.0:.2f}",
            "PipeGor_vs_Gorgeous_QPS_pct": f"{(pg['qps'] / g['qps'] - 1.0) * 100.0:.2f}",
            "PipeANN_Recall": f"{p['recall']:.2f}",
            "Gorgeous_Recall": f"{g['recall']:.2f}",
            "PipeGor_Recall": f"{pg['recall']:.2f}",
            "PipeGor_vs_PipeANN_Recall_delta": f"{pg['recall'] - p['recall']:.2f}",
            "PipeGor_vs_Gorgeous_Recall_delta": f"{pg['recall'] - g['recall']:.2f}",
            "PipeANN_GraphIO": f"{p['graph_io']:.2f}",
            "Gorgeous_GraphIO": f"{g['graph_io']:.2f}",
            "PipeGor_GraphIO": f"{pg['graph_io']:.2f}",
            "PipeGor_PipeW": f"{pg['pipe_w']:.2f}",
            "PipeGor_PipeMax": f"{pg['pipe_max']:.2f}",
            "PipeGor_PipeAdj": f"{pg['pipe_adj']:.2f}",
        })

print(compare_csv.read_text(), end="")
PY

echo "RESULT_DIR=${RUN_DIR}"
