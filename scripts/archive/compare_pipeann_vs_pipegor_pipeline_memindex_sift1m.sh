#!/usr/bin/env bash
set -euo pipefail

PIPEANN_EXE="${PIPEANN_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/search_disk_index}"
PIPEANN_BUILD_MEM_EXE="${PIPEANN_BUILD_MEM_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/build_memory_index}"
PIPEGOR_EXE="${PIPEGOR_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"

THREADS="${THREADS:-8}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
L_LIST="${L_LIST:-15 18 20 25 30 35}"

PIPEANN_INDEX_PREFIX="${PIPEANN_INDEX_PREFIX:-/home/dell/data/pg_ann_from_gorgeous/sift1m/sift1m}"
PIPEANN_DATA_FILE="${PIPEANN_DATA_FILE:-/home/dell/data/sift/sift_base.fbin}"
PIPEANN_TAGS_FILE="${PIPEANN_TAGS_FILE:-${PIPEANN_INDEX_PREFIX}_disk.index.tags}"
PIPEANN_MEM_INDEX="${PIPEANN_MEM_INDEX:-${PIPEANN_INDEX_PREFIX}_mem.index}"
PIPEANN_MEM_R="${PIPEANN_MEM_R:-64}"
PIPEANN_MEM_BUILD_L="${PIPEANN_MEM_BUILD_L:-128}"
PIPEANN_MEM_ALPHA="${PIPEANN_MEM_ALPHA:-1.2}"
PIPEANN_MEM_THREADS="${PIPEANN_MEM_THREADS:-16}"

PIPEGOR_INDEX_PREFIX="${PIPEGOR_INDEX_PREFIX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/}"
PIPEGOR_PQ_PREFIX="${PIPEGOR_PQ_PREFIX:-/home/dell/data/gorgeous/sift1M/PQ/C4/}"
PIPEGOR_DISK_FILE="${PIPEGOR_DISK_FILE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index}"
PIPEGOR_MEM_INDEX="${PIPEGOR_MEM_INDEX:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index}"
PIPEGOR_MEM_SAMPLE="${PIPEGOR_MEM_SAMPLE:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin}"
PIPEGOR_DISK_GRAPH="${PIPEGOR_DISK_GRAPH:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/}"
PIPEGOR_GRAPH_REP="${PIPEGOR_GRAPH_REP:-/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/}"
PIPEGOR_USE_GRAPH_REP_INDEX="${PIPEGOR_USE_GRAPH_REP_INDEX:-1}"
PIPEGOR_MEM_GRAPH_USE_RATIO="${PIPEGOR_MEM_GRAPH_USE_RATIO:-0.0}"
PIPEGOR_MEM_EMB_USE_RATIO="${PIPEGOR_MEM_EMB_USE_RATIO:-0.0}"
PIPEGOR_EMB_SEARCH_RATIO="${PIPEGOR_EMB_SEARCH_RATIO:-1.0}"
PIPEGOR_USE_RATIO="${PIPEGOR_USE_RATIO:-0.3}"
PIPEGOR_PQ_RATIO="${PIPEGOR_PQ_RATIO:-1.0}"

QUERY_FILE="${QUERY_FILE:-/home/dell/data/sift/sift_query.fbin}"
GT_FILE="${GT_FILE:-/home/dell/data/sift/computed_gt_1000_sift1m.bin}"
K="${K:-10}"
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/pipeann_vs_pipegor_pipeline_memindex}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}/pipegor"

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

cat > "${RUN_DIR}/PARAMS.txt" <<EOF
Dataset: SIFT1M
Query file: ${QUERY_FILE}
Ground truth: ${GT_FILE}

Comparison goal:
  PipeANN baseline vs PipeGor_ANN
  same threads / beam width / mem index depth
  dynamic cache disabled
  pipeline enabled for PipeGor_ANN
  PipeANN-aligned dynamic pipeline only

PipeANN baseline:
  exe = ${PIPEANN_EXE}
  index_prefix = ${PIPEANN_INDEX_PREFIX}
  mem_L = ${MEM_L}
  mem_index = ${PIPEANN_MEM_INDEX}

PipeGor_ANN:
  exe = ${PIPEGOR_EXE}
  index_prefix = ${PIPEGOR_INDEX_PREFIX}
  mem_L = ${MEM_L}
  dynamic graph cache ratio = 0
  PipeANN state machine = 1
  PipeANN scheduler = 1
  PipeANN dynamic width = 1
  PipeANN feedback = 1
  PipeANN start width = 4 (auto)
  PipeANN min width = 4 (auto)
  PipeANN resource aware = 0
  use_graph_rep_index = ${PIPEGOR_USE_GRAPH_REP_INDEX}
  mem_graph_use_ratio = ${PIPEGOR_MEM_GRAPH_USE_RATIO}
  mem_emb_use_ratio = ${PIPEGOR_MEM_EMB_USE_RATIO}
  emb_search_ratio = ${PIPEGOR_EMB_SEARCH_RATIO}

Common:
  threads = ${THREADS}
  width = ${WIDTH}
  K = ${K}
  L values = ${L_LIST}
EOF

ensure_pipeann_mem_index

PIPEANN_LOG="${RUN_DIR}/pipeann.log"
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
  GORGEOUS_PIPELINED_GRAPH_IO=0 \
  GORGEOUS_PIPELINED_REFINE_IO=0 \
  GORGEOUS_EARLY_REFINE_PREFETCH=0 \
  "${PIPEGOR_EXE}" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "${PIPEGOR_INDEX_PREFIX}" \
    --pq_path_prefix "${PIPEGOR_PQ_PREFIX}" \
    --result_path "${RUN_DIR}/pipegor/result" \
    --query_file "${QUERY_FILE}" \
    --gt_file "${GT_FILE}" \
    -K "${K}" \
    -L ${L_LIST} \
    -T "${THREADS}" \
    -W "${WIDTH}" \
    --mem_L "${MEM_L}" \
    --sector_len 4096 \
    --mem_index_path "${PIPEGOR_MEM_INDEX}" \
    --mem_sample_path "${PIPEGOR_MEM_SAMPLE}" \
    --use_page_search 1 \
    --use_ratio "${PIPEGOR_USE_RATIO}" \
    --pq_ratio "${PIPEGOR_PQ_RATIO}" \
    --disk_file_path "${PIPEGOR_DISK_FILE}" \
    --disk_graph_prefix "${PIPEGOR_DISK_GRAPH}" \
    --graph_rep_index_prefix "${PIPEGOR_GRAPH_REP}" \
    --deco_impl 1 \
    --use_graph_rep_index "${PIPEGOR_USE_GRAPH_REP_INDEX}" \
    --mem_graph_use_ratio "${PIPEGOR_MEM_GRAPH_USE_RATIO}" \
    --mem_emb_use_ratio "${PIPEGOR_MEM_EMB_USE_RATIO}" \
    --emb_search_ratio "${PIPEGOR_EMB_SEARCH_RATIO}" > "${PIPEGOR_LOG}" 2>&1

python3 - "${PIPEANN_LOG}" "${PIPEGOR_LOG}" "${RUN_DIR}/compare.csv" <<'PY'
import csv
import pathlib
import sys

pipeann_log = pathlib.Path(sys.argv[1])
pipegor_log = pathlib.Path(sys.argv[2])
compare_csv = pathlib.Path(sys.argv[3])


def parse_pipeann(path: pathlib.Path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            lval = parts[0]
            rows[lval] = {
                "QPS": float(parts[2]),
                "MeanLtc": float(parts[3]),
                "P99": float(parts[4]),
                "MeanIO": float(parts[6]),
                "Recall": float(parts[7]),
            }
    return rows


def parse_pipegor(path: pathlib.Path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 25 and parts[0].isdigit():
            lval = parts[0]
            rows[lval] = {
                "QPS": float(parts[2]),
                "MeanLtc": float(parts[3]),
                "P999": float(parts[4]),
                "GraphIO": float(parts[5]),
                "Recall": float(parts[17]),
                "PipeSub": float(parts[18]),
                "PipeUsePct": float(parts[19]),
                "PipeWst": float(parts[20]),
                "PipeStl": float(parts[21]),
                "PipeW": float(parts[22]),
                "PipeMax": float(parts[23]),
                "PipeAdj": float(parts[24]),
            }
    return rows


pipeann = parse_pipeann(pipeann_log)
pipegor = parse_pipegor(pipegor_log)

fields = [
    "L",
    "PipeANN_QPS",
    "PipeGor_QPS",
    "QPS_Gain_Pct",
    "PipeANN_MeanLtc",
    "PipeGor_MeanLtc",
    "MeanLtc_Improve_Pct",
    "PipeANN_P99Ltc",
    "PipeGor_P999Ltc",
    "PipeANN_Recall",
    "PipeGor_Recall",
    "Recall_Delta",
    "PipeANN_MeanIO",
    "PipeGor_GraphIO",
    "PipeGor_PipeUsePct",
    "PipeGor_PipeW",
    "PipeGor_PipeMax",
    "PipeGor_PipeAdj",
]

with compare_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for lval in sorted(set(pipeann) & set(pipegor), key=lambda x: int(x)):
        p = pipeann[lval]
        g = pipegor[lval]
        mean_improve = (1.0 - g["MeanLtc"] / p["MeanLtc"]) * 100.0 if p["MeanLtc"] else 0.0
        writer.writerow({
            "L": lval,
            "PipeANN_QPS": f"{p['QPS']:.2f}",
            "PipeGor_QPS": f"{g['QPS']:.2f}",
            "QPS_Gain_Pct": f"{(g['QPS'] / p['QPS'] - 1.0) * 100.0:.2f}",
            "PipeANN_MeanLtc": f"{p['MeanLtc']:.2f}",
            "PipeGor_MeanLtc": f"{g['MeanLtc']:.2f}",
            "MeanLtc_Improve_Pct": f"{mean_improve:.2f}",
            "PipeANN_P99Ltc": f"{p['P99']:.2f}",
            "PipeGor_P999Ltc": f"{g['P999']:.2f}",
            "PipeANN_Recall": f"{p['Recall']:.2f}",
            "PipeGor_Recall": f"{g['Recall']:.2f}",
            "Recall_Delta": f"{g['Recall'] - p['Recall']:.2f}",
            "PipeANN_MeanIO": f"{p['MeanIO']:.2f}",
            "PipeGor_GraphIO": f"{g['GraphIO']:.2f}",
            "PipeGor_PipeUsePct": f"{g['PipeUsePct']:.2f}",
            "PipeGor_PipeW": f"{g['PipeW']:.2f}",
            "PipeGor_PipeMax": f"{g['PipeMax']:.2f}",
            "PipeGor_PipeAdj": f"{g['PipeAdj']:.2f}",
        })

print(compare_csv.read_text(), end="")
PY

echo "RESULT_DIR=${RUN_DIR}"
