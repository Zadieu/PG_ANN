#!/usr/bin/env bash
set -euo pipefail

PIPEANN_EXE="${PIPEANN_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/search_disk_index}"
PIPEANN_BUILD_MEM_EXE="${PIPEANN_BUILD_MEM_EXE:-/home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/build_memory_index}"
GORGEOUS_EXE="${GORGEOUS_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"

THREADS="${THREADS:-8}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
L_LIST="${L_LIST:-18 20 25 30 35 40}"

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
GORGEOUS_MEM_GRAPH_USE_RATIO="${GORGEOUS_MEM_GRAPH_USE_RATIO:-0.0}"
GORGEOUS_MEM_EMB_USE_RATIO="${GORGEOUS_MEM_EMB_USE_RATIO:-0.0}"
GORGEOUS_EMB_SEARCH_RATIO="${GORGEOUS_EMB_SEARCH_RATIO:-0.4}"
GORGEOUS_USE_RATIO="${GORGEOUS_USE_RATIO:-0.3}"
GORGEOUS_PQ_RATIO="${GORGEOUS_PQ_RATIO:-1.0}"

QUERY_FILE="${QUERY_FILE:-/home/dell/data/sift/sift_query.fbin}"
GT_FILE="${GT_FILE:-/home/dell/data/sift/computed_gt_1000_sift1m.bin}"
K="${K:-10}"
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/pipeann_vs_gorgeous_meml_pipeline}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}/gorgeous"

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

PipeANN baseline:
  exe = ${PIPEANN_EXE}
  index_prefix = ${PIPEANN_INDEX_PREFIX}
  mem_L = ${MEM_L}
  mem_index = ${PIPEANN_MEM_INDEX}

Current Gorgeous:
  exe = ${GORGEOUS_EXE}
  index_prefix = ${GORGEOUS_INDEX_PREFIX}
  mem_L = ${MEM_L}
  dynamic graph cache ratio = 0
  PipeANN state machine = 1
  PipeANN scheduler = 1
  PipeANN dynamic width = 1
  PipeANN feedback = 1
  PipeANN resource aware = 0
  mem_graph_use_ratio = ${GORGEOUS_MEM_GRAPH_USE_RATIO}
  mem_emb_use_ratio = ${GORGEOUS_MEM_EMB_USE_RATIO}
  emb_search_ratio = ${GORGEOUS_EMB_SEARCH_RATIO}

Common:
  threads = ${THREADS}
  width = ${WIDTH}
  K = ${K}
  L values = ${L_LIST}
EOF

ensure_pipeann_mem_index

PIPEANN_LOG="${RUN_DIR}/pipeann.log"
GORGEOUS_LOG="${RUN_DIR}/gorgeous.log"

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

echo "[RUN] Gorgeous current"
env \
  GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
  GORGEOUS_PIPEANN_STATE_MACHINE=1 \
  GORGEOUS_PIPEANN_STATE_SCHEDULER=1 \
  GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
  GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=1 \
  GORGEOUS_PIPEANN_PIPE_START=auto \
  GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10 \
  GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5 \
  GORGEOUS_PIPEANN_PIPE_FEEDBACK=1 \
  GORGEOUS_PIPEANN_RESOURCE_AWARE=0 \
  GORGEOUS_PIPELINED_GRAPH_IO=0 \
  GORGEOUS_PIPELINED_REFINE_IO=0 \
  GORGEOUS_EARLY_REFINE_PREFETCH=0 \
  "${GORGEOUS_EXE}" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "${GORGEOUS_INDEX_PREFIX}" \
    --pq_path_prefix "${GORGEOUS_PQ_PREFIX}" \
    --result_path "${RUN_DIR}/gorgeous/result" \
    --query_file "${QUERY_FILE}" \
    --gt_file "${GT_FILE}" \
    -K "${K}" \
    -L ${L_LIST} \
    -T "${THREADS}" \
    -W "${WIDTH}" \
    --mem_L "${MEM_L}" \
    --sector_len 4096 \
    --mem_index_path "${GORGEOUS_MEM_INDEX}" \
    --mem_sample_path "${GORGEOUS_MEM_SAMPLE}" \
    --use_page_search 1 \
    --use_ratio "${GORGEOUS_USE_RATIO}" \
    --pq_ratio "${GORGEOUS_PQ_RATIO}" \
    --disk_file_path "${GORGEOUS_DISK_FILE}" \
    --disk_graph_prefix "${GORGEOUS_DISK_GRAPH}" \
    --graph_rep_index_prefix "${GORGEOUS_GRAPH_REP}" \
    --deco_impl 1 \
    --use_graph_rep_index 0 \
    --mem_graph_use_ratio "${GORGEOUS_MEM_GRAPH_USE_RATIO}" \
    --mem_emb_use_ratio "${GORGEOUS_MEM_EMB_USE_RATIO}" \
    --emb_search_ratio "${GORGEOUS_EMB_SEARCH_RATIO}" > "${GORGEOUS_LOG}" 2>&1

python3 - "${PIPEANN_LOG}" "${GORGEOUS_LOG}" "${RUN_DIR}/compare.csv" <<'PY'
import csv
import pathlib
import re
import sys

pipeann_log = pathlib.Path(sys.argv[1])
gorgeous_log = pathlib.Path(sys.argv[2])
compare_csv = pathlib.Path(sys.argv[3])

def parse_pipeann(path: pathlib.Path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit():
            lval = parts[0]
            rows[lval] = {
                "QPS": float(parts[2]),
                "MeanLtc": float(parts[3]),
                "P99": float(parts[4]),
                "MeanIO": float(parts[6]),
                "Recall": float(parts[7]) if len(parts) >= 8 else 0.0,
            }
    return rows

def parse_gorgeous(path: pathlib.Path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 24 and parts[0].isdigit():
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
                "PipeAdj": float(parts[24]) if len(parts) >= 25 else 0.0,
            }
    return rows

pipeann = parse_pipeann(pipeann_log)
gorgeous = parse_gorgeous(gorgeous_log)

fields = [
    "L",
    "PipeANN_QPS",
    "Gorgeous_QPS",
    "GorgeousGain_vs_PipeANN_pct",
    "PipeANN_Recall",
    "Gorgeous_Recall",
    "Recall_Delta",
    "PipeANN_MeanIO",
    "Gorgeous_GraphIO",
    "Gorgeous_PipeUsePct",
    "Gorgeous_PipeW",
    "Gorgeous_PipeMax",
    "Gorgeous_PipeAdj",
]

with compare_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for lval in sorted(set(pipeann) & set(gorgeous), key=lambda x: int(x)):
      p = pipeann[lval]
      g = gorgeous[lval]
      writer.writerow({
          "L": lval,
          "PipeANN_QPS": f"{p['QPS']:.2f}",
          "Gorgeous_QPS": f"{g['QPS']:.2f}",
          "GorgeousGain_vs_PipeANN_pct": f"{(g['QPS'] / p['QPS'] - 1.0) * 100.0:.2f}",
          "PipeANN_Recall": f"{p['Recall']:.2f}",
          "Gorgeous_Recall": f"{g['Recall']:.2f}",
          "Recall_Delta": f"{g['Recall'] - p['Recall']:.2f}",
          "PipeANN_MeanIO": f"{p['MeanIO']:.2f}",
          "Gorgeous_GraphIO": f"{g['GraphIO']:.2f}",
          "Gorgeous_PipeUsePct": f"{g['PipeUsePct']:.2f}",
          "Gorgeous_PipeW": f"{g['PipeW']:.2f}",
          "Gorgeous_PipeMax": f"{g['PipeMax']:.2f}",
          "Gorgeous_PipeAdj": f"{g['PipeAdj']:.2f}",
      })

print(compare_csv.read_text(), end="")
PY

echo "RESULT_DIR=${RUN_DIR}"
