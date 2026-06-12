#!/usr/bin/env bash
set -euo pipefail

GORGEOUS_EXE="${GORGEOUS_EXE:-/home/dell/projects/Gorgeous-baseline/build/tests/search_disk_index}"
PIPEGOR_EXE="${PIPEGOR_EXE:-/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index}"

THREADS_LIST="${THREADS_LIST:-8 16 32 64}"
L_LIST="${L_LIST:-15 20 25 30 35 40}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
K="${K:-10}"

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
RESULT_ROOT="${RESULT_ROOT:-/home/dell/data/gorgeous/sift1M/graph_rep_compare}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}"

COMMON_ARGS=(
  --data_type float
  --dist_fn l2
  --index_path_prefix "${GORGEOUS_INDEX_PREFIX}"
  --pq_path_prefix "${GORGEOUS_PQ_PREFIX}"
  --query_file "${QUERY_FILE}"
  --gt_file "${GT_FILE}"
  -K "${K}"
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
  --use_graph_rep_index 1
  --mem_graph_use_ratio 0.1
  --mem_emb_use_ratio 0.0
  --emb_search_ratio 0.4
)

cat > "${RUN_DIR}/PARAMS.txt" <<EOF
Dataset: SIFT1M
threads_list = ${THREADS_LIST}
L_list = ${L_LIST}
width = ${WIDTH}
mem_L = ${MEM_L}
K = ${K}
use_graph_rep_index = 1
disk_file = ${GORGEOUS_DISK_FILE}
graph_rep_prefix = ${GORGEOUS_GRAPH_REP}

Gorgeous exe = ${GORGEOUS_EXE}
PipeGorANN exe = ${PIPEGOR_EXE}
EOF

for threads in ${THREADS_LIST}; do
  case_dir="${RUN_DIR}/T${threads}"
  mkdir -p "${case_dir}/gorgeous" "${case_dir}/pipegor"

  echo "[RUN] Gorgeous graph-rep T=${threads}"
  env \
    GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
    GORGEOUS_PIPEANN_STATE_MACHINE=0 \
    GORGEOUS_PIPEANN_STATE_SCHEDULER=0 \
    GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=0 \
    GORGEOUS_PIPELINED_GRAPH_IO=0 \
    GORGEOUS_PIPELINED_REFINE_IO=0 \
    GORGEOUS_EARLY_REFINE_PREFETCH=0 \
    "${GORGEOUS_EXE}" \
      "${COMMON_ARGS[@]}" \
      -T "${threads}" \
      -L ${L_LIST} \
      --result_path "${case_dir}/gorgeous/result" \
      > "${case_dir}/gorgeous.log" 2>&1

  echo "[RUN] PipeGorANN graph-rep T=${threads}"
  env \
    GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
    GORGEOUS_PIPEANN_STATE_MACHINE=0 \
    GORGEOUS_PIPEANN_STATE_SCHEDULER=0 \
    GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
    GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=0 \
    GORGEOUS_PIPEANN_PIPE_START=auto \
    GORGEOUS_PIPEANN_PIPE_MIN=1 \
    GORGEOUS_PIPEANN_L_AWARE_PIPE_START=1 \
    GORGEOUS_PIPEANN_L_AWARE_LOW_L=12 \
    GORGEOUS_PIPEANN_L_AWARE_HIGH_L=18 \
    GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10 \
    GORGEOUS_PIPEANN_PIPE_DOWN_WASTE_THRESHOLD=0.35 \
    GORGEOUS_PIPEANN_PIPE_FEEDBACK_WINDOW=4 \
    GORGEOUS_PIPEANN_PIPE_MIN_MARKER=2 \
    GORGEOUS_PIPEANN_PIPE_FEEDBACK=1 \
    GORGEOUS_PIPEANN_RESOURCE_ADAPTIVE=0 \
    GORGEOUS_PIPELINED_GRAPH_IO=1 \
    GORGEOUS_PIPELINED_GRAPH_DYNAMIC_WIDTH=1 \
    GORGEOUS_PIPELINED_GRAPH_PIPE_MAX=auto \
    GORGEOUS_PIPELINED_GRAPH_QD_BUDGET=256 \
    GORGEOUS_PIPELINED_GRAPH_RAMP_STEP=auto \
    "${PIPEGOR_EXE}" \
      "${COMMON_ARGS[@]}" \
      -T "${threads}" \
      -L ${L_LIST} \
      --result_path "${case_dir}/pipegor/result" \
      > "${case_dir}/pipegor.log" 2>&1
done

python3 - "${RUN_DIR}" <<'PY'
import csv
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])

def parse_gorgeous_like(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 18 and parts[0].isdigit():
            rows[int(parts[0])] = {
                "qps": float(parts[2]),
                "mean_ltc": float(parts[3]),
                "p99": float(parts[4]),
                "graph_io": float(parts[5]),
                "recall": float(parts[17]),
                "pipe_w": float(parts[22]) if len(parts) > 22 else 0.0,
                "pipe_max": float(parts[23]) if len(parts) > 23 else 0.0,
                "pipe_adj": float(parts[24]) if len(parts) > 24 else 0.0,
                "pipe_skip": float(parts[25]) if len(parts) > 25 else 0.0,
            }
    return rows

rows = []
for case_dir in sorted(run_dir.glob("T*"), key=lambda p: int(p.name[1:])):
    threads = int(case_dir.name[1:])
    gorgeous = parse_gorgeous_like(case_dir / "gorgeous.log")
    pipegor = parse_gorgeous_like(case_dir / "pipegor.log")
    for l_value in sorted(set(gorgeous) | set(pipegor)):
        g = gorgeous.get(l_value, {})
        p = pipegor.get(l_value, {})
        g_qps = g.get("qps", 0.0)
        p_qps = p.get("qps", 0.0)
        gain = ((p_qps / g_qps) - 1.0) * 100.0 if g_qps else 0.0
        rows.append({
            "threads": threads,
            "L": l_value,
            "gorgeous_qps": g.get("qps", ""),
            "pipegor_qps": p.get("qps", ""),
            "qps_gain_pct": gain,
            "gorgeous_mean_us": g.get("mean_ltc", ""),
            "pipegor_mean_us": p.get("mean_ltc", ""),
            "gorgeous_p99_us": g.get("p99", ""),
            "pipegor_p99_us": p.get("p99", ""),
            "gorgeous_recall": g.get("recall", ""),
            "pipegor_recall": p.get("recall", ""),
            "gorgeous_graph_io": g.get("graph_io", ""),
            "pipegor_graph_io": p.get("graph_io", ""),
            "pipegor_pipe_w": p.get("pipe_w", ""),
            "pipegor_pipe_max": p.get("pipe_max", ""),
            "pipegor_pipe_adj": p.get("pipe_adj", ""),
            "pipegor_pipe_skip": p.get("pipe_skip", ""),
        })

out = run_dir / "graph_rep_compare.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
        "threads", "L", "gorgeous_qps", "pipegor_qps", "qps_gain_pct",
        "gorgeous_mean_us", "pipegor_mean_us", "gorgeous_p99_us", "pipegor_p99_us",
        "gorgeous_recall", "pipegor_recall", "gorgeous_graph_io", "pipegor_graph_io",
        "pipegor_pipe_w", "pipegor_pipe_max", "pipegor_pipe_adj", "pipegor_pipe_skip",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(out)
for row in rows:
    print(
        f"T={row['threads']:>2} L={row['L']:>2} "
        f"G={row['gorgeous_qps']} P={row['pipegor_qps']} "
        f"gain={row['qps_gain_pct']:.2f}% "
        f"lat(us) {row['gorgeous_mean_us']}->{row['pipegor_mean_us']} "
        f"recall {row['gorgeous_recall']}->{row['pipegor_recall']}"
    )
PY

echo "[DONE] ${RUN_DIR}"
