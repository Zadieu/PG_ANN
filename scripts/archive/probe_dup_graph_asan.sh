#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-/home/dell/data/gorgeous/sift1M/asan_dup_graph_probe_fix1}"
BIN="${BIN:-./build_debug/tests/search_disk_index}"
L_LIST="${L_LIST:-15}"
THREADS="${THREADS:-8}"
WIDTH="${WIDTH:-8}"
MEM_L="${MEM_L:-10}"
PIPEANN_STATE_MACHINE="${PIPEANN_STATE_MACHINE:-1}"
PIPEANN_STATE_SCHEDULER="${PIPEANN_STATE_SCHEDULER:-1}"
PIPEANN_DYNAMIC_PIPE_WIDTH="${PIPEANN_DYNAMIC_PIPE_WIDTH:-1}"
PIPEANN_PIPE_FEEDBACK="${PIPEANN_PIPE_FEEDBACK:-1}"
PIPEANN_SCHEDULER_WINDOW="${PIPEANN_SCHEDULER_WINDOW:-auto}"
PIPEANN_PIPE_START="${PIPEANN_PIPE_START:-auto}"
PIPEANN_PIPE_MIN="${PIPEANN_PIPE_MIN:-auto}"
PIPELINED_GRAPH_IO="${PIPELINED_GRAPH_IO:-0}"
PIPEANN_RECALL_SAFE_WINDOW="${PIPEANN_RECALL_SAFE_WINDOW:-1}"
PIPEANN_STALE_OUTSIDE_WINDOW="${PIPEANN_STALE_OUTSIDE_WINDOW:-0}"
EMB_SEARCH_RATIO="${EMB_SEARCH_RATIO:-1.0}"

mkdir -p "${OUTDIR}"
cd /home/dell/projects/PipeGor_ANN

export ASAN_OPTIONS="${ASAN_OPTIONS:-abort_on_error=1:detect_leaks=0:halt_on_error=1}"
export GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0
export GORGEOUS_PIPEANN_STATE_MACHINE="${PIPEANN_STATE_MACHINE}"
export GORGEOUS_PIPEANN_STATE_SCHEDULER="${PIPEANN_STATE_SCHEDULER}"
export GORGEOUS_PIPEANN_SCHEDULER_WINDOW="${PIPEANN_SCHEDULER_WINDOW}"
export GORGEOUS_PIPEANN_RECALL_SAFE_WINDOW="${PIPEANN_RECALL_SAFE_WINDOW}"
export GORGEOUS_PIPEANN_STALE_OUTSIDE_WINDOW="${PIPEANN_STALE_OUTSIDE_WINDOW}"
export GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH="${PIPEANN_DYNAMIC_PIPE_WIDTH}"
export GORGEOUS_PIPEANN_PIPE_START="${PIPEANN_PIPE_START}"
export GORGEOUS_PIPEANN_PIPE_MIN="${PIPEANN_PIPE_MIN}"
export GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10
export GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5
export GORGEOUS_PIPEANN_PIPE_FEEDBACK="${PIPEANN_PIPE_FEEDBACK}"
export GORGEOUS_PIPEANN_RESOURCE_AWARE=0
export GORGEOUS_PIPELINED_GRAPH_IO="${PIPELINED_GRAPH_IO}"
export GORGEOUS_PIPELINED_REFINE_IO=0
export GORGEOUS_EARLY_REFINE_PREFETCH=0

"${BIN}" \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/ \
  --pq_path_prefix /home/dell/data/gorgeous/sift1M/PQ/C4/ \
  --result_path "${OUTDIR}/result" \
  --query_file /home/dell/data/sift/sift_query.fbin \
  --gt_file /home/dell/data/sift/computed_gt_1000_sift1m.bin \
  -K 10 \
  -L ${L_LIST} \
  -T "${THREADS}" \
  -W "${WIDTH}" \
  --mem_L "${MEM_L}" \
  --sector_len 4096 \
  --mem_index_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index \
  --mem_sample_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin \
  --use_page_search 1 \
  --use_ratio 0.3 \
  --pq_ratio 1.0 \
  --disk_file_path /home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index \
  --disk_graph_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/ \
  --graph_rep_index_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/ \
  --deco_impl 1 \
  --use_graph_rep_index 1 \
  --mem_graph_use_ratio 0.0 \
  --mem_emb_use_ratio 0.0 \
  --emb_search_ratio "${EMB_SEARCH_RATIO}" \
  > "${OUTDIR}/pipegor.log" 2>&1
