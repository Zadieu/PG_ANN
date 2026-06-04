#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR=/home/dell/projects/Gorgeous-original-baseline
CURRENT_DIR=/home/dell/projects/Gorgeous-main
RESULT_ROOT=/home/dell/projects/gorgeous_demo_results
OUT="$RESULT_ROOT/original_vs_current_sift1m_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT"
DETAIL="$OUT/detail.csv"
AVG="$OUT/avg.csv"
COMPARE="$OUT/compare.csv"

echo "round,version,L,BW,QPS,MeanLtc,P999Ltc,GraphIO,EmbIO,ExtCmp,PQCmp,Recall10" > "$DETAIL"

COMMON_ARGS=(
  --data_type float
  --dist_fn l2
  --index_path_prefix /home/dell/data/gorgeous/sift1M/M4_R64_L128/
  --pq_path_prefix /home/dell/data/gorgeous/sift1M/PQ/C4/
  --query_file /home/dell/data/sift/sift_query.fbin
  --gt_file /home/dell/data/sift/computed_gt_1000_sift1m.bin
  -K 10
  -T 8
  -W 8
  --mem_L 0
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
)

extract_row() {
  local round="$1"
  local version="$2"
  local L="$3"
  local log="$4"
  awk -v round="$round" -v version="$version" -v lval="$L" '
    $1 == lval && $2 == 8 {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        round, version, $1, $2, $3, $4, $5, $6, $7, $8, $9, $18
    }' "$log" >> "$DETAIL"
}

run_original() {
  local round="$1"
  local L="$2"
  local result_dir="$OUT/original/r${round}_L${L}"
  local log="$result_dir.log"
  mkdir -p "$result_dir"
  echo "RUN original round=$round L=$L"
  (
    cd "$BASELINE_DIR"
    env \
      GORGEOUS_PIPEANN_STATE_MACHINE=0 \
      GORGEOUS_PIPEANN_STATE_SCHEDULER=0 \
      GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=0 \
      GORGEOUS_PIPELINED_GRAPH_IO=0 \
      GORGEOUS_PIPELINED_REFINE_IO=0 \
      GORGEOUS_EARLY_REFINE_PREFETCH=0 \
      GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0 \
      ./build/tests/search_disk_index "${COMMON_ARGS[@]}" -L "$L" --result_path "$result_dir/result"
  ) > "$log" 2>&1
  extract_row "$round" original "$L" "$log"
}

run_current() {
  local round="$1"
  local L="$2"
  local result_dir="$OUT/current/r${round}_L${L}"
  local log="$result_dir.log"
  mkdir -p "$result_dir"
  echo "RUN current round=$round L=$L"
  (
    cd "$CURRENT_DIR"
    env \
      GORGEOUS_PIPEANN_STATE_MACHINE=1 \
      GORGEOUS_PIPEANN_STATE_SCHEDULER=1 \
      GORGEOUS_PIPEANN_SCHEDULER_WINDOW=auto \
      GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=1 \
      GORGEOUS_PIPEANN_PIPE_START=auto \
      GORGEOUS_PIPEANN_PIPE_CONVERGE_RANK=5 \
      GORGEOUS_PIPELINED_GRAPH_IO=0 \
      GORGEOUS_PIPELINED_REFINE_IO=0 \
      GORGEOUS_EARLY_REFINE_PREFETCH=0 \
      GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto \
      ./build/tests/search_disk_index "${COMMON_ARGS[@]}" -L "$L" --result_path "$result_dir/result"
  ) > "$log" 2>&1
  extract_row "$round" current "$L" "$log"
}

for round in 1 2 3; do
  for L in 18 20 25 30 35 40; do
    run_original "$round" "$L"
    run_current "$round" "$L"
  done
done

awk -F, '
  NR == 1 { next }
  {
    key = $2 "," $3
    cnt[key]++
    qps[key] += $5
    mean[key] += $6
    p999[key] += $7
    gio[key] += $8
    eio[key] += $9
    ext[key] += $10
    pq[key] += $11
    rec[key] += $12
  }
  END {
    print "version,L,n,QPS,MeanLtc,P999Ltc,GraphIO,EmbIO,ExtCmp,PQCmp,Recall10"
    for (key in cnt) {
      split(key, parts, ",")
      printf "%s,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
        parts[1], parts[2], cnt[key], qps[key]/cnt[key], mean[key]/cnt[key],
        p999[key]/cnt[key], gio[key]/cnt[key], eio[key]/cnt[key],
        ext[key]/cnt[key], pq[key]/cnt[key], rec[key]/cnt[key]
    }
  }' "$DETAIL" | sort -t, -k2,2n -k1,1 > "$AVG"

awk -F, '
  NR == 1 { next }
  $1 == "original" {
    oq[$2] = $4; om[$2] = $5; op[$2] = $6; og[$2] = $7; orc[$2] = $11
  }
  $1 == "current" {
    cq[$2] = $4; cm[$2] = $5; cp[$2] = $6; cg[$2] = $7; crc[$2] = $11
  }
  END {
    print "L,OriginalQPS,CurrentQPS,QPSGainPct,OriginalRecall,CurrentRecall,RecallDelta,OriginalGraphIO,CurrentGraphIO,GraphIODelta"
    for (L = 18; L <= 40; L += (L == 18 ? 2 : 5)) {
      if (oq[L] == 0) {
        continue
      }
      printf "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
        L, oq[L], cq[L], (cq[L] - oq[L]) * 100 / oq[L],
        orc[L], crc[L], crc[L] - orc[L], og[L], cg[L], cg[L] - og[L]
    }
  }' "$AVG" > "$COMPARE"

echo "RESULT_DIR=$OUT"
echo "DETAIL=$DETAIL"
echo "AVG=$AVG"
echo "COMPARE=$COMPARE"
cat "$COMPARE"
