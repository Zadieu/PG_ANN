# PipeGor_ANN

PipeGor_ANN is a disk-resident approximate nearest neighbor search project built
on top of Gorgeous and extended with a PipeANN-style graph I/O scheduler.

The current mainline is intentionally conservative:

```text
Gorgeous graph-replicated layout
+ PipeANN-style state scheduler
+ dynamic pipe width
+ L-aware pipe-width start
```

The project is used to study how runtime pipelining can work with Gorgeous's
graph-replicated disk layout without introducing excessive speculative I/O.

## Motivation

Gorgeous already performs a form of structural prefetching through its
graph-replicated layout. A disk page for node `u` contains:

```text
u exact vector
u adjacency list
selected neighbors' adjacency lists
```

This is different from PipeANN's original setting, where runtime prefetching is
mainly used to overlap future graph accesses. If PipeANN-style prefetching is
applied to Gorgeous naively, each wrong prediction may fetch a heavier
graph-replicated page and may duplicate adjacency information that has already
been brought back by Gorgeous's layout.

PipeGor_ANN therefore keeps Gorgeous's graph-replicated layout as the data
layout foundation and uses a more restrained scheduler:

- issue graph I/O only for candidates that are actually inside the current
  scheduling window;
- use dynamic pipe width to avoid over-prefetching at small `L`;
- use L-aware initial pipe width so low-recall and high-recall searches can use
  different degrees of overlap;
- keep refinement on the original batch exact-vector read path.

## Current Mainline

The current tested PipeGor configuration is:

```bash
GORGEOUS_PIPEANN_STATE_SCHEDULER=1
GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH=1
GORGEOUS_PIPEANN_L_AWARE_PIPE_START=1
GORGEOUS_PIPEANN_L_AWARE_LOW_L=12
GORGEOUS_PIPEANN_L_AWARE_HIGH_L=18
GORGEOUS_PIPEANN_PIPE_FEEDBACK=1
GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD=0.10
GORGEOUS_PIPEANN_PIPE_MIN_MARKER=5
```

The graph-replicated search path is implemented mainly in:

```text
src/gorgeous/index_search_dup_graph.cpp
```

Older exploratory ideas such as early refinement prefetch, refinement pipeline,
grouped refinement, rank-ready expansion, and resource-aware pipe shrinking have
been removed from the graph-replicated mainline to keep the implementation
focused and reproducible.

## Repository Layout

```text
include/                 Public headers and search utilities
src/gorgeous/            Gorgeous/PipeGor search implementation
tests/                   Build/search utilities and executable entry points
scripts/                 Main benchmark and comparison scripts
scripts/archive/         Old exploratory scripts kept for reference
graph_partition/         Graph partitioning and oneTBB-related components
assets/                  Existing figures used by the original project README
```

The most important executable for search experiments is:

```text
build/tests/search_disk_index
```

## Dependencies

Install the common system dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  g++ \
  libaio-dev \
  libboost-all-dev \
  libgoogle-perftools-dev \
  libmkl-full-dev
```

The project also depends on oneTBB. In our local setup, oneTBB is already placed
under the project tree through the existing Gorgeous build structure.

## Build

From the project root:

```bash
cd /home/dell/projects/PipeGor_ANN
cmake --build build -j2 --target search_disk_index
```

If the build directory does not exist yet, configure it first:

```bash
cd /home/dell/projects/PipeGor_ANN
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target search_disk_index
```

## Run The Three-Way Comparison

The main comparison script is:

```text
scripts/compare_three_way_sift1m_t8.sh
```

It compares:

```text
PipeANN baseline
Gorgeous original baseline
PipeGor_ANN
```

Default baseline executable paths:

```text
PipeANN:
  /home/dell/projects/PipeANN-official-20260604/PipeANN-main/build/tests/search_disk_index

Gorgeous original:
  /home/dell/projects/Gorgeous-original-baseline/build/tests/search_disk_index

PipeGor_ANN:
  /home/dell/projects/PipeGor_ANN/build/tests/search_disk_index
```

Example SIFT1M run:

```bash
cd /home/dell/projects/PipeGor_ANN

env \
  RESULT_ROOT=/home/dell/data/gorgeous/sift1M/three_way_t8 \
  L_LIST='18 20 24 28 32 40 50 64 80 96 128 160' \
  MEM_L=10 \
  THREADS=8 \
  WIDTH=8 \
  bash scripts/compare_three_way_sift1m_t8.sh
```

Each run creates a timestamped directory:

```text
${RESULT_ROOT}/run_YYYYmmdd_HHMMSS/
```

Important output files:

```text
PARAMS.txt
pipeann.log
gorgeous_original.log
pipegor.log
compare.csv
```

## Key Script Parameters

```text
L_LIST
  Search list depths to test.

THREADS
  Number of search threads.

WIDTH
  Beam width / I/O width.

MEM_L
  Memory index search depth.

PIPEGOR_L_AWARE_LOW_L
  L value at which PipeGor starts from a smaller pipe width.
  Default: 12

PIPEGOR_L_AWARE_HIGH_L
  L value at which PipeGor starts from full beam width.
  Default: 18
```

Dataset and index paths can also be overridden through environment variables in
`scripts/compare_three_way_sift1m_t8.sh`.

## Latest Local SIFT1M Check

After cleanup, we reran PipeGor_ANN against both baselines on SIFT1M with:

```text
T = 8
W = 8
MEM_L = 10
L = 18 20 24 28 32 40 50 64 80 96 128 160
```

Representative high-recall results:

```text
L=64   PipeGor Recall@10 99.43  QPS 2987.69
       vs PipeANN +34.89%, vs Gorgeous +1.99%

L=96   PipeGor Recall@10 99.81  QPS 2199.52
       vs PipeANN +42.47%, vs Gorgeous +2.19%

L=128  PipeGor Recall@10 99.91  QPS 1773.50
       vs PipeANN +55.19%, vs Gorgeous +5.61%

L=160  PipeGor Recall@10 99.96  QPS 1436.05
       vs PipeANN +52.52%, vs Gorgeous +4.86%
```

The latest local chart artifacts were generated under the Codex work directory:

```text
work/final_three_way_cleanup_t8_summary.csv
work/final_three_way_cleanup_t8_recall.svg
work/final_three_way_cleanup_t8_qps.svg
work/final_three_way_cleanup_t8_latency.svg
```

## Notes For Development

- Keep the graph-replicated mainline simple. New experimental ideas should start
  behind a small, clearly named branch or script, then be deleted or archived if
  they do not help.
- Re-run both baselines whenever comparing performance. The current comparison
  script intentionally reruns PipeANN and Gorgeous in the same test round.
- Prefer recall-matched QPS comparisons over only same-`L` comparisons when
  presenting final results.
- Use `scripts/archive/` only as historical reference; main experiments should
  use `scripts/compare_three_way_sift1m_t8.sh`.

## Citation

PipeGor_ANN is an experimental project built on Gorgeous. For the original data
layout idea, cite Gorgeous:

```bibtex
@article{yin2025gorgeous,
  title={Gorgeous: Revisiting the Data Layout for Disk-Resident High-Dimensional Vector Search},
  author={Yin, Peiqi and Yan, Xiao and Zhou, Qihui and Li, Hui and Li, Xiaolu and Zhang, Lin and Wang, Meiling and Yao, Xin and Cheng, James},
  journal={arXiv preprint arXiv:2508.15290},
  year={2025}
}
```

## License

This repository follows the license files included in the project.
