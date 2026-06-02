# PipeANN + Gorgeous Layout

这个仓库现在是一个更纯粹的 `PipeANN + Gorgeous layout` 工程：

- 建盘直接调用 vendored `PipeANN` 原生构建逻辑。
- 页面布局使用 `Gorgeous` 分区结果与 relayout。
- 搜索走 vendored `PipeANN` 的原生 `beam/page/pipe` 路径。
- 项目内 CLI 主要保留 `build / relayout / ground_truth / bench / merge / compare`。

`hybrid search`、`search_backend` 分发、`search` / `inspect` 入口以及 `graphrep` 兼容导出不再是当前默认工程的一部分。

## 依赖

- `CMake >= 3.16`
- `C++17`
- `BLAS`
- `OpenMP`
- 可选 `libaio`

仓库会直接编译以下 vendored 代码：

- `third_party/pipeann`
- `third_party/gorgeous`

## 构建

```bash
cd /home/adieu/OS/project
cmake -S . -B build
cmake --build build -j
```

当前默认保留的主要可执行文件：

- `build/pipeann_gorgeous_build`
- `build/pipeann_gorgeous_relayout`
- `build/pipeann_gorgeous_ground_truth`
- `build/pipeann_gorgeous_bench`
- `build/pipeann_gorgeous_bench_pipeann`
- `build/pipeann_gorgeous_bench_merge`
- `build/pipeann_gorgeous_bench_compare`

## 工作流

```text
vectors
  -> PipeANN disk build
  -> Gorgeous partition
  -> Gorgeous relayout
  -> PipeANN equal/gorgeous layout activation
  -> PipeANN beam/page/pipe search
```

其中最关键的产物是：

- `<prefix>.bin`
  - PipeANN 基础向量数据。
- `<prefix>_disk.index`
  - 当前激活中的 PipeANN 磁盘索引。
- `<prefix>_disk.index.equal`
  - equal layout 版本。
- `<prefix>_disk.index.gorgeous`
  - gorgeous layout 版本。
- `<prefix>_partition.bin`
  - Gorgeous GP 分区文件。
- `<prefix>_graph_relayout.index`
  - 独立输出的 Gorgeous relayout 索引。
- `<prefix>_pq_pivots.bin`
  - PipeANN PQ codebook。
- `<prefix>_pq_compressed.bin`
  - PipeANN PQ compressed codes。

`pipeann_layout_activate` 会在 bench 前自动把 `.equal` 或 `.gorgeous` 切换到 `<prefix>_disk.index`，所以通常不需要手动复制文件。

## 输入格式

`build`、`ground_truth` 和 `bench` 支持以下向量输入格式：

- `text`
- `fvecs`
- `bvecs`
- `bin`

`bin` 格式为：

```text
uint32_t num_points
uint32_t dim
float payload[num_points][dim]
```

## Build

查看帮助：

```bash
./build/pipeann_gorgeous_build --help
```

示例：

```bash
./build/pipeann_gorgeous_build \
  --mode fvecs \
  --input /home/adieu/OS/data/sift/sift/sift_base.fvecs \
  --train_query_mode fvecs \
  --train_query_path /home/adieu/OS/data/sift/sift/sift_learn.fvecs \
  --output_dir /home/adieu/OS/data/sift/out \
  --dataset_name sift1m \
  --degree 32 \
  --dense_degree 64 \
  --build_l 128 \
  --build_ram_budget_gb 16 \
  --build_threads 8 \
  --page_nodes 8 \
  --partition_scale 0 \
  --partition_ldg_times 4 \
  --pq_subspaces 16
```

常用参数：

- `--mode text|fvecs|bvecs|bin`
- `--input PATH`
- `--train_query_mode` / `--train_query_path`
- `--degree` / `--dense_degree`
- `--r_ood` / `--l_ood`
- `--build_l` / `--build_ram_budget_gb` / `--build_threads`
- `--page_nodes` / `--partition_scale` / `--partition_ldg_times`
- `--entry_id`
- `--pq_subspaces` / `--pq_centroids` / `--pq_iterations`

构建完成后会打印这些关键路径：

- `pipeann_index_prefix`
- `pipeann_disk_index`
- `pipeann_equal_layout`
- `pipeann_gorgeous_layout`
- `gorgeous_partition`
- `gorgeous_relayout`
- `pipeann_pq_pivots`
- `pipeann_pq_compressed`

## Relayout

`build` 已经会自动产出 gorgeous relayout 结果；`relayout` 入口主要用于把已有的 PipeANN equal-layout 索引配合现成 GP 分区文件重新生成 gorgeous layout。

查看帮助：

```bash
./build/pipeann_gorgeous_relayout --help
```

示例：

```bash
./build/pipeann_gorgeous_relayout \
  --disk_index /home/adieu/OS/data/sift/out/sift1m_disk.index.equal \
  --partition /home/adieu/OS/data/sift/out/sift1m_partition.bin \
  --output /home/adieu/OS/data/sift/out/sift1m_disk.index.gorgeous
```

## Ground Truth

查看帮助：

```bash
./build/pipeann_gorgeous_ground_truth --help
```

示例：

```bash
./build/pipeann_gorgeous_ground_truth \
  --index_prefix /home/adieu/OS/data/sift/out/sift1m \
  --queries /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --top_k 10 \
  --output /home/adieu/OS/data/sift/results/ground_truth.txt
```

这里的 `--index_prefix` 指向 workflow 前缀，例如 `.../sift1m`。默认 exact ground truth 会直接读取 `<prefix>.bin`；如果基础向量数据另有位置，可以通过 `--approx PATH` 覆盖。

## Bench

查看帮助：

```bash
./build/pipeann_gorgeous_bench --help
```

单次 bench：

```bash
./build/pipeann_gorgeous_bench \
  --index_prefix /home/adieu/OS/data/sift/out/sift1m \
  --pipeann_layout gorgeous \
  --gp_partition /home/adieu/OS/data/sift/out/sift1m_partition.bin \
  --pipeann_mode 2 \
  --queries /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --generate_ground_truth /home/adieu/OS/data/sift/results/ground_truth.txt \
  --ground_truth_k 10 \
  --top_k 10 \
  --beam_width 8 \
  --l_search 20 \
  --threads 1 \
  --mem_l 0 \
  --export /home/adieu/OS/data/sift/results/pipeann_summary.tsv
```

sweep 示例：

```bash
./build/pipeann_gorgeous_bench \
  --index_prefix /home/adieu/OS/data/sift/out/sift1m \
  --queries /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --ground_truth /home/adieu/OS/data/sift/results/ground_truth.txt \
  --top_k 10 \
  --beam_widths 4,8,16 \
  --l_search_values 16,20,32 \
  --thread_counts 1,8 \
  --pipeann_modes 0,1,2 \
  --pipeann_layouts equal,gorgeous \
  --export /home/adieu/OS/data/sift/results/pipeann_sweep.tsv
```

参数说明：

- `--index_prefix`
  - 指向 PipeANN workflow 前缀，实际使用 `<prefix>_disk.index*`、`<prefix>_partition.bin`、`<prefix>.bin`。
- `--pipeann_layout equal|gorgeous`
  - 指定 bench 前激活哪一种 layout。
- `--gp_partition PATH`
  - gorgeous 模式下显式指定分区文件；不传时默认推导为 `<prefix>_partition.bin`。
- `--pipeann_mode 0|1|2`
  - 分别对应 `beam / page / pipe`。
- `--mem_l`
  - 与 PipeANN `MEML0` 语义对齐。

导出的 TSV 列为：

```text
scale layout mode threads beam mem_l L qps avg_lat_us p99_lat_us mean_hops mean_ios unique_pg_io dup_pg_hit polls recall_at_k log
```

## Standalone Bench

`pipeann_gorgeous_bench_pipeann` 提供单次运行入口，参数风格更接近 PipeANN 原生命令，适合调试、性能对齐和 `strace`。

```bash
./build/pipeann_gorgeous_bench_pipeann \
  --index_prefix /home/adieu/OS/data/sift/out/sift1m \
  --layout gorgeous \
  --gp_partition /home/adieu/OS/data/sift/out/sift1m_partition.bin \
  --mode 2 \
  --queries /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --ground_truth /home/adieu/OS/data/sift/results/ground_truth.txt \
  --ground_truth_k 10 \
  --top_k 10 \
  --beamwidth 8 \
  --L 20 \
  --threads 1 \
  --mem_L 0
```

## Merge And Compare

合并多个 summary：

```bash
./build/pipeann_gorgeous_bench_merge \
  --inputs results/run_a.tsv,results/run_b.tsv \
  --output results/all.tsv
```

比较两组实验：

```bash
./build/pipeann_gorgeous_bench_compare \
  --baseline results/baseline.tsv \
  --candidate results/gorgeous.tsv \
  --output results/compare.md \
  --format markdown
```

也支持 `--format tsv`。

## 目录概览

- `include/`
- `src/`
- `tests/`
- `third_party/pipeann/`
- `third_party/gorgeous/`
- `docs/`

`tests/` 目录仍保留在工作区中，但不再接入当前默认构建。
