# PipeANN x Gorgeous Hybrid

这个项目现在不是“只做概念验证”的最小拼接仓库，而是一个已经具备完整构建、检索、检视、benchmark 与实验导出链路的 C++ 工程。

当前实现的核心方向是：

- 以 vendored `PipeANN` 原始建盘逻辑为骨架生成磁盘索引。
- 以 vendored `Gorgeous` 分区与 relayout 逻辑生成原生图复制布局。
- 以项目内统一 CLI 暴露 `build / search / inspect / ground_truth / bench / compare` 工具链。
- 以 `Gorgeous native relayout` 作为主输出格式，并保留 `graphrep` 兼容导出。

## 当前状态

目前仓库已经实现并验证了这些能力：

- 支持从 `toy`、`text`、`fvecs`、`bvecs`、`bin` 输入建盘。
- 默认工作流已经收拢到 `vectors -> PipeANN disk build -> Gorgeous partition -> Gorgeous relayout -> PQ artifacts`。
- 原生索引加载以 `Gorgeous` 原始产物为主，`_partition.bin` / `_reorder.bin` 仅作为辅助 sidecar，不再是核心语义来源。
- native full-precision payload 采用按需读取，不会在 `Load()` 时整包 materialize 到内存。
- 搜索链路支持 `PQ` 近似距离与全精度 fallback。
- 查询、ground truth、benchmark 都支持 `text`、`fvecs`、`bvecs`、`bin` 输入。
- benchmark 支持 sweep、实验目录导出、结果汇总和两组实验对比。

如果你想看当前构建出的目标和测试目标，直接看 `CMakeLists.txt` 即可。

## 依赖

构建依赖：

- CMake >= 3.16
- C++17 编译器
- BLAS
- OpenMP
- 可选 `libaio`

项目当前会 vendored 编译：

- `third_party/pipeann`
- `third_party/gorgeous`

如果系统没有 `libaio`，页读取后端会回退到 `std::async + pread` 兼容路径。

## 构建

```bash
cd /home/adieu/OS/project
cmake -S . -B build
cmake --build build -j
```

构建后会得到这些主要工具：

- `build/pipeann_gorgeous_build`
- `build/pipeann_gorgeous_search`
- `build/pipeann_gorgeous_inspect`
- `build/pipeann_gorgeous_ground_truth`
- `build/pipeann_gorgeous_bench`
- `build/pipeann_gorgeous_bench_merge`
- `build/pipeann_gorgeous_bench_compare`
- `build/pipeann_gorgeous_hybrid`

## 快速验证

可以先跑内置 toy 链路，确认构建、索引加载和搜索都通：

```bash
cd /home/adieu/OS/project
./build/pipeann_gorgeous_hybrid
```

默认会在 `sample_data/` 里生成 toy 数据的索引与 PQ 产物，并打印 `PQ` 路径和全精度路径的搜索结果。

## 数据与索引格式

### 输入格式

支持五种输入模式：

- `--mode toy`
- `--mode text --input /path/to/vectors.txt`
- `--mode fvecs --input /path/to/base.fvecs`
- `--mode bvecs --input /path/to/base.bvecs`
- `--mode bin --input /path/to/base.bin`

其中 `bin` 的格式为：

```text
uint32_t num_points
uint32_t dim
float payload[num_points][dim]
```

### 构建主链

当前默认主链是：

```text
vectors
  -> PipeANN disk build
  -> Gorgeous partition
  -> Gorgeous relayout
  -> PQ artifacts
```

当前构建产物里最重要的几个文件是：

- `<prefix>.bin`
  - PipeANN 基础向量数据。
- `<prefix>_train.bin`
  - 可选的训练查询数据。
- `<prefix>_disk.index`
  - PipeANN 原始磁盘索引。
- `<prefix>_partition.bin`
  - Gorgeous 分区产物。
- `<prefix>_graph_relayout.index`
  - Gorgeous native relayout 索引，也是默认主输出。
- `<prefix>_pq_pivots.bin`
  - PipeANN PQ pivots。
- `<prefix>_pq_compressed.bin`
  - PipeANN PQ compressed codes。

如果指定 `--project_compatible_output`，还会额外导出：

- `<prefix>.graphrep`
- `<prefix>.graphrep.partition`
- `<prefix>.graphrep.reorder`

## Native 与 GraphRep

当前项目支持两类索引表示：

- `gorgeous-native`
  - 默认输出。
  - 文件名通常是 `<prefix>_graph_relayout.index`。
  - 优先直接承载原始 `Gorgeous` 页布局与元数据。
- `project-compatible`
  - 可选兼容导出。
  - 文件名通常是 `<prefix>.graphrep`。
  - 主要用于兼容旧的项目内消费方式和一些对比路径。

推荐优先使用 `gorgeous-native`。

## 构建 CLI

建盘命令入口：

```bash
./build/pipeann_gorgeous_build --help
```

最小 toy 示例：

```bash
./build/pipeann_gorgeous_build \
  --mode toy \
  --output_dir build_data \
  --dataset_name toy_run
```

文本向量示例：

```bash
./build/pipeann_gorgeous_build \
  --mode text \
  --input data/vectors.txt \
  --output_dir build_data \
  --dataset_name sample \
  --degree 16 \
  --dense_degree 32 \
  --r_ood 4 \
  --build_l 128 \
  --build_candidates 256 \
  --build_alpha 1.2 \
  --build_threads 8 \
  --page_nodes 8 \
  --pq_subspaces 4
```

常用参数：

- `--degree`
  - 总的一跳邻接预算。
- `--dense_degree`
  - 盘上 `DiskNode` 的稠密邻接容量。
- `--r_ood`
  - PipeANN refine tail 宽度。
- `--build_l`
  - 建图搜索列表大小。
- `--build_candidates`
  - alpha-RNG 剪枝候选上限。
- `--build_alpha`
  - alpha-RNG 剪枝参数。
- `--build_ram_budget_gb`
  - 构建 RAM 预算。
- `--build_threads`
  - 构建线程数。
- `--page_nodes`
  - 每个复制页最多保留多少个节点布局。
- `--partition_scale`
  - Gorgeous 分区 scale。
- `--partition_ldg_times`
  - Gorgeous 分区 LDG 迭代次数。
- `--entry_id`
  - 显式覆盖入口点。
- `--pq_subspaces`
  - PQ 子空间数，必须整除维度。
- `--project_compatible_output`
  - 导出 `graphrep` 兼容产物。
- `--gorgeous_native_output`
  - 显式要求 native 输出，当前也是默认值。

构建完成后，CLI 会打印完整 artifact 路径，包括：

- `pipeann_base_data=...`
- `pipeann_train_query=...`
- `pipeann_disk_index=...`
- `gorgeous_partition=...`
- `gorgeous_relayout=...`
- `index=...`
- `full_data=...`
- `pipeann_pq_pivots=...`
- `pipeann_pq_compressed=...`

## 查询 CLI

查询命令入口：

```bash
./build/pipeann_gorgeous_search --help
```

单条文本查询示例：

```bash
./build/pipeann_gorgeous_search \
  --index build_data/sample_graph_relayout.index \
  --query "0.95 0.15 0.0" \
  --approx_kind pq \
  --top_k 10 \
  --beam_width 8 \
  --l_search 64
```

从文件读取查询：

```bash
./build/pipeann_gorgeous_search \
  --index build_data/sample_graph_relayout.index \
  --query_file data/query.fvecs \
  --query_format fvecs \
  --query_index 0 \
  --top_k 10 \
  --beam_width 8 \
  --l_search 64
```

说明：

- `--approx_kind pq|full`
  - 选择 `PQ` 路径或全精度路径。
- `--approx`
  - 可选地显式指定 full-precision 数据文件。
- `--pq_codebook` / `--pq_codes`
  - 可选地显式覆盖 PQ 路径；不提供时会根据索引路径自动推导。

输出会包含：

- `approx_backend`
- top-k 结果
- 聚合搜索统计，如 `reads`、`completed_pages`、`approx_evals`、`exact_evals`

## 检视 CLI

检视命令入口：

```bash
./build/pipeann_gorgeous_inspect --help
```

示例：

```bash
./build/pipeann_gorgeous_inspect \
  --index build_data/sample_graph_relayout.index \
  --page_id 0
```

它会输出：

- 通用索引元数据，如 `points`、`pages`、`dim`、`entry_id`、`page_size`
- `storage_format`
- native 元数据，如 `native_nodes_per_sector`、`native_range`、`native_range_dense`
- native full-precision payload 的布局边界
- 是否存在 `partition`、`reorder`、`pipeann refine`
- 指定页的布局节点、base degree、dense degree
- 可选的 PQ 元数据与文件大小

## Ground Truth CLI

如果只想先生成 ground truth：

```bash
./build/pipeann_gorgeous_ground_truth \
  --index build_data/sample_graph_relayout.index \
  --queries data/query.fvecs \
  --query_format fvecs \
  --top_k 10 \
  --output results/ground_truth.txt
```

输出格式是纯文本：

- 每行一条查询
- 每行按空格分隔若干真实近邻 ID

这个格式可以直接被 `bench --ground_truth` 消费。

## Benchmark CLI

benchmark 命令入口：

```bash
./build/pipeann_gorgeous_bench --help
```

单次 benchmark 示例：

```bash
./build/pipeann_gorgeous_bench \
  --index build_data/sample_graph_relayout.index \
  --queries data/query.fvecs \
  --query_format fvecs \
  --generate_ground_truth results/ground_truth.txt \
  --ground_truth_k 10 \
  --approx_kind pq \
  --top_k 10 \
  --beam_width 8 \
  --l_search 64 \
  --export results/bench.tsv \
  --experiment_root results/experiments \
  --experiment_name run_001
```

参数 sweep 示例：

```bash
./build/pipeann_gorgeous_bench \
  --index build_data/sample_graph_relayout.index \
  --queries data/query.fvecs \
  --query_format fvecs \
  --ground_truth results/ground_truth.txt \
  --approx_kinds full,pq \
  --beam_widths 4,8,16 \
  --l_search_values 32,64,128 \
  --top_k 10 \
  --export results/sweep.tsv
```

`bench` 支持：

- 自动生成或加载 ground truth
- 单配置运行
- 多组 `beam_width` / `l_search` / `approx_kind` sweep
- 导出 TSV
- 导出结构化实验目录

实验目录中通常会包含：

- `summary.tsv`
- `manifest.txt`
- `ground_truth.txt`
- `runs/run_000.txt`

## Bench Merge 与 Compare

合并多个实验结果：

```bash
./build/pipeann_gorgeous_bench_merge \
  --inputs results/exp_a,results/exp_b,results/single_summary.tsv \
  --output results/all_runs.tsv
```

比较两组实验：

```bash
./build/pipeann_gorgeous_bench_compare \
  --baseline results/exp_a \
  --candidate results/exp_b \
  --output results/compare.md \
  --format markdown
```

也支持输出 TSV：

```bash
./build/pipeann_gorgeous_bench_compare \
  --baseline results/exp_a/summary.tsv \
  --candidate results/exp_b/summary.tsv \
  --output results/compare.tsv \
  --format tsv
```

## 用 SIFT1M 测试

项目已经可以直接吃标准 `fvecs`，所以 `SIFT1M` 的最短路径就是：

```text
下载 sift.tar.gz
  -> 用 sift_base.fvecs 建盘
  -> 用 sift_query.fvecs 查询
  -> 可选用 sift_learn.fvecs 参与训练查询输入
```

下载数据：

```bash
mkdir -p /home/adieu/OS/data/sift
cd /home/adieu/OS/data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

如果你使用的是 `OpenBLAS`，建议先限制 BLAS 线程，避免和 OpenMP 构建线程互相嵌套：

```bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=8
```

建索引：

```bash
cd /home/adieu/OS/project

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
  --build_threads 8 \
  --build_ram_budget_gb 16 \
  --page_nodes 8 \
  --pq_subspaces 16
```

单条查询：

```bash
./build/pipeann_gorgeous_search \
  --index /home/adieu/OS/data/sift/out/sift1m_graph_relayout.index \
  --query_file /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --query_index 0 \
  --top_k 10 \
  --beam_width 8 \
  --l_search 64 \
  --approx_kind pq
```

批量 benchmark：

```bash
mkdir -p /home/adieu/OS/data/sift/results

./build/pipeann_gorgeous_bench \
  --index /home/adieu/OS/data/sift/out/sift1m_graph_relayout.index \
  --queries /home/adieu/OS/data/sift/sift/sift_query.fvecs \
  --query_format fvecs \
  --generate_ground_truth /home/adieu/OS/data/sift/results/ground_truth.txt \
  --ground_truth_k 10 \
  --top_k 10 \
  --beam_width 8 \
  --l_search 64 \
  --approx_kind pq \
  --export /home/adieu/OS/data/sift/results/bench.tsv
```

## 测试

当前仓库内已经接入这些测试目标：

- `pipeann_gorgeous_search_test`
- `pipeann_gorgeous_build_test`
- `pipeann_gorgeous_tool_cli_test`
- `pipeann_gorgeous_integration_test`

运行方式：

```bash
cd /home/adieu/OS/project
ctest --test-dir build --output-on-failure
```

如果只想跑单个测试，也可以直接执行对应二进制。

## 目录概览

- `include/`
  - 项目公开头文件。
- `src/`
  - 项目实现与各 CLI 入口。
- `tests/`
  - 构建、搜索、CLI、集成测试。
- `third_party/pipeann/`
  - vendored PipeANN 源码。
- `third_party/gorgeous/`
  - vendored Gorgeous 源码。
- `docs/`
  - 路线图和其他设计文档。

## 后续文档

如果你想看项目后续演进方向，优先参考：

- [路线图](file:///home/adieu/OS/project/docs/ROADMAP.md)

这份 README 主要描述“当前已经能做什么、应该怎么用”；更细的演进计划放在 `docs/` 中维护。
