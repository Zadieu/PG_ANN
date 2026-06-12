# PipeGor_ANN

## 中文说明

PipeGor_ANN 是一个基于 Gorgeous 改造的磁盘常驻近似最近邻搜索项目。我们的核心目标不是重新设计一套磁盘索引格式，而是在 Gorgeous 已有磁盘结构和查询逻辑基础上，引入 PipeANN 风格的运行时流水线读取，使 SSD 读取和 CPU 计算尽可能重叠。

简单来说：

```text
Gorgeous:
  按搜索过程一批一批读取磁盘页，读完后再做 CPU 计算和候选扩展。

PipeGor_ANN:
  保留 Gorgeous 的磁盘布局和邻接表缓存思想，
  但在查询过程中把未来可能需要的图页提前发起 I/O，
  让磁盘读取和 CPU 处理并行进行。
```

## 与 Gorgeous 的关系

本项目目前使用的 Gorgeous baseline 是：

```text
/home/dell/projects/Gorgeous-baseline
```

在对比实验中，Gorgeous baseline 和 PipeGor_ANN 可以使用同一份磁盘文件：

```text
/home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index
/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/_graph_rep.index
```

其中：

```text
_disk.index
  原始 Gorgeous/Starling 风格磁盘文件。每个磁盘页保存节点向量和对应邻接表。

_graph_rep.index
  Gorgeous 论文中的 graph-replicated 磁盘结构。
  它除了保存当前节点的信息，还会额外复制部分邻居的邻接表。
```

需要强调的是：PipeGor_ANN 的主要贡献不是创造新的磁盘文件格式。我们在实验中尽量使用和 Gorgeous baseline 相同的磁盘结构，然后只改变查询执行策略。这样对比更公平，性能差异主要来自流水线调度，而不是磁盘布局不同。

## 我们做了什么

### 1. 查询阶段的流水线图读取

Gorgeous 原始查询路径更偏向同步批处理：搜索到一批候选点后，读取它们所在的磁盘页，然后再继续做邻接表展开、PQ 距离计算和精排。

PipeGor_ANN 在这个过程中增加了流水线图读取：

```text
CPU 正在处理当前已经读回来的节点
        +
SSD 同时读取后续可能要访问的图页
```

这样做的好处是：当 CPU 计算和 SSD I/O 都有空闲时，可以隐藏一部分 I/O 等待时间，提高 QPS 并降低平均延迟。

主要实现位置：

```text
src/gorgeous/index_search.cpp
src/gorgeous/index_search_dup_graph.cpp
```

其中：

```text
index_search.cpp
  对应普通 _disk.index 查询路径。

index_search_dup_graph.cpp
  对应 _graph_rep.index 查询路径。
```

### 2. 面向 Gorgeous 磁盘结构的邻接表筛选

Gorgeous 的 graph-replicated 磁盘页可能已经把部分邻居的邻接表一起读回来了。因此 PipeGor_ANN 在流水线调度时不会盲目再次读取同一个邻接表，而是先判断这个邻接表是否已经随着之前的磁盘页被带回。

也就是说，我们不是简单照搬 PipeANN 的提前读取，而是利用 Gorgeous 的磁盘结构做筛选：

```text
如果邻接表已经在已读页面中：
  直接展开，避免重复 I/O

如果邻接表还没有被读回：
  再作为流水线候选提交给 SSD
```

这部分机制用于减少无效读取，尤其适合 `_graph_rep.index` 这种已经包含邻接表复制的磁盘结构。

### 3. 动态流水线宽度

直接把流水线宽度一开始开到 `beam_width` 并不稳定。我们在实验中发现，尤其是在小 `L` 或高线程下，过早提交太多 I/O 会带来大量无效预读，导致带宽浪费、延迟上升，甚至 recall 下降。

因此 PipeGor_ANN 增加了动态流水线宽度：

```text
搜索开始时使用较小宽度
  -> 随着搜索候选逐渐稳定，慢慢增大宽度
  -> 高线程下根据全局 QD budget 限制每个 query 的最大宽度
```

当前 graph-replicated 路径支持以下环境变量：

```bash
GORGEOUS_PIPELINED_GRAPH_IO=1
GORGEOUS_PIPELINED_GRAPH_DYNAMIC_WIDTH=1
GORGEOUS_PIPELINED_GRAPH_PIPE_MAX=auto
GORGEOUS_PIPELINED_GRAPH_QD_BUDGET=256
GORGEOUS_PIPELINED_GRAPH_RAMP_STEP=auto
GORGEOUS_PIPEANN_PIPE_START=auto
GORGEOUS_PIPEANN_PIPE_MIN=1
GORGEOUS_PIPEANN_L_AWARE_PIPE_START=1
GORGEOUS_PIPEANN_L_AWARE_LOW_L=12
GORGEOUS_PIPEANN_L_AWARE_HIGH_L=18
```

最近一次 SIFT1M graph-rep 对比中，加入动态宽度后效果明显更稳定：

```text
测试矩阵:
  T = 8, 16, 32, 64
  L = 15, 20, 25, 30, 35, 40

修改前:
  平均 QPS gain:    +0.66%
  平均延迟改善:     +0.48%
  胜场:             11 / 24
  平均 recall 差:   -1.10

加入 graph-rep 动态流水线宽度后:
  平均 QPS gain:    +4.03%
  平均延迟改善:     +3.53%
  胜场:             18 / 24
  平均 recall 差:   -1.05
```

结果文件示例：

```text
/home/dell/data/gorgeous/sift1M/graph_rep_after_dynamic/run_20260612_143607/graph_rep_compare.csv
```

### 4. 离线自适应参数预测器

除了运行时动态宽度，我们还实现了一个 VDTuner 风格的离线参数预测器。它不是在每个 query 执行时调用神经网络，而是在正式测试之前使用历史实验记录学习参数和性能之间的关系，然后推荐下一轮应该测试或使用的流水线参数。

整体流程是：

```text
1. 离线采样
   对不同 T、L、pipe_max、qd_budget、pipe_start、ramp_step 等参数组合进行小规模测试。

2. 记录实验历史
   保存每个组合相对 baseline 的 QPS gain、recall drop、latency、graph I/O 等指标。

3. 训练/拟合推荐器
   使用历史记录拟合一个 surrogate model 或 MLP 模型。

4. 生成推荐 profile
   对新的 T/L 或新的数据集，预测哪些参数组合可能收益更高且 recall 损失可控。

5. 回测验证
   把推荐 profile 交给 offline_tune_pipegor.py，再实际运行验证。
```

相关脚本：

```text
scripts/offline_tune_pipegor.py
  离线采样和回测工具。可以选择 baseline 为 PipeGorANN pipeline-off 或 Gorgeous。

scripts/model_tune_pipegor.py
  VDTuner 风格的轻量 surrogate 推荐器，基于历史记录和近邻加权预测。

scripts/mlp_tune_pipegor.py
  sklearn MLP 版本的推荐器。

scripts/torch_mlp_tune_pipegor.py
  PyTorch MLP 版本的推荐器，支持带 recall 约束的目标函数。
```

当前效果较好的预测器是 PyTorch MLP 约束版本。它不只是预测 `gain_pct` 和 `recall_drop`，而是使用带约束的目标：

```text
score = predicted_gain - recall_penalty * max(0, predicted_recall_drop - max_allowed_drop)
```

这样可以让推荐器优先选择：

```text
QPS 提升明显
recall 损失不超过阈值
参数不要过于激进
```

本地较好的配置示例：

```text
recall_penalty = 80
min_pred_gain = 5
max_recall_drop = 1.0
```

对应的一轮验证结果中，推荐器选择的参数整体平均收益为正，并且直接推荐为 pipeline 的候选大多数满足 recall 约束。这个工具后续可以扩展到其他数据集，而不需要每次都手工遍历完整参数网格。

### 5. fallback 与 baseline 的含义

在预测器里，`fallback` 表示当模型认为流水线在某个 `T/L` 下不划算，或者 recall 风险过高时，推荐退回 baseline 查询方式。

这里的 baseline 可以有两种：

```text
PipeGorANN pipeline-off
  使用 PipeGorANN 可执行文件，但关闭流水线。
  这个模式用于判断我们的改动本身有没有额外开销。

Gorgeous baseline
  使用 /home/dell/projects/Gorgeous-baseline 中的 Gorgeous 原始实现。
  这个模式用于和真正的 Gorgeous 原项目进行性能对比。
```

`graph-rep` 则表示是否使用 `_graph_rep.index` 磁盘结构。它不是 PipeGor_ANN 的必要条件；PipeGor_ANN 也可以在普通 `_disk.index` 路径上运行流水线机制。

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
  /home/dell/projects/Gorgeous-baseline/build/tests/search_disk_index

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
