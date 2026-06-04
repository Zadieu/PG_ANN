# SIFT1M Memory Navigation Cache Compare

Experiment date: 2026-06-01

## Goal

Evaluate whether dynamic adjacency-list cache is still beneficial when the optional in-memory navigation graph is enabled.

## Setup

- Dataset: SIFT1M
- Memory navigation: enabled with `MEM_L=10`
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Threads / beam width: `T=8`, `BW=8`
- L queue: `25 30 35 40 45`
- Static baseline: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO` unset
- Dynamic cache: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto`
- Repeats: 3 runs per configuration
- Raw logs: `/home/dell/data/gorgeous/sift1M/memnav_cache_compare`

## Auto Selection

With memory navigation enabled, auto mode still selected the same SIFT1M cache split:

```text
auto dynamic graph cache ratio selected: 0.01
enabled dynamic graph cache. static slots: 99000, dynamic slots: 1000
```

## Mean Results

| Config | L | QPS mean | QPS std | Recall@10 mean | Graph IO mean | Mean latency | P99.9 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static + MEM_L=10 | 25 | 4198.43 | 66.13 | 90.43 | 24.77 | 1902 | 6068 |
| Static + MEM_L=10 | 30 | 3833.80 | 166.16 | 92.73 | 27.68 | 2084 | 8396 |
| Static + MEM_L=10 | 35 | 3523.90 | 95.05 | 94.35 | 30.71 | 2266 | 8015 |
| Static + MEM_L=10 | 40 | 3203.62 | 258.71 | 95.48 | 33.85 | 2504 | 17471 |
| Static + MEM_L=10 | 45 | 3058.00 | 160.90 | 96.39 | 37.09 | 2616 | 8565 |
| Auto dynamic + MEM_L=10 | 25 | 4110.99 | 265.10 | 90.38 | 24.75 | 1948 | 7758 |
| Auto dynamic + MEM_L=10 | 30 | 3700.34 | 274.01 | 92.63 | 27.46 | 2166 | 10361 |
| Auto dynamic + MEM_L=10 | 35 | 3318.93 | 306.87 | 94.27 | 30.39 | 2420 | 10731 |
| Auto dynamic + MEM_L=10 | 40 | 3126.71 | 203.04 | 95.43 | 33.49 | 2560 | 12514 |
| Auto dynamic + MEM_L=10 | 45 | 2865.01 | 215.43 | 96.32 | 36.72 | 2799 | 9511 |

## Average Across L Values

| Config | Avg QPS | Avg Recall@10 | Avg Graph IO | Avg Mean Latency |
|---|---:|---:|---:|---:|
| Static + MEM_L=10 | 3563.55 | 93.88 | 30.82 | 2274 |
| Auto dynamic + MEM_L=10 | 3424.40 | 93.81 | 30.56 | 2379 |

Compared with static cache under `MEM_L=10`, auto dynamic cache changed the averages by:

- QPS: -3.9%
- Graph IO: -0.8%
- Mean latency: +4.6%

## Takeaway

On SIFT1M, memory navigation and dynamic adjacency-list cache are not strongly complementary under `MEM_L=10`. The navigation graph already reduces graph IO substantially, leaving little additional IO for the dynamic cache to save. The dynamic cache still reduces graph IO slightly, but its lookup, lock, and replacement overhead outweigh the small IO reduction.

This suggests that adaptive dynamic cache sizing should also consider whether memory navigation is enabled. A reasonable next rule is to reduce or disable dynamic cache when `MEM_L > 0` and the graph IO is already low.
