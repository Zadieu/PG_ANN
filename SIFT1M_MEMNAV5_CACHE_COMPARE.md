# SIFT1M MEM_L=5 Cache Compare

Experiment date: 2026-06-01

## Goal

Test whether dynamic adjacency-list cache is still beneficial with a weaker in-memory navigation graph than `MEM_L=10`.

## Setup

- Dataset: SIFT1M
- Memory navigation: enabled with `MEM_L=5`
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Threads / beam width: `T=8`, `BW=8`
- L queue: `25 30 35 40 45`
- Static baseline: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO` unset
- Dynamic cache: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto`
- Repeats: 3 runs per configuration
- Raw logs: `/home/dell/data/gorgeous/sift1M/memnav5_cache_compare`

## Auto Selection

Adaptive mode selected the same SIFT1M split:

```text
auto dynamic graph cache ratio selected: 0.01
enabled dynamic graph cache. static slots: 99000, dynamic slots: 1000
```

## Mean Results

| Config | L | QPS mean | QPS std | Recall@10 mean | Graph IO mean | Mean latency | P99.9 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static + MEM_L=5 | 25 | 4240.90 | 120.30 | 90.42 | 25.56 | 1884 | 3853 |
| Static + MEM_L=5 | 30 | 3956.95 | 85.52 | 92.77 | 28.46 | 2019 | 4835 |
| Static + MEM_L=5 | 35 | 3653.71 | 48.99 | 94.37 | 31.44 | 2186 | 4309 |
| Static + MEM_L=5 | 40 | 3363.28 | 25.99 | 95.53 | 34.59 | 2375 | 4561 |
| Static + MEM_L=5 | 45 | 3116.91 | 57.94 | 96.41 | 37.81 | 2563 | 4815 |
| Auto dynamic + MEM_L=5 | 25 | 4209.30 | 106.16 | 90.41 | 25.50 | 1898 | 5610 |
| Auto dynamic + MEM_L=5 | 30 | 3726.61 | 58.46 | 92.68 | 28.23 | 2143 | 6281 |
| Auto dynamic + MEM_L=5 | 35 | 3477.92 | 43.32 | 94.28 | 31.11 | 2296 | 4941 |
| Auto dynamic + MEM_L=5 | 40 | 3223.99 | 60.46 | 95.46 | 34.23 | 2478 | 5894 |
| Auto dynamic + MEM_L=5 | 45 | 3001.90 | 36.55 | 96.34 | 37.42 | 2661 | 5154 |

## Average Across L Values

| Config | Avg QPS | Avg Recall@10 | Avg Graph IO | Avg Mean Latency |
|---|---:|---:|---:|---:|
| Static + MEM_L=5 | 3666.35 | 93.90 | 31.57 | 2205 |
| Auto dynamic + MEM_L=5 | 3527.94 | 93.84 | 31.30 | 2295 |

Compared with static cache under `MEM_L=5`, auto dynamic cache changed the averages by:

- QPS: -3.8%
- Graph IO: -0.9%
- Mean latency: +4.1%

## Summary Across MEM_L Values

| MEM_L | Static Avg QPS | Auto Avg QPS | Auto QPS vs Static | Static Graph IO | Auto Graph IO | Static Latency | Auto Latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2723.62 | 3360.10 | +23.4% | 49.16 | 36.94 | 2948 | 2396 |
| 5 | 3666.35 | 3527.94 | -3.8% | 31.57 | 31.30 | 2205 | 2295 |
| 10 | 3563.55 | 3424.40 | -3.9% | 30.82 | 30.56 | 2274 | 2379 |

## Takeaway

On SIFT1M, dynamic adjacency-list cache is strongly beneficial when memory navigation is disabled, but loses its benefit once the in-memory navigation graph is enabled. Even at `MEM_L=5`, memory navigation reduces graph IO enough that the dynamic cache only saves about 0.9% additional graph IO, while its cache-management overhead reduces QPS.

This supports adding `MEM_L` awareness to adaptive dynamic cache sizing. For the current SIFT1M configuration, a better rule is:

```text
if MEM_L == 0:
    use adaptive dynamic cache, which selects about 1%
else:
    prefer static graph cache or choose a much smaller dynamic ratio
```
