# SIFT1M 0.2% Dynamic Cache With Memory Navigation

Experiment date: 2026-06-01

## Goal

Test whether `0.2%` dynamic graph cache is too small, especially when memory navigation is enabled.

## Setup

- Dataset: SIFT1M
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Dynamic ratio: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.002`
- Dynamic cache slots: `200`
- Static cache slots: `99800`
- Threads / beam width: `T=8`, `BW=8`
- L queue: `25 30 35 40 45`
- Tested memory navigation settings: `MEM_L=0`, `MEM_L=5`, `MEM_L=10`
- Repeats: 3 runs per configuration
- Raw logs: `/home/dell/data/gorgeous/sift1M/dynamic002_memnav_sweep`

## Per-L Results For 0.2% Dynamic Cache

| MEM_L | L | QPS mean | QPS std | Recall@10 mean | Graph IO mean | Mean latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25 | 3144.72 | 31.89 | 90.65 | 34.77 | 2539 |
| 0 | 30 | 2824.49 | 135.06 | 92.91 | 37.57 | 2831 |
| 0 | 35 | 2702.79 | 165.60 | 94.53 | 40.48 | 2964 |
| 0 | 40 | 2616.32 | 5.22 | 95.66 | 43.51 | 3054 |
| 0 | 45 | 2389.67 | 153.47 | 96.50 | 46.71 | 3354 |
| 5 | 25 | 3936.70 | 293.36 | 90.43 | 25.56 | 2037 |
| 5 | 30 | 3714.35 | 130.94 | 92.74 | 28.42 | 2150 |
| 5 | 35 | 3432.12 | 162.53 | 94.38 | 31.40 | 2329 |
| 5 | 40 | 2995.12 | 407.55 | 95.52 | 34.53 | 2703 |
| 5 | 45 | 2798.67 | 426.08 | 96.39 | 37.78 | 2903 |
| 10 | 25 | 3904.23 | 446.24 | 90.40 | 24.74 | 2065 |
| 10 | 30 | 3538.97 | 368.61 | 92.69 | 27.58 | 2273 |
| 10 | 35 | 3202.91 | 292.64 | 94.33 | 30.56 | 2507 |
| 10 | 40 | 2915.93 | 243.04 | 95.47 | 33.68 | 2751 |
| 10 | 45 | 2725.03 | 206.49 | 96.34 | 36.94 | 2942 |

## Average Comparison

| MEM_L | Static Avg QPS | 0.2% Dynamic Avg QPS | 0.5% Dynamic Avg QPS | 1% Auto Avg QPS | Best Setting |
|---:|---:|---:|---:|---:|---|
| 0 | 2723.62 | 2735.60 | 2918.12 | 3360.10 | 1% Auto |
| 5 | 3666.35 | 3375.39 | 3621.30 | 3527.94 | Static |
| 10 | 3563.55 | 3257.42 | 3650.48 | 3424.40 | 0.5% Dynamic |

## Average Metrics

| MEM_L | Setting | Avg QPS | QPS vs Static | Avg Recall@10 | Avg Graph IO | Avg Mean Latency |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Static | 2723.62 | - | 94.03 | 49.16 | 2948 |
| 0 | 0.2% Dynamic | 2735.60 | +0.4% | 94.05 | 40.61 | 2949 |
| 0 | 0.5% Dynamic | 2918.12 | +7.1% | 94.05 | 38.63 | 2769 |
| 0 | 1% Auto | 3360.10 | +23.4% | 94.08 | 36.94 | 2396 |
| 5 | Static | 3666.35 | - | 93.90 | 31.57 | 2205 |
| 5 | 0.2% Dynamic | 3375.39 | -7.9% | 93.89 | 31.54 | 2425 |
| 5 | 0.5% Dynamic | 3621.30 | -1.2% | 93.87 | 31.42 | 2231 |
| 5 | 1% Auto | 3527.94 | -3.8% | 93.84 | 31.30 | 2295 |
| 10 | Static | 3563.55 | - | 93.88 | 30.82 | 2274 |
| 10 | 0.2% Dynamic | 3257.42 | -8.6% | 93.85 | 30.70 | 2508 |
| 10 | 0.5% Dynamic | 3650.48 | +2.4% | 93.83 | 30.65 | 2217 |
| 10 | 1% Auto | 3424.40 | -3.9% | 93.81 | 30.56 | 2379 |

## Takeaway

The `0.2%` dynamic cache ratio is too small for this SIFT1M setup:

- With `MEM_L=0`, it is only 0.4% faster than static and far worse than 0.5% or 1%.
- With `MEM_L=5` and `MEM_L=10`, it is clearly worse than static.

This suggests that 200 dynamic slots do not capture enough reusable graph-neighborhood hot spots, but still add dynamic-cache lookup and replacement overhead. For SIFT1M, a practical dynamic-cache range is:

```text
MEM_L=0:  around 1%
MEM_L>0: around 0.5%, not 0.2%
```
