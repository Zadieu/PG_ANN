# SIFT1M 0.5% Dynamic Cache With Memory Navigation

Experiment date: 2026-06-01

## Goal

Test whether a smaller dynamic graph cache ratio, `0.5%`, works better when the in-memory navigation graph is enabled.

## Setup

- Dataset: SIFT1M
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Dynamic ratio: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.005`
- Dynamic cache slots: `500`
- Static cache slots: `99500`
- Threads / beam width: `T=8`, `BW=8`
- L queue: `25 30 35 40 45`
- Tested memory navigation settings: `MEM_L=0`, `MEM_L=5`, `MEM_L=10`
- Repeats: 3 runs per configuration
- Raw logs: `/home/dell/data/gorgeous/sift1M/dynamic005_memnav_sweep`

## Per-L Results For 0.5% Dynamic Cache

| MEM_L | L | QPS mean | QPS std | Recall@10 mean | Graph IO mean | Mean latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25 | 3392.10 | 131.52 | 90.65 | 32.77 | 2356 |
| 0 | 30 | 3104.99 | 183.24 | 92.88 | 35.50 | 2579 |
| 0 | 35 | 2857.68 | 98.60 | 94.56 | 38.50 | 2798 |
| 0 | 40 | 2711.23 | 43.58 | 95.67 | 41.57 | 2947 |
| 0 | 45 | 2524.61 | 31.67 | 96.51 | 44.81 | 3165 |
| 5 | 25 | 4167.24 | 77.05 | 90.45 | 25.55 | 1917 |
| 5 | 30 | 3900.09 | 35.80 | 92.71 | 28.30 | 2046 |
| 5 | 35 | 3635.61 | 32.33 | 94.34 | 31.27 | 2196 |
| 5 | 40 | 3322.28 | 41.00 | 95.48 | 34.39 | 2404 |
| 5 | 45 | 3081.31 | 33.81 | 96.36 | 37.60 | 2592 |
| 10 | 25 | 4289.41 | 55.80 | 90.41 | 24.74 | 1862 |
| 10 | 30 | 3894.43 | 145.73 | 92.66 | 27.53 | 2052 |
| 10 | 35 | 3625.27 | 78.49 | 94.29 | 30.51 | 2203 |
| 10 | 40 | 3343.76 | 71.30 | 95.46 | 33.62 | 2389 |
| 10 | 45 | 3099.54 | 78.11 | 96.34 | 36.86 | 2579 |

## Average Comparison

| MEM_L | Static Avg QPS | 0.5% Dynamic Avg QPS | 1% Auto Avg QPS | Best Setting |
|---:|---:|---:|---:|---|
| 0 | 2723.62 | 2918.12 | 3360.10 | 1% Auto |
| 5 | 3666.35 | 3621.30 | 3527.94 | Static |
| 10 | 3563.55 | 3650.48 | 3424.40 | 0.5% Dynamic |

## Average Metrics

| MEM_L | Setting | Avg QPS | QPS vs Static | Avg Recall@10 | Avg Graph IO | Avg Mean Latency |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Static | 2723.62 | - | 94.03 | 49.16 | 2948 |
| 0 | 0.5% Dynamic | 2918.12 | +7.1% | 94.05 | 38.63 | 2769 |
| 0 | 1% Auto | 3360.10 | +23.4% | 94.08 | 36.94 | 2396 |
| 5 | Static | 3666.35 | - | 93.90 | 31.57 | 2205 |
| 5 | 0.5% Dynamic | 3621.30 | -1.2% | 93.87 | 31.42 | 2231 |
| 5 | 1% Auto | 3527.94 | -3.8% | 93.84 | 31.30 | 2295 |
| 10 | Static | 3563.55 | - | 93.88 | 30.82 | 2274 |
| 10 | 0.5% Dynamic | 3650.48 | +2.4% | 93.83 | 30.65 | 2217 |
| 10 | 1% Auto | 3424.40 | -3.9% | 93.81 | 30.56 | 2379 |

## Takeaway

The fixed 0.5% dynamic cache ratio is not strong enough when memory navigation is disabled; 1% remains much better for `MEM_L=0`. However, when memory navigation is enabled, 0.5% reduces dynamic-cache overhead compared with 1% and becomes competitive:

- `MEM_L=5`: 0.5% is close to static, while 1% is clearly worse.
- `MEM_L=10`: 0.5% outperforms static by 2.4%, while 1% is worse.

This suggests the adaptive rule should be memory-navigation aware:

```text
if MEM_L == 0:
    use around 1% dynamic cache on SIFT1M
elif MEM_L > 0:
    try a smaller dynamic ratio around 0.5%
```

The current heuristic selected 1% because it does not yet include `MEM_L` as an input. A better version should pass `MEM_L` into the adaptive selector or use a short profiling phase to choose between 0%, 0.5%, and 1%.
