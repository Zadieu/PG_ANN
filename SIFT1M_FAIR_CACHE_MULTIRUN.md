# SIFT1M Fair Cache Multirun Report

Experiment date: 2026-05-31

## Setup

- Dataset: SIFT1M
- Repeats: 3 independent search runs per configuration
- L queue: 25, 30, 35, 40, 45
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Disk graph cache index: `USE_DISK_GRAPH_CACHE_INDEX=0`
- Threads / beam width: `T=8`, `BW=8`
- Dynamic ratio env: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO`
- Raw logs: `/home/dell/data/gorgeous/sift1M/fair_cache_compare_multirun`

## Dynamic Cache Policy

The current dynamic graph cache is not pure LRU. It uses a two-hit admission rule plus LFU replacement with LRU tie-breaking:

1. Static cached nodes are never inserted into the dynamic region.
2. A disk-read graph node becomes a candidate on first observation, but is admitted only after it is seen at least twice.
3. If the dynamic region has free slots, the admitted node is inserted directly.
4. If full, the victim is the slot with the lowest hit count; ties are broken by older access timestamp.
5. A candidate is rejected if its candidate hit count is lower than the selected victim's hit count.

## Mean Results Across 3 Runs

| Config | L | QPS mean | QPS std | Recall@10 mean | Graph IO mean | Read(T) mean | QPS vs Static | Graph IO reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static baseline | 25 | 2949.59 | 74.53 | 90.60 | 43.39 | 2228.45 | - | - |
| Static baseline | 30 | 2878.30 | 108.08 | 92.88 | 46.12 | 2263.34 | - | - |
| Static baseline | 35 | 2748.21 | 50.14 | 94.52 | 49.00 | 2348.69 | - | - |
| Static baseline | 40 | 2585.02 | 11.80 | 95.65 | 52.07 | 2496.55 | - | - |
| Static baseline | 45 | 2456.98 | 45.75 | 96.52 | 55.21 | 2624.12 | - | - |
| Dynamic 1% | 25 | 3809.40 | 69.97 | 90.67 | 31.12 | 1564.50 | 29.2% | 28.3% |
| Dynamic 1% | 30 | 3587.40 | 85.64 | 92.96 | 33.73 | 1588.34 | 24.6% | 26.9% |
| Dynamic 1% | 35 | 3316.93 | 149.87 | 94.56 | 36.73 | 1700.74 | 20.7% | 25.0% |
| Dynamic 1% | 40 | 3129.68 | 85.40 | 95.68 | 39.87 | 1781.26 | 21.1% | 23.4% |
| Dynamic 1% | 45 | 2918.13 | 56.85 | 96.53 | 43.11 | 1910.14 | 18.8% | 21.9% |
| Dynamic 2% | 25 | 3811.15 | 83.91 | 90.69 | 29.99 | 1504.11 | 29.2% | 30.9% |
| Dynamic 2% | 30 | 3544.42 | 78.87 | 92.96 | 32.35 | 1409.51 | 23.1% | 29.9% |
| Dynamic 2% | 35 | 3291.12 | 85.95 | 94.54 | 35.37 | 1455.92 | 19.8% | 27.8% |
| Dynamic 2% | 40 | 3078.38 | 40.32 | 95.70 | 38.51 | 1545.39 | 19.1% | 26.0% |
| Dynamic 2% | 45 | 2896.20 | 30.72 | 96.54 | 41.75 | 1654.80 | 17.9% | 24.4% |
| Dynamic 3% | 25 | 3872.57 | 27.56 | 90.69 | 29.47 | 1336.89 | 31.3% | 32.1% |
| Dynamic 3% | 30 | 3356.80 | 46.24 | 92.94 | 31.75 | 1108.59 | 16.6% | 31.1% |
| Dynamic 3% | 35 | 3157.20 | 58.09 | 94.52 | 34.69 | 1246.04 | 14.9% | 29.2% |
| Dynamic 3% | 40 | 2974.34 | 20.43 | 95.66 | 37.80 | 1360.46 | 15.1% | 27.4% |
| Dynamic 3% | 45 | 2745.02 | 62.17 | 96.54 | 41.04 | 1519.91 | 11.7% | 25.7% |
| Dynamic 5% | 25 | 3641.38 | 115.44 | 90.70 | 29.11 | 1161.95 | 23.5% | 32.9% |
| Dynamic 5% | 30 | 2590.80 | 72.89 | 92.90 | 31.21 | 1195.85 | -10.0% | 32.3% |
| Dynamic 5% | 35 | 2509.24 | 41.61 | 94.48 | 34.05 | 1348.15 | -8.7% | 30.5% |
| Dynamic 5% | 40 | 2447.06 | 20.74 | 95.62 | 37.07 | 1416.67 | -5.3% | 28.8% |
| Dynamic 5% | 45 | 2376.22 | 18.52 | 96.47 | 40.20 | 1409.78 | -3.3% | 27.2% |
| Dynamic 10% | 25 | 2896.63 | 8.66 | 90.62 | 28.68 | 1099.52 | -1.8% | 33.9% |
| Dynamic 10% | 30 | 1679.37 | 24.18 | 92.64 | 29.79 | 432.19 | -41.7% | 35.4% |
| Dynamic 10% | 35 | 1641.32 | 17.58 | 94.25 | 32.44 | 471.29 | -40.3% | 33.8% |
| Dynamic 10% | 40 | 1619.08 | 18.30 | 95.46 | 35.46 | 608.19 | -37.4% | 31.9% |
| Dynamic 10% | 45 | 1574.67 | 20.58 | 96.31 | 38.61 | 846.70 | -35.9% | 30.1% |

## Average Over L Values

| Config | Avg QPS | Avg Recall@10 | Avg Graph IO | Avg QPS vs Static |
|---|---:|---:|---:|---:|
| Static baseline | 2723.62 | 94.03 | 49.16 | - |
| Dynamic 1% | 3352.31 | 94.08 | 36.91 | 23.1% |
| Dynamic 2% | 3324.26 | 94.09 | 35.59 | 22.1% |
| Dynamic 3% | 3221.18 | 94.07 | 34.95 | 18.3% |
| Dynamic 5% | 2712.94 | 94.03 | 34.33 | -0.4% |
| Dynamic 10% | 1882.21 | 93.85 | 33.00 | -30.9% |

## Takeaways

- Dynamic 1% and 2% are the strongest stable settings in this run. Dynamic 1% improves average QPS by 23.1% over static across `L=25..45`, while Dynamic 2% improves it by 22.1%.
- Dynamic 3% is still beneficial on all L values, but its QPS gain drops at larger L.
- Dynamic 5% and 10% reduce Graph IO further, but QPS declines for most L values, likely because replacement bookkeeping, mutex contention, and dynamic-cache lookup overhead begin to dominate the saved IO on SIFT1M.
- Recall remains essentially unchanged across static and dynamic variants, so the improvement mainly comes from reducing disk graph IO rather than changing search quality.
