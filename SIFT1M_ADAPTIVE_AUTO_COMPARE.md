# SIFT1M Adaptive Dynamic Cache Test

Experiment date: 2026-06-01

## Goal

Evaluate whether `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto` can:

1. Select a suitable dynamic graph cache ratio for the current dataset.
2. Outperform the original all-static graph cache baseline.

## Setup

- Dataset: SIFT1M
- Graph cache budget: `MEM_GRAPH_USE_RATIO=0.1`
- Auto mode: `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto`
- Threads / beam width: `T=8`, `BW=8`
- L queue: `25 30 35 40 45`
- Repeats: 3 auto runs
- Auto logs: `/home/dell/data/gorgeous/sift1M/adaptive_auto_compare`
- Baseline/fixed-ratio logs: `/home/dell/data/gorgeous/sift1M/fair_cache_compare_multirun`

## Auto Selection

For SIFT1M, adaptive mode selected:

```text
auto dynamic graph cache ratio selected: 0.01
enabled dynamic graph cache. static slots: 99000, dynamic slots: 1000
```

This matches the strongest region found in the fixed-ratio sweep, where 1%-2% dynamic cache performed best on average.

## Average Results Across L Values

| Config | Avg QPS | Avg Recall@10 | Avg Graph IO | Avg Mean Latency | QPS vs Static |
|---|---:|---:|---:|---:|---:|
| Static baseline | 2723.62 | 94.03 | 49.16 | 2948 | - |
| Dynamic 1% | 3352.31 | 94.08 | 36.91 | 2406 | +23.1% |
| Dynamic 2% | 3324.26 | 94.09 | 35.59 | 2427 | +22.1% |
| Dynamic 3% | 3221.18 | 94.07 | 34.95 | 2514 | +18.3% |
| Dynamic 5% | 2712.94 | 94.03 | 34.33 | 3018 | -0.4% |
| Dynamic 10% | 1882.21 | 93.85 | 33.00 | 4479 | -30.9% |
| Adaptive auto | 3360.10 | 94.08 | 36.94 | 2396 | +23.4% |

## Per-L Auto Results

| L | Auto QPS | Auto Recall@10 | Auto Graph IO | Auto Mean Latency |
|---:|---:|---:|---:|---:|
| 25 | 3739.40 | 90.72 | 31.16 | 2137 |
| 30 | 3599.38 | 92.92 | 33.74 | 2219 |
| 35 | 3381.37 | 94.54 | 36.75 | 2363 |
| 40 | 3133.33 | 95.68 | 39.91 | 2550 |
| 45 | 2947.05 | 96.54 | 43.15 | 2711 |

## Conclusion

For SIFT1M, adaptive mode selected 1% dynamic cache, which is the best average-performing region from the fixed-ratio sweep. Compared with the all-static graph cache baseline, adaptive mode improved average QPS by 23.4%, reduced average graph IO from 49.16 to 36.94 per query, and reduced average latency from 2948 to 2396.

This validates the adaptive rule on SIFT1M. To claim dataset-general optimality, the same test should be repeated on larger and higher-dimensional datasets such as GIST1M, Deep10M/100M, or Laion-style embeddings.
