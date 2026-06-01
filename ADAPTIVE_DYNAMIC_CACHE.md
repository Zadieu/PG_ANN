# Adaptive Dynamic Graph Cache

This project supports both fixed and adaptive dynamic graph cache sizing.

## Modes

- Static-only baseline: leave `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO` unset.
- Fixed dynamic cache: set `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO` to a number, for example `0.01`.
- Adaptive dynamic cache: set `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto`.

The ratio is interpreted as the fraction of the graph-cache slots reserved for dynamic replacement. For example, with `MEM_GRAPH_USE_RATIO=0.1` and `GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=0.01`, 1% of the 10% graph cache is dynamic and the remaining 99% is static.

## Adaptive Rule

The adaptive mode chooses a conservative dynamic ratio from dataset scale, vector dimensionality, and graph-cache budget:

- Small datasets start around 0.5%-1%.
- Larger datasets increase toward 2%-3%.
- High-dimensional datasets receive a small additional dynamic budget because disk-resident searches are more sensitive to random graph I/O.
- The final dynamic ratio is clamped to 0.5%-5% to avoid the performance drop observed with overly large dynamic regions.

For SIFT1M with `MEM_GRAPH_USE_RATIO=0.1`, adaptive mode selects approximately 1%, matching the strongest region observed in the fair multirun experiment.

## Usage

```bash
export GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO=auto
./scripts/run_benchmark.sh release search
```

The search log prints the selected ratio and slot allocation, for example:

```text
auto dynamic graph cache ratio selected: 0.01 (...)
enabled dynamic graph cache. static slots: 99000, dynamic slots: 1000
```
