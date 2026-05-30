# PG_ANN Handoff

This document summarizes the current handoff state after the M1-M4 work.
It is intended for the next student to continue from the current `main`
branch without reconstructing the implementation history from chat logs.

## Current Scope

Implemented and validated locally:

- M1 instrumentation:
  - Search counters now include bytes read, resident page hits, replicated-page
    hits, graph-cache counters, refinement counters, scheduler counters, and
    dynamic beam counters.
  - Search, bench, summary export, and comparison export include the new fields.
- M2 graph cache:
  - `GraphAdjacencyCache` stores adjacency-only cached graph nodes.
  - `SearchConfig::graph_cache_budget_bytes` enables cache use.
  - `graph_cache_policy` supports `entry_bfs` and `page_layout`.
  - Cache hits expand graph neighbors without submitting traversal page reads.
  - Final exact rerank still goes through the existing refinement path.
- M3 explicit refinement:
  - `refine_k`, `refine_ratio`, and `defer_exact_until_refinement` are explicit
    search and bench variables.
  - Approximate traversal candidates and exact refinement candidates are now
    separately counted.
- M4 scheduler experiment controls:
  - `scheduler_policy` supports `conservative`, `bounded`, and `aggressive`.
  - `scheduler_policy_limit` optionally overrides the computed pending-work
    limit.
  - `dynamic_beam_policy` supports `adaptive` and `fixed`.
  - The search loop exports observed `inflight + ready_unexpanded` pressure via
    `scheduler_pending_max`, `scheduler_ready_unexpanded_max`, and
    `scheduler_limit_hits`.

## Key Files

- `include/pipeline_search.h`
  - Main search config and stats surface.
- `include/search/graph_cache.h`
  - Graph cache policy and adjacency-cache interface.
- `src/search/graph_cache.cpp`
  - Entry-BFS and page-layout cache builders.
- `src/search/search_session.cpp`
  - Search execution, graph-cache expansion, deferred exact refinement, and
    scheduler policy accounting.
- `src/tools/tool_cli.cpp`
  - Bench sweep, TSV export/load, experiment manifest/run details, and
    comparison output.
- `src/search_main.cpp` and `src/bench_main.cpp`
  - User-facing CLI argument parsing and stats printing.
- `tests/pipeline_search_test.cpp`
  - Search-level coverage for graph cache, page-layout policy, deferred
    refinement, and scheduler policy.
- `tests/tool_cli_test.cpp`
  - Tooling coverage for sweep/export/load/compare fields.
- `docs/RESEARCH_IMPLEMENTATION_PLAN.md`
  - Research roadmap and implementation notes.

## Local Validation

Recommended WSL validation:

```bash
cmake -S . -B build_wsl
cmake --build build_wsl -j2
ctest --test-dir build_wsl --output-on-failure
```

Last observed local result:

- Full build completed.
- `ctest --test-dir build_wsl --output-on-failure`: 4/4 tests passed.
- `git diff --check` passed, with only CRLF conversion warnings from Git.

Additional CLI smoke used during handoff:

```bash
./build_wsl/pipeann_gorgeous_bench \
  --index build_wsl/test_tool_cli_data/tool_cli_graph_relayout.index \
  --queries build_wsl/test_tool_cli_data/queries.txt \
  --query_format text \
  --generate_ground_truth build_wsl/m4_smoke_ground_truth.txt \
  --ground_truth_k 2 \
  --top_k 2 \
  --approx_kinds full \
  --beam_widths 2 \
  --l_search_values 6 \
  --graph_cache_bytes_values 0,8192 \
  --graph_cache_policies entry_bfs,page_layout \
  --refine_k_values 2 \
  --defer_exact_values 0,1 \
  --scheduler_policies conservative,bounded \
  --scheduler_policy_limit_values 0 \
  --dynamic_beam_policies adaptive,fixed \
  --experiment_dir build_wsl/m4_smoke

./build_wsl/pipeann_gorgeous_bench_compare \
  --baseline build_wsl/m4_smoke \
  --candidate build_wsl/m4_smoke \
  --output build_wsl/m4_smoke_compare.tsv \
  --format tsv
```

This smoke produced 32 runs and confirmed that comparison loading/export works
with graph-cache, refinement, and scheduler fields.

## Example M2-M4 Sweep

Use a real dataset path and query file for meaningful numbers:

```bash
./build/pipeann_gorgeous_bench \
  --index build_data/sample_graph_relayout.index \
  --queries data/query.fvecs \
  --query_format fvecs \
  --ground_truth results/ground_truth.txt \
  --top_k 10 \
  --approx_kinds full,pq \
  --beam_widths 4,8 \
  --l_search_values 32,64 \
  --graph_cache_bytes_values 0,1048576 \
  --graph_cache_policies entry_bfs,page_layout \
  --refine_k_values 10,32 \
  --defer_exact_values 0,1 \
  --scheduler_policies conservative,bounded \
  --dynamic_beam_policies adaptive,fixed \
  --experiment_root results/experiments \
  --experiment_name m2_m4_sweep
```

Important exported columns to inspect:

- I/O and layout: `async_reads`, `bytes_read`, `page_resident_hits`,
  `graph_replicated_hits`.
- Graph cache: `graph_cache_hits`, `graph_cache_misses`,
  `graph_cache_avoided_reads`, `graph_cache_resident_bytes`,
  `graph_cache_entries`, `graph_cache_build_page_reads`.
- Refinement: `refinement_bound`, `refinement_reads`,
  `refinement_already_exact`, `refinement_exactified`,
  `deferred_exact_candidates`.
- Scheduler: `scheduler_policy_limit_observed`, `scheduler_pending_max`,
  `scheduler_ready_unexpanded_max`, `scheduler_limit_hits`,
  `max_inflight_reads`, `max_beam_width`, `beam_width_increases`.

## Caveats

- Current validation is correctness evidence on toy/small local workloads only.
  Do not claim SSD or large-dataset performance from the local WSL tests.
- The M4 scheduler work exposes comparable policy controls and measurements,
  but it is still a simplified PipeANN-style loop, not a full queue refactor
  into separate submit/loading/ready executor components.
- `scheduler_policy_limit` is an experiment knob. A value of `0` means use the
  policy-derived limit.
- Cache policy is only meaningful when `graph_cache_budget_bytes` is non-zero.
  Sweeping cache policy with zero cache budget is useful for matrix symmetry but
  should be interpreted as a no-cache run.

## Suggested Next Work

1. Run a small real-dataset smoke, such as SIFT-scale if available, and keep the
   exported experiment directory under `results/`.
2. Add p50/p95/p99 latency export. Current bench reports average latency and
   total elapsed time, which is not enough for tail-latency claims.
3. Split scheduler accounting further into submit, wait, ready-queue, expansion,
   and refinement time if the next objective is a stronger PipeANN execution
   comparison.
4. Start M5 only after preserving M1-M4 counters across any I/O backend changes.
