# PipeANN x Gorgeous Research and Implementation Plan

## 1. Research Position

This project should not be framed as a loose concatenation of two systems. The
right target is a disk-resident graph ANN search engine whose query executor is
PipeANN-like and whose data/cache layout is Gorgeous-like.

The fused hypothesis is:

- PipeANN reduces critical-path latency by decoupling I/O issue, I/O completion,
  and graph expansion.
- Gorgeous reduces the number of disk reads by prioritizing adjacency lists over
  exact vectors in both memory cache and disk page layout.
- The combined system should schedule reads with a PipeANN-style pipeline while
  treating graph adjacency as the primary cacheable/search-driving object and
  exact vectors as a later refinement resource.

## 2. Literature Takeaways

Primary sources checked:

- PipeANN, OSDI 2025:
  https://www.usenix.org/system/files/osdi25-guo.pdf
- PipeANN source:
  https://github.com/thustorage/PipeANN
- Gorgeous paper:
  https://arxiv.org/abs/2508.15290
- Gorgeous source:
  https://github.com/yinpeiqi/Gorgeous
- DiskANN baseline:
  https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/
- Starling:
  https://arxiv.org/abs/2401.02116
- SPANN:
  https://arxiv.org/abs/2111.08566
- GoVector:
  https://arxiv.org/abs/2508.15694
- FreshDiskANN:
  https://arxiv.org/abs/2105.09613

Key implications for this repo:

- DiskANN/Vamana is the baseline execution and storage model: compressed vectors
  stay in memory for approximate distance, while full vectors and graph records
  are disk-resident.
- PipeANN's core contribution is not simply "async I/O"; it is speculative I/O
  issue from the current candidate pool, polling completions independently of
  graph expansion, dynamic pipeline width, and bounding read-but-unexpanded
  waste.
- Gorgeous's core contribution is not simply "reordered pages"; it is separating
  adjacency-list demand from exact-vector demand, then using graph-prioritized
  memory cache and graph-replicated disk blocks to reduce traversal I/O.
- Starling and GoVector are useful comparison points for locality and caching,
  but their ideas should be treated as evaluation baselines or optional variants,
  not as the central fused design.
- SPANN is a cluster/inverted-index baseline. It is useful for literature context,
  but less directly aligned with the current graph-index codebase.

## 3. Current Code Evidence

The README says the current build chain is:

```text
vectors -> PipeANN disk build -> Gorgeous partition -> Gorgeous relayout -> PQ artifacts
```

The code matches this at a high level:

- `src/tools/build_pipeline.cpp` runs `BuildPipeannDiskIndex`, loads the flat
  PipeANN graph, calls `BuildGorgeousGraphReplicaIndex`, then builds PipeANN PQ
  artifacts.
- `src/integrations/pipeann_builder.cpp` calls vendored PipeANN build and PQ
  build code.
- `src/integrations/gorgeous_original.cpp` writes a Gorgeous-compatible
  partition input, invokes the graph replica partitioner, normalizes the
  partition file, and writes graph-replicated pages.
- `src/gorgeous_layout.cpp` supports native Gorgeous loading, page views, node
  views, sidecar validation, and on-demand full-precision payload reads.
- `src/search/search_session.cpp` already has a query session with candidate
  pool, page read submission, completion polling, resident page registration,
  graph expansion, approximate distance evaluation, and final refinement.
- `src/io/page_reader.cpp` provides async `pread` and PipeANN-style Linux AIO
  backends behind `IPageReader`.
- `src/quant/approx_distance.cpp` consumes PipeANN PQ pivots/compressed codes
  and also supports a full-precision fallback path.

Existing tests cover meaningful behavior: search, PQ, native relayout loading,
filter/range variants, build workflow, CLI utilities, benchmark export, and
integration modules. The current WSL validation path is
`Ubuntu-OSLab-Recovered` at `/mnt/d/OS比赛读优化/PG_ANN`, using a separate
`build_wsl` directory so the older `build` cache is not reused across source
paths.

## 4. Main Gaps

### 4.1 Gorgeous Semantics

The current search uses graph-replicated pages, but it does not yet implement a
true Gorgeous graph-prioritized memory cache. Resident pages in the query buffer
can avoid repeated reads during one query, but there is no persistent
adjacency-only cache with a memory budget, no cache admission policy, and no
separate graph-cache hit metric.

The current refinement path exactifies candidates after traversal, but the
search/refinement split is still too implicit. Gorgeous's two-stage model should
be made explicit: approximate graph traversal first, then batched exact-vector
rerank for top approximate candidates.

### 4.2 PipeANN Semantics

The current session issues async page reads and polls completions, but it is
still a simplified PipeSearch/PipeANN kernel. The missing pieces are:

- stronger query scratch reuse across queries;
- clearer separation of I/O submit, completion, ready queue, and expansion;
- a principled bound on ongoing plus read-but-unexpanded candidates;
- a dynamic pipeline-width policy that is parameterized and measurable;
- stronger multi-query execution and tail-latency instrumentation.

### 4.3 I/O Backend

The current default is reliable async `pread`; Linux AIO exists but is not the
default. There is no `io_uring` backend yet, and I/O stats do not distinguish
submit latency, wait latency, completion latency, ready-queue time, and CPU
expansion time well enough for research claims.

### 4.4 Quantization

PipeANN PQ artifacts are consumed, which is a good baseline. The remaining work
is to turn `PQ` into one implementation of a broader approximate-distance
interface that can later host RaBitQ or other quantizers without changing the
search loop.

### 4.5 Experiments

The toolchain exports benchmark summaries, but it needs research-grade fields:
reads/query, exact-vector reads/query, graph-cache hits, graph-replicated page
hits, in-flight depth, ready-queue depth, I/O wait time, CPU expansion time,
refinement reads, and recall/latency/QPS under identical parameter sweeps.

## 5. Implementation Roadmap

### M0: Reproducible Local Baseline

- Make the local WSL path and build/test command unambiguous.
- Run `cmake`, build all targets, and run `ctest --output-on-failure`.
- Keep this as correctness evidence only; do not claim performance from the
  small local VM.

Exit criteria:

- All current tests pass in WSL or failures are classified as environment,
  dependency, or logic issues.
- README/test commands match the actual local workflow.

### M1: Instrumentation Before Optimization

- Extend `SearchStats` with graph-cache hits, page-resident hits,
  page-replicated adjacency hits, exact-vector reads, refinement reads,
  in-flight depth samples, ready-queue depth samples, submit/completion/wait CPU
  timings, and dynamic beam-width trace.
- Export these fields through search and benchmark CLI.
- Add a small deterministic test that proves counters move in expected
  directions.

Exit criteria:

- We can tell whether a speedup comes from fewer I/Os, better overlap, fewer
  exact evaluations, or measurement noise.

### M2: Gorgeous Graph-Prioritized Cache

- Add a memory-budgeted adjacency-list cache independent from exact vectors.
- Start with deterministic cache construction from entry-neighborhood/BFS and
  optional frequency files; later add query-log or warmup-based admission.
- Teach `SearchSession` to expand a candidate directly from graph cache without
  disk I/O.
- Track graph-cache hit/miss and avoided page-read counts.

Exit criteria:

- Turning graph cache on/off changes read counts in expected directions.
- Cache memory is expressed in bytes and can be swept.

### M3: Explicit Two-Stage Search/Refinement

- Separate approximate traversal state from exact result state.
- Add parameters for refinement candidate count or refinement ratio.
- Batch exact-vector reads for top approximate candidates not already available
  from loaded pages or payload cache.
- Keep final top-k ranked by exact distance.

Exit criteria:

- We can sweep refinement ratio/count and observe recall/latency/read tradeoffs.

### M4: PipeANN-Style Scheduler Refinement

- Refactor the scheduler into submit queue, loading queue, ready queue, and
  expansion loop.
- Implement a measurable bound on missed neighbors:
  `inflight + ready_unexpanded <= policy_limit`.
- Make dynamic width policy configurable and exported.
- Keep the default conservative for local testing; reserve aggressive settings
  for SSD experiments.

Exit criteria:

- Different scheduler policies can be compared with the same cache/layout and
  quantizer settings.

### M5: I/O Backend Upgrade

- Keep async `pread` as portable fallback.
- Add an `io_uring` backend under Linux/WSL when dependencies are available.
- Preserve aligned and unaligned offset semantics in `IORequest`.
- Export backend name and backend-specific counters.

Exit criteria:

- The search loop is backend-agnostic.
- `pread`, Linux AIO, and `io_uring` can be selected without touching search
  logic.

### M6: Quantizer Abstraction and RaBitQ Path

- Convert PQ/full into a cleaner `IApproximateDistanceComputer` hierarchy with
  metadata, batch distance, and per-query scratch ownership.
- Add tests for default artifact inference and explicit override paths.
- Evaluate RaBitQ only after M1-M4 counters are stable.

Exit criteria:

- Adding a quantizer requires no changes to candidate scheduling or page
  expansion.

### M7: Experiment Matrix

Local WSL:

- toy and small synthetic tests for correctness;
- optional SIFT1M smoke if resources allow;
- no strong performance claims.

Remote hardware:

- DiskANN/PipeANN-like baseline;
- current hybrid baseline;
- graph cache on/off;
- graph replicated layout on/off or page_nodes sweep;
- dynamic pipeline policy sweep;
- refinement ratio/count sweep;
- PQ/full and later RaBitQ comparison.

Required metrics:

- recall@k;
- mean/p50/p95/p99 latency;
- QPS under fixed thread count;
- reads/query and bytes/query;
- exact-vector reads/query;
- graph-cache hit rate;
- replicated-page adjacency hit rate;
- I/O overlap ratio;
- CPU vs I/O time split.

## 6. Recommended Next Step

Start with M0 and M1. The current code already has enough functionality that
blindly adding caches or scheduler heuristics would make results hard to
interpret. Instrumentation first gives a defensible path to show that each
subsequent change improves the intended mechanism.

After M1, implement M2 before M4. Gorgeous's graph cache should reduce I/O
pressure; only then does it make sense to tune PipeANN-style scheduling against
the new, lower-I/O workload.

## 7. Implementation Notes

Current local progress:

- M1 instrumentation has been added to `SearchStats`, search/bench CLI output,
  experiment TSV export, and comparison TSV export. The new fields distinguish
  page reads, bytes read, replicated-page hits, refinement reads, exact distance
  source, dynamic beam behavior, and cache-related counters.
- M2 now has an adjacency-only `GraphAdjacencyCache`. A non-zero
  `SearchConfig::graph_cache_budget_bytes` enables cached graph expansion
  without submitting a page read. `graph_cache_policy` selects either
  `entry_bfs` or `page_layout`, so cache budget and physical-layout order can be
  studied independently.
- M3 exposes `refine_k`, `refine_ratio`, and
  `defer_exact_until_refinement`. This makes approximate traversal and exact
  reranking separate experiment dimensions while preserving the old defaults.
- M4 adds scheduler-level experiment controls:
  `scheduler_policy`, `scheduler_policy_limit`, and `dynamic_beam_policy`.
  Search now records the observed pending-work limit, max
  `inflight + ready_unexpanded`, ready-unexpanded max, and limit-hit count.
- Bench supports graph-cache policy, cache budget, refinement width/ratio,
  deferred-exact, scheduler policy, scheduler limit, and dynamic-beam sweeps
  together with `beam_width`, `l_search`, and `approx_kind`.

Verification status:

- Configured successfully with
  `wsl -d Ubuntu-OSLab-Recovered -- bash -lc "cmake -S . -B build_wsl"`.
- Built the focused search/bench test targets, then built the remaining local
  test targets.
- Passed `ctest --test-dir build_wsl --output-on-failure`: 4/4 tests passed.
  This is correctness evidence on toy/small workloads only; it is not a remote
  SSD performance claim.
