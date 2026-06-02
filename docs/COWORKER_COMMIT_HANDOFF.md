# Coworker Commit Review and Handoff

Date: 2026-06-02

Remote commit reviewed: `origin/main` at `30fe084` (`increase multi-thread concurrency`).

## Decision

Do not merge or cherry-pick the remote commit wholesale.

The useful concurrency ideas were selectively ported into the current tree, but the remote
commit also:

- deletes `CMakeLists.txt`;
- deletes current source and test files;
- commits many `build/` binaries and generated test artifacts;
- contains parity code that references stale `BenchToolConfig` fields;
- does not include the parity test source files shown by its build artifacts.

## Ported Locally

- Bench query-level OpenMP execution via `BenchToolConfig::num_threads`.
- CLI support for `--threads` and `--thread_counts`.
- `threads` included in summary keys, TSV export/import, experiment manifests, and compare output.
- Shared `GraphAdjacencyCache` across query workers.
- Shared loaded PipeANN PQ object across query workers; per-query distance tables remain per session.
- Thread-local `IndexReader` reload for `NativeGorgeousIndex` when `num_threads > 1`.
- Explicit-resource search overload that accepts a per-query page reader, graph cache, and shared PQ.
- Linux AIO context refcounting per reader instance for explicit Linux AIO usage.
- Per-query latency summaries: `mean_latency_us`, `p95_latency_us`, `p99_latency_us`.

Best-effort page reads intentionally remain on the async pread backend. The Linux AIO path still
requires O_DIRECT-compatible aligned buffers, while current `SearchSession` page buffers are
plain `std::vector<char>`.

## Verified

WSL environment: `Ubuntu-OSLab-Recovered`

```bash
cmake --build build_wsl -j2
ctest --test-dir build_wsl --output-on-failure
```

Result: 4/4 tests passed.

Small thread sweep smoke:

```bash
./pipeann_gorgeous_bench \
  --index test_tool_cli_data/tool_cli_graph_relayout.index \
  --approx test_tool_cli_data/tool_cli.bin \
  --queries thread_smoke_queries.txt \
  --query_format text \
  --top_k 4 \
  --beam_widths 3 \
  --l_search_values 8 \
  --approx_kinds pq \
  --pq_codebook test_tool_cli_data/tool_cli_pq_pivots.bin \
  --pq_codes test_tool_cli_data/tool_cli_pq_compressed.bin \
  --thread_counts 1,2 \
  --export thread_smoke.tsv
```

Result: 2 runs exported. `threads`, `mean_latency_us`, `p95_latency_us`, and
`p99_latency_us` are present in the TSV.

## Next Good Work

1. Clean PipeANN parity baseline

   Reimplement this independently instead of cherry-picking the remote version. The goal is a
   small tool that loads the vendored PipeANN `SSDIndex`, runs `beam_search`, `page_search`,
   or `pipe_search`, and exports comparable QPS, recall, mean latency, p95/p99 latency, hops,
   and IO counters.

2. Query buffer pool

   Useful for reducing per-query allocation churn, but it should wait until buffer ownership is
   clear. If Linux AIO is enabled for the default path, page buffers must be aligned.

3. Native Gorgeous reader reuse

   The current multi-thread fallback reopens a native index per worker. A better long-term path
   is thread-local vector streams/cache state inside the native reader, with tests covering
   concurrent search on native and graphrep indexes.

4. Larger benchmark pass

   Run `thread_counts=1,2,4,8` on a real small dataset and compare both QPS and p99 latency.
   QPS alone can hide tail-latency regressions.

## Known Dirty State

`README.md` had an unrelated pre-existing deletion of the Graph Cache options section. It was
left untouched by this port.
