#pragma once

#include <cstdint>
#include <vector>

#include "integrations/pipeann_parity_index.h"
#include "integrations/pipeann_parity_searcher.h"

namespace hybrid::pipeann_parity {

struct PipeannParityBenchConfig {
  PipeannParityIndexConfig index;
  PipeannParitySearchConfig search;
  uint32_t num_threads = 1;
  std::vector<std::vector<float>> queries;
  // Optional contiguous query layout (queries.size() * dim floats). When populated,
  // the benchmark hot path avoids nested-vector indexing and extra copies.
  std::vector<float> flat_query_data;
  std::vector<std::vector<uint32_t>> ground_truth_ids;
  uint32_t recall_at_k = 10;
};

void EnsureFlatQueryData(PipeannParityBenchConfig &config, uint32_t dim);

struct PipeannParityLResult {
  uint32_t L = 0;
  double qps = 0;
  double mean_latency_us = 0;
  double p99_latency_us = 0;
  double mean_hops = 0;
  double mean_ios = 0;
  double mean_unique_pg_io = 0;
  double mean_dup_pg_hit = 0;
  double mean_polls = 0;
  double recall_at_k = 0;
};

struct PipeannParityBenchSummary {
  std::vector<PipeannParityLResult> per_l;
};

PipeannParityLResult RunPipeannParityQueries(PipeannParityIndex &index,
                                             const PipeannParityBenchConfig &config);

std::vector<PipeannParityLResult> RunPipeannParitySweep(
    PipeannParityIndex &index,
    const PipeannParityBenchConfig &base,
    const std::vector<uint32_t> &l_values);

}  // namespace hybrid::pipeann_parity
