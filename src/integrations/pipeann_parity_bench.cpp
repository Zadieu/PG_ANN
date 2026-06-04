#include "integrations/pipeann_parity_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace hybrid::pipeann_parity {

namespace {

double MeanLatencyUs(const std::vector<uint64_t> &latency_us) {
  if (latency_us.empty()) {
    return 0.0;
  }
  const double sum =
      std::accumulate(latency_us.begin(), latency_us.end(), 0.0, [](double acc, uint64_t value) {
        return acc + static_cast<double>(value);
      });
  return sum / static_cast<double>(latency_us.size());
}

double P99LatencyUs(std::vector<uint64_t> latency_us) {
  if (latency_us.empty()) {
    return 0.0;
  }
  std::sort(latency_us.begin(), latency_us.end());
  const size_t rank =
      static_cast<size_t>(std::ceil(0.99 * static_cast<double>(latency_us.size()))) - 1;
  return static_cast<double>(latency_us[std::min(rank, latency_us.size() - 1)]);
}

double MeanStats(const std::vector<pipeann::QueryStats> &stats,
                 const std::function<double(const pipeann::QueryStats &)> &member) {
  if (stats.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const auto &stat : stats) {
    sum += member(stat);
  }
  return sum / static_cast<double>(stats.size());
}

void ValidateBenchConfig(const PipeannParityBenchConfig &config, uint32_t expected_dim) {
  if (config.num_threads == 0) {
    throw std::runtime_error("PipeANN parity benchmark num_threads must be positive");
  }
  if (config.search.top_k == 0 || config.search.l_search == 0 || config.search.beam_width == 0) {
    throw std::runtime_error("PipeANN parity benchmark search config must be positive");
  }
  if (config.recall_at_k == 0) {
    throw std::runtime_error("PipeANN parity benchmark recall_at_k must be positive");
  }
  if (config.search.top_k < config.recall_at_k) {
    throw std::runtime_error("PipeANN parity benchmark top_k must be >= recall_at_k");
  }
  if (config.queries.size() != config.ground_truth_ids.size()) {
    throw std::runtime_error("PipeANN parity benchmark queries and ground_truth_ids must have the same size");
  }

  for (size_t i = 0; i < config.queries.size(); ++i) {
    if (config.queries[i].size() != expected_dim) {
      throw std::runtime_error("PipeANN parity benchmark query dimension does not match index dimension");
    }
    if (config.ground_truth_ids[i].size() < config.recall_at_k) {
      throw std::runtime_error("PipeANN parity benchmark ground truth width is smaller than recall_at_k");
    }
  }
}

double CalculateRecallAtK(const PipeannParityBenchConfig &config,
                          const std::vector<uint32_t> &flat_results) {
  if (config.queries.empty()) {
    return 0.0;
  }

  const size_t gt_width = config.ground_truth_ids.front().size();
  for (const auto &row : config.ground_truth_ids) {
    if (row.size() != gt_width) {
      throw std::runtime_error("PipeANN parity benchmark ground truth rows must have a consistent width");
    }
  }

  std::vector<unsigned> flat_ground_truth(config.queries.size() * gt_width, 0);
  std::vector<unsigned> flat_results_unsigned(flat_results.size(), 0);
  for (size_t i = 0; i < config.ground_truth_ids.size(); ++i) {
    for (size_t j = 0; j < gt_width; ++j) {
      flat_ground_truth[i * gt_width + j] = static_cast<unsigned>(config.ground_truth_ids[i][j]);
    }
  }
  for (size_t i = 0; i < flat_results.size(); ++i) {
    flat_results_unsigned[i] = static_cast<unsigned>(flat_results[i]);
  }

  return pipeann::calculate_recall(static_cast<unsigned>(config.queries.size()),
                                   flat_ground_truth.data(),
                                   nullptr,
                                   static_cast<unsigned>(gt_width),
                                   flat_results_unsigned.data(),
                                   config.search.top_k,
                                   config.recall_at_k);
}

const float *ResolveQueryBase(const PipeannParityBenchConfig &config,
                              uint32_t dim,
                              std::vector<float> &owned_flat_queries) {
  const size_t n = config.queries.size();
  const size_t expected_bytes = n * static_cast<size_t>(dim);
  if (config.flat_query_data.size() == expected_bytes) {
    return config.flat_query_data.data();
  }

  owned_flat_queries.resize(expected_bytes);
  float *dst = owned_flat_queries.data();
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(dst + i * dim, config.queries[i].data(), static_cast<size_t>(dim) * sizeof(float));
  }
  return owned_flat_queries.data();
}

}  // namespace

void EnsureFlatQueryData(PipeannParityBenchConfig &config, uint32_t dim) {
  const size_t expected_bytes = config.queries.size() * static_cast<size_t>(dim);
  if (config.flat_query_data.size() == expected_bytes) {
    return;
  }
  config.flat_query_data.resize(expected_bytes);
  for (size_t i = 0; i < config.queries.size(); ++i) {
    std::memcpy(config.flat_query_data.data() + i * dim,
                config.queries[i].data(),
                static_cast<size_t>(dim) * sizeof(float));
  }
}

PipeannParityLResult RunPipeannParityQueries(PipeannParityIndex &index,
                                             const PipeannParityBenchConfig &config) {
  const uint32_t dim = static_cast<uint32_t>(index.index().meta_.data_dim);
  ValidateBenchConfig(config, dim);

  PipeannParityLResult result;
  result.L = config.search.l_search;

  const size_t n = config.queries.size();
  if (n == 0) {
    return result;
  }

  std::vector<float> owned_flat_queries;
  const float *query_base = ResolveQueryBase(config, dim, owned_flat_queries);
  const size_t top_k = static_cast<size_t>(config.search.top_k);

  std::vector<pipeann::QueryStats> stats(n);
  std::vector<uint64_t> latency_us(n, 0);
  std::vector<uint32_t> flat_results(n * top_k, 0);

  omp_set_num_threads(static_cast<int>(config.num_threads));
  const auto started = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(static_cast<int>(config.num_threads))
  {
    std::vector<uint32_t> ids(top_k);
    std::vector<float> distances(top_k);
#pragma omp for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
      const auto query_started = std::chrono::high_resolution_clock::now();
      pipeann::QueryStats query_stats{};
      SearchInto(index,
                 query_base + static_cast<size_t>(i) * dim,
                 dim,
                 config.search,
                 ids.data(),
                 distances.data(),
                 &query_stats);
      const uint64_t total_us = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
                                                                query_started)
              .count());
      query_stats.total_us = static_cast<double>(total_us);
      stats[static_cast<size_t>(i)] = query_stats;
      latency_us[static_cast<size_t>(i)] = total_us;
      std::memcpy(flat_results.data() + static_cast<size_t>(i) * top_k,
                  ids.data(),
                  top_k * sizeof(uint32_t));
    }
  }
  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count();

  result.qps = elapsed_s > 0.0 ? static_cast<double>(n) / elapsed_s : 0.0;
  result.mean_latency_us = MeanLatencyUs(latency_us);
  result.p99_latency_us = P99LatencyUs(latency_us);
  result.mean_hops = MeanStats(stats, [](const pipeann::QueryStats &s) { return s.n_hops; });
  result.mean_ios = MeanStats(stats, [](const pipeann::QueryStats &s) { return s.n_ios; });
  // Vendored QueryStats does not expose pipe-mode page dedup counters.
  result.mean_unique_pg_io = 0.0;
  result.mean_dup_pg_hit = 0.0;
  result.mean_polls = 0.0;
  result.recall_at_k = CalculateRecallAtK(config, flat_results);
  return result;
}

std::vector<PipeannParityLResult> RunPipeannParitySweep(
    PipeannParityIndex &index,
    const PipeannParityBenchConfig &base,
    const std::vector<uint32_t> &l_values) {
  const uint32_t dim = static_cast<uint32_t>(index.index().meta_.data_dim);
  PipeannParityBenchConfig config = base;
  EnsureFlatQueryData(config, dim);

  std::vector<PipeannParityLResult> results;
  results.reserve(l_values.size());
  for (uint32_t l_value : l_values) {
    config.search.l_search = l_value;
    results.push_back(RunPipeannParityQueries(index, config));
  }
  return results;
}

}  // namespace hybrid::pipeann_parity
