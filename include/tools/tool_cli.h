#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "integrations/pipeann_layout_activate.h"
#include "integrations/pipeann_parity_bench.h"
#include "integrations/pipeann_parity_searcher.h"

namespace hybrid {

enum class QueryInputMode {
  kText = 0,
  kFvecs = 1,
  kBvecs = 2,
  kBin = 3,
};

struct BenchResultEntry {
  uint32_t id = 0;
  float distance = 0.0f;
};

struct BenchToolConfig {
  std::string index_prefix;
  std::string approx_path;
  std::vector<std::vector<float>> queries;
  std::vector<std::vector<uint32_t>> ground_truth_ids;
  uint32_t top_k = 10;
  uint32_t beam_width = 8;
  uint32_t l_search = 20;
  pipeann_parity::PipeannSearchMode pipeann_mode = pipeann_parity::PipeannSearchMode::kPipe;
  pipeann_parity::PipeannLayout pipeann_layout = pipeann_parity::PipeannLayout::kGorgeous;
  std::string gp_partition_path;
  uint32_t mem_l = 0;
  uint32_t recall_at_k = 0;
  uint32_t num_threads = 1;
};

struct BenchToolSummary {
  std::string index_prefix;
  uint32_t num_queries = 0;
  uint32_t num_threads = 1;
  uint32_t top_k = 0;
  uint32_t beam_width = 0;
  uint32_t l_search = 0;
  double elapsed_ms = 0.0;
  double mean_latency_us = 0.0;
  double p99_latency_us = 0.0;
  double average_latency_ms = 0.0;
  double p99_latency_ms = 0.0;
  double qps = 0.0;
  pipeann_parity::PipeannSearchMode pipeann_mode = pipeann_parity::PipeannSearchMode::kPipe;
  pipeann_parity::PipeannLayout pipeann_layout = pipeann_parity::PipeannLayout::kGorgeous;
  uint32_t mem_l = 0;
  double mean_hops = 0.0;
  double mean_ios = 0.0;
  double unique_pg_io = 0.0;
  double dup_pg_hit = 0.0;
  double polls = 0.0;
  bool has_recall = false;
  double average_recall = 0.0;
  std::vector<BenchResultEntry> first_query_results;
};

struct BenchSweepConfig {
  BenchToolConfig base_config;
  std::vector<uint32_t> beam_widths;
  std::vector<uint32_t> l_search_values;
  std::vector<uint32_t> thread_counts;
  std::vector<pipeann_parity::PipeannSearchMode> pipeann_modes;
  std::vector<pipeann_parity::PipeannLayout> pipeann_layouts;
};

struct BenchSweepSummary {
  std::vector<BenchToolSummary> runs;
};

struct BenchComparisonRow {
  BenchToolSummary baseline;
  BenchToolSummary candidate;
  double delta_elapsed_ms = 0.0;
  double delta_mean_latency_us = 0.0;
  double delta_p99_latency_us = 0.0;
  double delta_qps = 0.0;
  double delta_average_recall = 0.0;
  double delta_mean_hops = 0.0;
  double delta_mean_ios = 0.0;
  double delta_unique_pg_io = 0.0;
  double delta_dup_pg_hit = 0.0;
  double delta_polls = 0.0;
};

struct BenchComparisonSummary {
  std::string baseline_label;
  std::string candidate_label;
  std::vector<BenchComparisonRow> rows;
};

std::vector<float> ParseFloatVector(const std::string &text);
std::vector<uint32_t> ParseUint32List(const std::string &text);
std::vector<std::vector<float>> LoadTextVectors(const std::string &path);
std::vector<std::vector<float>> LoadFvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBinVectors(const std::string &path);
std::vector<std::vector<float>> LoadQueryVectors(const std::string &path, QueryInputMode mode);
std::vector<std::vector<uint32_t>> LoadGroundTruthIds(const std::string &path);
std::vector<std::vector<uint32_t>> GenerateGroundTruthIds(const std::string &index_prefix,
                                                          const std::string &base_data_path,
                                                          const std::vector<std::vector<float>> &queries,
                                                          uint32_t top_k);
void WriteGroundTruthIds(const std::string &path, const std::vector<std::vector<uint32_t>> &ground_truth_ids);
std::string CreateExperimentDirectory(const std::string &root_directory,
                                      const BenchSweepConfig &config,
                                      const std::string &experiment_name);
BenchToolSummary RunBenchTool(const BenchToolConfig &config);
BenchToolSummary RunPipeannParityBenchTool(const BenchToolConfig &config);
BenchSweepSummary RunBenchSweep(const BenchSweepConfig &config);
void ExportBenchSummariesTsv(const std::string &path, const std::vector<BenchToolSummary> &summaries);
void ExportBenchExperiment(const std::string &directory,
                           const BenchSweepConfig &config,
                           const BenchSweepSummary &summary);
void MergeBenchSummaryTsvFiles(const std::string &output_path, const std::vector<std::string> &input_paths);
std::vector<BenchToolSummary> LoadBenchSummariesTsv(const std::string &path);
BenchComparisonSummary CompareBenchSummaries(const std::string &baseline_label,
                                             const std::vector<BenchToolSummary> &baseline,
                                             const std::string &candidate_label,
                                             const std::vector<BenchToolSummary> &candidate);
void ExportBenchComparisonMarkdown(const std::string &path, const BenchComparisonSummary &summary);
void ExportBenchComparisonTsv(const std::string &path, const BenchComparisonSummary &summary);

}  // namespace hybrid
