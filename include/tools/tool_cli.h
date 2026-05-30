#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gorgeous_layout.h"
#include "integrations/pipeann_builder.h"
#include "pipeline_search.h"
#include "quant/approx_distance.h"

namespace hybrid {

enum class QueryInputMode {
  kText = 0,
  kFvecs = 1,
  kBvecs = 2,
  kBin = 3,
};

struct SearchToolConfig {
  std::string index_path;
  std::string approx_path;
  std::string pq_codebook_path;
  std::string pq_codes_path;
  std::vector<float> query;
  SearchConfig search_config{};
  ApproxDistanceKind approx_kind = ApproxDistanceKind::kProductQuantization;
};

struct SearchToolSummary {
  std::string approx_backend_name;
  std::vector<SearchResult> results;
  SearchStats stats{};
};

struct BenchToolConfig {
  std::string index_path;
  std::string approx_path;
  std::string pq_codebook_path;
  std::string pq_codes_path;
  std::vector<std::vector<float>> queries;
  std::vector<std::vector<uint32_t>> ground_truth_ids;
  SearchConfig search_config{};
  ApproxDistanceKind approx_kind = ApproxDistanceKind::kProductQuantization;
  uint32_t recall_at_k = 0;
};

struct BenchToolSummary {
  std::string approx_backend_name;
  uint32_t num_queries = 0;
  double elapsed_ms = 0.0;
  double average_latency_ms = 0.0;
  double qps = 0.0;
  SearchStats aggregate_stats{};
  std::vector<SearchResult> first_query_results;
  SearchConfig search_config{};
  ApproxDistanceKind approx_kind = ApproxDistanceKind::kProductQuantization;
  bool has_recall = false;
  double average_recall = 0.0;
};

struct BenchSweepConfig {
  BenchToolConfig base_config;
  std::vector<uint32_t> beam_widths;
  std::vector<uint32_t> l_search_values;
  std::vector<uint64_t> graph_cache_budget_bytes_values;
  std::vector<GraphCacheBuildPolicy> graph_cache_policies;
  std::vector<uint32_t> refine_k_values;
  std::vector<float> refine_ratio_values;
  std::vector<uint8_t> defer_exact_until_refinement_values;
  std::vector<SearchConfig::SchedulerPolicy> scheduler_policies;
  std::vector<uint32_t> scheduler_policy_limit_values;
  std::vector<SearchConfig::DynamicBeamPolicy> dynamic_beam_policies;
  std::vector<ApproxDistanceKind> approx_kinds;
};

struct BenchSweepSummary {
  std::vector<BenchToolSummary> runs;
};

struct BenchComparisonRow {
  BenchToolSummary baseline;
  BenchToolSummary candidate;
  double delta_elapsed_ms = 0.0;
  double delta_average_latency_ms = 0.0;
  double delta_qps = 0.0;
  double delta_average_recall = 0.0;
  int64_t delta_async_reads = 0;
  int64_t delta_pages_completed = 0;
  int64_t delta_resident_expansions = 0;
  int64_t delta_bytes_read = 0;
  int64_t delta_page_resident_hits = 0;
  int64_t delta_graph_replicated_hits = 0;
  int64_t delta_graph_cache_hits = 0;
  int64_t delta_graph_cache_misses = 0;
  int64_t delta_graph_cache_expansions = 0;
  int64_t delta_graph_cache_avoided_reads = 0;
  int64_t delta_graph_cache_resident_bytes = 0;
  int64_t delta_graph_cache_entries = 0;
  int64_t delta_graph_cache_build_page_reads = 0;
  int64_t delta_exact_from_page = 0;
  int64_t delta_exact_from_payload = 0;
  int64_t delta_refinement_candidates = 0;
  int64_t delta_refinement_reads = 0;
  int64_t delta_approximate_candidates = 0;
  int64_t delta_refinement_bound = 0;
  int64_t delta_refinement_already_exact = 0;
  int64_t delta_refinement_exactified = 0;
  int64_t delta_deferred_exact_candidates = 0;
  int64_t delta_read_hits_in_pool = 0;
  int64_t delta_read_waste_out_of_pool = 0;
  int64_t delta_max_inflight_reads = 0;
  int64_t delta_max_beam_width = 0;
  int64_t delta_beam_width_increases = 0;
  int64_t delta_scheduler_pending_max = 0;
  int64_t delta_scheduler_ready_unexpanded_max = 0;
  int64_t delta_scheduler_limit_hits = 0;
  int64_t delta_poll_calls = 0;
  int64_t delta_drain_calls = 0;
  int64_t delta_approx_evals = 0;
  int64_t delta_exact_evals = 0;
  int64_t delta_n_ios = 0;
  int64_t delta_n_cmps = 0;
  int64_t delta_n_hops = 0;
  int64_t delta_cpu_us = 0;
  int64_t delta_io_us = 0;
};

struct BenchComparisonSummary {
  std::string baseline_label;
  std::string candidate_label;
  std::vector<BenchComparisonRow> rows;
};

struct InspectToolConfig {
  std::string index_path;
  std::string approx_path;
  std::string pq_codebook_path;
  std::string pq_codes_path;
  bool inspect_page = false;
  uint32_t page_id = 0;
};

struct InspectPageSummary {
  uint32_t page_id = 0;
  uint32_t target_id = 0;
  std::vector<uint32_t> layout;
  std::vector<uint16_t> nnbrs;
  std::vector<uint16_t> n_dense_nbrs;
};

struct InspectToolSummary {
  SearchIndexMetadata index_metadata{};
  IndexStorageFormat storage_format = IndexStorageFormat::kProjectGraphReplicated;
  bool has_project_metadata = false;
  GraphReplicatedMetadata project_metadata{};
  bool has_native_metadata = false;
  gorgeous_integration::DiskIndexMetadata native_metadata{};
  NativePayloadInfo native_payload{};
  std::vector<uint64_t> native_page_boundaries;
  uint64_t index_file_size = 0;
  uint64_t approx_file_size = 0;
  bool has_partition = false;
  bool has_reorder = false;
  uint64_t reorder_entries = 0;
  bool has_page = false;
  InspectPageSummary page;
  bool has_pipeann_refine = false;
  pipeann_integration::PipeannRefineSidecarHeader pipeann_refine_header{};
  uint32_t pipeann_refine_nodes = 0;
  uint32_t pipeann_refine_edges = 0;
  bool has_pipeann_refine_nodes = false;
  std::vector<uint32_t> page_refine_neighbors;
  std::vector<uint16_t> page_refine_ehs;
  uint32_t page_refine_degree = 0;
  uint16_t page_refine_max_eh = 0;
  bool has_pq = false;
  ProductQuantizationMetadata pq_metadata{};
  uint64_t pq_codebook_file_size = 0;
  uint64_t pq_codes_file_size = 0;
};

std::vector<float> ParseFloatVector(const std::string &text);
std::vector<uint32_t> ParseUint32List(const std::string &text);
std::vector<uint64_t> ParseUint64List(const std::string &text);
std::vector<float> ParseFloatList(const std::string &text);
std::vector<uint8_t> ParseBoolList(const std::string &text);
std::vector<ApproxDistanceKind> ParseApproxKindList(const std::string &text);
std::vector<GraphCacheBuildPolicy> ParseGraphCachePolicyList(const std::string &text);
std::vector<SearchConfig::SchedulerPolicy> ParseSchedulerPolicyList(const std::string &text);
std::vector<SearchConfig::DynamicBeamPolicy> ParseDynamicBeamPolicyList(const std::string &text);
std::vector<std::vector<float>> LoadQueryVectors(const std::string &path, QueryInputMode mode);
std::vector<std::vector<uint32_t>> LoadGroundTruthIds(const std::string &path);
std::vector<std::vector<uint32_t>> GenerateGroundTruthIds(const std::string &index_path,
                                                          const std::string &approx_path,
                                                          const std::vector<std::vector<float>> &queries,
                                                          uint32_t top_k);
void WriteGroundTruthIds(const std::string &path, const std::vector<std::vector<uint32_t>> &ground_truth_ids);
std::string CreateExperimentDirectory(const std::string &root_directory,
                                      const BenchSweepConfig &config,
                                      const std::string &experiment_name);
SearchToolSummary RunSearchTool(const SearchToolConfig &config);
BenchToolSummary RunBenchTool(const BenchToolConfig &config);
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
InspectToolSummary RunInspectTool(const InspectToolConfig &config);

}  // namespace hybrid
