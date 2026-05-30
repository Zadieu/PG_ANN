#include "tools/tool_cli.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "tools/build_pipeline.h"

namespace hybrid {

namespace {

double ComputeRecallAtK(const std::vector<SearchResult> &results,
                        const std::vector<uint32_t> &ground_truth,
                        uint32_t k) {
  if (k == 0 || ground_truth.empty()) {
    return 0.0;
  }
  const size_t cutoff = std::min<size_t>(k, ground_truth.size());
  std::unordered_set<uint32_t> truth;
  truth.reserve(cutoff);
  for (size_t i = 0; i < cutoff; ++i) {
    truth.insert(ground_truth[i]);
  }

  size_t hits = 0;
  for (size_t i = 0; i < results.size() && i < cutoff; ++i) {
    if (truth.find(results[i].id) != truth.end()) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(cutoff);
}

const char *ApproxKindName(ApproxDistanceKind kind) {
  switch (kind) {
    case ApproxDistanceKind::kFullPrecision:
      return "full";
    case ApproxDistanceKind::kProductQuantization:
      return "pq";
  }
  throw std::runtime_error("unsupported approximate distance kind");
}

std::string JoinResultIds(const std::vector<SearchResult> &results) {
  std::ostringstream out;
  for (size_t i = 0; i < results.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << results[i].id;
  }
  return out.str();
}

std::string JoinUint32Values(const std::vector<uint32_t> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  return out.str();
}

std::string JoinUint64Values(const std::vector<uint64_t> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  return out.str();
}

std::string JoinFloatValues(const std::vector<float> &values) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  return out.str();
}

std::string JoinBoolValues(const std::vector<uint8_t> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << (values[i] != 0 ? 1 : 0);
  }
  return out.str();
}

std::string JoinApproxKinds(const std::vector<ApproxDistanceKind> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << ApproxKindName(values[i]);
  }
  return out.str();
}

uint32_t GroundTruthWidth(const BenchToolConfig &config) {
  const uint32_t recall_k = config.recall_at_k == 0 ? config.search_config.top_k : config.recall_at_k;
  return std::max(config.search_config.top_k, recall_k);
}

std::string SanitizeLabel(const std::string &text) {
  std::string sanitized;
  sanitized.reserve(text.size());
  for (char ch : text) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      sanitized.push_back(static_cast<char>(std::tolower(uch)));
    } else if (ch == '-' || ch == '_') {
      sanitized.push_back(ch);
    } else {
      sanitized.push_back('_');
    }
  }
  if (sanitized.empty()) {
    return "experiment";
  }
  return sanitized;
}

std::string DefaultExperimentStem(const BenchSweepConfig &config) {
  const std::filesystem::path index_path(config.base_config.index_path);
  const std::string dataset = index_path.stem().empty() ? "index" : index_path.stem().string();
  return SanitizeLabel(dataset + "_q" + std::to_string(config.base_config.queries.size()));
}

std::string UtcTimestampString() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(_WIN32)
  gmtime_s(&utc_tm, &now_time);
#else
  gmtime_r(&now_time, &utc_tm);
#endif
  std::ostringstream out;
  out << std::put_time(&utc_tm, "%Y%m%d_%H%M%S");
  return out.str();
}

std::filesystem::path ResolveSummaryPath(const std::string &path_text) {
  const std::filesystem::path path(path_text);
  if (std::filesystem::is_directory(path)) {
    return path / "summary.tsv";
  }
  return path;
}

std::vector<std::string> SplitTabSeparatedLine(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream in(line);
  while (std::getline(in, field, '\t')) {
    fields.push_back(field);
  }
  return fields;
}

std::unordered_map<std::string, size_t> BuildHeaderIndex(const std::vector<std::string> &header_fields) {
  std::unordered_map<std::string, size_t> columns;
  columns.reserve(header_fields.size());
  for (size_t i = 0; i < header_fields.size(); ++i) {
    if (!columns.emplace(header_fields[i], i).second) {
      throw std::runtime_error("bench summary header contains a duplicate column: " + header_fields[i]);
    }
  }
  return columns;
}

const std::string &RequiredTsvField(const std::vector<std::string> &fields,
                                    const std::unordered_map<std::string, size_t> &columns,
                                    const char *name) {
  const auto it = columns.find(name);
  if (it == columns.end()) {
    throw std::runtime_error("bench summary header is missing column: " + std::string(name));
  }
  if (it->second >= fields.size()) {
    throw std::runtime_error("bench summary row is missing column: " + std::string(name));
  }
  return fields[it->second];
}

ApproxDistanceKind ParseApproxKind(const std::string &text) {
  if (text == "full") {
    return ApproxDistanceKind::kFullPrecision;
  }
  if (text == "pq") {
    return ApproxDistanceKind::kProductQuantization;
  }
  throw std::runtime_error("unsupported approx kind in summary: " + text);
}

uint32_t ParseUint32Field(const std::string &name, const std::string &text) {
  try {
    return static_cast<uint32_t>(std::stoul(text));
  } catch (const std::exception &) {
    throw std::runtime_error("failed to parse uint32 field: " + name);
  }
}

uint64_t ParseUint64Field(const std::string &name, const std::string &text) {
  try {
    return static_cast<uint64_t>(std::stoull(text));
  } catch (const std::exception &) {
    throw std::runtime_error("failed to parse uint64 field: " + name);
  }
}

double ParseDoubleField(const std::string &name, const std::string &text) {
  try {
    return std::stod(text);
  } catch (const std::exception &) {
    throw std::runtime_error("failed to parse double field: " + name);
  }
}

uint64_t ParseOptionalUint64Column(const std::vector<std::string> &fields,
                                   const std::unordered_map<std::string, size_t> &columns,
                                   const char *name,
                                   uint64_t default_value = 0) {
  const auto it = columns.find(name);
  if (it == columns.end() || it->second >= fields.size() || fields[it->second].empty()) {
    return default_value;
  }
  return ParseUint64Field(name, fields[it->second]);
}

uint32_t ParseOptionalUint32Column(const std::vector<std::string> &fields,
                                   const std::unordered_map<std::string, size_t> &columns,
                                   const char *name,
                                   uint32_t default_value = 0) {
  const auto it = columns.find(name);
  if (it == columns.end() || it->second >= fields.size() || fields[it->second].empty()) {
    return default_value;
  }
  return ParseUint32Field(name, fields[it->second]);
}

double ParseOptionalDoubleColumn(const std::vector<std::string> &fields,
                                 const std::unordered_map<std::string, size_t> &columns,
                                 const char *name,
                                 double default_value = 0.0) {
  const auto it = columns.find(name);
  if (it == columns.end() || it->second >= fields.size() || fields[it->second].empty()) {
    return default_value;
  }
  return ParseDoubleField(name, fields[it->second]);
}

std::string SummaryRunKey(const BenchToolSummary &summary) {
  std::ostringstream out;
  out << std::setprecision(9)
      << ApproxKindName(summary.approx_kind) << '|'
      << summary.search_config.top_k << '|'
      << summary.search_config.beam_width << '|'
      << summary.search_config.l_search << '|'
      << summary.search_config.graph_cache_budget_bytes << '|'
      << GraphCacheBuildPolicyName(summary.search_config.graph_cache_policy) << '|'
      << summary.search_config.refine_k << '|'
      << summary.search_config.refine_ratio << '|'
      << (summary.search_config.defer_exact_until_refinement ? 1 : 0) << '|'
      << SchedulerPolicyName(summary.search_config.scheduler_policy) << '|'
      << summary.search_config.scheduler_policy_limit << '|'
      << DynamicBeamPolicyName(summary.search_config.dynamic_beam_policy) << '|'
      << summary.num_queries;
  return out.str();
}

std::string JoinGraphCachePolicies(const std::vector<GraphCacheBuildPolicy> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << GraphCacheBuildPolicyName(values[i]);
  }
  return out.str();
}

std::string JoinSchedulerPolicies(const std::vector<SearchConfig::SchedulerPolicy> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << SchedulerPolicyName(values[i]);
  }
  return out.str();
}

std::string JoinDynamicBeamPolicies(const std::vector<SearchConfig::DynamicBeamPolicy> &values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << DynamicBeamPolicyName(values[i]);
  }
  return out.str();
}

void AccumulateSearchStats(SearchStats *total, const SearchStats &delta) {
  if (total == nullptr) {
    return;
  }
  total->async_reads += delta.async_reads;
  total->pages_completed += delta.pages_completed;
  total->resident_expansions += delta.resident_expansions;
  total->bytes_read += delta.bytes_read;
  total->page_resident_hits += delta.page_resident_hits;
  total->graph_replicated_hits += delta.graph_replicated_hits;
  total->graph_cache_hits += delta.graph_cache_hits;
  total->graph_cache_misses += delta.graph_cache_misses;
  total->graph_cache_expansions += delta.graph_cache_expansions;
  total->graph_cache_avoided_reads += delta.graph_cache_avoided_reads;
  total->graph_cache_resident_bytes = std::max(total->graph_cache_resident_bytes, delta.graph_cache_resident_bytes);
  total->graph_cache_entries = std::max(total->graph_cache_entries, delta.graph_cache_entries);
  total->graph_cache_build_page_reads =
      std::max(total->graph_cache_build_page_reads, delta.graph_cache_build_page_reads);
  total->exact_from_page += delta.exact_from_page;
  total->exact_from_payload += delta.exact_from_payload;
  total->refinement_candidates += delta.refinement_candidates;
  total->refinement_reads += delta.refinement_reads;
  total->approximate_candidates += delta.approximate_candidates;
  total->refinement_bound += delta.refinement_bound;
  total->refinement_already_exact += delta.refinement_already_exact;
  total->refinement_exactified += delta.refinement_exactified;
  total->deferred_exact_candidates += delta.deferred_exact_candidates;
  total->read_hits_in_pool += delta.read_hits_in_pool;
  total->read_waste_out_of_pool += delta.read_waste_out_of_pool;
  total->max_inflight_reads = std::max(total->max_inflight_reads, delta.max_inflight_reads);
  total->max_beam_width = std::max(total->max_beam_width, delta.max_beam_width);
  total->beam_width_increases += delta.beam_width_increases;
  total->scheduler_policy_limit = std::max(total->scheduler_policy_limit, delta.scheduler_policy_limit);
  total->scheduler_pending_max = std::max(total->scheduler_pending_max, delta.scheduler_pending_max);
  total->scheduler_ready_unexpanded_max =
      std::max(total->scheduler_ready_unexpanded_max, delta.scheduler_ready_unexpanded_max);
  total->scheduler_limit_hits += delta.scheduler_limit_hits;
  total->poll_calls += delta.poll_calls;
  total->drain_calls += delta.drain_calls;
  total->exact_distance_evals += delta.exact_distance_evals;
  total->approx_distance_evals += delta.approx_distance_evals;
  total->n_ios += delta.n_ios;
  total->n_cmps += delta.n_cmps;
  total->n_hops += delta.n_hops;
  total->cpu_us += delta.cpu_us;
  total->io_us += delta.io_us;
  total->cpu_us1 += delta.cpu_us1;
  total->cpu_us2 += delta.cpu_us2;
  total->io_us1 += delta.io_us1;
  total->total_us += delta.total_us;
  total->range_stop = total->range_stop || delta.range_stop;
  for (size_t i = 0; i < 4; ++i) {
    total->estimated_filter_reads[i] += delta.estimated_filter_reads[i];
    total->estimated_filter_cmps[i] += delta.estimated_filter_cmps[i];
    total->filter_reads[i] += delta.filter_reads[i];
    total->filter_accessed_vectors[i] += delta.filter_accessed_vectors[i];
    total->filter_false_positives[i] += delta.filter_false_positives[i];
    total->filter_io_us[i] += delta.filter_io_us[i];
    total->filter_cpu_us[i] += delta.filter_cpu_us[i];
    total->filter_io_us1[i] += delta.filter_io_us1[i];
  }
}

int64_t CounterDelta(uint64_t candidate, uint64_t baseline) {
  return static_cast<int64_t>(candidate) - static_cast<int64_t>(baseline);
}

}  // namespace

std::vector<float> ParseFloatVector(const std::string &text) {
  std::istringstream in(text);
  std::vector<float> values;
  float value = 0.0f;
  while (in >> value) {
    values.push_back(value);
  }
  if (values.empty()) {
    throw std::runtime_error("query vector must not be empty");
  }
  return values;
}

std::vector<uint32_t> ParseUint32List(const std::string &text) {
  std::vector<uint32_t> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(static_cast<uint32_t>(std::stoul(token)));
    } catch (const std::exception &) {
      throw std::runtime_error("failed to parse uint32 list");
    }
  }
  if (values.empty()) {
    throw std::runtime_error("uint32 list must not be empty");
  }
  return values;
}

std::vector<uint64_t> ParseUint64List(const std::string &text) {
  std::vector<uint64_t> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(static_cast<uint64_t>(std::stoull(token)));
    } catch (const std::exception &) {
      throw std::runtime_error("failed to parse uint64 list");
    }
  }
  if (values.empty()) {
    throw std::runtime_error("uint64 list must not be empty");
  }
  return values;
}

std::vector<float> ParseFloatList(const std::string &text) {
  std::vector<float> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(std::stof(token));
    } catch (const std::exception &) {
      throw std::runtime_error("failed to parse float list");
    }
  }
  if (values.empty()) {
    throw std::runtime_error("float list must not be empty");
  }
  return values;
}

std::vector<uint8_t> ParseBoolList(const std::string &text) {
  std::vector<uint8_t> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    if (token == "1" || token == "true" || token == "yes") {
      values.push_back(1);
    } else if (token == "0" || token == "false" || token == "no") {
      values.push_back(0);
    } else {
      throw std::runtime_error("failed to parse bool list");
    }
  }
  if (values.empty()) {
    throw std::runtime_error("bool list must not be empty");
  }
  return values;
}

std::vector<ApproxDistanceKind> ParseApproxKindList(const std::string &text) {
  std::vector<ApproxDistanceKind> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token == "full") {
      values.push_back(ApproxDistanceKind::kFullPrecision);
    } else if (token == "pq") {
      values.push_back(ApproxDistanceKind::kProductQuantization);
    } else if (!token.empty()) {
      throw std::runtime_error("failed to parse approx kind list");
    }
  }
  if (values.empty()) {
    throw std::runtime_error("approx kind list must not be empty");
  }
  return values;
}

std::vector<GraphCacheBuildPolicy> ParseGraphCachePolicyList(const std::string &text) {
  std::vector<GraphCacheBuildPolicy> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    values.push_back(ParseGraphCacheBuildPolicy(token));
  }
  if (values.empty()) {
    throw std::runtime_error("graph cache policy list must not be empty");
  }
  return values;
}

std::vector<SearchConfig::SchedulerPolicy> ParseSchedulerPolicyList(const std::string &text) {
  std::vector<SearchConfig::SchedulerPolicy> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    values.push_back(ParseSchedulerPolicy(token));
  }
  if (values.empty()) {
    throw std::runtime_error("scheduler policy list must not be empty");
  }
  return values;
}

std::vector<SearchConfig::DynamicBeamPolicy> ParseDynamicBeamPolicyList(const std::string &text) {
  std::vector<SearchConfig::DynamicBeamPolicy> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) {
      continue;
    }
    values.push_back(ParseDynamicBeamPolicy(token));
  }
  if (values.empty()) {
    throw std::runtime_error("dynamic beam policy list must not be empty");
  }
  return values;
}

std::vector<std::vector<float>> LoadQueryVectors(const std::string &path, QueryInputMode mode) {
  switch (mode) {
    case QueryInputMode::kText:
      return LoadTextVectors(path);
    case QueryInputMode::kFvecs:
      return LoadFvecsVectors(path);
    case QueryInputMode::kBvecs:
      return LoadBvecsVectors(path);
    case QueryInputMode::kBin:
      return LoadBinVectors(path);
  }
  throw std::runtime_error("unsupported query input mode");
}

std::vector<std::vector<uint32_t>> LoadGroundTruthIds(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open ground truth file");
  }
  std::vector<std::vector<uint32_t>> truth;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::vector<uint32_t> ids;
    uint32_t value = 0;
    while (row >> value) {
      ids.push_back(value);
    }
    if (!ids.empty()) {
      truth.push_back(std::move(ids));
    }
  }
  if (truth.empty()) {
    throw std::runtime_error("ground truth file does not contain any rows");
  }
  return truth;
}

std::vector<std::vector<uint32_t>> GenerateGroundTruthIds(const std::string &index_path,
                                                          const std::string &approx_path,
                                                          const std::vector<std::vector<float>> &queries,
                                                          uint32_t top_k) {
  if (queries.empty()) {
    throw std::runtime_error("ground truth generation requires at least one query");
  }
  if (top_k == 0) {
    throw std::runtime_error("ground truth generation requires top_k > 0");
  }

  std::unique_ptr<IndexReader> index = LoadIndexReader(index_path, approx_path);

  const size_t num_points = index->search_metadata().num_points;
  const uint32_t cutoff = std::min<uint32_t>(top_k, index->search_metadata().num_points);
  std::vector<std::vector<uint32_t>> ground_truth;
  ground_truth.reserve(queries.size());
  for (const auto &query : queries) {
    if (query.size() != index->search_metadata().dim) {
      throw std::runtime_error("query dimensionality does not match index during ground truth generation");
    }

    std::vector<SearchResult> exact_results;
    exact_results.reserve(num_points);
    for (uint32_t point_id = 0; point_id < index->search_metadata().num_points; ++point_id) {
      exact_results.push_back({point_id, L2Distance(query, index->exact_vector(point_id))});
    }
    std::sort(exact_results.begin(), exact_results.end(), [](const SearchResult &lhs, const SearchResult &rhs) {
      if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
      }
      return lhs.id < rhs.id;
    });

    std::vector<uint32_t> ids;
    ids.reserve(cutoff);
    for (uint32_t i = 0; i < cutoff; ++i) {
      ids.push_back(exact_results[i].id);
    }
    ground_truth.push_back(std::move(ids));
  }
  return ground_truth;
}

void WriteGroundTruthIds(const std::string &path, const std::vector<std::vector<uint32_t>> &ground_truth_ids) {
  std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open ground truth output file");
  }
  for (const auto &row : ground_truth_ids) {
    for (size_t i = 0; i < row.size(); ++i) {
      out << (i == 0 ? "" : " ") << row[i];
    }
    out << '\n';
  }
}

std::string CreateExperimentDirectory(const std::string &root_directory,
                                      const BenchSweepConfig &config,
                                      const std::string &experiment_name) {
  if (root_directory.empty()) {
    throw std::runtime_error("experiment root directory must not be empty");
  }

  const std::filesystem::path root(root_directory);
  std::filesystem::create_directories(root);

  const std::string stem =
      experiment_name.empty() ? DefaultExperimentStem(config) + "_" + UtcTimestampString() : SanitizeLabel(experiment_name);
  std::filesystem::path candidate = root / stem;
  uint32_t suffix = 1;
  while (std::filesystem::exists(candidate)) {
    std::ostringstream name;
    name << stem << "_" << suffix++;
    candidate = root / name.str();
  }
  std::filesystem::create_directories(candidate);
  return candidate.string();
}

SearchToolSummary RunSearchTool(const SearchToolConfig &config) {
  std::unique_ptr<IndexReader> index = LoadIndexReader(config.index_path, config.approx_path);

  SearchToolSummary summary;
  PipelinedGraphReplicatedSearcher searcher(*index);
  summary.approx_backend_name =
      config.approx_kind == ApproxDistanceKind::kProductQuantization ? "pipeann_pq" : "full_precision";
  summary.results = searcher.Search(
      config.query, config.search_config, config.approx_kind, &summary.stats, config.pq_codebook_path, config.pq_codes_path);
  return summary;
}

BenchToolSummary RunBenchTool(const BenchToolConfig &config) {
  std::unique_ptr<IndexReader> index = LoadIndexReader(config.index_path, config.approx_path);

  BenchToolSummary summary;
  summary.num_queries = static_cast<uint32_t>(config.queries.size());
  summary.search_config = config.search_config;
  summary.approx_kind = config.approx_kind;
  if (config.queries.empty()) {
    throw std::runtime_error("bench requires at least one query");
  }
  if (!config.ground_truth_ids.empty() && config.ground_truth_ids.size() != config.queries.size()) {
    throw std::runtime_error("ground truth row count must match query count");
  }

  PipelinedGraphReplicatedSearcher searcher(*index);
  double recall_sum = 0.0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t query_id = 0; query_id < config.queries.size(); ++query_id) {
    SearchStats stats;
    summary.approx_backend_name =
        config.approx_kind == ApproxDistanceKind::kProductQuantization ? "pipeann_pq" : "full_precision";
    const std::vector<SearchResult> results = searcher.Search(config.queries[query_id],
                                                              config.search_config,
                                                              config.approx_kind,
                                                              &stats,
                                                              config.pq_codebook_path,
                                                              config.pq_codes_path);

    if (query_id == 0) {
      summary.first_query_results = results;
    }
    if (!config.ground_truth_ids.empty()) {
      const uint32_t k = config.recall_at_k == 0 ? config.search_config.top_k : config.recall_at_k;
      recall_sum += ComputeRecallAtK(results, config.ground_truth_ids[query_id], k);
    }
    AccumulateSearchStats(&summary.aggregate_stats, stats);
  }
  const auto end = std::chrono::steady_clock::now();
  summary.elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  summary.average_latency_ms = summary.elapsed_ms / static_cast<double>(summary.num_queries);
  summary.qps = summary.elapsed_ms > 0.0 ? static_cast<double>(summary.num_queries) * 1000.0 / summary.elapsed_ms
                                         : 0.0;
  if (!config.ground_truth_ids.empty()) {
    summary.has_recall = true;
    summary.average_recall = recall_sum / static_cast<double>(summary.num_queries);
  }
  return summary;
}

BenchSweepSummary RunBenchSweep(const BenchSweepConfig &config) {
  BenchSweepSummary summary;
  const std::vector<uint32_t> beam_widths =
      config.beam_widths.empty() ? std::vector<uint32_t>{config.base_config.search_config.beam_width}
                                 : config.beam_widths;
  const std::vector<uint32_t> l_search_values =
      config.l_search_values.empty() ? std::vector<uint32_t>{config.base_config.search_config.l_search}
                                     : config.l_search_values;
  const std::vector<uint64_t> graph_cache_budget_bytes_values =
      config.graph_cache_budget_bytes_values.empty()
          ? std::vector<uint64_t>{config.base_config.search_config.graph_cache_budget_bytes}
          : config.graph_cache_budget_bytes_values;
  const std::vector<GraphCacheBuildPolicy> graph_cache_policies =
      config.graph_cache_policies.empty()
          ? std::vector<GraphCacheBuildPolicy>{config.base_config.search_config.graph_cache_policy}
          : config.graph_cache_policies;
  const std::vector<uint32_t> refine_k_values =
      config.refine_k_values.empty() ? std::vector<uint32_t>{config.base_config.search_config.refine_k}
                                     : config.refine_k_values;
  const std::vector<float> refine_ratio_values =
      config.refine_ratio_values.empty() ? std::vector<float>{config.base_config.search_config.refine_ratio}
                                         : config.refine_ratio_values;
  const std::vector<uint8_t> defer_exact_until_refinement_values =
      config.defer_exact_until_refinement_values.empty()
          ? std::vector<uint8_t>{static_cast<uint8_t>(config.base_config.search_config.defer_exact_until_refinement ? 1 : 0)}
          : config.defer_exact_until_refinement_values;
  const std::vector<SearchConfig::SchedulerPolicy> scheduler_policies =
      config.scheduler_policies.empty()
          ? std::vector<SearchConfig::SchedulerPolicy>{config.base_config.search_config.scheduler_policy}
          : config.scheduler_policies;
  const std::vector<uint32_t> scheduler_policy_limit_values =
      config.scheduler_policy_limit_values.empty()
          ? std::vector<uint32_t>{config.base_config.search_config.scheduler_policy_limit}
          : config.scheduler_policy_limit_values;
  const std::vector<SearchConfig::DynamicBeamPolicy> dynamic_beam_policies =
      config.dynamic_beam_policies.empty()
          ? std::vector<SearchConfig::DynamicBeamPolicy>{config.base_config.search_config.dynamic_beam_policy}
          : config.dynamic_beam_policies;
  const std::vector<ApproxDistanceKind> approx_kinds =
      config.approx_kinds.empty() ? std::vector<ApproxDistanceKind>{config.base_config.approx_kind}
                                  : config.approx_kinds;

  for (ApproxDistanceKind kind : approx_kinds) {
    for (uint32_t beam_width : beam_widths) {
      for (uint32_t l_search : l_search_values) {
        for (uint64_t graph_cache_budget_bytes : graph_cache_budget_bytes_values) {
          for (GraphCacheBuildPolicy graph_cache_policy : graph_cache_policies) {
            for (uint32_t refine_k : refine_k_values) {
              for (float refine_ratio : refine_ratio_values) {
                for (uint8_t defer_exact : defer_exact_until_refinement_values) {
                  for (SearchConfig::SchedulerPolicy scheduler_policy : scheduler_policies) {
                    for (uint32_t scheduler_policy_limit : scheduler_policy_limit_values) {
                      for (SearchConfig::DynamicBeamPolicy dynamic_beam_policy : dynamic_beam_policies) {
                        BenchToolConfig run_config = config.base_config;
                        run_config.approx_kind = kind;
                        run_config.search_config.beam_width = beam_width;
                        run_config.search_config.l_search = l_search;
                        run_config.search_config.graph_cache_budget_bytes = graph_cache_budget_bytes;
                        run_config.search_config.graph_cache_policy = graph_cache_policy;
                        run_config.search_config.refine_k = refine_k;
                        run_config.search_config.refine_ratio = refine_ratio;
                        run_config.search_config.defer_exact_until_refinement = defer_exact != 0;
                        run_config.search_config.scheduler_policy = scheduler_policy;
                        run_config.search_config.scheduler_policy_limit = scheduler_policy_limit;
                        run_config.search_config.dynamic_beam_policy = dynamic_beam_policy;
                        summary.runs.push_back(RunBenchTool(run_config));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return summary;
}

void ExportBenchSummariesTsv(const std::string &path, const std::vector<BenchToolSummary> &summaries) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench export file");
  }
  out << "approx_kind\tapprox_backend\ttop_k\tbeam_width\tl_search\tgraph_cache_budget_bytes"
         "\tgraph_cache_policy\trefine_k\trefine_ratio\tdefer_exact_until_refinement"
         "\tscheduler_policy\tscheduler_policy_limit\tdynamic_beam_policy"
         "\tqueries\telapsed_ms\taverage_latency_ms\tqps"
         "\tasync_reads\tpages_completed\tresident_expansions\tapprox_evals\texact_evals"
         "\tn_ios\tn_cmps\tn_hops\tcpu_us\tio_us\tbytes_read\tpage_resident_hits"
         "\tgraph_replicated_hits\tgraph_cache_hits\tgraph_cache_misses\tgraph_cache_expansions"
         "\tgraph_cache_avoided_reads\tgraph_cache_resident_bytes\tgraph_cache_entries"
         "\tgraph_cache_build_page_reads\texact_from_page\texact_from_payload\trefinement_candidates"
         "\trefinement_reads\tapproximate_candidates\trefinement_bound\trefinement_already_exact"
         "\trefinement_exactified\tdeferred_exact_candidates\tread_hits_in_pool\tread_waste_out_of_pool\tmax_inflight_reads"
         "\tmax_beam_width\tbeam_width_increases\tscheduler_policy_limit_observed\tscheduler_pending_max"
         "\tscheduler_ready_unexpanded_max\tscheduler_limit_hits\tpoll_calls\tdrain_calls\trange_stop\taverage_recall"
         "\tfirst_query_result_ids\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &summary : summaries) {
    out << ApproxKindName(summary.approx_kind) << '\t' << summary.approx_backend_name << '\t'
        << summary.search_config.top_k << '\t' << summary.search_config.beam_width << '\t'
        << summary.search_config.l_search << '\t' << summary.search_config.graph_cache_budget_bytes << '\t'
        << GraphCacheBuildPolicyName(summary.search_config.graph_cache_policy) << '\t'
        << summary.search_config.refine_k << '\t'
        << summary.search_config.refine_ratio << '\t'
        << (summary.search_config.defer_exact_until_refinement ? 1 : 0) << '\t'
        << SchedulerPolicyName(summary.search_config.scheduler_policy) << '\t'
        << summary.search_config.scheduler_policy_limit << '\t'
        << DynamicBeamPolicyName(summary.search_config.dynamic_beam_policy) << '\t'
        << summary.num_queries << '\t' << summary.elapsed_ms << '\t' << summary.average_latency_ms << '\t'
        << summary.qps << '\t'
        << summary.aggregate_stats.async_reads << '\t' << summary.aggregate_stats.pages_completed << '\t'
        << summary.aggregate_stats.resident_expansions << '\t'
        << summary.aggregate_stats.approx_distance_evals << '\t'
        << summary.aggregate_stats.exact_distance_evals << '\t'
        << summary.aggregate_stats.n_ios << '\t'
        << summary.aggregate_stats.n_cmps << '\t'
        << summary.aggregate_stats.n_hops << '\t'
        << summary.aggregate_stats.cpu_us << '\t'
        << summary.aggregate_stats.io_us << '\t'
        << summary.aggregate_stats.bytes_read << '\t'
        << summary.aggregate_stats.page_resident_hits << '\t'
        << summary.aggregate_stats.graph_replicated_hits << '\t'
        << summary.aggregate_stats.graph_cache_hits << '\t'
        << summary.aggregate_stats.graph_cache_misses << '\t'
        << summary.aggregate_stats.graph_cache_expansions << '\t'
        << summary.aggregate_stats.graph_cache_avoided_reads << '\t'
        << summary.aggregate_stats.graph_cache_resident_bytes << '\t'
        << summary.aggregate_stats.graph_cache_entries << '\t'
        << summary.aggregate_stats.graph_cache_build_page_reads << '\t'
        << summary.aggregate_stats.exact_from_page << '\t'
        << summary.aggregate_stats.exact_from_payload << '\t'
        << summary.aggregate_stats.refinement_candidates << '\t'
        << summary.aggregate_stats.refinement_reads << '\t'
        << summary.aggregate_stats.approximate_candidates << '\t'
        << summary.aggregate_stats.refinement_bound << '\t'
        << summary.aggregate_stats.refinement_already_exact << '\t'
        << summary.aggregate_stats.refinement_exactified << '\t'
        << summary.aggregate_stats.deferred_exact_candidates << '\t'
        << summary.aggregate_stats.read_hits_in_pool << '\t'
        << summary.aggregate_stats.read_waste_out_of_pool << '\t'
        << summary.aggregate_stats.max_inflight_reads << '\t'
        << summary.aggregate_stats.max_beam_width << '\t'
        << summary.aggregate_stats.beam_width_increases << '\t'
        << summary.aggregate_stats.scheduler_policy_limit << '\t'
        << summary.aggregate_stats.scheduler_pending_max << '\t'
        << summary.aggregate_stats.scheduler_ready_unexpanded_max << '\t'
        << summary.aggregate_stats.scheduler_limit_hits << '\t'
        << summary.aggregate_stats.poll_calls << '\t'
        << summary.aggregate_stats.drain_calls << '\t'
        << (summary.aggregate_stats.range_stop ? 1 : 0) << '\t'
        << (summary.has_recall ? summary.average_recall : -1.0) << '\t'
        << JoinResultIds(summary.first_query_results) << '\n';
  }
}

void ExportBenchExperiment(const std::string &directory,
                           const BenchSweepConfig &config,
                           const BenchSweepSummary &summary) {
  const std::filesystem::path root(directory);
  std::filesystem::create_directories(root);

  const std::filesystem::path summary_path = root / "summary.tsv";
  ExportBenchSummariesTsv(summary_path.string(), summary.runs);

  const std::filesystem::path manifest_path = root / "manifest.txt";
  std::ofstream manifest(manifest_path, std::ios::trunc);
  if (!manifest) {
    throw std::runtime_error("failed to open bench manifest file");
  }

  const std::vector<uint32_t> beam_widths =
      config.beam_widths.empty() ? std::vector<uint32_t>{config.base_config.search_config.beam_width}
                                 : config.beam_widths;
  const std::vector<uint32_t> l_search_values =
      config.l_search_values.empty() ? std::vector<uint32_t>{config.base_config.search_config.l_search}
                                     : config.l_search_values;
  const std::vector<uint64_t> graph_cache_budget_bytes_values =
      config.graph_cache_budget_bytes_values.empty()
          ? std::vector<uint64_t>{config.base_config.search_config.graph_cache_budget_bytes}
          : config.graph_cache_budget_bytes_values;
  const std::vector<GraphCacheBuildPolicy> graph_cache_policies =
      config.graph_cache_policies.empty()
          ? std::vector<GraphCacheBuildPolicy>{config.base_config.search_config.graph_cache_policy}
          : config.graph_cache_policies;
  const std::vector<uint32_t> refine_k_values =
      config.refine_k_values.empty() ? std::vector<uint32_t>{config.base_config.search_config.refine_k}
                                     : config.refine_k_values;
  const std::vector<float> refine_ratio_values =
      config.refine_ratio_values.empty() ? std::vector<float>{config.base_config.search_config.refine_ratio}
                                         : config.refine_ratio_values;
  const std::vector<uint8_t> defer_exact_until_refinement_values =
      config.defer_exact_until_refinement_values.empty()
          ? std::vector<uint8_t>{static_cast<uint8_t>(config.base_config.search_config.defer_exact_until_refinement ? 1 : 0)}
          : config.defer_exact_until_refinement_values;
  const std::vector<SearchConfig::SchedulerPolicy> scheduler_policies =
      config.scheduler_policies.empty()
          ? std::vector<SearchConfig::SchedulerPolicy>{config.base_config.search_config.scheduler_policy}
          : config.scheduler_policies;
  const std::vector<uint32_t> scheduler_policy_limit_values =
      config.scheduler_policy_limit_values.empty()
          ? std::vector<uint32_t>{config.base_config.search_config.scheduler_policy_limit}
          : config.scheduler_policy_limit_values;
  const std::vector<SearchConfig::DynamicBeamPolicy> dynamic_beam_policies =
      config.dynamic_beam_policies.empty()
          ? std::vector<SearchConfig::DynamicBeamPolicy>{config.base_config.search_config.dynamic_beam_policy}
          : config.dynamic_beam_policies;
  const std::vector<ApproxDistanceKind> approx_kinds =
      config.approx_kinds.empty() ? std::vector<ApproxDistanceKind>{config.base_config.approx_kind}
                                  : config.approx_kinds;
  const std::string full_data_path =
      config.base_config.approx_path.empty() ? DefaultPipeannBaseDataPath(config.base_config.index_path)
                                             : config.base_config.approx_path;
  const std::string pq_pivots_path =
      config.base_config.pq_codebook_path.empty() ? DefaultPipeannPqPivotsPath(config.base_config.index_path)
                                                  : config.base_config.pq_codebook_path;
  const std::string pq_compressed_path =
      config.base_config.pq_codes_path.empty() ? DefaultPipeannPqCompressedPath(config.base_config.index_path)
                                               : config.base_config.pq_codes_path;

  manifest << "index=" << config.base_config.index_path << '\n';
  manifest << "pipeann_base_data=" << full_data_path << '\n';
  manifest << "full_data=" << full_data_path << '\n';
  manifest << "pipeann_pq_pivots=" << pq_pivots_path << '\n';
  manifest << "pipeann_pq_compressed=" << pq_compressed_path << '\n';
  manifest << "query_count=" << config.base_config.queries.size() << '\n';
  manifest << "top_k=" << config.base_config.search_config.top_k << '\n';
  manifest << "recall_at_k=" << config.base_config.recall_at_k << '\n';
  manifest << "beam_widths=" << JoinUint32Values(beam_widths) << '\n';
  manifest << "l_search_values=" << JoinUint32Values(l_search_values) << '\n';
  manifest << "graph_cache_budget_bytes_values=" << JoinUint64Values(graph_cache_budget_bytes_values) << '\n';
  manifest << "graph_cache_policies=" << JoinGraphCachePolicies(graph_cache_policies) << '\n';
  manifest << "refine_k_values=" << JoinUint32Values(refine_k_values) << '\n';
  manifest << "refine_ratio_values=" << JoinFloatValues(refine_ratio_values) << '\n';
  manifest << "defer_exact_until_refinement_values=" << JoinBoolValues(defer_exact_until_refinement_values) << '\n';
  manifest << "scheduler_policies=" << JoinSchedulerPolicies(scheduler_policies) << '\n';
  manifest << "scheduler_policy_limit_values=" << JoinUint32Values(scheduler_policy_limit_values) << '\n';
  manifest << "dynamic_beam_policies=" << JoinDynamicBeamPolicies(dynamic_beam_policies) << '\n';
  manifest << "approx_kinds=" << JoinApproxKinds(approx_kinds) << '\n';
  manifest << "summary_tsv=" << summary_path.string() << '\n';

  if (!config.base_config.ground_truth_ids.empty()) {
    const std::filesystem::path ground_truth_path = root / "ground_truth.txt";
    WriteGroundTruthIds(ground_truth_path.string(), config.base_config.ground_truth_ids);
    manifest << "ground_truth=" << ground_truth_path.string() << '\n';
    manifest << "ground_truth_width=" << GroundTruthWidth(config.base_config) << '\n';
  }

  const std::filesystem::path runs_dir = root / "runs";
  std::filesystem::create_directories(runs_dir);
  for (size_t i = 0; i < summary.runs.size(); ++i) {
    const auto &run = summary.runs[i];
    std::ostringstream name;
    name << "run_" << std::setfill('0') << std::setw(3) << i << ".txt";
    std::ofstream run_out(runs_dir / name.str(), std::ios::trunc);
    if (!run_out) {
      throw std::runtime_error("failed to open bench run detail file");
    }

    run_out << "approx_kind=" << ApproxKindName(run.approx_kind) << '\n';
    run_out << "approx_backend=" << run.approx_backend_name << '\n';
    run_out << "queries=" << run.num_queries << '\n';
    run_out << "top_k=" << run.search_config.top_k << '\n';
    run_out << "beam_width=" << run.search_config.beam_width << '\n';
    run_out << "l_search=" << run.search_config.l_search << '\n';
    run_out << "graph_cache_budget_bytes=" << run.search_config.graph_cache_budget_bytes << '\n';
    run_out << "graph_cache_policy=" << GraphCacheBuildPolicyName(run.search_config.graph_cache_policy) << '\n';
    run_out << "refine_k=" << run.search_config.refine_k << '\n';
    run_out << "refine_ratio=" << run.search_config.refine_ratio << '\n';
    run_out << "defer_exact_until_refinement=" << (run.search_config.defer_exact_until_refinement ? 1 : 0) << '\n';
    run_out << "scheduler_policy=" << SchedulerPolicyName(run.search_config.scheduler_policy) << '\n';
    run_out << "scheduler_policy_limit=" << run.search_config.scheduler_policy_limit << '\n';
    run_out << "dynamic_beam_policy=" << DynamicBeamPolicyName(run.search_config.dynamic_beam_policy) << '\n';
    run_out << std::fixed << std::setprecision(6);
    run_out << "elapsed_ms=" << run.elapsed_ms << '\n';
    run_out << "average_latency_ms=" << run.average_latency_ms << '\n';
    run_out << "qps=" << run.qps << '\n';
    run_out << "average_recall=" << (run.has_recall ? run.average_recall : -1.0) << '\n';
    run_out << "async_reads=" << run.aggregate_stats.async_reads << '\n';
    run_out << "pages_completed=" << run.aggregate_stats.pages_completed << '\n';
    run_out << "resident_expansions=" << run.aggregate_stats.resident_expansions << '\n';
    run_out << "approx_evals=" << run.aggregate_stats.approx_distance_evals << '\n';
    run_out << "exact_evals=" << run.aggregate_stats.exact_distance_evals << '\n';
    run_out << "n_ios=" << run.aggregate_stats.n_ios << '\n';
    run_out << "n_cmps=" << run.aggregate_stats.n_cmps << '\n';
    run_out << "n_hops=" << run.aggregate_stats.n_hops << '\n';
    run_out << "cpu_us=" << run.aggregate_stats.cpu_us << '\n';
    run_out << "io_us=" << run.aggregate_stats.io_us << '\n';
    run_out << "bytes_read=" << run.aggregate_stats.bytes_read << '\n';
    run_out << "page_resident_hits=" << run.aggregate_stats.page_resident_hits << '\n';
    run_out << "graph_replicated_hits=" << run.aggregate_stats.graph_replicated_hits << '\n';
    run_out << "graph_cache_hits=" << run.aggregate_stats.graph_cache_hits << '\n';
    run_out << "graph_cache_misses=" << run.aggregate_stats.graph_cache_misses << '\n';
    run_out << "graph_cache_expansions=" << run.aggregate_stats.graph_cache_expansions << '\n';
    run_out << "graph_cache_avoided_reads=" << run.aggregate_stats.graph_cache_avoided_reads << '\n';
    run_out << "graph_cache_resident_bytes=" << run.aggregate_stats.graph_cache_resident_bytes << '\n';
    run_out << "graph_cache_entries=" << run.aggregate_stats.graph_cache_entries << '\n';
    run_out << "graph_cache_build_page_reads=" << run.aggregate_stats.graph_cache_build_page_reads << '\n';
    run_out << "exact_from_page=" << run.aggregate_stats.exact_from_page << '\n';
    run_out << "exact_from_payload=" << run.aggregate_stats.exact_from_payload << '\n';
    run_out << "refinement_candidates=" << run.aggregate_stats.refinement_candidates << '\n';
    run_out << "refinement_reads=" << run.aggregate_stats.refinement_reads << '\n';
    run_out << "approximate_candidates=" << run.aggregate_stats.approximate_candidates << '\n';
    run_out << "refinement_bound=" << run.aggregate_stats.refinement_bound << '\n';
    run_out << "refinement_already_exact=" << run.aggregate_stats.refinement_already_exact << '\n';
    run_out << "refinement_exactified=" << run.aggregate_stats.refinement_exactified << '\n';
    run_out << "deferred_exact_candidates=" << run.aggregate_stats.deferred_exact_candidates << '\n';
    run_out << "read_hits_in_pool=" << run.aggregate_stats.read_hits_in_pool << '\n';
    run_out << "read_waste_out_of_pool=" << run.aggregate_stats.read_waste_out_of_pool << '\n';
    run_out << "max_inflight_reads=" << run.aggregate_stats.max_inflight_reads << '\n';
    run_out << "max_beam_width=" << run.aggregate_stats.max_beam_width << '\n';
    run_out << "beam_width_increases=" << run.aggregate_stats.beam_width_increases << '\n';
    run_out << "scheduler_policy_limit_observed=" << run.aggregate_stats.scheduler_policy_limit << '\n';
    run_out << "scheduler_pending_max=" << run.aggregate_stats.scheduler_pending_max << '\n';
    run_out << "scheduler_ready_unexpanded_max=" << run.aggregate_stats.scheduler_ready_unexpanded_max << '\n';
    run_out << "scheduler_limit_hits=" << run.aggregate_stats.scheduler_limit_hits << '\n';
    run_out << "poll_calls=" << run.aggregate_stats.poll_calls << '\n';
    run_out << "drain_calls=" << run.aggregate_stats.drain_calls << '\n';
    run_out << "range_stop=" << (run.aggregate_stats.range_stop ? 1 : 0) << '\n';
    run_out << "first_query_result_ids=" << JoinResultIds(run.first_query_results) << '\n';
  }
}

void MergeBenchSummaryTsvFiles(const std::string &output_path, const std::vector<std::string> &input_paths) {
  if (input_paths.empty()) {
    throw std::runtime_error("bench merge requires at least one input path");
  }

  const std::filesystem::path output_file(output_path);
  if (output_file.has_parent_path()) {
    std::filesystem::create_directories(output_file.parent_path());
  }
  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open merged bench summary output");
  }

  bool wrote_header = false;
  for (const std::string &input_path_text : input_paths) {
    const std::filesystem::path input_path = ResolveSummaryPath(input_path_text);
    std::ifstream in(input_path);
    if (!in) {
      throw std::runtime_error("failed to open bench summary input: " + input_path.string());
    }

    std::string line;
    bool is_first_line = true;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      if (is_first_line) {
        if (!wrote_header) {
          out << line << '\n';
          wrote_header = true;
        }
        is_first_line = false;
        continue;
      }
      out << line << '\n';
    }
  }

  if (!wrote_header) {
    throw std::runtime_error("merged bench summary did not contain any header");
  }
}

std::vector<BenchToolSummary> LoadBenchSummariesTsv(const std::string &path) {
  const std::filesystem::path input_path = ResolveSummaryPath(path);
  std::ifstream in(input_path);
  if (!in) {
    throw std::runtime_error("failed to open bench summary file: " + input_path.string());
  }

  std::string header;
  if (!std::getline(in, header)) {
    throw std::runtime_error("bench summary file is empty");
  }

  const std::vector<std::string> header_fields = SplitTabSeparatedLine(header);
  const std::unordered_map<std::string, size_t> columns = BuildHeaderIndex(header_fields);

  std::vector<BenchToolSummary> summaries;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = SplitTabSeparatedLine(line);
    if (fields.size() > header_fields.size()) {
      throw std::runtime_error("bench summary row has unexpected column count");
    }

    BenchToolSummary summary;
    summary.approx_kind = ParseApproxKind(RequiredTsvField(fields, columns, "approx_kind"));
    summary.approx_backend_name = RequiredTsvField(fields, columns, "approx_backend");
    summary.search_config.top_k = ParseUint32Field("top_k", RequiredTsvField(fields, columns, "top_k"));
    summary.search_config.beam_width =
        ParseUint32Field("beam_width", RequiredTsvField(fields, columns, "beam_width"));
    summary.search_config.l_search = ParseUint32Field("l_search", RequiredTsvField(fields, columns, "l_search"));
    summary.search_config.graph_cache_budget_bytes =
        ParseOptionalUint64Column(fields, columns, "graph_cache_budget_bytes");
    const auto graph_cache_policy_column = columns.find("graph_cache_policy");
    if (graph_cache_policy_column != columns.end() && graph_cache_policy_column->second < fields.size() &&
        !fields[graph_cache_policy_column->second].empty()) {
      summary.search_config.graph_cache_policy =
          ParseGraphCacheBuildPolicy(fields[graph_cache_policy_column->second]);
    }
    summary.search_config.refine_k = ParseOptionalUint32Column(fields, columns, "refine_k");
    summary.search_config.refine_ratio =
        static_cast<float>(ParseOptionalDoubleColumn(fields, columns, "refine_ratio"));
    summary.search_config.defer_exact_until_refinement =
        ParseOptionalUint64Column(fields, columns, "defer_exact_until_refinement") != 0;
    const auto scheduler_policy_column = columns.find("scheduler_policy");
    if (scheduler_policy_column != columns.end() && scheduler_policy_column->second < fields.size() &&
        !fields[scheduler_policy_column->second].empty()) {
      summary.search_config.scheduler_policy = ParseSchedulerPolicy(fields[scheduler_policy_column->second]);
    }
    summary.search_config.scheduler_policy_limit =
        ParseOptionalUint32Column(fields, columns, "scheduler_policy_limit");
    const auto dynamic_beam_policy_column = columns.find("dynamic_beam_policy");
    if (dynamic_beam_policy_column != columns.end() && dynamic_beam_policy_column->second < fields.size() &&
        !fields[dynamic_beam_policy_column->second].empty()) {
      summary.search_config.dynamic_beam_policy =
          ParseDynamicBeamPolicy(fields[dynamic_beam_policy_column->second]);
    }
    summary.num_queries = ParseUint32Field("queries", RequiredTsvField(fields, columns, "queries"));
    summary.elapsed_ms = ParseDoubleField("elapsed_ms", RequiredTsvField(fields, columns, "elapsed_ms"));
    summary.average_latency_ms =
        ParseDoubleField("average_latency_ms", RequiredTsvField(fields, columns, "average_latency_ms"));
    summary.qps = ParseDoubleField("qps", RequiredTsvField(fields, columns, "qps"));
    summary.aggregate_stats.async_reads =
        ParseUint64Field("async_reads", RequiredTsvField(fields, columns, "async_reads"));
    summary.aggregate_stats.pages_completed =
        ParseUint64Field("pages_completed", RequiredTsvField(fields, columns, "pages_completed"));
    summary.aggregate_stats.resident_expansions =
        ParseUint64Field("resident_expansions", RequiredTsvField(fields, columns, "resident_expansions"));
    summary.aggregate_stats.approx_distance_evals =
        ParseUint64Field("approx_evals", RequiredTsvField(fields, columns, "approx_evals"));
    summary.aggregate_stats.exact_distance_evals =
        ParseUint64Field("exact_evals", RequiredTsvField(fields, columns, "exact_evals"));
    summary.aggregate_stats.n_ios = ParseUint64Field("n_ios", RequiredTsvField(fields, columns, "n_ios"));
    summary.aggregate_stats.n_cmps = ParseUint64Field("n_cmps", RequiredTsvField(fields, columns, "n_cmps"));
    summary.aggregate_stats.n_hops = ParseUint64Field("n_hops", RequiredTsvField(fields, columns, "n_hops"));
    summary.aggregate_stats.cpu_us = ParseUint64Field("cpu_us", RequiredTsvField(fields, columns, "cpu_us"));
    summary.aggregate_stats.io_us = ParseUint64Field("io_us", RequiredTsvField(fields, columns, "io_us"));
    summary.aggregate_stats.bytes_read = ParseOptionalUint64Column(fields, columns, "bytes_read");
    summary.aggregate_stats.page_resident_hits = ParseOptionalUint64Column(fields, columns, "page_resident_hits");
    summary.aggregate_stats.graph_replicated_hits =
        ParseOptionalUint64Column(fields, columns, "graph_replicated_hits");
    summary.aggregate_stats.graph_cache_hits = ParseOptionalUint64Column(fields, columns, "graph_cache_hits");
    summary.aggregate_stats.graph_cache_misses = ParseOptionalUint64Column(fields, columns, "graph_cache_misses");
    summary.aggregate_stats.graph_cache_expansions =
        ParseOptionalUint64Column(fields, columns, "graph_cache_expansions");
    summary.aggregate_stats.graph_cache_avoided_reads =
        ParseOptionalUint64Column(fields, columns, "graph_cache_avoided_reads");
    summary.aggregate_stats.graph_cache_resident_bytes =
        ParseOptionalUint64Column(fields, columns, "graph_cache_resident_bytes");
    summary.aggregate_stats.graph_cache_entries = ParseOptionalUint64Column(fields, columns, "graph_cache_entries");
    summary.aggregate_stats.graph_cache_build_page_reads =
        ParseOptionalUint64Column(fields, columns, "graph_cache_build_page_reads");
    summary.aggregate_stats.exact_from_page = ParseOptionalUint64Column(fields, columns, "exact_from_page");
    summary.aggregate_stats.exact_from_payload = ParseOptionalUint64Column(fields, columns, "exact_from_payload");
    summary.aggregate_stats.refinement_candidates =
        ParseOptionalUint64Column(fields, columns, "refinement_candidates");
    summary.aggregate_stats.refinement_reads = ParseOptionalUint64Column(fields, columns, "refinement_reads");
    summary.aggregate_stats.approximate_candidates =
        ParseOptionalUint64Column(fields, columns, "approximate_candidates");
    summary.aggregate_stats.refinement_bound = ParseOptionalUint64Column(fields, columns, "refinement_bound");
    summary.aggregate_stats.refinement_already_exact =
        ParseOptionalUint64Column(fields, columns, "refinement_already_exact");
    summary.aggregate_stats.refinement_exactified =
        ParseOptionalUint64Column(fields, columns, "refinement_exactified");
    summary.aggregate_stats.deferred_exact_candidates =
        ParseOptionalUint64Column(fields, columns, "deferred_exact_candidates");
    summary.aggregate_stats.read_hits_in_pool = ParseOptionalUint64Column(fields, columns, "read_hits_in_pool");
    summary.aggregate_stats.read_waste_out_of_pool =
        ParseOptionalUint64Column(fields, columns, "read_waste_out_of_pool");
    summary.aggregate_stats.max_inflight_reads = ParseOptionalUint64Column(fields, columns, "max_inflight_reads");
    summary.aggregate_stats.max_beam_width = ParseOptionalUint64Column(fields, columns, "max_beam_width");
    summary.aggregate_stats.beam_width_increases =
        ParseOptionalUint64Column(fields, columns, "beam_width_increases");
    summary.aggregate_stats.scheduler_policy_limit =
        ParseOptionalUint64Column(fields, columns, "scheduler_policy_limit_observed");
    summary.aggregate_stats.scheduler_pending_max =
        ParseOptionalUint64Column(fields, columns, "scheduler_pending_max");
    summary.aggregate_stats.scheduler_ready_unexpanded_max =
        ParseOptionalUint64Column(fields, columns, "scheduler_ready_unexpanded_max");
    summary.aggregate_stats.scheduler_limit_hits =
        ParseOptionalUint64Column(fields, columns, "scheduler_limit_hits");
    summary.aggregate_stats.poll_calls = ParseOptionalUint64Column(fields, columns, "poll_calls");
    summary.aggregate_stats.drain_calls = ParseOptionalUint64Column(fields, columns, "drain_calls");
    summary.aggregate_stats.range_stop =
        ParseUint64Field("range_stop", RequiredTsvField(fields, columns, "range_stop")) != 0;
    summary.average_recall =
        ParseDoubleField("average_recall", RequiredTsvField(fields, columns, "average_recall"));
    summary.has_recall = summary.average_recall >= 0.0;
    summaries.push_back(std::move(summary));
  }

  if (summaries.empty()) {
    throw std::runtime_error("bench summary file does not contain any rows");
  }
  return summaries;
}

BenchComparisonSummary CompareBenchSummaries(const std::string &baseline_label,
                                             const std::vector<BenchToolSummary> &baseline,
                                             const std::string &candidate_label,
                                             const std::vector<BenchToolSummary> &candidate) {
  if (baseline.empty() || candidate.empty()) {
    throw std::runtime_error("bench comparison requires non-empty baseline and candidate summaries");
  }

  std::unordered_map<std::string, BenchToolSummary> baseline_by_key;
  baseline_by_key.reserve(baseline.size());
  for (const auto &summary : baseline) {
    baseline_by_key.emplace(SummaryRunKey(summary), summary);
  }

  BenchComparisonSummary result;
  result.baseline_label = baseline_label;
  result.candidate_label = candidate_label;
  for (const auto &candidate_summary : candidate) {
    const std::string key = SummaryRunKey(candidate_summary);
    auto it = baseline_by_key.find(key);
    if (it == baseline_by_key.end()) {
      continue;
    }

    BenchComparisonRow row;
    row.baseline = it->second;
    row.candidate = candidate_summary;
    row.delta_elapsed_ms = candidate_summary.elapsed_ms - row.baseline.elapsed_ms;
    row.delta_average_latency_ms = candidate_summary.average_latency_ms - row.baseline.average_latency_ms;
    row.delta_qps = candidate_summary.qps - row.baseline.qps;
    const double baseline_recall = row.baseline.has_recall ? row.baseline.average_recall : 0.0;
    const double candidate_recall = candidate_summary.has_recall ? candidate_summary.average_recall : 0.0;
    row.delta_average_recall = candidate_recall - baseline_recall;
    row.delta_async_reads =
        CounterDelta(candidate_summary.aggregate_stats.async_reads, row.baseline.aggregate_stats.async_reads);
    row.delta_pages_completed =
        CounterDelta(candidate_summary.aggregate_stats.pages_completed, row.baseline.aggregate_stats.pages_completed);
    row.delta_resident_expansions = CounterDelta(candidate_summary.aggregate_stats.resident_expansions,
                                                row.baseline.aggregate_stats.resident_expansions);
    row.delta_bytes_read =
        CounterDelta(candidate_summary.aggregate_stats.bytes_read, row.baseline.aggregate_stats.bytes_read);
    row.delta_page_resident_hits = CounterDelta(candidate_summary.aggregate_stats.page_resident_hits,
                                               row.baseline.aggregate_stats.page_resident_hits);
    row.delta_graph_replicated_hits = CounterDelta(candidate_summary.aggregate_stats.graph_replicated_hits,
                                                  row.baseline.aggregate_stats.graph_replicated_hits);
    row.delta_graph_cache_hits = CounterDelta(candidate_summary.aggregate_stats.graph_cache_hits,
                                             row.baseline.aggregate_stats.graph_cache_hits);
    row.delta_graph_cache_misses = CounterDelta(candidate_summary.aggregate_stats.graph_cache_misses,
                                               row.baseline.aggregate_stats.graph_cache_misses);
    row.delta_graph_cache_expansions = CounterDelta(candidate_summary.aggregate_stats.graph_cache_expansions,
                                                   row.baseline.aggregate_stats.graph_cache_expansions);
    row.delta_graph_cache_avoided_reads = CounterDelta(candidate_summary.aggregate_stats.graph_cache_avoided_reads,
                                                      row.baseline.aggregate_stats.graph_cache_avoided_reads);
    row.delta_graph_cache_resident_bytes = CounterDelta(candidate_summary.aggregate_stats.graph_cache_resident_bytes,
                                                       row.baseline.aggregate_stats.graph_cache_resident_bytes);
    row.delta_graph_cache_entries = CounterDelta(candidate_summary.aggregate_stats.graph_cache_entries,
                                                row.baseline.aggregate_stats.graph_cache_entries);
    row.delta_graph_cache_build_page_reads =
        CounterDelta(candidate_summary.aggregate_stats.graph_cache_build_page_reads,
                     row.baseline.aggregate_stats.graph_cache_build_page_reads);
    row.delta_exact_from_page =
        CounterDelta(candidate_summary.aggregate_stats.exact_from_page, row.baseline.aggregate_stats.exact_from_page);
    row.delta_exact_from_payload = CounterDelta(candidate_summary.aggregate_stats.exact_from_payload,
                                               row.baseline.aggregate_stats.exact_from_payload);
    row.delta_refinement_candidates = CounterDelta(candidate_summary.aggregate_stats.refinement_candidates,
                                                  row.baseline.aggregate_stats.refinement_candidates);
    row.delta_refinement_reads =
        CounterDelta(candidate_summary.aggregate_stats.refinement_reads, row.baseline.aggregate_stats.refinement_reads);
    row.delta_approximate_candidates = CounterDelta(candidate_summary.aggregate_stats.approximate_candidates,
                                                    row.baseline.aggregate_stats.approximate_candidates);
    row.delta_refinement_bound =
        CounterDelta(candidate_summary.aggregate_stats.refinement_bound, row.baseline.aggregate_stats.refinement_bound);
    row.delta_refinement_already_exact = CounterDelta(candidate_summary.aggregate_stats.refinement_already_exact,
                                                     row.baseline.aggregate_stats.refinement_already_exact);
    row.delta_refinement_exactified = CounterDelta(candidate_summary.aggregate_stats.refinement_exactified,
                                                  row.baseline.aggregate_stats.refinement_exactified);
    row.delta_deferred_exact_candidates = CounterDelta(candidate_summary.aggregate_stats.deferred_exact_candidates,
                                                      row.baseline.aggregate_stats.deferred_exact_candidates);
    row.delta_read_hits_in_pool = CounterDelta(candidate_summary.aggregate_stats.read_hits_in_pool,
                                              row.baseline.aggregate_stats.read_hits_in_pool);
    row.delta_read_waste_out_of_pool = CounterDelta(candidate_summary.aggregate_stats.read_waste_out_of_pool,
                                                   row.baseline.aggregate_stats.read_waste_out_of_pool);
    row.delta_max_inflight_reads = CounterDelta(candidate_summary.aggregate_stats.max_inflight_reads,
                                               row.baseline.aggregate_stats.max_inflight_reads);
    row.delta_max_beam_width =
        CounterDelta(candidate_summary.aggregate_stats.max_beam_width, row.baseline.aggregate_stats.max_beam_width);
    row.delta_beam_width_increases = CounterDelta(candidate_summary.aggregate_stats.beam_width_increases,
                                                 row.baseline.aggregate_stats.beam_width_increases);
    row.delta_scheduler_pending_max = CounterDelta(candidate_summary.aggregate_stats.scheduler_pending_max,
                                                  row.baseline.aggregate_stats.scheduler_pending_max);
    row.delta_scheduler_ready_unexpanded_max =
        CounterDelta(candidate_summary.aggregate_stats.scheduler_ready_unexpanded_max,
                     row.baseline.aggregate_stats.scheduler_ready_unexpanded_max);
    row.delta_scheduler_limit_hits = CounterDelta(candidate_summary.aggregate_stats.scheduler_limit_hits,
                                                 row.baseline.aggregate_stats.scheduler_limit_hits);
    row.delta_poll_calls =
        CounterDelta(candidate_summary.aggregate_stats.poll_calls, row.baseline.aggregate_stats.poll_calls);
    row.delta_drain_calls =
        CounterDelta(candidate_summary.aggregate_stats.drain_calls, row.baseline.aggregate_stats.drain_calls);
    row.delta_approx_evals = CounterDelta(candidate_summary.aggregate_stats.approx_distance_evals,
                                         row.baseline.aggregate_stats.approx_distance_evals);
    row.delta_exact_evals = CounterDelta(candidate_summary.aggregate_stats.exact_distance_evals,
                                        row.baseline.aggregate_stats.exact_distance_evals);
    row.delta_n_ios = CounterDelta(candidate_summary.aggregate_stats.n_ios, row.baseline.aggregate_stats.n_ios);
    row.delta_n_cmps = CounterDelta(candidate_summary.aggregate_stats.n_cmps, row.baseline.aggregate_stats.n_cmps);
    row.delta_n_hops = CounterDelta(candidate_summary.aggregate_stats.n_hops, row.baseline.aggregate_stats.n_hops);
    row.delta_cpu_us = CounterDelta(candidate_summary.aggregate_stats.cpu_us, row.baseline.aggregate_stats.cpu_us);
    row.delta_io_us = CounterDelta(candidate_summary.aggregate_stats.io_us, row.baseline.aggregate_stats.io_us);
    result.rows.push_back(std::move(row));
  }

  if (result.rows.empty()) {
    throw std::runtime_error("bench comparison did not find any matching runs");
  }

  std::sort(result.rows.begin(), result.rows.end(), [](const BenchComparisonRow &lhs, const BenchComparisonRow &rhs) {
    return SummaryRunKey(lhs.candidate) < SummaryRunKey(rhs.candidate);
  });
  return result;
}

void ExportBenchComparisonMarkdown(const std::string &path, const BenchComparisonSummary &summary) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench comparison markdown output");
  }

  out << "# Bench Comparison\n\n";
  out << "- Baseline: `" << summary.baseline_label << "`\n";
  out << "- Candidate: `" << summary.candidate_label << "`\n";
  out << "- Matched runs: " << summary.rows.size() << "\n\n";
  out << "| approx_kind | beam_width | l_search | graph_cache_budget_bytes | graph_cache_policy | refine_k | refine_ratio | defer_exact | scheduler_policy | scheduler_limit | dynamic_beam | baseline_qps | candidate_qps | delta_qps | baseline_recall | candidate_recall | delta_recall | baseline_latency_ms | candidate_latency_ms | delta_latency_ms |\n";
  out << "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &row : summary.rows) {
    out << "| " << ApproxKindName(row.candidate.approx_kind)
        << " | " << row.candidate.search_config.beam_width
        << " | " << row.candidate.search_config.l_search
        << " | " << row.candidate.search_config.graph_cache_budget_bytes
        << " | " << GraphCacheBuildPolicyName(row.candidate.search_config.graph_cache_policy)
        << " | " << row.candidate.search_config.refine_k
        << " | " << row.candidate.search_config.refine_ratio
        << " | " << (row.candidate.search_config.defer_exact_until_refinement ? 1 : 0)
        << " | " << SchedulerPolicyName(row.candidate.search_config.scheduler_policy)
        << " | " << row.candidate.search_config.scheduler_policy_limit
        << " | " << DynamicBeamPolicyName(row.candidate.search_config.dynamic_beam_policy)
        << " | " << row.baseline.qps
        << " | " << row.candidate.qps
        << " | " << row.delta_qps
        << " | " << (row.baseline.has_recall ? row.baseline.average_recall : -1.0)
        << " | " << (row.candidate.has_recall ? row.candidate.average_recall : -1.0)
        << " | " << row.delta_average_recall
        << " | " << row.baseline.average_latency_ms
        << " | " << row.candidate.average_latency_ms
        << " | " << row.delta_average_latency_ms
        << " |\n";
  }
}

void ExportBenchComparisonTsv(const std::string &path, const BenchComparisonSummary &summary) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench comparison TSV output");
  }

  out << "baseline_label\tcandidate_label\tapprox_kind\ttop_k\tbeam_width\tl_search\tgraph_cache_budget_bytes"
         "\tgraph_cache_policy\trefine_k\trefine_ratio\tdefer_exact_until_refinement"
         "\tscheduler_policy\tscheduler_policy_limit\tdynamic_beam_policy\tqueries"
         "\tbaseline_qps\tcandidate_qps\tdelta_qps"
         "\tbaseline_average_latency_ms\tcandidate_average_latency_ms\tdelta_average_latency_ms"
         "\tbaseline_average_recall\tcandidate_average_recall\tdelta_average_recall"
         "\tdelta_async_reads\tdelta_pages_completed\tdelta_resident_expansions\tdelta_approx_evals\tdelta_exact_evals"
         "\tdelta_bytes_read\tdelta_page_resident_hits\tdelta_graph_replicated_hits"
         "\tdelta_graph_cache_hits\tdelta_graph_cache_misses\tdelta_graph_cache_expansions"
         "\tdelta_graph_cache_avoided_reads\tdelta_graph_cache_resident_bytes\tdelta_graph_cache_entries"
         "\tdelta_graph_cache_build_page_reads"
         "\tdelta_exact_from_page\tdelta_exact_from_payload\tdelta_refinement_candidates"
         "\tdelta_refinement_reads\tdelta_approximate_candidates\tdelta_refinement_bound"
         "\tdelta_refinement_already_exact\tdelta_refinement_exactified\tdelta_deferred_exact_candidates"
         "\tdelta_read_hits_in_pool\tdelta_read_waste_out_of_pool"
         "\tdelta_max_inflight_reads\tdelta_max_beam_width\tdelta_beam_width_increases"
         "\tdelta_scheduler_pending_max\tdelta_scheduler_ready_unexpanded_max\tdelta_scheduler_limit_hits"
         "\tdelta_poll_calls\tdelta_drain_calls"
         "\tdelta_n_ios\tdelta_n_cmps\tdelta_n_hops\tdelta_cpu_us\tdelta_io_us\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &row : summary.rows) {
    out << summary.baseline_label << '\t'
        << summary.candidate_label << '\t'
        << ApproxKindName(row.candidate.approx_kind) << '\t'
        << row.candidate.search_config.top_k << '\t'
        << row.candidate.search_config.beam_width << '\t'
        << row.candidate.search_config.l_search << '\t'
        << row.candidate.search_config.graph_cache_budget_bytes << '\t'
        << GraphCacheBuildPolicyName(row.candidate.search_config.graph_cache_policy) << '\t'
        << row.candidate.search_config.refine_k << '\t'
        << row.candidate.search_config.refine_ratio << '\t'
        << (row.candidate.search_config.defer_exact_until_refinement ? 1 : 0) << '\t'
        << SchedulerPolicyName(row.candidate.search_config.scheduler_policy) << '\t'
        << row.candidate.search_config.scheduler_policy_limit << '\t'
        << DynamicBeamPolicyName(row.candidate.search_config.dynamic_beam_policy) << '\t'
        << row.candidate.num_queries << '\t'
        << row.baseline.qps << '\t'
        << row.candidate.qps << '\t'
        << row.delta_qps << '\t'
        << row.baseline.average_latency_ms << '\t'
        << row.candidate.average_latency_ms << '\t'
        << row.delta_average_latency_ms << '\t'
        << (row.baseline.has_recall ? row.baseline.average_recall : -1.0) << '\t'
        << (row.candidate.has_recall ? row.candidate.average_recall : -1.0) << '\t'
        << row.delta_average_recall << '\t'
        << row.delta_async_reads << '\t'
        << row.delta_pages_completed << '\t'
        << row.delta_resident_expansions << '\t'
        << row.delta_approx_evals << '\t'
        << row.delta_exact_evals << '\t'
        << row.delta_bytes_read << '\t'
        << row.delta_page_resident_hits << '\t'
        << row.delta_graph_replicated_hits << '\t'
        << row.delta_graph_cache_hits << '\t'
        << row.delta_graph_cache_misses << '\t'
        << row.delta_graph_cache_expansions << '\t'
        << row.delta_graph_cache_avoided_reads << '\t'
        << row.delta_graph_cache_resident_bytes << '\t'
        << row.delta_graph_cache_entries << '\t'
        << row.delta_graph_cache_build_page_reads << '\t'
        << row.delta_exact_from_page << '\t'
        << row.delta_exact_from_payload << '\t'
        << row.delta_refinement_candidates << '\t'
        << row.delta_refinement_reads << '\t'
        << row.delta_approximate_candidates << '\t'
        << row.delta_refinement_bound << '\t'
        << row.delta_refinement_already_exact << '\t'
        << row.delta_refinement_exactified << '\t'
        << row.delta_deferred_exact_candidates << '\t'
        << row.delta_read_hits_in_pool << '\t'
        << row.delta_read_waste_out_of_pool << '\t'
        << row.delta_max_inflight_reads << '\t'
        << row.delta_max_beam_width << '\t'
        << row.delta_beam_width_increases << '\t'
        << row.delta_scheduler_pending_max << '\t'
        << row.delta_scheduler_ready_unexpanded_max << '\t'
        << row.delta_scheduler_limit_hits << '\t'
        << row.delta_poll_calls << '\t'
        << row.delta_drain_calls << '\t'
        << row.delta_n_ios << '\t'
        << row.delta_n_cmps << '\t'
        << row.delta_n_hops << '\t'
        << row.delta_cpu_us << '\t'
        << row.delta_io_us << '\n';
  }
}

InspectToolSummary RunInspectTool(const InspectToolConfig &config) {
  std::unique_ptr<IndexReader> index = LoadIndexReader(config.index_path, config.approx_path);

  InspectToolSummary summary;
  summary.storage_format = index->storage_format();
  summary.index_metadata = index->search_metadata();
  summary.index_file_size = std::filesystem::file_size(config.index_path);
  summary.approx_file_size = index->ApproxFileSizeBytes();
  summary.has_partition = index->HasPartitionData();
  summary.has_reorder = index->HasReorderData();
  summary.reorder_entries = static_cast<uint64_t>(index->reorder_ids().size());
  if (const auto *project_index = dynamic_cast<const GraphReplicatedIndex *>(index.get())) {
    summary.has_project_metadata = true;
    summary.project_metadata = project_index->metadata();
  }
  if (const auto *native_index = dynamic_cast<const NativeGorgeousIndex *>(index.get())) {
    summary.has_native_metadata = true;
    summary.native_metadata = native_index->native_metadata();
    summary.native_payload = native_index->payload_info();
    summary.native_page_boundaries = native_index->page_boundaries();
  }

  const std::string pipeann_refine_sidecar_path =
      pipeann_integration::DefaultPipeannRefineSidecarPath(config.index_path);
  if (std::filesystem::exists(pipeann_refine_sidecar_path)) {
    const pipeann_integration::PipeannRefineSidecar refine =
        pipeann_integration::ReadPipeannRefineSidecar(pipeann_refine_sidecar_path);
    summary.has_pipeann_refine = true;
    summary.pipeann_refine_header = refine.header;
    for (const auto &node : refine.nodes) {
      if (!node.refine_neighbors.empty()) {
        ++summary.pipeann_refine_nodes;
        summary.pipeann_refine_edges += static_cast<uint32_t>(node.refine_neighbors.size());
      }
    }
  }
  const std::string pipeann_refine_nodes_path = config.index_path + ".pipeann.refine.nodes.tsv";
  summary.has_pipeann_refine_nodes = std::filesystem::exists(pipeann_refine_nodes_path);

  if (config.inspect_page) {
    std::ifstream in(config.index_path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("failed to open index file for page inspection");
    }
    std::vector<char> page_bytes(index->search_metadata().page_size);
    in.seekg(static_cast<std::streamoff>(index->PageOffset(config.page_id)));
    in.read(page_bytes.data(), static_cast<std::streamsize>(page_bytes.size()));
    if (!in) {
      throw std::runtime_error("failed to read index page for inspection");
    }
    summary.has_page = true;
    const PageView page = index->ViewPage(config.page_id, page_bytes);
    summary.page.page_id = page.page_id;
    summary.page.target_id = page.target_id;
    summary.page.layout = index->CopyPageLayout(page);
    summary.page.nnbrs.reserve(page.layout_size);
    summary.page.n_dense_nbrs.reserve(page.layout_size);
    for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
      const DiskNodeView node = index->ViewNode(page, slot);
      summary.page.nnbrs.push_back(node.nnbrs);
      summary.page.n_dense_nbrs.push_back(node.n_dense_nbrs);
    }
    if (summary.has_pipeann_refine) {
      const uint32_t target_id = summary.page.target_id;
      const pipeann_integration::PipeannRefineSidecar refine =
          pipeann_integration::ReadPipeannRefineSidecar(pipeann_refine_sidecar_path);
      if (target_id < refine.nodes.size()) {
        summary.page_refine_neighbors = refine.nodes[target_id].refine_neighbors;
        summary.page_refine_ehs = refine.nodes[target_id].ehs;
        summary.page_refine_degree = static_cast<uint32_t>(summary.page_refine_neighbors.size());
        for (uint16_t eh : summary.page_refine_ehs) {
          summary.page_refine_max_eh = std::max<uint16_t>(summary.page_refine_max_eh, eh);
        }
      }
    }
  }

  const bool has_explicit_pq_paths = !config.pq_codebook_path.empty() || !config.pq_codes_path.empty();
  const std::string pq_codebook_path =
      config.pq_codebook_path.empty() ? DefaultPipeannPqPivotsPath(config.index_path) : config.pq_codebook_path;
  const std::string pq_codes_path =
      config.pq_codes_path.empty() ? DefaultPipeannPqCompressedPath(config.index_path) : config.pq_codes_path;
  if (has_explicit_pq_paths || (std::filesystem::exists(pq_codebook_path) && std::filesystem::exists(pq_codes_path))) {
    auto pq = std::make_unique<ProductQuantizationDistanceComputer>(*index);
    pq->Load(pq_codebook_path, pq_codes_path);
    summary.has_pq = true;
    summary.pq_metadata = pq->metadata();
    summary.pq_codebook_file_size = std::filesystem::file_size(pq_codebook_path);
    summary.pq_codes_file_size = std::filesystem::file_size(pq_codes_path);
  }

  return summary;
}

}  // namespace hybrid
