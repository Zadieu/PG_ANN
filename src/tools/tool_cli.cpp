#include "tools/tool_cli.h"

#include "tools/vector_io.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace hybrid {
namespace {

namespace fs = std::filesystem;

std::string DefaultGpPartitionPath(const std::string &index_prefix) {
  const fs::path gp_part = fs::path(index_prefix).string() + "_gp_part.bin";
  if (fs::exists(gp_part)) {
    return gp_part.string();
  }
  return fs::path(index_prefix).string() + "_partition.bin";
}

std::string ScaleLabelFromIndexPrefix(const std::string &index_prefix) {
  return fs::path(index_prefix).filename().string();
}

const char *PipeannModeName(pipeann_parity::PipeannSearchMode mode) {
  switch (mode) {
    case pipeann_parity::PipeannSearchMode::kBeam:
      return "beam";
    case pipeann_parity::PipeannSearchMode::kPage:
      return "page";
    case pipeann_parity::PipeannSearchMode::kPipe:
      return "pipe";
  }
  throw std::runtime_error("unsupported pipeann mode");
}

int PipeannModeNumber(pipeann_parity::PipeannSearchMode mode) {
  switch (mode) {
    case pipeann_parity::PipeannSearchMode::kBeam:
      return 0;
    case pipeann_parity::PipeannSearchMode::kPage:
      return 1;
    case pipeann_parity::PipeannSearchMode::kPipe:
      return 2;
  }
  throw std::runtime_error("unsupported pipeann mode");
}

pipeann_parity::PipeannSearchMode ParsePipeannModeName(const std::string &text) {
  if (text == "0" || text == "beam") {
    return pipeann_parity::PipeannSearchMode::kBeam;
  }
  if (text == "1" || text == "page") {
    return pipeann_parity::PipeannSearchMode::kPage;
  }
  if (text == "2" || text == "pipe") {
    return pipeann_parity::PipeannSearchMode::kPipe;
  }
  throw std::runtime_error("unsupported pipeann mode label: " + text);
}

const char *PipeannLayoutName(pipeann_parity::PipeannLayout layout) {
  switch (layout) {
    case pipeann_parity::PipeannLayout::kEqual:
      return "equal";
    case pipeann_parity::PipeannLayout::kGorgeous:
      return "gorgeous";
  }
  throw std::runtime_error("unsupported pipeann layout");
}

pipeann_parity::PipeannLayout ParsePipeannLayoutName(const std::string &text) {
  if (text == "equal") {
    return pipeann_parity::PipeannLayout::kEqual;
  }
  if (text == "gorgeous") {
    return pipeann_parity::PipeannLayout::kGorgeous;
  }
  throw std::runtime_error("unsupported pipeann layout label: " + text);
}

float SquaredL2Distance(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  float sum = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float delta = lhs[i] - rhs[i];
    sum += delta * delta;
  }
  return sum;
}

std::string SummaryRunKey(const BenchToolSummary &summary) {
  std::ostringstream key;
  key << PipeannLayoutName(summary.pipeann_layout) << '\t' << PipeannModeName(summary.pipeann_mode) << '\t'
      << summary.num_threads << '\t' << summary.beam_width << '\t' << summary.mem_l << '\t' << summary.l_search;
  return key.str();
}

BenchToolSummary MakeSummaryFromParityResult(const BenchToolConfig &config,
                                             const pipeann_parity::PipeannParityLResult &parity_result) {
  BenchToolSummary summary;
  summary.index_prefix = config.index_prefix;
  summary.num_queries = static_cast<uint32_t>(config.queries.size());
  summary.num_threads = config.num_threads == 0 ? 1 : config.num_threads;
  summary.top_k = config.top_k;
  summary.beam_width = config.beam_width;
  summary.l_search = config.l_search;
  summary.pipeann_mode = config.pipeann_mode;
  summary.pipeann_layout = config.pipeann_layout;
  summary.mem_l = config.mem_l;
  summary.elapsed_ms =
      parity_result.qps > 0.0 ? static_cast<double>(summary.num_queries) * 1000.0 / parity_result.qps : 0.0;
  summary.mean_latency_us = parity_result.mean_latency_us;
  summary.p99_latency_us = parity_result.p99_latency_us;
  summary.average_latency_ms = summary.mean_latency_us / 1000.0;
  summary.p99_latency_ms = summary.p99_latency_us / 1000.0;
  summary.qps = parity_result.qps;
  summary.mean_hops = parity_result.mean_hops;
  summary.mean_ios = parity_result.mean_ios;
  summary.unique_pg_io = parity_result.mean_unique_pg_io;
  summary.dup_pg_hit = parity_result.mean_dup_pg_hit;
  summary.polls = parity_result.mean_polls;
  if (!config.ground_truth_ids.empty()) {
    summary.has_recall = true;
    summary.average_recall = parity_result.recall_at_k;
  }
  return summary;
}

}  // namespace

std::vector<float> ParseFloatVector(const std::string &text) {
  std::vector<float> values;
  std::istringstream in(text);
  float value = 0.0f;
  while (in >> value) {
    values.push_back(value);
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
    values.push_back(static_cast<uint32_t>(std::stoul(token)));
  }
  if (values.empty()) {
    throw std::runtime_error("integer list must not be empty");
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

std::vector<std::vector<uint32_t>> GenerateGroundTruthIds(const std::string &index_prefix,
                                                          const std::string &base_data_path,
                                                          const std::vector<std::vector<float>> &queries,
                                                          uint32_t top_k) {
  if (queries.empty()) {
    throw std::runtime_error("ground truth generation requires at least one query");
  }
  if (top_k == 0) {
    throw std::runtime_error("ground truth generation requires top_k > 0");
  }

  const std::string base_path = base_data_path.empty() ? index_prefix + ".bin" : base_data_path;
  const std::vector<std::vector<float>> base_vectors = LoadBinVectors(base_path);
  if (base_vectors.empty()) {
    throw std::runtime_error("base vector file is empty");
  }
  const uint32_t dim = static_cast<uint32_t>(base_vectors.front().size());
  const uint32_t cutoff = std::min<uint32_t>(top_k, static_cast<uint32_t>(base_vectors.size()));

  std::vector<std::vector<uint32_t>> ground_truth;
  ground_truth.reserve(queries.size());
  for (const auto &query : queries) {
    if (query.size() != dim) {
      throw std::runtime_error("query dimensionality does not match base data during ground truth generation");
    }
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(base_vectors.size());
    for (uint32_t point_id = 0; point_id < base_vectors.size(); ++point_id) {
      ranked.emplace_back(SquaredL2Distance(query, base_vectors[point_id]), point_id);
    }
    std::partial_sort(ranked.begin(),
                      ranked.begin() + static_cast<std::ptrdiff_t>(cutoff),
                      ranked.end(),
                      [](const auto &lhs, const auto &rhs) {
                        if (lhs.first != rhs.first) {
                          return lhs.first < rhs.first;
                        }
                        return lhs.second < rhs.second;
                      });
    std::vector<uint32_t> ids;
    ids.reserve(cutoff);
    for (uint32_t i = 0; i < cutoff; ++i) {
      ids.push_back(ranked[i].second);
    }
    ground_truth.push_back(std::move(ids));
  }
  return ground_truth;
}

void WriteGroundTruthIds(const std::string &path, const std::vector<std::vector<uint32_t>> &ground_truth_ids) {
  const fs::path output_path(path);
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
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
  fs::create_directories(root_directory);
  const std::string stem =
      experiment_name.empty() ? (ScaleLabelFromIndexPrefix(config.base_config.index_prefix) + "_experiment")
                              : experiment_name;
  fs::path candidate = fs::path(root_directory) / stem;
  uint32_t suffix = 1;
  while (fs::exists(candidate)) {
    candidate = fs::path(root_directory) / (stem + "_" + std::to_string(suffix++));
  }
  fs::create_directories(candidate);
  return candidate.string();
}

BenchToolSummary RunPipeannParityBenchToolOnIndex(pipeann_parity::PipeannParityIndex &parity_index,
                                                  const BenchToolConfig &config) {
  if (config.index_prefix.empty()) {
    throw std::runtime_error("pipeann parity bench requires index_prefix");
  }
  if (config.queries.empty()) {
    throw std::runtime_error("pipeann parity bench requires at least one query");
  }
  if (config.top_k == 0 || config.beam_width == 0 || config.l_search == 0) {
    throw std::runtime_error("pipeann parity bench requires positive top_k, beam_width, and l_search");
  }
  if (!config.ground_truth_ids.empty() && config.ground_truth_ids.size() != config.queries.size()) {
    throw std::runtime_error("ground truth row count must match query count");
  }

  pipeann_parity::PipeannParityBenchConfig bench_config;
  bench_config.index = parity_index.config();
  bench_config.search.mode = config.pipeann_mode;
  bench_config.search.top_k = config.top_k;
  bench_config.search.l_search = config.l_search;
  bench_config.search.mem_l = config.mem_l;
  bench_config.search.beam_width = config.beam_width;
  bench_config.num_threads = config.num_threads == 0 ? 1 : config.num_threads;
  bench_config.queries = config.queries;
  bench_config.ground_truth_ids = config.ground_truth_ids;
  bench_config.recall_at_k = config.recall_at_k == 0 ? config.top_k : config.recall_at_k;
  pipeann_parity::EnsureFlatQueryData(bench_config, static_cast<uint32_t>(parity_index.index().meta_.data_dim));

  const pipeann_parity::PipeannParityLResult parity_result =
      pipeann_parity::RunPipeannParityQueries(parity_index, bench_config);

  BenchToolSummary summary = MakeSummaryFromParityResult(config, parity_result);

  if (!config.queries.empty()) {
    const uint32_t dim = static_cast<uint32_t>(parity_index.index().meta_.data_dim);
    const pipeann_parity::PipeannParitySearchConfig search = bench_config.search;
    const auto first_query = pipeann_parity::SearchOne(parity_index,
                                                       bench_config.flat_query_data.data(),
                                                       dim,
                                                       search);
    summary.first_query_results.reserve(first_query.ids.size());
    for (size_t i = 0; i < first_query.ids.size(); ++i) {
      summary.first_query_results.push_back({first_query.ids[i], first_query.distances[i]});
    }
  }
  return summary;
}

BenchToolSummary RunPipeannParityBenchTool(const BenchToolConfig &config) {
  pipeann_parity::PipeannParityIndexConfig index_config;
  index_config.index_prefix = config.index_prefix;
  index_config.metric = pipeann::Metric::L2;
  index_config.max_threads = config.num_threads == 0 ? 1 : config.num_threads;
  index_config.layout = config.pipeann_layout;
  index_config.gp_partition_path =
      config.gp_partition_path.empty() ? DefaultGpPartitionPath(config.index_prefix) : config.gp_partition_path;

  pipeann_parity::PipeannParityIndex parity_index(index_config);
  parity_index.Load();
  return RunPipeannParityBenchToolOnIndex(parity_index, config);
}

BenchToolSummary RunBenchTool(const BenchToolConfig &config) {
  return RunPipeannParityBenchTool(config);
}

BenchSweepSummary RunBenchSweep(const BenchSweepConfig &config) {
  BenchSweepSummary summary;
  const std::vector<uint32_t> beam_widths =
      config.beam_widths.empty() ? std::vector<uint32_t>{config.base_config.beam_width} : config.beam_widths;
  const std::vector<uint32_t> l_search_values =
      config.l_search_values.empty() ? std::vector<uint32_t>{config.base_config.l_search} : config.l_search_values;
  const std::vector<uint32_t> thread_counts =
      config.thread_counts.empty() ? std::vector<uint32_t>{config.base_config.num_threads == 0 ? 1u
                                                                                               : config.base_config.num_threads}
                                   : config.thread_counts;
  const std::vector<pipeann_parity::PipeannSearchMode> pipeann_modes =
      config.pipeann_modes.empty() ? std::vector<pipeann_parity::PipeannSearchMode>{config.base_config.pipeann_mode}
                                   : config.pipeann_modes;
  const std::vector<pipeann_parity::PipeannLayout> pipeann_layouts =
      config.pipeann_layouts.empty() ? std::vector<pipeann_parity::PipeannLayout>{config.base_config.pipeann_layout}
                                     : config.pipeann_layouts;

  for (pipeann_parity::PipeannLayout layout : pipeann_layouts) {
    for (pipeann_parity::PipeannSearchMode mode : pipeann_modes) {
      for (uint32_t thread_count : thread_counts) {
        pipeann_parity::PipeannParityIndexConfig index_config;
        index_config.index_prefix = config.base_config.index_prefix;
        index_config.metric = pipeann::Metric::L2;
        index_config.max_threads = thread_count;
        index_config.layout = layout;
        index_config.gp_partition_path = config.base_config.gp_partition_path.empty()
                                             ? DefaultGpPartitionPath(config.base_config.index_prefix)
                                             : config.base_config.gp_partition_path;

        pipeann_parity::PipeannParityIndex parity_index(index_config);
        parity_index.Load();

        for (uint32_t beam_width : beam_widths) {
          for (uint32_t l_search : l_search_values) {
            BenchToolConfig run_config = config.base_config;
            run_config.pipeann_layout = layout;
            run_config.pipeann_mode = mode;
            run_config.num_threads = thread_count;
            run_config.beam_width = beam_width;
            run_config.l_search = l_search;
            summary.runs.push_back(RunPipeannParityBenchToolOnIndex(parity_index, run_config));
          }
        }
      }
    }
  }
  return summary;
}

void ExportBenchSummariesTsv(const std::string &path, const std::vector<BenchToolSummary> &summaries) {
  const fs::path output_path(path);
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench export file");
  }
  out << "scale\tlayout\tmode\tthreads\tbeam\tmem_l\tL\tqps\tavg_lat_us\tp99_lat_us\tmean_hops\tmean_ios\t"
         "unique_pg_io\tdup_pg_hit\tpolls\trecall_at_k\tlog\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &summary : summaries) {
    out << ScaleLabelFromIndexPrefix(summary.index_prefix) << '\t' << PipeannLayoutName(summary.pipeann_layout) << '\t'
        << PipeannModeNumber(summary.pipeann_mode) << '\t' << summary.num_threads << '\t' << summary.beam_width << '\t'
        << summary.mem_l << '\t' << summary.l_search << '\t' << summary.qps << '\t' << summary.mean_latency_us << '\t'
        << summary.p99_latency_us << '\t' << summary.mean_hops << '\t' << summary.mean_ios << '\t'
        << summary.unique_pg_io << '\t' << summary.dup_pg_hit << '\t' << summary.polls << '\t'
        << (summary.has_recall ? summary.average_recall : -1.0) << '\t' << path << '\n';
  }
}

void ExportBenchExperiment(const std::string &directory,
                           const BenchSweepConfig &config,
                           const BenchSweepSummary &summary) {
  const fs::path root(directory);
  fs::create_directories(root);
  ExportBenchSummariesTsv((root / "summary.tsv").string(), summary.runs);
  std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
  manifest << "index_prefix=" << config.base_config.index_prefix << '\n';
  manifest << "queries=" << config.base_config.queries.size() << '\n';
}

void MergeBenchSummaryTsvFiles(const std::string &output_path, const std::vector<std::string> &input_paths) {
  if (input_paths.empty()) {
    throw std::runtime_error("merge requires at least one input file");
  }
  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open merge output file");
  }
  bool wrote_header = false;
  for (const auto &input_path : input_paths) {
    std::ifstream in(input_path);
    if (!in) {
      throw std::runtime_error("failed to open merge input file: " + input_path);
    }
    std::string line;
    if (!std::getline(in, line)) {
      continue;
    }
    if (!wrote_header) {
      out << line << '\n';
      wrote_header = true;
    }
    while (std::getline(in, line)) {
      if (!line.empty()) {
        out << line << '\n';
      }
    }
  }
}

std::vector<BenchToolSummary> LoadBenchSummariesTsv(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open bench summary file");
  }
  std::string header;
  if (!std::getline(in, header)) {
    throw std::runtime_error("bench summary file is empty");
  }
  std::vector<BenchToolSummary> summaries;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::string scale;
    std::string layout;
    int mode = 0;
    BenchToolSummary summary;
    std::string mem_l_text;
    std::string recall_text;
    std::string log_path;
    if (!(row >> scale >> layout >> mode >> summary.num_threads >> summary.beam_width >> mem_l_text >>
          summary.l_search >> summary.qps >> summary.mean_latency_us >> summary.p99_latency_us >> summary.mean_hops >>
          summary.mean_ios >> summary.unique_pg_io >> summary.dup_pg_hit >> summary.polls >> recall_text)) {
      continue;
    }
    std::getline(row, log_path);
    if (!log_path.empty() && log_path.front() == '\t') {
      log_path.erase(log_path.begin());
    }
    summary.index_prefix = scale;
    summary.pipeann_layout = ParsePipeannLayoutName(layout);
    summary.pipeann_mode = ParsePipeannModeName(std::to_string(mode));
    summary.mem_l = static_cast<uint32_t>(std::stoul(mem_l_text));
    summary.average_latency_ms = summary.mean_latency_us / 1000.0;
    summary.p99_latency_ms = summary.p99_latency_us / 1000.0;
    summary.elapsed_ms = summary.qps > 0.0 ? 1000.0 / summary.qps : 0.0;
    if (recall_text != "-1.000000" && recall_text != "-1") {
      summary.has_recall = true;
      summary.average_recall = std::stod(recall_text);
    }
    summaries.push_back(summary);
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
    const auto it = baseline_by_key.find(SummaryRunKey(candidate_summary));
    if (it == baseline_by_key.end()) {
      continue;
    }
    BenchComparisonRow row;
    row.baseline = it->second;
    row.candidate = candidate_summary;
    row.delta_elapsed_ms = candidate_summary.elapsed_ms - row.baseline.elapsed_ms;
    row.delta_mean_latency_us = candidate_summary.mean_latency_us - row.baseline.mean_latency_us;
    row.delta_p99_latency_us = candidate_summary.p99_latency_us - row.baseline.p99_latency_us;
    row.delta_qps = candidate_summary.qps - row.baseline.qps;
    row.delta_average_recall =
        (candidate_summary.has_recall ? candidate_summary.average_recall : 0.0) -
        (row.baseline.has_recall ? row.baseline.average_recall : 0.0);
    row.delta_mean_hops = candidate_summary.mean_hops - row.baseline.mean_hops;
    row.delta_mean_ios = candidate_summary.mean_ios - row.baseline.mean_ios;
    row.delta_unique_pg_io = candidate_summary.unique_pg_io - row.baseline.unique_pg_io;
    row.delta_dup_pg_hit = candidate_summary.dup_pg_hit - row.baseline.dup_pg_hit;
    row.delta_polls = candidate_summary.polls - row.baseline.polls;
    result.rows.push_back(std::move(row));
  }
  if (result.rows.empty()) {
    throw std::runtime_error("bench comparison did not find any matching runs");
  }
  return result;
}

void ExportBenchComparisonMarkdown(const std::string &path, const BenchComparisonSummary &summary) {
  const fs::path output_path(path);
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench comparison markdown output");
  }
  out << "# Bench Comparison\n\n";
  out << "- Baseline: `" << summary.baseline_label << "`\n";
  out << "- Candidate: `" << summary.candidate_label << "`\n";
  out << "- Matched runs: " << summary.rows.size() << "\n\n";
  out << "| layout | mode | threads | beam | L | baseline_qps | candidate_qps | delta_qps | baseline_recall | "
         "candidate_recall | delta_recall | baseline_lat_us | candidate_lat_us | delta_lat_us |\n";
  out << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &row : summary.rows) {
    out << "| " << PipeannLayoutName(row.candidate.pipeann_layout) << " | "
        << PipeannModeName(row.candidate.pipeann_mode) << " | " << row.candidate.num_threads << " | "
        << row.candidate.beam_width << " | " << row.candidate.l_search << " | " << row.baseline.qps << " | "
        << row.candidate.qps << " | " << row.delta_qps << " | "
        << (row.baseline.has_recall ? row.baseline.average_recall : -1.0) << " | "
        << (row.candidate.has_recall ? row.candidate.average_recall : -1.0) << " | " << row.delta_average_recall
        << " | " << row.baseline.mean_latency_us << " | " << row.candidate.mean_latency_us << " | "
        << row.delta_mean_latency_us << " |\n";
  }
}

void ExportBenchComparisonTsv(const std::string &path, const BenchComparisonSummary &summary) {
  const fs::path output_path(path);
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open bench comparison TSV output");
  }
  out << "layout\tmode\tthreads\tbeam\tmem_l\tL\tdelta_qps\tdelta_avg_lat_us\tdelta_p99_lat_us\tdelta_recall\t"
         "delta_mean_hops\tdelta_mean_ios\tdelta_unique_pg_io\tdelta_dup_pg_hit\tdelta_polls\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &row : summary.rows) {
    out << PipeannLayoutName(row.candidate.pipeann_layout) << '\t' << PipeannModeName(row.candidate.pipeann_mode)
        << '\t' << row.candidate.num_threads << '\t' << row.candidate.beam_width << '\t' << row.candidate.mem_l << '\t'
        << row.candidate.l_search << '\t' << row.delta_qps << '\t' << row.delta_mean_latency_us << '\t'
        << row.delta_p99_latency_us << '\t' << row.delta_average_recall << '\t' << row.delta_mean_hops << '\t'
        << row.delta_mean_ios << '\t' << row.delta_unique_pg_io << '\t' << row.delta_dup_pg_hit << '\t'
        << row.delta_polls << '\n';
  }
}

}  // namespace hybrid
