#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_bench [options]\n"
      << "  --index_prefix PATH\n"
      << "  --approx PATH (optional full-precision data override)\n"
      << "  --queries PATH\n"
      << "  --query_format text|fvecs|bvecs|bin\n"
      << "  --ground_truth PATH\n"
      << "  --generate_ground_truth PATH\n"
      << "  --ground_truth_k N\n"
      << "  --top_k N\n"
      << "  --beam_width N\n"
      << "  --beam_widths v1,v2,...\n"
      << "  --l_search N\n"
      << "  --l_search_values v1,v2,...\n"
      << "  --threads N\n"
      << "  --thread_counts v1,v2,...\n"
      << "  --mem_l N\n"
      << "  --pipeann_layout equal|gorgeous\n"
      << "  --gp_partition PATH\n"
      << "  --pipeann_mode 0|1|2\n"
      << "  --pipeann_modes 0,1,2\n"
      << "  --pipeann_layouts equal,gorgeous\n"
      << "  --export PATH\n"
      << "  --experiment_dir PATH\n"
      << "  --experiment_root PATH\n"
      << "  --experiment_name NAME\n"
      << "  --help\n";
}

uint32_t ParseUint32(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint32_t>(std::stoul(value));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + name);
  }
}

hybrid::QueryInputMode ParseQueryInputMode(const std::string &value) {
  if (value == "text") {
    return hybrid::QueryInputMode::kText;
  }
  if (value == "fvecs") {
    return hybrid::QueryInputMode::kFvecs;
  }
  if (value == "bvecs") {
    return hybrid::QueryInputMode::kBvecs;
  }
  if (value == "bin") {
    return hybrid::QueryInputMode::kBin;
  }
  throw std::runtime_error("unsupported query_format: " + value);
}

hybrid::pipeann_parity::PipeannSearchMode ParsePipeannModeArg(const std::string &value) {
  if (value == "0") {
    return hybrid::pipeann_parity::PipeannSearchMode::kBeam;
  }
  if (value == "1") {
    return hybrid::pipeann_parity::PipeannSearchMode::kPage;
  }
  if (value == "2") {
    return hybrid::pipeann_parity::PipeannSearchMode::kPipe;
  }
  throw std::runtime_error("unsupported pipeann_mode: " + value);
}

hybrid::pipeann_parity::PipeannLayout ParsePipeannLayoutArg(const std::string &value) {
  if (value == "equal") {
    return hybrid::pipeann_parity::PipeannLayout::kEqual;
  }
  if (value == "gorgeous") {
    return hybrid::pipeann_parity::PipeannLayout::kGorgeous;
  }
  throw std::runtime_error("unsupported pipeann_layout: " + value);
}

std::vector<hybrid::pipeann_parity::PipeannSearchMode> ParsePipeannModeList(const std::string &text) {
  std::vector<hybrid::pipeann_parity::PipeannSearchMode> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (!token.empty()) {
      values.push_back(ParsePipeannModeArg(token));
    }
  }
  if (values.empty()) {
    throw std::runtime_error("pipeann mode list must not be empty");
  }
  return values;
}

std::vector<hybrid::pipeann_parity::PipeannLayout> ParsePipeannLayoutList(const std::string &text) {
  std::vector<hybrid::pipeann_parity::PipeannLayout> values;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (!token.empty()) {
      values.push_back(ParsePipeannLayoutArg(token));
    }
  }
  if (values.empty()) {
    throw std::runtime_error("pipeann layout list must not be empty");
  }
  return values;
}

const char *PipeannModeName(hybrid::pipeann_parity::PipeannSearchMode mode) {
  switch (mode) {
    case hybrid::pipeann_parity::PipeannSearchMode::kBeam:
      return "beam";
    case hybrid::pipeann_parity::PipeannSearchMode::kPage:
      return "page";
    case hybrid::pipeann_parity::PipeannSearchMode::kPipe:
      return "pipe";
  }
  throw std::runtime_error("unsupported pipeann mode");
}

const char *PipeannLayoutName(hybrid::pipeann_parity::PipeannLayout layout) {
  switch (layout) {
    case hybrid::pipeann_parity::PipeannLayout::kEqual:
      return "equal";
    case hybrid::pipeann_parity::PipeannLayout::kGorgeous:
      return "gorgeous";
  }
  throw std::runtime_error("unsupported pipeann layout");
}

struct ParsedBenchArgs {
  hybrid::BenchSweepConfig sweep_config;
  std::string export_path;
  std::string generate_ground_truth_path;
  std::string experiment_dir;
  std::string experiment_root;
  std::string experiment_name;
};

ParsedBenchArgs ParseArgs(int argc, char **argv) {
  ParsedBenchArgs parsed;
  hybrid::BenchToolConfig &config = parsed.sweep_config.base_config;
  std::string queries_path;
  hybrid::QueryInputMode query_format = hybrid::QueryInputMode::kText;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--help") {
      PrintUsage();
      std::exit(0);
    }
    if (arg == "--index_prefix") {
      config.index_prefix = need_value("--index_prefix");
      continue;
    }
    if (arg == "--approx") {
      config.approx_path = need_value("--approx");
      continue;
    }
    if (arg == "--queries") {
      queries_path = need_value("--queries");
      continue;
    }
    if (arg == "--query_format") {
      query_format = ParseQueryInputMode(need_value("--query_format"));
      continue;
    }
    if (arg == "--ground_truth") {
      config.ground_truth_ids = hybrid::LoadGroundTruthIds(need_value("--ground_truth"));
      continue;
    }
    if (arg == "--generate_ground_truth") {
      parsed.generate_ground_truth_path = need_value("--generate_ground_truth");
      continue;
    }
    if (arg == "--ground_truth_k") {
      config.recall_at_k = ParseUint32("--ground_truth_k", need_value("--ground_truth_k"));
      continue;
    }
    if (arg == "--top_k") {
      config.top_k = ParseUint32("--top_k", need_value("--top_k"));
      continue;
    }
    if (arg == "--beam_width") {
      config.beam_width = ParseUint32("--beam_width", need_value("--beam_width"));
      continue;
    }
    if (arg == "--beam_widths") {
      parsed.sweep_config.beam_widths = hybrid::ParseUint32List(need_value("--beam_widths"));
      continue;
    }
    if (arg == "--l_search") {
      config.l_search = ParseUint32("--l_search", need_value("--l_search"));
      continue;
    }
    if (arg == "--l_search_values") {
      parsed.sweep_config.l_search_values = hybrid::ParseUint32List(need_value("--l_search_values"));
      continue;
    }
    if (arg == "--threads") {
      config.num_threads = ParseUint32("--threads", need_value("--threads"));
      continue;
    }
    if (arg == "--thread_counts") {
      parsed.sweep_config.thread_counts = hybrid::ParseUint32List(need_value("--thread_counts"));
      continue;
    }
    if (arg == "--mem_l") {
      config.mem_l = ParseUint32("--mem_l", need_value("--mem_l"));
      continue;
    }
    if (arg == "--pipeann_layout") {
      config.pipeann_layout = ParsePipeannLayoutArg(need_value("--pipeann_layout"));
      continue;
    }
    if (arg == "--gp_partition") {
      config.gp_partition_path = need_value("--gp_partition");
      continue;
    }
    if (arg == "--pipeann_mode") {
      config.pipeann_mode = ParsePipeannModeArg(need_value("--pipeann_mode"));
      continue;
    }
    if (arg == "--pipeann_modes") {
      parsed.sweep_config.pipeann_modes = ParsePipeannModeList(need_value("--pipeann_modes"));
      continue;
    }
    if (arg == "--pipeann_layouts") {
      parsed.sweep_config.pipeann_layouts = ParsePipeannLayoutList(need_value("--pipeann_layouts"));
      continue;
    }
    if (arg == "--export") {
      parsed.export_path = need_value("--export");
      continue;
    }
    if (arg == "--experiment_dir") {
      parsed.experiment_dir = need_value("--experiment_dir");
      continue;
    }
    if (arg == "--experiment_root") {
      parsed.experiment_root = need_value("--experiment_root");
      continue;
    }
    if (arg == "--experiment_name") {
      parsed.experiment_name = need_value("--experiment_name");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (config.index_prefix.empty()) {
    throw std::runtime_error("--index_prefix is required");
  }
  if (queries_path.empty()) {
    throw std::runtime_error("--queries is required");
  }
  if (!config.ground_truth_ids.empty() && !parsed.generate_ground_truth_path.empty()) {
    throw std::runtime_error("--ground_truth and --generate_ground_truth cannot be used together");
  }
  if (!parsed.experiment_dir.empty() && !parsed.experiment_root.empty()) {
    throw std::runtime_error("--experiment_dir and --experiment_root cannot be used together");
  }
  config.queries = hybrid::LoadQueryVectors(queries_path, query_format);
  return parsed;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    ParsedBenchArgs parsed = ParseArgs(argc, argv);
    if (!parsed.experiment_root.empty()) {
      parsed.experiment_dir =
          hybrid::CreateExperimentDirectory(parsed.experiment_root, parsed.sweep_config, parsed.experiment_name);
    }
    if (!parsed.generate_ground_truth_path.empty()) {
      const uint32_t ground_truth_k =
          parsed.sweep_config.base_config.recall_at_k == 0
              ? parsed.sweep_config.base_config.top_k
              : std::max(parsed.sweep_config.base_config.top_k, parsed.sweep_config.base_config.recall_at_k);
      parsed.sweep_config.base_config.ground_truth_ids = hybrid::GenerateGroundTruthIds(
          parsed.sweep_config.base_config.index_prefix,
          parsed.sweep_config.base_config.approx_path,
          parsed.sweep_config.base_config.queries,
          ground_truth_k);
      hybrid::WriteGroundTruthIds(parsed.generate_ground_truth_path,
                                  parsed.sweep_config.base_config.ground_truth_ids);
    }

    const hybrid::BenchSweepSummary sweep = hybrid::RunBenchSweep(parsed.sweep_config);
    if (!parsed.export_path.empty()) {
      hybrid::ExportBenchSummariesTsv(parsed.export_path, sweep.runs);
    }
    if (!parsed.experiment_dir.empty()) {
      hybrid::ExportBenchExperiment(parsed.experiment_dir, parsed.sweep_config, sweep);
    }

    std::cout << "Bench completed\n";
    std::cout << "  engine=pipeann_gorgeous_layout\n";
    std::cout << "  index_prefix=" << parsed.sweep_config.base_config.index_prefix << '\n';
    std::cout << "  runs=" << sweep.runs.size() << '\n';
    if (!parsed.generate_ground_truth_path.empty()) {
      std::cout << "  generated_ground_truth=" << parsed.generate_ground_truth_path << '\n';
    }
    if (!parsed.export_path.empty()) {
      std::cout << "  export_tsv=" << parsed.export_path << '\n';
    }
    if (!parsed.experiment_dir.empty()) {
      std::cout << "  experiment_dir=" << parsed.experiment_dir << '\n';
    }
    for (const auto &summary : sweep.runs) {
      std::cout << "  run: layout=" << PipeannLayoutName(summary.pipeann_layout)
                << " mode=" << PipeannModeName(summary.pipeann_mode)
                << " threads=" << summary.num_threads
                << " beam_width=" << summary.beam_width
                << " l_search=" << summary.l_search
                << " mem_l=" << summary.mem_l
                << " qps=" << summary.qps
                << " avg_lat_us=" << summary.mean_latency_us
                << " p99_lat_us=" << summary.p99_latency_us
                << " mean_hops=" << summary.mean_hops
                << " mean_ios=" << summary.mean_ios
                << " unique_pg_io=" << summary.unique_pg_io
                << " dup_pg_hit=" << summary.dup_pg_hit
                << " polls=" << summary.polls;
      if (summary.has_recall) {
        std::cout << " recall_at_k=" << summary.average_recall;
      }
      std::cout << '\n';
      std::cout << "    first_query_results:\n";
      for (const auto &result : summary.first_query_results) {
        std::cout << "      id=" << result.id << " dist=" << result.distance << '\n';
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Bench failed: " << e.what() << '\n';
    return 1;
  }
}
