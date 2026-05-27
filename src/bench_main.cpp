#include <algorithm>
#include <exception>
#include <iostream>
#include <string>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_bench [options]\n"
      << "  --index PATH\n"
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
      << "  --mem_l N\n"
      << "  --range_partial F\n"
      << "  --approx_kind full|pq (default: pq)\n"
      << "  --approx_kinds full,pq\n"
      << "  --pq_codebook PATH (optional PQ pivots override)\n"
      << "  --pq_codes PATH (optional PQ compressed override)\n"
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

float ParseFloat(const std::string &name, const std::string &value) {
  try {
    return std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid float value for " + name);
  }
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
    if (arg == "--index") {
      config.index_path = need_value("--index");
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
      const std::string value = need_value("--query_format");
      if (value == "text") {
        query_format = hybrid::QueryInputMode::kText;
      } else if (value == "fvecs") {
        query_format = hybrid::QueryInputMode::kFvecs;
      } else if (value == "bvecs") {
        query_format = hybrid::QueryInputMode::kBvecs;
      } else if (value == "bin") {
        query_format = hybrid::QueryInputMode::kBin;
      } else {
        throw std::runtime_error("unsupported query_format: " + value);
      }
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
      config.search_config.top_k = ParseUint32("--top_k", need_value("--top_k"));
      continue;
    }
    if (arg == "--beam_width") {
      config.search_config.beam_width = ParseUint32("--beam_width", need_value("--beam_width"));
      continue;
    }
    if (arg == "--beam_widths") {
      parsed.sweep_config.beam_widths = hybrid::ParseUint32List(need_value("--beam_widths"));
      continue;
    }
    if (arg == "--l_search") {
      config.search_config.l_search = ParseUint32("--l_search", need_value("--l_search"));
      continue;
    }
    if (arg == "--l_search_values") {
      parsed.sweep_config.l_search_values = hybrid::ParseUint32List(need_value("--l_search_values"));
      continue;
    }
    if (arg == "--mem_l") {
      config.search_config.mem_l = ParseUint32("--mem_l", need_value("--mem_l"));
      continue;
    }
    if (arg == "--range_partial") {
      config.search_config.range_partial = ParseFloat("--range_partial", need_value("--range_partial"));
      continue;
    }
    if (arg == "--approx_kind") {
      const std::string value = need_value("--approx_kind");
      if (value == "full") {
        config.approx_kind = hybrid::ApproxDistanceKind::kFullPrecision;
      } else if (value == "pq") {
        config.approx_kind = hybrid::ApproxDistanceKind::kProductQuantization;
      } else {
        throw std::runtime_error("unsupported approx_kind: " + value);
      }
      continue;
    }
    if (arg == "--approx_kinds") {
      parsed.sweep_config.approx_kinds = hybrid::ParseApproxKindList(need_value("--approx_kinds"));
      continue;
    }
    if (arg == "--pq_codebook") {
      config.pq_codebook_path = need_value("--pq_codebook");
      continue;
    }
    if (arg == "--pq_codes") {
      config.pq_codes_path = need_value("--pq_codes");
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

  if (config.index_path.empty() || queries_path.empty()) {
    throw std::runtime_error("--index and --queries are required");
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
      parsed.experiment_dir = hybrid::CreateExperimentDirectory(parsed.experiment_root,
                                                                parsed.sweep_config,
                                                                parsed.experiment_name);
    }
    if (!parsed.generate_ground_truth_path.empty()) {
      const uint32_t ground_truth_k =
          parsed.sweep_config.base_config.recall_at_k == 0
              ? parsed.sweep_config.base_config.search_config.top_k
              : std::max(parsed.sweep_config.base_config.search_config.top_k,
                         parsed.sweep_config.base_config.recall_at_k);
      parsed.sweep_config.base_config.ground_truth_ids =
          hybrid::GenerateGroundTruthIds(parsed.sweep_config.base_config.index_path,
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
      std::cout << "  run: approx_kind=" << (summary.approx_kind == hybrid::ApproxDistanceKind::kProductQuantization ? "pq" : "full")
                << " beam_width=" << summary.search_config.beam_width
                << " l_search=" << summary.search_config.l_search
                << " queries=" << summary.num_queries
                << " elapsed_ms=" << summary.elapsed_ms
                << " avg_latency_ms=" << summary.average_latency_ms
                << " qps=" << summary.qps;
      if (summary.has_recall) {
        std::cout << " average_recall=" << summary.average_recall;
      }
      std::cout << '\n';
      std::cout << "    aggregate_stats: reads=" << summary.aggregate_stats.async_reads
                << " completed_pages=" << summary.aggregate_stats.pages_completed
                << " resident_expansions=" << summary.aggregate_stats.resident_expansions
                << " approx_evals=" << summary.aggregate_stats.approx_distance_evals
                << " exact_evals=" << summary.aggregate_stats.exact_distance_evals
                << " n_ios=" << summary.aggregate_stats.n_ios
                << " n_cmps=" << summary.aggregate_stats.n_cmps
                << " n_hops=" << summary.aggregate_stats.n_hops
                << " cpu_us=" << summary.aggregate_stats.cpu_us
                << " io_us=" << summary.aggregate_stats.io_us
                << " range_stop=" << (summary.aggregate_stats.range_stop ? 1 : 0) << '\n';
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
