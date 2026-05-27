#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_search [options]\n"
      << "  --index PATH\n"
      << "  --approx PATH (optional full-precision data override)\n"
      << "  --query \"f1 f2 ...\"\n"
      << "  --query_file PATH\n"
      << "  --query_format text|fvecs|bvecs|bin\n"
      << "  --query_index N\n"
      << "  --top_k N\n"
      << "  --beam_width N\n"
      << "  --l_search N\n"
      << "  --mem_l N\n"
      << "  --range_partial F\n"
      << "  --approx_kind full|pq (default: pq)\n"
      << "  --pq_codebook PATH (optional PQ pivots override)\n"
      << "  --pq_codes PATH (optional PQ compressed override)\n"
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

hybrid::SearchToolConfig ParseArgs(int argc, char **argv) {
  hybrid::SearchToolConfig config;
  std::string query_text;
  std::string query_file_path;
  hybrid::QueryInputMode query_format = hybrid::QueryInputMode::kText;
  uint32_t query_index = 0;
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
    if (arg == "--query") {
      query_text = need_value("--query");
      continue;
    }
    if (arg == "--query_file") {
      query_file_path = need_value("--query_file");
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
    if (arg == "--query_index") {
      query_index = ParseUint32("--query_index", need_value("--query_index"));
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
    if (arg == "--l_search") {
      config.search_config.l_search = ParseUint32("--l_search", need_value("--l_search"));
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
    if (arg == "--pq_codebook") {
      config.pq_codebook_path = need_value("--pq_codebook");
      continue;
    }
    if (arg == "--pq_codes") {
      config.pq_codes_path = need_value("--pq_codes");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (config.index_path.empty()) {
    throw std::runtime_error("--index is required");
  }
  if (query_text.empty() && query_file_path.empty()) {
    throw std::runtime_error("--query or --query_file is required");
  }
  if (!query_file_path.empty()) {
    const std::vector<std::vector<float>> queries = hybrid::LoadQueryVectors(query_file_path, query_format);
    if (query_index >= queries.size()) {
      throw std::runtime_error("query_index out of range");
    }
    config.query = queries[query_index];
  } else {
    config.query = hybrid::ParseFloatVector(query_text);
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const hybrid::SearchToolConfig config = ParseArgs(argc, argv);
    const hybrid::SearchToolSummary summary = hybrid::RunSearchTool(config);
    std::cout << "Search completed\n";
    std::cout << "  approx_backend=" << summary.approx_backend_name << '\n';
    std::cout << "  results:\n";
    for (const auto &result : summary.results) {
      std::cout << "    id=" << result.id << " dist=" << result.distance << '\n';
    }
    std::cout << "  stats: reads=" << summary.stats.async_reads
              << " completed_pages=" << summary.stats.pages_completed
              << " resident_expansions=" << summary.stats.resident_expansions
              << " approx_evals=" << summary.stats.approx_distance_evals
              << " exact_evals=" << summary.stats.exact_distance_evals
              << " n_ios=" << summary.stats.n_ios
              << " n_cmps=" << summary.stats.n_cmps
              << " n_hops=" << summary.stats.n_hops
              << " cpu_us=" << summary.stats.cpu_us
              << " io_us=" << summary.stats.io_us
              << " range_stop=" << (summary.stats.range_stop ? 1 : 0) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Search failed: " << e.what() << '\n';
    return 1;
  }
}
