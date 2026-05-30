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
      << "  --graph_cache_bytes N\n"
      << "  --graph_cache_policy entry_bfs|page_layout\n"
      << "  --refine_k N\n"
      << "  --refine_ratio F\n"
      << "  --defer_exact_until_refinement\n"
      << "  --scheduler_policy conservative|bounded|aggressive\n"
      << "  --scheduler_policy_limit N\n"
      << "  --dynamic_beam_policy adaptive|fixed\n"
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

uint64_t ParseUint64(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint64_t>(std::stoull(value));
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
    if (arg == "--graph_cache_bytes") {
      config.search_config.graph_cache_budget_bytes =
          ParseUint64("--graph_cache_bytes", need_value("--graph_cache_bytes"));
      continue;
    }
    if (arg == "--graph_cache_policy") {
      config.search_config.graph_cache_policy = hybrid::ParseGraphCacheBuildPolicy(need_value("--graph_cache_policy"));
      continue;
    }
    if (arg == "--refine_k") {
      config.search_config.refine_k = ParseUint32("--refine_k", need_value("--refine_k"));
      continue;
    }
    if (arg == "--refine_ratio") {
      config.search_config.refine_ratio = ParseFloat("--refine_ratio", need_value("--refine_ratio"));
      continue;
    }
    if (arg == "--defer_exact_until_refinement") {
      config.search_config.defer_exact_until_refinement = true;
      continue;
    }
    if (arg == "--scheduler_policy") {
      config.search_config.scheduler_policy = hybrid::ParseSchedulerPolicy(need_value("--scheduler_policy"));
      continue;
    }
    if (arg == "--scheduler_policy_limit") {
      config.search_config.scheduler_policy_limit =
          ParseUint32("--scheduler_policy_limit", need_value("--scheduler_policy_limit"));
      continue;
    }
    if (arg == "--dynamic_beam_policy") {
      config.search_config.dynamic_beam_policy = hybrid::ParseDynamicBeamPolicy(need_value("--dynamic_beam_policy"));
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
    std::cout << "  graph_cache_policy="
              << hybrid::GraphCacheBuildPolicyName(config.search_config.graph_cache_policy)
              << " refine_k=" << config.search_config.refine_k
              << " refine_ratio=" << config.search_config.refine_ratio
              << " defer_exact_until_refinement="
              << (config.search_config.defer_exact_until_refinement ? 1 : 0)
              << " scheduler_policy=" << hybrid::SchedulerPolicyName(config.search_config.scheduler_policy)
              << " scheduler_policy_limit=" << config.search_config.scheduler_policy_limit
              << " dynamic_beam_policy="
              << hybrid::DynamicBeamPolicyName(config.search_config.dynamic_beam_policy) << '\n';
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
              << " bytes_read=" << summary.stats.bytes_read
              << " page_resident_hits=" << summary.stats.page_resident_hits
              << " graph_replicated_hits=" << summary.stats.graph_replicated_hits
              << " graph_cache_hits=" << summary.stats.graph_cache_hits
              << " graph_cache_misses=" << summary.stats.graph_cache_misses
              << " graph_cache_expansions=" << summary.stats.graph_cache_expansions
              << " graph_cache_avoided_reads=" << summary.stats.graph_cache_avoided_reads
              << " graph_cache_resident_bytes=" << summary.stats.graph_cache_resident_bytes
              << " graph_cache_entries=" << summary.stats.graph_cache_entries
              << " graph_cache_build_page_reads=" << summary.stats.graph_cache_build_page_reads
              << " exact_from_page=" << summary.stats.exact_from_page
              << " exact_from_payload=" << summary.stats.exact_from_payload
              << " refinement_candidates=" << summary.stats.refinement_candidates
              << " refinement_reads=" << summary.stats.refinement_reads
              << " approximate_candidates=" << summary.stats.approximate_candidates
              << " refinement_bound=" << summary.stats.refinement_bound
              << " refinement_already_exact=" << summary.stats.refinement_already_exact
              << " refinement_exactified=" << summary.stats.refinement_exactified
              << " deferred_exact_candidates=" << summary.stats.deferred_exact_candidates
              << " read_hits_in_pool=" << summary.stats.read_hits_in_pool
              << " read_waste_out_of_pool=" << summary.stats.read_waste_out_of_pool
              << " max_inflight_reads=" << summary.stats.max_inflight_reads
              << " max_beam_width=" << summary.stats.max_beam_width
              << " beam_width_increases=" << summary.stats.beam_width_increases
              << " scheduler_policy_limit_observed=" << summary.stats.scheduler_policy_limit
              << " scheduler_pending_max=" << summary.stats.scheduler_pending_max
              << " scheduler_ready_unexpanded_max=" << summary.stats.scheduler_ready_unexpanded_max
              << " scheduler_limit_hits=" << summary.stats.scheduler_limit_hits
              << " poll_calls=" << summary.stats.poll_calls
              << " drain_calls=" << summary.stats.drain_calls
              << " range_stop=" << (summary.stats.range_stop ? 1 : 0) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Search failed: " << e.what() << '\n';
    return 1;
  }
}
