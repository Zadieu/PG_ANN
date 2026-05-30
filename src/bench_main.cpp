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
      << "  --graph_cache_bytes N\n"
      << "  --graph_cache_bytes_values v1,v2,...\n"
      << "  --graph_cache_policy entry_bfs|page_layout\n"
      << "  --graph_cache_policies entry_bfs,page_layout\n"
      << "  --refine_k N\n"
      << "  --refine_k_values v1,v2,...\n"
      << "  --refine_ratio F\n"
      << "  --refine_ratio_values v1,v2,...\n"
      << "  --defer_exact_until_refinement\n"
      << "  --defer_exact_values 0,1\n"
      << "  --scheduler_policy conservative|bounded|aggressive\n"
      << "  --scheduler_policies conservative,bounded,aggressive\n"
      << "  --scheduler_policy_limit N\n"
      << "  --scheduler_policy_limit_values v1,v2,...\n"
      << "  --dynamic_beam_policy adaptive|fixed\n"
      << "  --dynamic_beam_policies adaptive,fixed\n"
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
    if (arg == "--graph_cache_bytes") {
      config.search_config.graph_cache_budget_bytes =
          ParseUint64("--graph_cache_bytes", need_value("--graph_cache_bytes"));
      continue;
    }
    if (arg == "--graph_cache_bytes_values") {
      parsed.sweep_config.graph_cache_budget_bytes_values =
          hybrid::ParseUint64List(need_value("--graph_cache_bytes_values"));
      continue;
    }
    if (arg == "--graph_cache_policy") {
      config.search_config.graph_cache_policy = hybrid::ParseGraphCacheBuildPolicy(need_value("--graph_cache_policy"));
      continue;
    }
    if (arg == "--graph_cache_policies") {
      parsed.sweep_config.graph_cache_policies =
          hybrid::ParseGraphCachePolicyList(need_value("--graph_cache_policies"));
      continue;
    }
    if (arg == "--refine_k") {
      config.search_config.refine_k = ParseUint32("--refine_k", need_value("--refine_k"));
      continue;
    }
    if (arg == "--refine_k_values") {
      parsed.sweep_config.refine_k_values = hybrid::ParseUint32List(need_value("--refine_k_values"));
      continue;
    }
    if (arg == "--refine_ratio") {
      config.search_config.refine_ratio = ParseFloat("--refine_ratio", need_value("--refine_ratio"));
      continue;
    }
    if (arg == "--refine_ratio_values") {
      parsed.sweep_config.refine_ratio_values = hybrid::ParseFloatList(need_value("--refine_ratio_values"));
      continue;
    }
    if (arg == "--defer_exact_until_refinement") {
      config.search_config.defer_exact_until_refinement = true;
      continue;
    }
    if (arg == "--defer_exact_values") {
      parsed.sweep_config.defer_exact_until_refinement_values =
          hybrid::ParseBoolList(need_value("--defer_exact_values"));
      continue;
    }
    if (arg == "--scheduler_policy") {
      config.search_config.scheduler_policy = hybrid::ParseSchedulerPolicy(need_value("--scheduler_policy"));
      continue;
    }
    if (arg == "--scheduler_policies") {
      parsed.sweep_config.scheduler_policies = hybrid::ParseSchedulerPolicyList(need_value("--scheduler_policies"));
      continue;
    }
    if (arg == "--scheduler_policy_limit") {
      config.search_config.scheduler_policy_limit =
          ParseUint32("--scheduler_policy_limit", need_value("--scheduler_policy_limit"));
      continue;
    }
    if (arg == "--scheduler_policy_limit_values") {
      parsed.sweep_config.scheduler_policy_limit_values =
          hybrid::ParseUint32List(need_value("--scheduler_policy_limit_values"));
      continue;
    }
    if (arg == "--dynamic_beam_policy") {
      config.search_config.dynamic_beam_policy = hybrid::ParseDynamicBeamPolicy(need_value("--dynamic_beam_policy"));
      continue;
    }
    if (arg == "--dynamic_beam_policies") {
      parsed.sweep_config.dynamic_beam_policies =
          hybrid::ParseDynamicBeamPolicyList(need_value("--dynamic_beam_policies"));
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
                << " graph_cache_bytes=" << summary.search_config.graph_cache_budget_bytes
                << " graph_cache_policy=" << hybrid::GraphCacheBuildPolicyName(summary.search_config.graph_cache_policy)
                << " refine_k=" << summary.search_config.refine_k
                << " refine_ratio=" << summary.search_config.refine_ratio
                << " defer_exact_until_refinement="
                << (summary.search_config.defer_exact_until_refinement ? 1 : 0)
                << " scheduler_policy=" << hybrid::SchedulerPolicyName(summary.search_config.scheduler_policy)
                << " scheduler_policy_limit=" << summary.search_config.scheduler_policy_limit
                << " dynamic_beam_policy=" << hybrid::DynamicBeamPolicyName(summary.search_config.dynamic_beam_policy)
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
                << " bytes_read=" << summary.aggregate_stats.bytes_read
                << " page_resident_hits=" << summary.aggregate_stats.page_resident_hits
                << " graph_replicated_hits=" << summary.aggregate_stats.graph_replicated_hits
                << " graph_cache_hits=" << summary.aggregate_stats.graph_cache_hits
                << " graph_cache_misses=" << summary.aggregate_stats.graph_cache_misses
                << " graph_cache_expansions=" << summary.aggregate_stats.graph_cache_expansions
                << " graph_cache_avoided_reads=" << summary.aggregate_stats.graph_cache_avoided_reads
                << " graph_cache_resident_bytes=" << summary.aggregate_stats.graph_cache_resident_bytes
                << " graph_cache_entries=" << summary.aggregate_stats.graph_cache_entries
                << " graph_cache_build_page_reads=" << summary.aggregate_stats.graph_cache_build_page_reads
                << " exact_from_page=" << summary.aggregate_stats.exact_from_page
                << " exact_from_payload=" << summary.aggregate_stats.exact_from_payload
                << " refinement_candidates=" << summary.aggregate_stats.refinement_candidates
                << " refinement_reads=" << summary.aggregate_stats.refinement_reads
                << " approximate_candidates=" << summary.aggregate_stats.approximate_candidates
                << " refinement_bound=" << summary.aggregate_stats.refinement_bound
                << " refinement_already_exact=" << summary.aggregate_stats.refinement_already_exact
                << " refinement_exactified=" << summary.aggregate_stats.refinement_exactified
                << " deferred_exact_candidates=" << summary.aggregate_stats.deferred_exact_candidates
                << " read_hits_in_pool=" << summary.aggregate_stats.read_hits_in_pool
                << " read_waste_out_of_pool=" << summary.aggregate_stats.read_waste_out_of_pool
                << " max_inflight_reads=" << summary.aggregate_stats.max_inflight_reads
                << " max_beam_width=" << summary.aggregate_stats.max_beam_width
                << " beam_width_increases=" << summary.aggregate_stats.beam_width_increases
                << " scheduler_policy_limit_observed=" << summary.aggregate_stats.scheduler_policy_limit
                << " scheduler_pending_max=" << summary.aggregate_stats.scheduler_pending_max
                << " scheduler_ready_unexpanded_max=" << summary.aggregate_stats.scheduler_ready_unexpanded_max
                << " scheduler_limit_hits=" << summary.aggregate_stats.scheduler_limit_hits
                << " poll_calls=" << summary.aggregate_stats.poll_calls
                << " drain_calls=" << summary.aggregate_stats.drain_calls
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
