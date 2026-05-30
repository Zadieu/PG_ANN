#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tools/build_pipeline.h"
#include "tools/tool_cli.h"

namespace {

void WriteFvecs(const std::filesystem::path &path, const std::vector<std::vector<float>> &vectors) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  for (const auto &vector : vectors) {
    const uint32_t dim = static_cast<uint32_t>(vector.size());
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char *>(vector.data()), static_cast<std::streamsize>(dim * sizeof(float)));
  }
}

void WriteBvecs(const std::filesystem::path &path, const std::vector<std::vector<uint8_t>> &vectors) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  for (const auto &vector : vectors) {
    const uint32_t dim = static_cast<uint32_t>(vector.size());
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char *>(vector.data()), static_cast<std::streamsize>(dim));
  }
}

void WriteBin(const std::filesystem::path &path, const std::vector<std::vector<float>> &vectors) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  const uint32_t num_points = static_cast<uint32_t>(vectors.size());
  const uint32_t dim = static_cast<uint32_t>(vectors.front().size());
  out.write(reinterpret_cast<const char *>(&num_points), sizeof(num_points));
  out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  for (const auto &vector : vectors) {
    out.write(reinterpret_cast<const char *>(vector.data()), static_cast<std::streamsize>(dim * sizeof(float)));
  }
}

}  // namespace

int main() {
  const std::filesystem::path out_dir = std::filesystem::path("test_tool_cli_data");
  std::filesystem::create_directories(out_dir);

  const std::filesystem::path train_queries_path = out_dir / "train_queries.txt";
  {
    std::ofstream out(train_queries_path);
    assert(out);
    out << "1.0 0.05 0.0\n";
    out << "0.75 0.15 0.05\n";
    out << "0.05 0.95 0.2\n";
  }

  hybrid::BuildConfig build_config;
  build_config.input_mode = hybrid::VectorInputMode::kToy;
  build_config.train_query_mode = hybrid::VectorInputMode::kText;
  build_config.train_query_path = train_queries_path.string();
  build_config.output_dir = out_dir.string();
  build_config.dataset_name = "tool_cli";
  build_config.degree = 4;
  build_config.r_ood = 1;
  build_config.page_nodes = 4;
  build_config.entry_id = 0;
  build_config.pq_subspaces = 3;
  build_config.pq_centroids = 4;
  const hybrid::BuildArtifacts artifacts = hybrid::RunBuildPipeline(build_config);

  hybrid::InspectToolConfig inspect_config;
  inspect_config.index_path = artifacts.index_path;
  inspect_config.approx_path = artifacts.approx_path;
  inspect_config.inspect_page = true;
  inspect_config.page_id = 0;
  inspect_config.pq_codebook_path = artifacts.pq_codebook_path;
  inspect_config.pq_codes_path = artifacts.pq_codes_path;
  const hybrid::InspectToolSummary inspect_summary = hybrid::RunInspectTool(inspect_config);
  assert(inspect_summary.index_metadata.num_points == artifacts.num_points);
  assert(inspect_summary.index_metadata.num_pages == artifacts.num_points);
  assert(inspect_summary.index_metadata.dim == artifacts.dim);
  assert(inspect_summary.storage_format == hybrid::IndexStorageFormat::kGorgeousNative);
  assert(!inspect_summary.has_project_metadata);
  assert(inspect_summary.has_native_metadata);
  assert(inspect_summary.native_metadata.num_points == artifacts.num_points);
  assert(inspect_summary.native_page_boundaries.size() == artifacts.num_points + 1);
  assert(inspect_summary.has_partition);
  assert(inspect_summary.has_reorder);
  assert(inspect_summary.reorder_entries >= inspect_summary.index_metadata.num_pages);
  assert(inspect_summary.has_page);
  assert(inspect_summary.page.target_id == 0);
  assert(!inspect_summary.page.layout.empty());
  assert(!inspect_summary.has_pipeann_refine);
  assert(!inspect_summary.has_pipeann_refine_nodes);
  assert(inspect_summary.page_refine_degree == 0);
  assert(inspect_summary.page_refine_neighbors.empty());
  assert(inspect_summary.has_pq);
  assert(inspect_summary.pq_metadata.num_subspaces == 3);

  hybrid::SearchToolConfig pq_search_config;
  pq_search_config.index_path = artifacts.index_path;
  pq_search_config.approx_path = artifacts.approx_path;
  pq_search_config.query = {0.95f, 0.15f, 0.0f};
  pq_search_config.search_config.top_k = 4;
  pq_search_config.search_config.beam_width = 3;
  pq_search_config.search_config.l_search = 8;
  pq_search_config.pq_codebook_path = artifacts.pq_codebook_path;
  pq_search_config.pq_codes_path = artifacts.pq_codes_path;
  const hybrid::SearchToolSummary pq_summary = hybrid::RunSearchTool(pq_search_config);
  assert(!pq_summary.results.empty());
  assert(pq_summary.results.front().id == 0);
  assert(pq_summary.approx_backend_name == "pipeann_pq");

  hybrid::SearchToolConfig full_search_config = pq_search_config;
  full_search_config.approx_kind = hybrid::ApproxDistanceKind::kFullPrecision;
  const hybrid::SearchToolSummary full_summary = hybrid::RunSearchTool(full_search_config);
  assert(!full_summary.results.empty());
  assert(full_summary.results.front().id == 0);
  assert(full_summary.approx_backend_name == "full_precision");

  const std::vector<float> parsed = hybrid::ParseFloatVector("1.0 0.5 -0.25");
  assert(parsed.size() == 3);
  assert(parsed[2] == -0.25f);

  hybrid::BenchToolConfig bench_config;
  bench_config.index_path = artifacts.index_path;
  bench_config.approx_path = artifacts.approx_path;
  bench_config.pq_codebook_path = artifacts.pq_codebook_path;
  bench_config.pq_codes_path = artifacts.pq_codes_path;
  bench_config.search_config = pq_search_config.search_config;
  bench_config.queries = {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.3f},
      {-1.0f, 0.0f, 0.6f},
  };
  bench_config.recall_at_k = 1;
  bench_config.ground_truth_ids.clear();
  for (const auto &query : bench_config.queries) {
    hybrid::SearchToolConfig truth_config;
    truth_config.index_path = artifacts.index_path;
    truth_config.approx_path = artifacts.approx_path;
    truth_config.query = query;
    truth_config.search_config = bench_config.search_config;
    truth_config.approx_kind = hybrid::ApproxDistanceKind::kFullPrecision;
    const hybrid::SearchToolSummary truth_summary = hybrid::RunSearchTool(truth_config);
    assert(!truth_summary.results.empty());
    bench_config.ground_truth_ids.push_back({truth_summary.results.front().id});
  }

  hybrid::BenchToolConfig full_bench_config = bench_config;
  full_bench_config.approx_kind = hybrid::ApproxDistanceKind::kFullPrecision;
  const hybrid::BenchToolSummary full_bench_summary = hybrid::RunBenchTool(full_bench_config);
  assert(full_bench_summary.num_queries == 3);
  assert(!full_bench_summary.first_query_results.empty());
  assert(full_bench_summary.approx_backend_name == "full_precision");
  assert(full_bench_summary.has_recall);
  assert(full_bench_summary.average_recall >= 0.0);
  assert(full_bench_summary.average_recall <= 1.0);

  hybrid::BenchToolConfig pq_bench_config = bench_config;
  pq_bench_config.approx_kind = hybrid::ApproxDistanceKind::kProductQuantization;
  const hybrid::BenchToolSummary pq_bench_summary = hybrid::RunBenchTool(pq_bench_config);
  assert(pq_bench_summary.num_queries == 3);
  assert(!pq_bench_summary.first_query_results.empty());
  assert(pq_bench_summary.approx_backend_name == "pipeann_pq");
  assert(pq_bench_summary.aggregate_stats.bytes_read > 0);
  assert(pq_bench_summary.aggregate_stats.page_resident_hits ==
         pq_bench_summary.aggregate_stats.resident_expansions);
  assert(pq_bench_summary.aggregate_stats.exact_from_page +
             pq_bench_summary.aggregate_stats.exact_from_payload ==
         pq_bench_summary.aggregate_stats.exact_distance_evals);
  assert(pq_bench_summary.aggregate_stats.approx_distance_evals >= pq_bench_summary.aggregate_stats.exact_distance_evals);
  assert(pq_bench_summary.has_recall);
  assert(pq_bench_summary.average_recall >= 0.0);
  assert(pq_bench_summary.average_recall <= 1.0);

  const std::vector<std::vector<float>> file_queries = {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
  };
  const std::filesystem::path text_queries = out_dir / "queries.txt";
  {
    std::ofstream out(text_queries);
    assert(out);
    out << "1.0 2.0 3.0\n";
    out << "4.0 5.0 6.0\n";
  }
  assert(hybrid::LoadQueryVectors(text_queries.string(), hybrid::QueryInputMode::kText).size() == 2);

  const std::filesystem::path fvecs_queries = out_dir / "queries.fvecs";
  WriteFvecs(fvecs_queries, file_queries);
  const std::vector<std::vector<float>> loaded_fvecs =
      hybrid::LoadQueryVectors(fvecs_queries.string(), hybrid::QueryInputMode::kFvecs);
  assert(loaded_fvecs.size() == 2);
  assert(loaded_fvecs[1][2] == 6.0f);

  const std::filesystem::path bvecs_queries = out_dir / "queries.bvecs";
  WriteBvecs(bvecs_queries, {{1, 2, 3}, {4, 5, 6}});
  const std::vector<std::vector<float>> loaded_bvecs =
      hybrid::LoadQueryVectors(bvecs_queries.string(), hybrid::QueryInputMode::kBvecs);
  assert(loaded_bvecs.size() == 2);
  assert(loaded_bvecs[0][1] == 2.0f);

  const std::filesystem::path bin_queries = out_dir / "queries.bin";
  WriteBin(bin_queries, file_queries);
  const std::vector<std::vector<float>> loaded_bin =
      hybrid::LoadQueryVectors(bin_queries.string(), hybrid::QueryInputMode::kBin);
  assert(loaded_bin.size() == 2);
  assert(loaded_bin[0][0] == 1.0f);

  const std::vector<uint32_t> parsed_uints = hybrid::ParseUint32List("2,4,8");
  assert(parsed_uints.size() == 3);
  const std::vector<hybrid::ApproxDistanceKind> parsed_kinds = hybrid::ParseApproxKindList("full,pq");
  assert(parsed_kinds.size() == 2);

  const std::filesystem::path ground_truth_path = out_dir / "ground_truth.txt";
  {
    std::ofstream out(ground_truth_path);
    assert(out);
    out << "0\n";
    out << "3\n";
    out << "6\n";
  }
  const std::vector<std::vector<uint32_t>> loaded_truth = hybrid::LoadGroundTruthIds(ground_truth_path.string());
  assert(loaded_truth.size() == 3);
  assert(loaded_truth[1][0] == 3);

  const std::vector<std::vector<uint32_t>> generated_truth =
      hybrid::GenerateGroundTruthIds(artifacts.index_path, artifacts.approx_path, bench_config.queries, 2);
  assert(generated_truth.size() == bench_config.queries.size());
  assert(generated_truth[0].size() == 2);
  assert(generated_truth[0][0] == 0);
  const std::filesystem::path generated_truth_path = out_dir / "generated_ground_truth.txt";
  hybrid::WriteGroundTruthIds(generated_truth_path.string(), generated_truth);
  const std::vector<std::vector<uint32_t>> reloaded_generated_truth =
      hybrid::LoadGroundTruthIds(generated_truth_path.string());
  assert(reloaded_generated_truth == generated_truth);
  assert(hybrid::ParseGraphCachePolicyList("entry_bfs,page_layout").size() == 2);
  assert(hybrid::ParseBoolList("0,1,true,false").size() == 4);
  assert(hybrid::ParseSchedulerPolicyList("conservative,bounded,aggressive").size() == 3);
  assert(hybrid::ParseDynamicBeamPolicyList("adaptive,fixed").size() == 2);

  hybrid::BenchSweepConfig sweep_config;
  sweep_config.base_config = pq_bench_config;
  sweep_config.beam_widths = {2, 3};
  sweep_config.l_search_values = {6};
  sweep_config.graph_cache_budget_bytes_values = {0, 8192};
  sweep_config.graph_cache_policies = {hybrid::GraphCacheBuildPolicy::kEntryBfs};
  sweep_config.refine_k_values = {4};
  sweep_config.refine_ratio_values = {0.0f};
  sweep_config.defer_exact_until_refinement_values = {0, 1};
  sweep_config.scheduler_policies = {hybrid::SearchConfig::SchedulerPolicy::kConservative,
                                     hybrid::SearchConfig::SchedulerPolicy::kBounded};
  sweep_config.scheduler_policy_limit_values = {0};
  sweep_config.dynamic_beam_policies = {hybrid::SearchConfig::DynamicBeamPolicy::kAdaptive};
  sweep_config.approx_kinds = {hybrid::ApproxDistanceKind::kFullPrecision,
                               hybrid::ApproxDistanceKind::kProductQuantization};
  const hybrid::BenchSweepSummary sweep_summary = hybrid::RunBenchSweep(sweep_config);
  assert(sweep_summary.runs.size() == 32);
  bool saw_full_precision = false;
  bool saw_product_quantization = false;
  bool saw_graph_cache = false;
  bool saw_deferred_exact = false;
  bool saw_bounded_scheduler = false;
  for (const auto &run : sweep_summary.runs) {
    assert(run.has_recall);
    assert(run.average_recall >= 0.0);
    assert(run.average_recall <= 1.0);
    if (run.approx_kind == hybrid::ApproxDistanceKind::kFullPrecision) {
      saw_full_precision = true;
    }
    if (run.approx_kind == hybrid::ApproxDistanceKind::kProductQuantization) {
      saw_product_quantization = true;
    }
    if (run.search_config.graph_cache_budget_bytes != 0) {
      saw_graph_cache = true;
      assert(run.aggregate_stats.graph_cache_entries > 0);
      assert(run.aggregate_stats.graph_cache_hits > 0);
    }
    assert(run.search_config.refine_k == 4);
    assert(run.aggregate_stats.refinement_bound >= run.search_config.top_k);
    assert(run.aggregate_stats.scheduler_policy_limit > 0);
    assert(run.aggregate_stats.scheduler_pending_max > 0);
    if (run.search_config.scheduler_policy == hybrid::SearchConfig::SchedulerPolicy::kBounded) {
      saw_bounded_scheduler = true;
    }
    if (run.search_config.defer_exact_until_refinement) {
      saw_deferred_exact = true;
      assert(run.aggregate_stats.deferred_exact_candidates > 0);
      assert(run.aggregate_stats.refinement_exactified > 0);
    }
  }
  assert(saw_full_precision);
  assert(saw_product_quantization);
  assert(saw_graph_cache);
  assert(saw_deferred_exact);
  assert(saw_bounded_scheduler);

  const std::filesystem::path export_path = out_dir / "bench.tsv";
  hybrid::ExportBenchSummariesTsv(export_path.string(), sweep_summary.runs);
  assert(std::filesystem::exists(export_path));
  std::ifstream export_in(export_path);
  assert(export_in);
  std::stringstream buffer;
  buffer << export_in.rdbuf();
  const std::string export_text = buffer.str();
  assert(export_text.find("approx_kind") != std::string::npos);
  assert(export_text.find("average_recall") != std::string::npos);
  assert(export_text.find("bytes_read") != std::string::npos);
  assert(export_text.find("graph_replicated_hits") != std::string::npos);
  assert(export_text.find("graph_cache_hits") != std::string::npos);
  assert(export_text.find("graph_cache_budget_bytes") != std::string::npos);
  assert(export_text.find("graph_cache_policy") != std::string::npos);
  assert(export_text.find("refine_k") != std::string::npos);
  assert(export_text.find("defer_exact_until_refinement") != std::string::npos);
  assert(export_text.find("deferred_exact_candidates") != std::string::npos);
  assert(export_text.find("scheduler_policy") != std::string::npos);
  assert(export_text.find("scheduler_pending_max") != std::string::npos);

  const std::filesystem::path experiment_dir = out_dir / "experiment";
  hybrid::ExportBenchExperiment(experiment_dir.string(), sweep_config, sweep_summary);
  assert(std::filesystem::exists(experiment_dir / "summary.tsv"));
  assert(std::filesystem::exists(experiment_dir / "manifest.txt"));
  assert(std::filesystem::exists(experiment_dir / "ground_truth.txt"));
  assert(std::filesystem::exists(experiment_dir / "runs" / "run_000.txt"));
  std::ifstream manifest_in(experiment_dir / "manifest.txt");
  assert(manifest_in);
  std::stringstream manifest_buffer;
  manifest_buffer << manifest_in.rdbuf();
  const std::string manifest_text = manifest_buffer.str();
  assert(manifest_text.find("query_count=3") != std::string::npos);
  assert(manifest_text.find("approx_kinds=full,pq") != std::string::npos);
  assert(manifest_text.find("graph_cache_budget_bytes_values=0,8192") != std::string::npos);
  assert(manifest_text.find("graph_cache_policies=entry_bfs") != std::string::npos);
  assert(manifest_text.find("refine_k_values=4") != std::string::npos);
  assert(manifest_text.find("refine_ratio_values=0.000000") != std::string::npos);
  assert(manifest_text.find("defer_exact_until_refinement_values=0,1") != std::string::npos);
  assert(manifest_text.find("scheduler_policies=conservative,bounded") != std::string::npos);
  assert(manifest_text.find("scheduler_policy_limit_values=0") != std::string::npos);
  assert(manifest_text.find("dynamic_beam_policies=adaptive") != std::string::npos);
  assert(manifest_text.find("pipeann_base_data=") != std::string::npos);
  assert(manifest_text.find("full_data=") != std::string::npos);
  assert(manifest_text.find("pipeann_pq_pivots=") != std::string::npos);

  const std::filesystem::path auto_root = out_dir / "auto_experiments";
  const std::string auto_experiment_dir =
      hybrid::CreateExperimentDirectory(auto_root.string(), sweep_config, "toy sweep");
  assert(std::filesystem::exists(auto_experiment_dir));
  assert(std::filesystem::path(auto_experiment_dir).parent_path() == auto_root);
  assert(std::filesystem::path(auto_experiment_dir).filename().string().find("toy_sweep") != std::string::npos);

  const std::filesystem::path second_experiment_dir = out_dir / "experiment_second";
  hybrid::ExportBenchExperiment(second_experiment_dir.string(), sweep_config, sweep_summary);
  const std::filesystem::path merged_summary = out_dir / "merged.tsv";
  hybrid::MergeBenchSummaryTsvFiles(merged_summary.string(),
                                    {experiment_dir.string(), second_experiment_dir.string()});
  assert(std::filesystem::exists(merged_summary));
  std::ifstream merged_in(merged_summary);
  assert(merged_in);
  std::stringstream merged_buffer;
  merged_buffer << merged_in.rdbuf();
  const std::string merged_text = merged_buffer.str();
  assert(merged_text.find("approx_kind") != std::string::npos);
  assert(merged_text.find("pipeann_pq") != std::string::npos);

  const std::vector<hybrid::BenchToolSummary> loaded_baseline =
      hybrid::LoadBenchSummariesTsv(experiment_dir.string());
  const std::vector<hybrid::BenchToolSummary> loaded_candidate =
      hybrid::LoadBenchSummariesTsv(second_experiment_dir.string());
  assert(loaded_baseline.size() == sweep_summary.runs.size());
  assert(loaded_candidate.size() == sweep_summary.runs.size());
  const hybrid::BenchComparisonSummary comparison =
      hybrid::CompareBenchSummaries("baseline", loaded_baseline, "candidate", loaded_candidate);
  assert(!comparison.rows.empty());
  assert(comparison.rows.size() == sweep_summary.runs.size());
  const std::filesystem::path comparison_markdown = out_dir / "comparison.md";
  hybrid::ExportBenchComparisonMarkdown(comparison_markdown.string(), comparison);
  assert(std::filesystem::exists(comparison_markdown));
  std::ifstream markdown_in(comparison_markdown);
  assert(markdown_in);
  std::stringstream markdown_buffer;
  markdown_buffer << markdown_in.rdbuf();
  const std::string markdown_text = markdown_buffer.str();
  assert(markdown_text.find("# Bench Comparison") != std::string::npos);
  assert(markdown_text.find("baseline_qps") != std::string::npos);

  const std::filesystem::path comparison_tsv = out_dir / "comparison.tsv";
  hybrid::ExportBenchComparisonTsv(comparison_tsv.string(), comparison);
  assert(std::filesystem::exists(comparison_tsv));
  std::ifstream comparison_tsv_in(comparison_tsv);
  assert(comparison_tsv_in);
  std::stringstream comparison_tsv_buffer;
  comparison_tsv_buffer << comparison_tsv_in.rdbuf();
  const std::string comparison_tsv_text = comparison_tsv_buffer.str();
  assert(comparison_tsv_text.find("delta_qps") != std::string::npos);
  assert(comparison_tsv_text.find("delta_bytes_read") != std::string::npos);
  assert(comparison_tsv_text.find("delta_graph_cache_hits") != std::string::npos);
  assert(comparison_tsv_text.find("delta_deferred_exact_candidates") != std::string::npos);
  assert(comparison_tsv_text.find("delta_scheduler_pending_max") != std::string::npos);

  return 0;
}
