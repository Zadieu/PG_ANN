#include <iostream>
#include <memory>
#include <vector>

#include "gorgeous_layout.h"
#include "pipeline_search.h"
#include "tools/build_pipeline.h"

int main() {
  hybrid::BuildConfig build_config;
  build_config.input_mode = hybrid::VectorInputMode::kToy;
  build_config.output_dir = "sample_data";
  build_config.dataset_name = "toy";
  build_config.degree = 4;
  build_config.page_nodes = 4;
  build_config.entry_id = 0;
  build_config.pq_subspaces = 3;
  build_config.pq_centroids = 4;
  const hybrid::BuildArtifacts artifacts = hybrid::RunBuildPipeline(build_config);

  std::unique_ptr<hybrid::IndexReader> index = hybrid::LoadIndexReader(artifacts.index_path, artifacts.approx_path);

  hybrid::PipelinedGraphReplicatedSearcher searcher(*index);
  hybrid::SearchConfig config;
  config.top_k = 4;
  config.beam_width = 3;
  config.l_search = 8;
  config.graph_cache_budget_bytes = 8192;
  config.graph_cache_policy = hybrid::GraphCacheBuildPolicy::kEntryBfs;

  std::vector<float> query = {0.95f, 0.15f, 0.0f};
  hybrid::SearchStats stats;
  const std::vector<hybrid::SearchResult> results = searcher.Search(query, config, &stats);
  hybrid::SearchStats full_stats;
  const std::vector<hybrid::SearchResult> full_results =
      searcher.Search(query, config, hybrid::ApproxDistanceKind::kFullPrecision, &full_stats);

  std::cout << "Hybrid search over Gorgeous-style pages with a PipeANN-style I/O loop\n";
  std::cout << "Default PipeANN PQ results:\n";
  for (const auto &result : results) {
    std::cout << "  id=" << result.id << " dist=" << result.distance << '\n';
  }
  std::cout << "Stats: reads=" << stats.async_reads
            << " completed_pages=" << stats.pages_completed
            << " resident_expansions=" << stats.resident_expansions
            << " approx_evals=" << stats.approx_distance_evals
            << " exact_evals=" << stats.exact_distance_evals
            << " n_ios=" << stats.n_ios
            << " n_cmps=" << stats.n_cmps
            << " n_hops=" << stats.n_hops
            << " cpu_us=" << stats.cpu_us
            << " io_us=" << stats.io_us
            << " bytes_read=" << stats.bytes_read
            << " page_resident_hits=" << stats.page_resident_hits
            << " graph_replicated_hits=" << stats.graph_replicated_hits
            << " graph_cache_hits=" << stats.graph_cache_hits
            << " graph_cache_misses=" << stats.graph_cache_misses
            << " graph_cache_expansions=" << stats.graph_cache_expansions
            << " graph_cache_avoided_reads=" << stats.graph_cache_avoided_reads
            << " graph_cache_resident_bytes=" << stats.graph_cache_resident_bytes
            << " graph_cache_entries=" << stats.graph_cache_entries
            << " graph_cache_build_page_reads=" << stats.graph_cache_build_page_reads
            << " exact_from_page=" << stats.exact_from_page
            << " exact_from_payload=" << stats.exact_from_payload
            << " refinement_candidates=" << stats.refinement_candidates
            << " refinement_reads=" << stats.refinement_reads
            << " approximate_candidates=" << stats.approximate_candidates
            << " refinement_bound=" << stats.refinement_bound
            << " refinement_already_exact=" << stats.refinement_already_exact
            << " refinement_exactified=" << stats.refinement_exactified
            << " deferred_exact_candidates=" << stats.deferred_exact_candidates
            << " read_hits_in_pool=" << stats.read_hits_in_pool
            << " read_waste_out_of_pool=" << stats.read_waste_out_of_pool
            << " max_inflight_reads=" << stats.max_inflight_reads
            << " max_beam_width=" << stats.max_beam_width
            << " beam_width_increases=" << stats.beam_width_increases
            << " scheduler_policy_limit_observed=" << stats.scheduler_policy_limit
            << " scheduler_pending_max=" << stats.scheduler_pending_max
            << " scheduler_ready_unexpanded_max=" << stats.scheduler_ready_unexpanded_max
            << " scheduler_limit_hits=" << stats.scheduler_limit_hits
            << " poll_calls=" << stats.poll_calls
            << " drain_calls=" << stats.drain_calls
            << " range_stop=" << (stats.range_stop ? 1 : 0) << '\n';
  std::cout << "Full-precision fallback results:\n";
  for (const auto &result : full_results) {
    std::cout << "  id=" << result.id << " dist=" << result.distance << '\n';
  }
  std::cout << "Fallback stats: reads=" << full_stats.async_reads
            << " completed_pages=" << full_stats.pages_completed
            << " resident_expansions=" << full_stats.resident_expansions
            << " approx_evals=" << full_stats.approx_distance_evals
            << " exact_evals=" << full_stats.exact_distance_evals
            << " n_ios=" << full_stats.n_ios
            << " n_cmps=" << full_stats.n_cmps
            << " n_hops=" << full_stats.n_hops
            << " cpu_us=" << full_stats.cpu_us
            << " io_us=" << full_stats.io_us
            << " bytes_read=" << full_stats.bytes_read
            << " page_resident_hits=" << full_stats.page_resident_hits
            << " graph_replicated_hits=" << full_stats.graph_replicated_hits
            << " graph_cache_hits=" << full_stats.graph_cache_hits
            << " graph_cache_misses=" << full_stats.graph_cache_misses
            << " graph_cache_expansions=" << full_stats.graph_cache_expansions
            << " graph_cache_avoided_reads=" << full_stats.graph_cache_avoided_reads
            << " graph_cache_resident_bytes=" << full_stats.graph_cache_resident_bytes
            << " graph_cache_entries=" << full_stats.graph_cache_entries
            << " graph_cache_build_page_reads=" << full_stats.graph_cache_build_page_reads
            << " exact_from_page=" << full_stats.exact_from_page
            << " exact_from_payload=" << full_stats.exact_from_payload
            << " refinement_candidates=" << full_stats.refinement_candidates
            << " refinement_reads=" << full_stats.refinement_reads
            << " approximate_candidates=" << full_stats.approximate_candidates
            << " refinement_bound=" << full_stats.refinement_bound
            << " refinement_already_exact=" << full_stats.refinement_already_exact
            << " refinement_exactified=" << full_stats.refinement_exactified
            << " deferred_exact_candidates=" << full_stats.deferred_exact_candidates
            << " read_hits_in_pool=" << full_stats.read_hits_in_pool
            << " read_waste_out_of_pool=" << full_stats.read_waste_out_of_pool
            << " max_inflight_reads=" << full_stats.max_inflight_reads
            << " max_beam_width=" << full_stats.max_beam_width
            << " beam_width_increases=" << full_stats.beam_width_increases
            << " scheduler_policy_limit_observed=" << full_stats.scheduler_policy_limit
            << " scheduler_pending_max=" << full_stats.scheduler_pending_max
            << " scheduler_ready_unexpanded_max=" << full_stats.scheduler_ready_unexpanded_max
            << " scheduler_limit_hits=" << full_stats.scheduler_limit_hits
            << " poll_calls=" << full_stats.poll_calls
            << " drain_calls=" << full_stats.drain_calls
            << " range_stop=" << (full_stats.range_stop ? 1 : 0) << '\n';

  return 0;
}
