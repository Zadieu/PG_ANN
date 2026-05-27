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
            << " range_stop=" << (full_stats.range_stop ? 1 : 0) << '\n';

  return 0;
}
