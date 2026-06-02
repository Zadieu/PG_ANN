#include "search/shared_search_resources.h"

#include "tools/tool_cli.h"

namespace hybrid {

SharedSearchResources BuildSharedSearchResources(const BenchToolConfig &config) {
  SharedSearchResources shared;
  std::shared_ptr<IndexReader> index = LoadIndexReader(config.index_path, config.approx_path);
  shared.index = std::move(index);
  shared.page_reader_backend = config.page_reader_backend;

  if (config.search_config.graph_cache_budget_bytes != 0) {
    auto graph_cache = std::make_shared<GraphAdjacencyCache>(
        GraphAdjacencyCache::Build(*shared.index,
                                   config.search_config.graph_cache_budget_bytes,
                                   config.search_config.graph_cache_policy));
    if (!graph_cache->empty()) {
      shared.graph_cache = std::move(graph_cache);
    }
  }

  if (config.approx_kind == ApproxDistanceKind::kProductQuantization) {
    auto pq = std::make_shared<PipeannProductQuantization>(*shared.index);
    pq->Load(config.pq_codebook_path, config.pq_codes_path);
    shared.pq = std::move(pq);
  }

  return shared;
}

}  // namespace hybrid
