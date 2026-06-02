#include "search/search_thread_context.h"

#include <stdexcept>

namespace hybrid {

SearchThreadContext SearchThreadContext::Create(const SharedSearchResources &shared) {
  if (shared.index == nullptr) {
    throw std::runtime_error("shared search resources require a loaded index");
  }
  SearchThreadContext context;
  context.shared = shared;
  context.page_reader = CreateThreadLocalPageReader(*shared.index, shared.page_reader_backend);
  const SearchIndexMetadata &metadata = shared.index->search_metadata();
  context.query_buffer_pool.InitBuffers(2, metadata.page_size, metadata.num_points, metadata.num_pages);
  return context;
}

std::vector<SearchResult> SearchThreadContext::Search(const std::vector<float> &query,
                                                      const SearchConfig &config,
                                                      ApproxDistanceKind approx_kind,
                                                      SearchStats *stats,
                                                      const std::string &pq_codebook_path,
                                                      const std::string &pq_codes_path) {
  if (shared.index == nullptr) {
    throw std::runtime_error("search thread context requires shared index resources");
  }
  if (page_reader == nullptr) {
    page_reader = CreateThreadLocalPageReader(*shared.index, shared.page_reader_backend);
  }

  PipelinedGraphReplicatedSearcher searcher(*shared.index);
  QueryBufferLease query_buffer = query_buffer_pool.Acquire();
  return searcher.Search(query,
                         config,
                         approx_kind,
                         *page_reader,
                         shared.graph_cache.get(),
                         shared.pq.get(),
                         stats,
                         query_buffer.get(),
                         pq_codebook_path,
                         pq_codes_path);
}

}  // namespace hybrid
