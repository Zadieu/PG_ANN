#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pipeline_search.h"
#include "search/query_buffer_pool.h"
#include "search/shared_search_resources.h"

namespace hybrid {

struct SearchThreadContext {
  SharedSearchResources shared;
  std::unique_ptr<IPageReader> page_reader;
  QueryBufferPool query_buffer_pool;

  static SearchThreadContext Create(const SharedSearchResources &shared);
  std::vector<SearchResult> Search(const std::vector<float> &query,
                                   const SearchConfig &config,
                                   ApproxDistanceKind approx_kind,
                                   SearchStats *stats,
                                   const std::string &pq_codebook_path = {},
                                   const std::string &pq_codes_path = {});
};

}  // namespace hybrid
