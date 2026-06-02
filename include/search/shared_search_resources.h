#pragma once

#include <memory>

#include "gorgeous_layout.h"
#include "io/page_reader.h"
#include "quant/approx_distance.h"
#include "search/graph_cache.h"

namespace hybrid {

struct BenchToolConfig;

struct SharedSearchResources {
  std::shared_ptr<const IndexReader> index;
  std::shared_ptr<const GraphAdjacencyCache> graph_cache;
  std::shared_ptr<const PipeannProductQuantization> pq;
  PageReaderBackend page_reader_backend = PageReaderBackend::kLinuxAio;
};

SharedSearchResources BuildSharedSearchResources(const BenchToolConfig &config);

}  // namespace hybrid
