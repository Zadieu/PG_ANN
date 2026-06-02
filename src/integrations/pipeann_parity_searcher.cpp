#include "integrations/pipeann_parity_searcher.h"

#include <chrono>
#include <stdexcept>

namespace hybrid::pipeann_parity {

PipeannParityQueryResult SearchOne(PipeannParityIndex &parity_index,
                                   const float *query,
                                   uint32_t dim,
                                   const PipeannParitySearchConfig &config) {
  if (query == nullptr) {
    throw std::runtime_error("PipeANN parity query pointer must not be null");
  }
  if (dim == 0) {
    throw std::runtime_error("PipeANN parity query dimension must be positive");
  }
  if (config.top_k == 0 || config.l_search == 0 || config.beam_width == 0) {
    throw std::runtime_error("PipeANN parity search config must use positive top_k, l_search and beam_width");
  }

  PipeannParityQueryResult result;
  result.ids.resize(config.top_k);
  result.distances.resize(config.top_k);
  result.stats = {};

  pipeann::SSDIndex<float> &index = parity_index.index();
  const uint32_t index_dim = static_cast<uint32_t>(index.meta_.data_dim);
  if (dim != index_dim) {
    throw std::runtime_error("PipeANN parity query dimension does not match index dimension");
  }

  switch (config.mode) {
    case PipeannSearchMode::kBeam:
      index.beam_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        result.ids.data(),
                        result.distances.data(),
                        config.beam_width,
                        &result.stats);
      break;
    case PipeannSearchMode::kPage:
      index.page_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        result.ids.data(),
                        result.distances.data(),
                        config.beam_width,
                        &result.stats);
      break;
    case PipeannSearchMode::kPipe:
      index.pipe_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        result.ids.data(),
                        result.distances.data(),
                        config.beam_width,
                        &result.stats);
      break;
  }

  // Keep parity with search_disk_index where total_us is written by the outer benchmark loop.
  result.stats.total_us = 0;
  return result;
}

}  // namespace hybrid::pipeann_parity
