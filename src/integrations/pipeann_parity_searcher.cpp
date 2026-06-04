#include "integrations/pipeann_parity_searcher.h"

#include <stdexcept>

namespace hybrid::pipeann_parity {

void SearchInto(PipeannParityIndex &parity_index,
                const float *query,
                uint32_t dim,
                const PipeannParitySearchConfig &config,
                uint32_t *ids_out,
                float *distances_out,
                pipeann::QueryStats *stats) {
  if (query == nullptr || ids_out == nullptr || distances_out == nullptr || stats == nullptr) {
    throw std::runtime_error("PipeANN parity search outputs must not be null");
  }
  if (dim == 0) {
    throw std::runtime_error("PipeANN parity query dimension must be positive");
  }
  if (config.top_k == 0 || config.l_search == 0 || config.beam_width == 0) {
    throw std::runtime_error("PipeANN parity search config must use positive top_k, l_search and beam_width");
  }

  pipeann::SSDIndex<float> &index = parity_index.index();
  const uint32_t index_dim = static_cast<uint32_t>(index.meta_.data_dim);
  if (dim != index_dim) {
    throw std::runtime_error("PipeANN parity query dimension does not match index dimension");
  }

  *stats = pipeann::QueryStats{};
  switch (config.mode) {
    case PipeannSearchMode::kBeam:
      index.beam_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        ids_out,
                        distances_out,
                        config.beam_width,
                        stats);
      break;
    case PipeannSearchMode::kPage:
      index.page_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        ids_out,
                        distances_out,
                        config.beam_width,
                        stats);
      break;
    case PipeannSearchMode::kPipe:
      index.pipe_search(query,
                        config.top_k,
                        config.mem_l,
                        config.l_search,
                        ids_out,
                        distances_out,
                        config.beam_width,
                        stats);
      break;
  }

  // search_disk_index overwrites total_us in the outer benchmark loop.
  stats->total_us = 0;
}

PipeannParityQueryResult SearchOne(PipeannParityIndex &parity_index,
                                   const float *query,
                                   uint32_t dim,
                                   const PipeannParitySearchConfig &config) {
  PipeannParityQueryResult result;
  result.ids.resize(config.top_k);
  result.distances.resize(config.top_k);
  SearchInto(parity_index,
             query,
             dim,
             config,
             result.ids.data(),
             result.distances.data(),
             &result.stats);
  return result;
}

}  // namespace hybrid::pipeann_parity
