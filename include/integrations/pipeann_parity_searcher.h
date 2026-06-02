#pragma once

#include <cstdint>
#include <vector>

#include "integrations/pipeann_parity_index.h"
#include "utils/percentile_stats.h"

namespace hybrid::pipeann_parity {

enum class PipeannSearchMode {
  kBeam = 0,
  kPage = 1,
  kPipe = 2,
};

struct PipeannParitySearchConfig {
  PipeannSearchMode mode = PipeannSearchMode::kPipe;
  uint32_t top_k = 10;
  uint32_t l_search = 20;
  uint32_t mem_l = 0;
  uint64_t beam_width = 8;
};

struct PipeannParityQueryResult {
  std::vector<uint32_t> ids;
  std::vector<float> distances;
  pipeann::QueryStats stats{};
};

PipeannParityQueryResult SearchOne(PipeannParityIndex &index,
                                   const float *query,
                                   uint32_t dim,
                                   const PipeannParitySearchConfig &config);

}  // namespace hybrid::pipeann_parity
