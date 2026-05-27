#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hybrid::gorgeous_integration {

struct GorgeousOriginalPartitionResult {
  std::vector<std::vector<uint32_t>> layouts;
  std::vector<uint32_t> id_to_page;
  uint32_t page_capacity = 0;
  uint32_t num_points = 0;
  uint32_t num_pages = 0;
};

GorgeousOriginalPartitionResult BuildGorgeousGraphReplicaIndex(
    const std::vector<std::vector<float>> &vectors,
    const std::vector<std::vector<uint32_t>> &graph,
    uint32_t entry_id,
    const std::string &partition_path,
    const std::string &output_index_path,
    uint32_t partition_scale,
    uint32_t partition_ldg_times,
    uint64_t input_sector_len,
    uint64_t output_sector_len);

}  // namespace hybrid::gorgeous_integration
