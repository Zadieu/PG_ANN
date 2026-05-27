#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hybrid::gorgeous_integration {

struct DiskIndexMetadata {
  bool is_new_format = false;
  std::vector<uint64_t> raw_metadata;
  uint64_t file_size = 0;
  uint64_t page_region_sectors = 0;
  uint64_t num_points = 0;
  uint64_t dim = 0;
  uint64_t entry_point = 0;
  uint64_t max_node_len = 0;
  uint64_t nodes_per_sector = 0;
  uint64_t attr_size = 0;
  uint64_t range = 0;
  uint64_t range_dense = 0;
  uint64_t normal_node_len = 0;
  uint64_t r_ood = 0;
  uint64_t full_precision_payload_start_sector = 0;
  uint64_t full_precision_dim = 0;
  uint64_t full_precision_nodes_per_sector = 0;
};

DiskIndexMetadata ReadDiskIndexMetadata(const std::string &path);

void RelayoutDiskIndexToGraphReplica(const std::string &disk_index_path,
                                     const std::vector<std::vector<uint32_t>> &layouts,
                                     const std::string &output_path,
                                     size_t coord_bytes_per_dim,
                                     const std::vector<std::vector<float>> *reorder_vectors = nullptr,
                                     uint64_t input_sector_len = 4096,
                                     uint64_t output_sector_len = 4096);

}  // namespace hybrid::gorgeous_integration
