#include "integrations/disk_index_relayout.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hybrid::gorgeous_integration {

namespace {

constexpr int kExpectedPipeannMetaCount = 9;
constexpr int kExpectedNewMetaCountWithReorder = 12;
constexpr int kExpectedOldMetaCount = 11;
constexpr uint64_t kNativeMetaWithReorderCount = 12;

uint64_t RoundUpDiv(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

uint64_t NodeOffset(uint32_t node_id,
                    uint64_t nodes_per_sector,
                    uint64_t sector_len,
                    uint64_t max_node_len) {
  return (static_cast<uint64_t>(node_id) / nodes_per_sector + 1) * sector_len +
         (static_cast<uint64_t>(node_id) % nodes_per_sector) * max_node_len;
}

std::vector<char> ReadWholeFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open disk index file");
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to stat disk index file");
  }
  in.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  in.read(data.data(), size);
  if (!in) {
    throw std::runtime_error("failed to read disk index file");
  }
  return data;
}

template <typename T>
void WriteValue(char *dst, T value) {
  std::memcpy(dst, &value, sizeof(T));
}

void ValidatePipeannRawGraphBlock(const char *graph_block,
                                  uint64_t range,
                                  uint64_t range_dense,
                                  uint64_t attr_size,
                                  uint64_t num_points) {
  uint16_t nnbrs = 0;
  uint16_t n_dense_nbrs = 0;
  std::memcpy(&nnbrs, graph_block, sizeof(uint16_t));
  std::memcpy(&n_dense_nbrs, graph_block + sizeof(uint16_t), sizeof(uint16_t));
  if (nnbrs > range) {
    throw std::runtime_error("raw disk graph block base degree exceeds metadata range");
  }
  const uint64_t dense_capacity = range_dense >= range ? range_dense - range : 0;
  if (n_dense_nbrs > dense_capacity) {
    throw std::runtime_error("raw disk graph block dense degree exceeds metadata range");
  }

  const uint32_t *base_ptr = reinterpret_cast<const uint32_t *>(graph_block + sizeof(uint32_t));
  for (uint32_t i = 0; i < nnbrs; ++i) {
    const uint32_t neighbor = base_ptr[i];
    if (neighbor >= num_points) {
      throw std::runtime_error("raw disk graph block contains neighbor ids outside the dataset range");
    }
  }

  const uint32_t *dense_ptr =
      reinterpret_cast<const uint32_t *>(graph_block + sizeof(uint32_t) + range * sizeof(uint32_t) + attr_size);
  for (uint32_t i = 0; i < n_dense_nbrs; ++i) {
    const uint32_t neighbor = dense_ptr[i];
    if (neighbor >= num_points) {
      throw std::runtime_error("raw disk graph block contains dense neighbor ids outside the dataset range");
    }
  }
}

}  // namespace

DiskIndexMetadata ReadDiskIndexMetadata(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("failed to open disk index metadata file");
  }

  int meta_n = 0;
  int meta_dim = 0;
  fin.read(reinterpret_cast<char *>(&meta_n), sizeof(int));
  fin.read(reinterpret_cast<char *>(&meta_dim), sizeof(int));
  if (!fin) {
    throw std::runtime_error("failed to read disk index metadata header");
  }

  DiskIndexMetadata metadata;
  if (meta_n == kExpectedPipeannMetaCount || meta_n == kExpectedNewMetaCountWithReorder) {
    metadata.is_new_format = true;
    metadata.raw_metadata.resize(static_cast<size_t>(meta_n));
    fin.read(reinterpret_cast<char *>(metadata.raw_metadata.data()),
             static_cast<std::streamsize>(sizeof(uint64_t) * metadata.raw_metadata.size()));
  } else {
    metadata.is_new_format = false;
    metadata.raw_metadata.resize(kExpectedOldMetaCount);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(metadata.raw_metadata.data()),
             static_cast<std::streamsize>(sizeof(uint64_t) * metadata.raw_metadata.size()));
  }
  if (!fin) {
    throw std::runtime_error("failed to read full disk index metadata");
  }
  if (metadata.raw_metadata.size() < 5) {
    throw std::runtime_error("disk index metadata is too small");
  }

  if (metadata.is_new_format && metadata.raw_metadata.size() == static_cast<size_t>(kExpectedPipeannMetaCount)) {
    metadata.file_size = 0;
    metadata.page_region_sectors = 0;
    metadata.num_points = metadata.raw_metadata[0];
    metadata.dim = metadata.raw_metadata[1];
    metadata.entry_point = metadata.raw_metadata[2];
    metadata.max_node_len = metadata.raw_metadata[3];
    metadata.nodes_per_sector = metadata.raw_metadata[4];
    metadata.attr_size = metadata.raw_metadata[6];
    metadata.range = metadata.raw_metadata[7];
    metadata.range_dense =
        (metadata.max_node_len - metadata.dim * sizeof(float) - metadata.attr_size) / sizeof(uint32_t) - 1;
    metadata.normal_node_len =
        metadata.max_node_len - (metadata.range_dense - metadata.range) * sizeof(uint32_t);
    metadata.r_ood = metadata.raw_metadata[8];
    metadata.full_precision_payload_start_sector = 0;
    metadata.full_precision_dim = 0;
    metadata.full_precision_nodes_per_sector = 0;
  } else if (metadata.is_new_format &&
             metadata.raw_metadata.size() == static_cast<size_t>(kExpectedNewMetaCountWithReorder)) {
    metadata.file_size = metadata.raw_metadata.back();
    metadata.page_region_sectors = metadata.raw_metadata[8];
    metadata.num_points = metadata.raw_metadata[0];
    metadata.dim = metadata.raw_metadata[1];
    metadata.entry_point = metadata.raw_metadata[2];
    metadata.max_node_len = metadata.raw_metadata[3];
    metadata.nodes_per_sector = metadata.raw_metadata[4];
    metadata.attr_size = metadata.raw_metadata[6];
    metadata.range = metadata.raw_metadata[7];
    metadata.range_dense =
        (metadata.max_node_len - metadata.dim * sizeof(float) - metadata.attr_size) / sizeof(uint32_t) - 1;
    metadata.normal_node_len =
        metadata.max_node_len - (metadata.range_dense - metadata.range) * sizeof(uint32_t);
    metadata.r_ood = 0;
    if (metadata.raw_metadata[10] != 0 && metadata.raw_metadata[9] != 0 && metadata.raw_metadata[8] > 1) {
      metadata.full_precision_payload_start_sector = metadata.raw_metadata[8];
      metadata.full_precision_dim = metadata.raw_metadata[9];
      metadata.full_precision_nodes_per_sector = metadata.raw_metadata[10];
    } else {
      metadata.full_precision_payload_start_sector = 0;
      metadata.full_precision_dim = 0;
      metadata.full_precision_nodes_per_sector = 0;
    }
  } else {
    metadata.file_size = metadata.is_new_format ? metadata.raw_metadata.back() : metadata.raw_metadata[0];
    metadata.page_region_sectors = 0;
    metadata.num_points = metadata.is_new_format ? metadata.raw_metadata[0] : metadata.raw_metadata[1];
    metadata.dim = metadata.is_new_format ? metadata.raw_metadata[1] : metadata.raw_metadata[2];
    metadata.entry_point = metadata.is_new_format ? metadata.raw_metadata[2] : 0;
    metadata.max_node_len = metadata.raw_metadata[3];
    metadata.nodes_per_sector = metadata.raw_metadata[4];
    metadata.attr_size = 0;
    metadata.range = 0;
    metadata.range_dense = 0;
    metadata.normal_node_len = metadata.max_node_len;
    metadata.r_ood = 0;
    metadata.full_precision_payload_start_sector = 0;
    metadata.full_precision_dim = 0;
    metadata.full_precision_nodes_per_sector = 0;
  }
  if (metadata.num_points == 0 || metadata.dim == 0 || metadata.max_node_len == 0 ||
      metadata.nodes_per_sector == 0) {
    throw std::runtime_error("disk index metadata contains zero-valued required fields");
  }
  return metadata;
}

void RelayoutDiskIndexToGraphReplica(const std::string &disk_index_path,
                                     const std::vector<std::vector<uint32_t>> &layouts,
                                     const std::string &output_path,
                                     size_t coord_bytes_per_dim,
                                     const std::vector<std::vector<float>> *reorder_vectors,
                                     uint64_t input_sector_len,
                                     uint64_t output_sector_len) {
  if (layouts.empty()) {
    throw std::runtime_error("layouts must not be empty");
  }
  if (coord_bytes_per_dim == 0 || input_sector_len == 0 || output_sector_len == 0) {
    throw std::runtime_error("relayout parameters must be positive");
  }

  const DiskIndexMetadata metadata = ReadDiskIndexMetadata(disk_index_path);
  const uint64_t actual_file_size = std::filesystem::file_size(disk_index_path);
  if (metadata.file_size != 0 && actual_file_size != metadata.file_size) {
    throw std::runtime_error("disk index file size does not match metadata");
  }
  if (layouts.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("layout count exceeds supported page id range");
  }
  if (output_sector_len < input_sector_len) {
    throw std::runtime_error("output sector length must be at least as large as input sector length");
  }

  const uint64_t emb_node_len = coord_bytes_per_dim * metadata.dim;
  if (emb_node_len > metadata.max_node_len) {
    throw std::runtime_error("embedding bytes exceed max node length");
  }
  const uint64_t raw_graph_node_len = metadata.max_node_len - emb_node_len;
  const uint64_t graph_node_len = raw_graph_node_len;
  const uint64_t output_max_node_len = metadata.max_node_len;
  const size_t max_layout_size =
      std::max_element(layouts.begin(), layouts.end(),
                       [](const auto &lhs, const auto &rhs) { return lhs.size() < rhs.size(); })
          ->size();
  if (max_layout_size == 0) {
    throw std::runtime_error("layouts must contain at least one node per page");
  }

  const uint64_t required_bytes = emb_node_len + sizeof(uint32_t) +
                                  static_cast<uint64_t>(max_layout_size) * sizeof(uint32_t) +
                                  static_cast<uint64_t>(max_layout_size) * graph_node_len;
  if (required_bytes > output_sector_len) {
    throw std::runtime_error("output sector is too small for graph replica relayout");
  }
  if (reorder_vectors != nullptr) {
    if (reorder_vectors->size() != metadata.num_points) {
      throw std::runtime_error("reorder vectors size does not match disk index point count");
    }
    for (const auto &vector : *reorder_vectors) {
      if (vector.size() != metadata.dim) {
        throw std::runtime_error("reorder vectors dimension does not match disk index dimension");
      }
    }
  }

  const std::vector<char> input = ReadWholeFile(disk_index_path);
  std::vector<char> header(static_cast<size_t>(output_sector_len), 0);
  const bool append_reorder_data = reorder_vectors != nullptr;
  const uint64_t n_reorder_nodes_per_sector =
      append_reorder_data ? (output_sector_len / (metadata.dim * sizeof(float))) : 0;
  const uint64_t n_reorder_sectors =
      append_reorder_data ? RoundUpDiv(metadata.num_points, n_reorder_nodes_per_sector) : 0;
  const uint64_t page_region_sectors = static_cast<uint64_t>(layouts.size()) + 1;
  const uint64_t output_file_size =
      (page_region_sectors + n_reorder_sectors) * output_sector_len;
  std::fill(header.begin(), header.end(), 0);
  uint32_t nr = static_cast<uint32_t>(kNativeMetaWithReorderCount);
  uint32_t nc = 1;
  std::memcpy(header.data(), &nr, sizeof(uint32_t));
  std::memcpy(header.data() + sizeof(uint32_t), &nc, sizeof(uint32_t));
  uint64_t metas[kNativeMetaWithReorderCount] = {
      metadata.num_points,
      metadata.dim,
      metadata.entry_point,
      output_max_node_len,
      static_cast<uint64_t>(max_layout_size),
      metadata.num_points,
      metadata.attr_size,
      metadata.range,
      page_region_sectors,
      append_reorder_data ? metadata.dim : 0,
      n_reorder_nodes_per_sector,
      output_file_size,
  };
  std::memcpy(header.data() + 2 * sizeof(uint32_t), metas, sizeof(metas));

  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open relayout output file");
  }
  out.write(header.data(), static_cast<std::streamsize>(header.size()));

  for (uint32_t page_id = 0; page_id < layouts.size(); ++page_id) {
    const auto &layout = layouts[page_id];
    if (layout.empty()) {
      throw std::runtime_error("relayout page layout must not be empty");
    }
    const uint32_t target_id = layout.front();
    if (target_id >= metadata.num_points) {
      throw std::runtime_error("relayout target id exceeds disk index point count");
    }

    std::vector<char> sector(static_cast<size_t>(output_sector_len), 0);
    const uint64_t target_offset =
        NodeOffset(target_id, metadata.nodes_per_sector, input_sector_len, metadata.max_node_len);
    if (target_offset + emb_node_len > input.size()) {
      throw std::runtime_error("target node embedding offset exceeds disk index size");
    }
    std::memcpy(sector.data(), input.data() + target_offset, static_cast<size_t>(emb_node_len));

    uint64_t cursor = emb_node_len;
    const uint32_t layout_size = static_cast<uint32_t>(layout.size());
    WriteValue(sector.data() + cursor, layout_size);
    cursor += sizeof(uint32_t);
    std::memcpy(sector.data() + cursor, layout.data(), layout.size() * sizeof(uint32_t));
    cursor += static_cast<uint64_t>(max_layout_size) * sizeof(uint32_t);

    for (size_t slot = 0; slot < layout.size(); ++slot) {
      const uint32_t node_id = layout[slot];
      if (node_id >= metadata.num_points) {
        throw std::runtime_error("layout node id exceeds disk index point count");
      }
      const uint64_t node_offset =
          NodeOffset(node_id, metadata.nodes_per_sector, input_sector_len, metadata.max_node_len);
      const uint64_t graph_offset = node_offset + emb_node_len;
      if (graph_offset + raw_graph_node_len > input.size()) {
        throw std::runtime_error("graph block offset exceeds disk index size");
      }
      ValidatePipeannRawGraphBlock(input.data() + graph_offset,
                                   metadata.range,
                                   metadata.range_dense,
                                   metadata.attr_size,
                                   metadata.num_points);
      std::memcpy(sector.data() + cursor + slot * graph_node_len,
                  input.data() + graph_offset,
                  static_cast<size_t>(graph_node_len));
    }

    out.write(sector.data(), static_cast<std::streamsize>(sector.size()));
    if (!out) {
      throw std::runtime_error("failed while writing relayout output");
    }
  }

  if (append_reorder_data) {
    const size_t vec_len = static_cast<size_t>(metadata.dim) * sizeof(float);
    std::vector<char> sector(static_cast<size_t>(output_sector_len), 0);
    uint64_t node_id = 0;
    for (uint64_t sector_id = 0; sector_id < n_reorder_sectors; ++sector_id) {
      std::fill(sector.begin(), sector.end(), 0);
      for (uint64_t slot = 0; slot < n_reorder_nodes_per_sector && node_id < metadata.num_points; ++slot, ++node_id) {
        std::memcpy(sector.data() + slot * vec_len,
                    (*reorder_vectors)[static_cast<size_t>(node_id)].data(),
                    vec_len);
      }
      out.write(sector.data(), static_cast<std::streamsize>(sector.size()));
      if (!out) {
        throw std::runtime_error("failed while writing native reorder payload");
      }
    }
  }
}

}  // namespace hybrid::gorgeous_integration
