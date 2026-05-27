#include "integrations/gorgeous_original.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "partitioner.h"

namespace hybrid::gorgeous_integration {

namespace {

constexpr uint32_t kGorgeousMetaCountWithFileSize = 12;
constexpr uint32_t kGorgeousGraphReplicaMode = 3;

struct PartitionFileData {
  uint64_t page_capacity = 0;
  uint64_t num_pages = 0;
  uint64_t num_points = 0;
  std::vector<std::vector<uint32_t>> layouts;
  std::vector<uint32_t> id_to_page;
};

uint64_t RoundUpDiv(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

uint64_t NodeOffset(uint64_t node_id,
                    uint64_t nodes_per_sector,
                    uint64_t sector_len,
                    uint64_t max_node_len) {
  return (node_id / nodes_per_sector + 1) * sector_len +
         (node_id % nodes_per_sector) * max_node_len;
}

void WriteBytes(std::ofstream &out, const void *data, size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("failed to write Gorgeous integration file");
  }
}

std::vector<char> ReadWholeFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open Gorgeous integration input file");
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to stat Gorgeous integration input file");
  }
  in.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  in.read(data.data(), size);
  if (!in) {
    throw std::runtime_error("failed to read Gorgeous integration input file");
  }
  return data;
}

PartitionFileData LoadPartitionFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open Gorgeous partition file");
  }

  PartitionFileData data;
  in.read(reinterpret_cast<char *>(&data.page_capacity), sizeof(data.page_capacity));
  in.read(reinterpret_cast<char *>(&data.num_pages), sizeof(data.num_pages));
  in.read(reinterpret_cast<char *>(&data.num_points), sizeof(data.num_points));
  if (!in) {
    throw std::runtime_error("failed to read Gorgeous partition header");
  }

  data.layouts.resize(static_cast<size_t>(data.num_pages));
  for (uint64_t page_id = 0; page_id < data.num_pages; ++page_id) {
    uint32_t layout_size = 0;
    in.read(reinterpret_cast<char *>(&layout_size), sizeof(layout_size));
    if (!in) {
      throw std::runtime_error("failed to read Gorgeous partition layout size");
    }
    auto &layout = data.layouts[static_cast<size_t>(page_id)];
    layout.resize(layout_size);
    if (layout_size > 0) {
      in.read(reinterpret_cast<char *>(layout.data()),
              static_cast<std::streamsize>(layout_size * sizeof(uint32_t)));
      if (!in) {
        throw std::runtime_error("failed to read Gorgeous partition layout payload");
      }
    }
  }

  data.id_to_page.resize(static_cast<size_t>(data.num_points));
  if (!data.id_to_page.empty()) {
    in.read(reinterpret_cast<char *>(data.id_to_page.data()),
            static_cast<std::streamsize>(data.id_to_page.size() * sizeof(uint32_t)));
    if (!in) {
      throw std::runtime_error("failed to read Gorgeous partition id-to-page payload");
    }
  }
  return data;
}

void WritePartitionFile(const std::string &path, const PartitionFileData &data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open Gorgeous partition file for rewrite");
  }
  WriteBytes(out, &data.page_capacity, sizeof(data.page_capacity));
  WriteBytes(out, &data.num_pages, sizeof(data.num_pages));
  WriteBytes(out, &data.num_points, sizeof(data.num_points));
  for (const auto &layout : data.layouts) {
    const uint32_t size = static_cast<uint32_t>(layout.size());
    WriteBytes(out, &size, sizeof(size));
    if (size > 0) {
      WriteBytes(out, layout.data(), static_cast<size_t>(size) * sizeof(uint32_t));
    }
  }
  if (!data.id_to_page.empty()) {
    WriteBytes(out,
               data.id_to_page.data(),
               data.id_to_page.size() * sizeof(uint32_t));
  }
}

PartitionFileData NormalizePartitionFileData(PartitionFileData data) {
  if (data.layouts.size() != data.num_pages) {
    throw std::runtime_error("Gorgeous partition layout count does not match header");
  }
  std::vector<uint32_t> id_to_page(static_cast<size_t>(data.num_points),
                                   std::numeric_limits<uint32_t>::max());

  for (size_t page_id = 0; page_id < data.layouts.size(); ++page_id) {
    std::vector<uint32_t> normalized;
    normalized.reserve(data.layouts[page_id].size());
    std::vector<bool> seen(static_cast<size_t>(data.num_points), false);

    if (page_id < data.num_points) {
      normalized.push_back(static_cast<uint32_t>(page_id));
      seen[page_id] = true;
    }
    for (uint32_t node_id : data.layouts[page_id]) {
      if (node_id >= data.num_points || seen[node_id]) {
        continue;
      }
      seen[node_id] = true;
      normalized.push_back(node_id);
    }
    data.layouts[page_id].swap(normalized);
    for (uint32_t node_id : data.layouts[page_id]) {
      if (id_to_page[node_id] == std::numeric_limits<uint32_t>::max()) {
        id_to_page[node_id] = static_cast<uint32_t>(page_id);
      }
    }
  }

  for (uint32_t node_id = 0; node_id < data.num_points; ++node_id) {
    if (id_to_page[node_id] != std::numeric_limits<uint32_t>::max()) {
      continue;
    }
    const uint32_t fallback_page =
        !data.id_to_page.empty() && data.id_to_page[node_id] < data.num_pages
            ? data.id_to_page[node_id]
            : std::min<uint32_t>(node_id, static_cast<uint32_t>(data.num_pages - 1));
    auto &layout = data.layouts[fallback_page];
    if (std::find(layout.begin(), layout.end(), node_id) == layout.end()) {
      layout.push_back(node_id);
    }
    if (layout.front() != fallback_page) {
      auto it = std::find(layout.begin(), layout.end(), fallback_page);
      if (it != layout.end()) {
        std::iter_swap(layout.begin(), it);
      } else {
        layout.insert(layout.begin(), fallback_page);
      }
    }
    id_to_page[node_id] = fallback_page;
  }

  uint64_t max_layout_size = 0;
  for (size_t page_id = 0; page_id < data.layouts.size(); ++page_id) {
    if (data.layouts[page_id].empty()) {
      if (page_id >= data.num_points) {
        throw std::runtime_error("Gorgeous partition produced an empty page beyond the dataset range");
      }
      data.layouts[page_id].push_back(static_cast<uint32_t>(page_id));
      if (id_to_page[page_id] == std::numeric_limits<uint32_t>::max()) {
        id_to_page[page_id] = static_cast<uint32_t>(page_id);
      }
    }
    if (data.layouts[page_id].front() != page_id) {
      auto it = std::find(data.layouts[page_id].begin(), data.layouts[page_id].end(), page_id);
      if (it != data.layouts[page_id].end()) {
        std::iter_swap(data.layouts[page_id].begin(), it);
      } else {
        data.layouts[page_id].insert(data.layouts[page_id].begin(), static_cast<uint32_t>(page_id));
      }
    }
    max_layout_size = std::max<uint64_t>(max_layout_size, data.layouts[page_id].size());
  }

  data.page_capacity = max_layout_size;
  data.id_to_page = std::move(id_to_page);
  return data;
}

void WritePartitionInputDiskIndex(const std::vector<std::vector<float>> &vectors,
                                  const std::vector<std::vector<uint32_t>> &graph,
                                  uint32_t entry_id,
                                  const std::string &path,
                                  uint64_t sector_len) {
  if (vectors.empty() || vectors.size() != graph.size()) {
    throw std::runtime_error("Gorgeous partition input requires matching vectors and graph");
  }
  const uint32_t num_points = static_cast<uint32_t>(vectors.size());
  const uint32_t dim = static_cast<uint32_t>(vectors.front().size());
  if (dim == 0) {
    throw std::runtime_error("Gorgeous partition input vectors must be non-empty");
  }
  if (entry_id >= num_points) {
    throw std::runtime_error("Gorgeous partition input entry_id is out of range");
  }

  uint32_t max_degree = 0;
  for (size_t node_id = 0; node_id < graph.size(); ++node_id) {
    if (vectors[node_id].size() != dim) {
      throw std::runtime_error("Gorgeous partition input vectors must share one dimension");
    }
    for (uint32_t neighbor : graph[node_id]) {
      if (neighbor >= num_points) {
        throw std::runtime_error("Gorgeous partition input graph has out-of-range neighbors");
      }
    }
    max_degree = std::max<uint32_t>(max_degree, static_cast<uint32_t>(graph[node_id].size()));
  }
  if (max_degree == 0) {
    throw std::runtime_error("Gorgeous partition input graph must contain at least one edge");
  }

  const uint64_t max_node_len =
      static_cast<uint64_t>(dim) * sizeof(float) + sizeof(uint32_t) +
      static_cast<uint64_t>(max_degree) * sizeof(uint32_t);
  const uint64_t nodes_per_sector = sector_len / max_node_len;
  if (nodes_per_sector == 0) {
    throw std::runtime_error("Gorgeous partition input node size exceeds sector size");
  }

  const uint64_t file_size = (RoundUpDiv(num_points, nodes_per_sector) + 1) * sector_len;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open Gorgeous partition input disk index");
  }

  std::vector<char> header(static_cast<size_t>(sector_len), 0);
  const uint32_t meta_n = kGorgeousMetaCountWithFileSize;
  const uint32_t meta_dim = 1;
  std::memcpy(header.data(), &meta_n, sizeof(meta_n));
  std::memcpy(header.data() + sizeof(uint32_t), &meta_dim, sizeof(meta_dim));
  const uint64_t metas[kGorgeousMetaCountWithFileSize] = {
      num_points,
      dim,
      entry_id,
      max_node_len,
      nodes_per_sector,
      num_points,
      0,
      0,
      0,
      0,
      0,
      file_size,
  };
  std::memcpy(header.data() + 2 * sizeof(uint32_t), metas, sizeof(metas));
  WriteBytes(out, header.data(), header.size());

  std::vector<char> sector(static_cast<size_t>(sector_len), 0);
  for (uint32_t page = 0, node_id = 0; node_id < num_points; ++page) {
    std::fill(sector.begin(), sector.end(), 0);
    for (uint64_t slot = 0; slot < nodes_per_sector && node_id < num_points; ++slot, ++node_id) {
      char *node_buf = sector.data() + slot * max_node_len;
      std::memcpy(node_buf, vectors[node_id].data(), static_cast<size_t>(dim) * sizeof(float));
      const uint32_t degree = static_cast<uint32_t>(graph[node_id].size());
      std::memcpy(node_buf + static_cast<size_t>(dim) * sizeof(float), &degree, sizeof(degree));
      if (degree > 0) {
        std::memcpy(node_buf + static_cast<size_t>(dim) * sizeof(float) + sizeof(uint32_t),
                    graph[node_id].data(),
                    static_cast<size_t>(degree) * sizeof(uint32_t));
      }
    }
    WriteBytes(out, sector.data(), sector.size());
  }
}

void WriteGraphReplicaIndexFromOriginalRelayout(const std::string &partition_input_index_path,
                                                const std::string &partition_path,
                                                const std::string &output_index_path,
                                                uint64_t input_sector_len,
                                                uint64_t output_sector_len) {
  const PartitionFileData partition = LoadPartitionFile(partition_path);
  if (partition.layouts.empty()) {
    throw std::runtime_error("Gorgeous partition output must not be empty");
  }

  const std::vector<char> source = ReadWholeFile(partition_input_index_path);
  if (source.size() < input_sector_len) {
    throw std::runtime_error("Gorgeous partition input disk index is too small");
  }

  int meta_n = 0;
  int meta_dim = 0;
  std::memcpy(&meta_n, source.data(), sizeof(meta_n));
  std::memcpy(&meta_dim, source.data() + sizeof(int), sizeof(meta_dim));
  if (meta_n != static_cast<int>(kGorgeousMetaCountWithFileSize) || meta_dim != 1) {
    throw std::runtime_error("unexpected Gorgeous partition input metadata header");
  }

  const uint64_t *meta = reinterpret_cast<const uint64_t *>(source.data() + 2 * sizeof(int));
  const uint64_t num_points = meta[0];
  const uint64_t dim = meta[1];
  const uint64_t max_node_len = meta[3];
  const uint64_t nodes_per_sector = meta[4];
  if (num_points != partition.num_points) {
    throw std::runtime_error("Gorgeous partition output point count does not match relayout input");
  }
  if (dim == 0 || max_node_len == 0 || nodes_per_sector == 0) {
    throw std::runtime_error("Gorgeous partition input metadata has zero-valued fields");
  }

  const uint64_t emb_node_len = dim * sizeof(float);
  const uint64_t graph_node_len = max_node_len - emb_node_len;
  const uint64_t expected_input_size =
      (RoundUpDiv(num_points, nodes_per_sector) + 1) * input_sector_len;
  if (source.size() != expected_input_size) {
    throw std::runtime_error("Gorgeous partition input disk index size does not match metadata");
  }

  std::vector<char> header(static_cast<size_t>(output_sector_len), 0);
  std::memcpy(header.data(), source.data(), 2 * sizeof(int) +
                                        static_cast<size_t>(kGorgeousMetaCountWithFileSize) * sizeof(uint64_t));
  const uint64_t output_file_size =
      (static_cast<uint64_t>(partition.layouts.size()) + 1) * output_sector_len;
  uint64_t max_layout_size = 0;
  for (const auto &layout : partition.layouts) {
    max_layout_size = std::max<uint64_t>(max_layout_size, layout.size());
  }
  uint64_t *output_meta = reinterpret_cast<uint64_t *>(header.data() + 2 * sizeof(int));
  output_meta[4] = max_layout_size;
  output_meta[8] = static_cast<uint64_t>(partition.layouts.size()) + 1;
  output_meta[9] = 0;
  output_meta[10] = 0;
  output_meta[kGorgeousMetaCountWithFileSize - 1] = output_file_size;

  std::ofstream out(output_index_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open Gorgeous relayout output");
  }
  WriteBytes(out, header.data(), header.size());

  std::vector<char> sector(static_cast<size_t>(output_sector_len), 0);
  for (size_t page_id = 0; page_id < partition.layouts.size(); ++page_id) {
    const auto &layout = partition.layouts[page_id];
    if (layout.empty()) {
      throw std::runtime_error("Gorgeous partition layout must not contain empty pages");
    }
    if (layout.front() != page_id) {
      throw std::runtime_error("Gorgeous graph replica layout must use the page id as target id");
    }

    std::fill(sector.begin(), sector.end(), 0);
    size_t cursor = 0;
    const uint64_t target_offset =
        NodeOffset(page_id, nodes_per_sector, input_sector_len, max_node_len);
    if (target_offset + emb_node_len > source.size()) {
      throw std::runtime_error("Gorgeous graph replica target offset exceeds relayout input");
    }
    std::memcpy(sector.data(), source.data() + target_offset, static_cast<size_t>(emb_node_len));
    cursor += static_cast<size_t>(emb_node_len);

    const uint32_t layout_size = static_cast<uint32_t>(layout.size());
    std::memcpy(sector.data() + cursor, &layout_size, sizeof(layout_size));
    std::memcpy(sector.data() + cursor + sizeof(uint32_t),
                layout.data(),
                layout.size() * sizeof(uint32_t));
    cursor += sizeof(uint32_t) + static_cast<size_t>(partition.page_capacity) * sizeof(uint32_t);

    for (uint32_t node_id : layout) {
      const uint64_t graph_offset =
          NodeOffset(node_id, nodes_per_sector, input_sector_len, max_node_len) + emb_node_len;
      if (graph_offset + graph_node_len > source.size()) {
        throw std::runtime_error("Gorgeous graph replica source graph block exceeds relayout input");
      }
      std::memcpy(sector.data() + cursor, source.data() + graph_offset, static_cast<size_t>(graph_node_len));
      cursor += static_cast<size_t>(graph_node_len);
    }
    WriteBytes(out, sector.data(), sector.size());
  }
}

}  // namespace

GorgeousOriginalPartitionResult BuildGorgeousGraphReplicaIndex(
    const std::vector<std::vector<float>> &vectors,
    const std::vector<std::vector<uint32_t>> &graph,
    uint32_t entry_id,
    const std::string &partition_path,
    const std::string &output_index_path,
    uint32_t partition_scale,
    uint32_t partition_ldg_times,
    uint64_t input_sector_len,
    uint64_t output_sector_len) {
  if (input_sector_len == 0 || output_sector_len == 0) {
    throw std::runtime_error("Gorgeous sector lengths must be positive");
  }
  if (output_sector_len < input_sector_len || output_sector_len % input_sector_len != 0) {
    throw std::runtime_error("Gorgeous output sector length must be a positive multiple of the input sector length");
  }
  if (partition_ldg_times == 0) {
    throw std::runtime_error("Gorgeous partition_ldg_times must be positive");
  }

  const std::filesystem::path partition_file(partition_path);
  const std::filesystem::path temp_index_path =
      partition_file.parent_path() / (partition_file.stem().string() + ".gorgeous.partition_input.tmp");

  WritePartitionInputDiskIndex(vectors, graph, entry_id, temp_index_path.string(), input_sector_len);

  const unsigned block_size = static_cast<unsigned>(output_sector_len / input_sector_len);
  GP::graph_partitioner partitioner(temp_index_path.string().c_str(),
                                    "float",
                                    true,
                                    block_size,
                                    false,
                                    std::string(),
                                    INF,
                                    false,
                                    GP::Mode::GRAPH_REPLICA,
                                    static_cast<unsigned>(input_sector_len),
                                    static_cast<unsigned>(output_sector_len));
  partitioner.graph_partition(partition_path.c_str(),
                              static_cast<int>(partition_ldg_times),
                              static_cast<int>(partition_scale),
                              0);
  PartitionFileData partition = NormalizePartitionFileData(LoadPartitionFile(partition_path));
  WritePartitionFile(partition_path, partition);

  WriteGraphReplicaIndexFromOriginalRelayout(temp_index_path.string(),
                                             partition_path,
                                             output_index_path,
                                             input_sector_len,
                                             output_sector_len);
  std::filesystem::remove(temp_index_path);

  GorgeousOriginalPartitionResult result;
  result.layouts = partition.layouts;
  result.id_to_page = partition.id_to_page;
  result.page_capacity = static_cast<uint32_t>(partition.page_capacity);
  result.num_points = static_cast<uint32_t>(partition.num_points);
  result.num_pages = static_cast<uint32_t>(partition.num_pages);
  return result;
}

}  // namespace hybrid::gorgeous_integration
