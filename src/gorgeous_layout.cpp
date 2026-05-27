#include "gorgeous_layout.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "ssd_index_defs.h"

namespace hybrid {

namespace {

constexpr uint64_t kMagic = 0x4859425249443031ULL;
constexpr uint32_t kAlignment = 4096;
constexpr uint32_t kSupportedVersion = 4;
constexpr uint32_t kNativeSectorLen = 4096;

struct PartitionSidecar {
  uint64_t page_capacity = 0;
  uint64_t num_pages = 0;
  uint64_t num_points = 0;
  std::vector<std::vector<uint32_t>> layouts;
  std::vector<uint32_t> id_to_page;
};

struct ReorderSidecar {
  uint64_t num_pages = 0;
  uint64_t num_entries = 0;
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> ids;
};

struct LayoutScanResult {
  std::vector<std::vector<uint32_t>> layouts;
  std::vector<uint32_t> id_to_page;
};

std::string RemoveKnownIndexSuffix(const std::string &index_path) {
  static const char *kKnownSuffixes[] = {
      "_graph_relayout.index",
      ".graphrep.relayout",
      ".graphrep",
      ".gorgeous",
  };
  for (const char *suffix : kKnownSuffixes) {
    const std::string suffix_str(suffix);
    if (index_path.size() >= suffix_str.size() &&
        index_path.compare(index_path.size() - suffix_str.size(), suffix_str.size(), suffix_str) == 0) {
      return index_path.substr(0, index_path.size() - suffix_str.size());
    }
  }
  return index_path;
}

std::string ResolveExistingSidecarPath(const std::string &preferred_path,
                                       const std::string &alternate_path) {
  if (std::filesystem::exists(preferred_path)) {
    return preferred_path;
  }
  if (std::filesystem::exists(alternate_path)) {
    return alternate_path;
  }
  return preferred_path;
}

uint32_t AlignUp(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *what) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    throw std::runtime_error(what);
  }
  return lhs + rhs;
}

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *what) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw std::runtime_error(what);
  }
  return lhs * rhs;
}

uint64_t RoundUpDiv(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

SearchIndexMetadata ToSearchMetadata(const GraphReplicatedMetadata &meta) {
  SearchIndexMetadata result;
  result.dim = meta.dim;
  result.num_points = meta.num_points;
  result.num_pages = meta.num_pages;
  result.entry_id = meta.entry_id;
  result.page_size = meta.page_size;
  return result;
}

void ValidateSearchMetadata(const SearchIndexMetadata &meta) {
  if (meta.dim == 0 || meta.num_points == 0) {
    throw std::runtime_error("index metadata must define a non-empty dataset");
  }
  if (meta.num_pages == 0) {
    throw std::runtime_error("index metadata must define at least one page");
  }
  if (meta.entry_id >= meta.num_points) {
    throw std::runtime_error("index metadata entry_id is out of range");
  }
  if (meta.page_size == 0 || meta.page_size % kAlignment != 0) {
    throw std::runtime_error("index metadata page size must be aligned");
  }
}

uint64_t MinRawPageBytes(const GraphReplicatedMetadata &meta) {
  uint64_t bytes = sizeof(uint32_t);
  bytes = CheckedAdd(bytes,
                     CheckedMul(meta.max_page_nodes, sizeof(uint32_t), "page size overflow"),
                     "page size overflow");
  const uint64_t per_node_bytes =
      CheckedAdd(CheckedMul(meta.dim, sizeof(float), "page size overflow"),
                 CheckedAdd(sizeof(uint32_t),
                            CheckedMul(meta.max_degree, sizeof(uint32_t), "page size overflow"),
                            "page size overflow"),
                 "page size overflow");
  bytes = CheckedAdd(bytes,
                     CheckedMul(meta.max_page_nodes, per_node_bytes, "page size overflow"),
                     "page size overflow");
  return bytes;
}

void ValidateGraphReplicatedMetadata(const GraphReplicatedMetadata &meta) {
  if (meta.magic != kMagic) {
    throw std::runtime_error("unexpected index magic");
  }
  if (meta.version != kSupportedVersion) {
    throw std::runtime_error("unsupported index version");
  }
  if (meta.max_degree == 0) {
    throw std::runtime_error("index metadata must allow at least one graph edge slot");
  }
  if (meta.max_base_degree > meta.max_degree) {
    throw std::runtime_error("index metadata base degree exceeds total graph edge capacity");
  }
  if (meta.max_page_nodes == 0) {
    throw std::runtime_error("index metadata must allow at least one node per page");
  }
  ValidateSearchMetadata(ToSearchMetadata(meta));
  if (meta.page_size < MinRawPageBytes(meta)) {
    throw std::runtime_error("index metadata page size is too small for the declared layout");
  }
}

uint64_t ExpectedGraphReplicatedIndexFileSize(const GraphReplicatedMetadata &meta) {
  return CheckedMul(static_cast<uint64_t>(meta.num_pages) + 1,
                    static_cast<uint64_t>(meta.page_size),
                    "index file size overflow");
}

bool HasGraphReplicatedMagic(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open index file");
  }
  uint64_t magic = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (!in) {
    throw std::runtime_error("failed to read index file header");
  }
  return magic == kMagic;
}

uint64_t ExpectedApproxFileSize(const SearchIndexMetadata &meta) {
  const uint64_t payload =
      CheckedMul(meta.num_points,
                 CheckedMul(meta.dim, sizeof(float), "approx file size overflow"),
                 "approx file size overflow");
  return CheckedAdd(sizeof(uint32_t) * 2, payload, "approx file size overflow");
}

uint64_t GetFileSize(const std::string &path, const char *what) {
  std::error_code error;
  const uint64_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error(what);
  }
  return size;
}

void ValidateNeighborIds(const std::vector<uint32_t> &neighbors, uint32_t num_points, const char *what) {
  for (uint32_t neighbor : neighbors) {
    if (neighbor >= num_points) {
      throw std::runtime_error(what);
    }
  }
}

void ValidateNeighborIds(const uint32_t *neighbors,
                        uint32_t count,
                        uint32_t num_points,
                        const char *what) {
  for (uint32_t i = 0; i < count; ++i) {
    if (neighbors[i] >= num_points) {
      throw std::runtime_error(what);
    }
  }
}

void ValidatePageLayout(uint32_t num_points,
                        const std::vector<uint32_t> &layout,
                        const char *what) {
  if (layout.empty()) {
    throw std::runtime_error(what);
  }
  std::unordered_set<uint32_t> seen;
  seen.reserve(layout.size());
  for (uint32_t node_id : layout) {
    if (node_id >= num_points) {
      throw std::runtime_error(what);
    }
    if (!seen.insert(node_id).second) {
      throw std::runtime_error("page layout must not contain duplicate node ids");
    }
  }
}

void WriteBytes(std::ofstream &out, const void *data, size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("failed to write output file");
  }
}

template <typename T>
void WritePod(std::ofstream &out, const T &value) {
  WriteBytes(out, &value, sizeof(T));
}

template <typename T>
T ReadPod(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("failed to read input file");
  }
  return value;
}

PartitionSidecar ReadPartitionSidecar(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open partition sidecar");
  }

  PartitionSidecar sidecar;
  sidecar.page_capacity = ReadPod<uint64_t>(in);
  sidecar.num_pages = ReadPod<uint64_t>(in);
  sidecar.num_points = ReadPod<uint64_t>(in);
  sidecar.layouts.resize(static_cast<size_t>(sidecar.num_pages));
  for (uint64_t page_id = 0; page_id < sidecar.num_pages; ++page_id) {
    const uint32_t size = ReadPod<uint32_t>(in);
    sidecar.layouts[page_id].resize(size);
    if (size > 0) {
      in.read(reinterpret_cast<char *>(sidecar.layouts[page_id].data()),
              static_cast<std::streamsize>(size * sizeof(uint32_t)));
      if (!in) {
        throw std::runtime_error("failed to read partition layout payload");
      }
    }
  }
  sidecar.id_to_page.resize(static_cast<size_t>(sidecar.num_points));
  if (!sidecar.id_to_page.empty()) {
    in.read(reinterpret_cast<char *>(sidecar.id_to_page.data()),
            static_cast<std::streamsize>(sidecar.id_to_page.size() * sizeof(uint32_t)));
    if (!in) {
      throw std::runtime_error("failed to read partition id-to-page mapping");
    }
  }
  return sidecar;
}

ReorderSidecar ReadReorderSidecar(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open reorder sidecar");
  }

  ReorderSidecar sidecar;
  sidecar.num_pages = ReadPod<uint64_t>(in);
  sidecar.num_entries = ReadPod<uint64_t>(in);
  sidecar.offsets.resize(static_cast<size_t>(sidecar.num_pages) + 1);
  if (!sidecar.offsets.empty()) {
    in.read(reinterpret_cast<char *>(sidecar.offsets.data()),
            static_cast<std::streamsize>(sidecar.offsets.size() * sizeof(uint64_t)));
    if (!in) {
      throw std::runtime_error("failed to read reorder sidecar offsets");
    }
  }
  sidecar.ids.resize(static_cast<size_t>(sidecar.num_entries));
  if (!sidecar.ids.empty()) {
    in.read(reinterpret_cast<char *>(sidecar.ids.data()),
            static_cast<std::streamsize>(sidecar.ids.size() * sizeof(uint32_t)));
    if (!in) {
      throw std::runtime_error("failed to read reorder sidecar ids");
    }
  }
  return sidecar;
}

std::vector<std::vector<float>> LoadApproxVectors(const std::string &approx_path, const SearchIndexMetadata &meta) {
  std::ifstream approx_in(approx_path, std::ios::binary);
  if (!approx_in) {
    throw std::runtime_error("failed to open approx vector file");
  }
  const uint32_t num_points = ReadPod<uint32_t>(approx_in);
  const uint32_t dim = ReadPod<uint32_t>(approx_in);
  if (num_points != meta.num_points || dim != meta.dim) {
    throw std::runtime_error("approx vector metadata does not match index metadata");
  }
  const uint64_t approx_file_size = GetFileSize(approx_path, "failed to stat approx vector file");
  if (approx_file_size != ExpectedApproxFileSize(meta)) {
    throw std::runtime_error("approx vector file size does not match metadata");
  }

  std::vector<std::vector<float>> approx_vectors(num_points, std::vector<float>(dim, 0.0f));
  for (uint32_t id = 0; id < num_points; ++id) {
    approx_in.read(reinterpret_cast<char *>(approx_vectors[id].data()),
                   static_cast<std::streamsize>(dim * sizeof(float)));
    if (!approx_in) {
      throw std::runtime_error("failed to read approx vector payload");
    }
  }
  return approx_vectors;
}

void EnsureStreamOpen(const std::string &path, std::ifstream *stream, const char *what) {
  if (stream == nullptr) {
    throw std::runtime_error("vector stream must not be null");
  }
  if (stream->is_open()) {
    return;
  }
  stream->open(path, std::ios::binary);
  if (!*stream) {
    throw std::runtime_error(what);
  }
}

pipeann::SSDIndexMetadata<float> BuildPageDiskNodeMetadata(const GraphReplicatedMetadata &meta) {
  pipeann::SSDIndexMetadata<float> disk_meta;
  disk_meta.npoints = static_cast<uint64_t>(meta.num_pages) * static_cast<uint64_t>(meta.max_page_nodes);
  disk_meta.data_dim = meta.dim;
  disk_meta.entry_point = meta.entry_id;
  disk_meta.max_node_len =
      static_cast<uint64_t>(meta.dim) * sizeof(float) + sizeof(uint32_t) +
      static_cast<uint64_t>(meta.max_degree) * sizeof(uint32_t);
  disk_meta.nnodes_per_sector = meta.max_page_nodes;
  disk_meta.npts_cur_shard = disk_meta.npoints;
  disk_meta.attr_size = 0;
  disk_meta.range = meta.max_base_degree;
  disk_meta.R_ood = 0;
  disk_meta.init_temporary_fields();
  disk_meta.range_dense = meta.max_degree;
  disk_meta.normal_node_len =
      disk_meta.max_node_len -
      static_cast<uint64_t>(disk_meta.range_dense - disk_meta.range) * sizeof(uint32_t);
  return disk_meta;
}

pipeann::SSDIndexMetadata<float> BuildNativeDiskNodeMetadata(
    const gorgeous_integration::DiskIndexMetadata &meta,
    uint64_t layout_capacity) {
  pipeann::SSDIndexMetadata<float> disk_meta;
  disk_meta.npoints = meta.num_points;
  disk_meta.data_dim = meta.dim;
  disk_meta.entry_point = meta.entry_point;
  disk_meta.max_node_len = meta.max_node_len;
  disk_meta.nnodes_per_sector = layout_capacity;
  disk_meta.npts_cur_shard = meta.num_points;
  disk_meta.attr_size = meta.attr_size;
  disk_meta.range = meta.range;
  disk_meta.R_ood = meta.r_ood;
  disk_meta.init_temporary_fields();
  disk_meta.range_dense = meta.range_dense;
  disk_meta.normal_node_len = meta.normal_node_len;
  return disk_meta;
}

std::vector<uint32_t> CopyNeighborsVector(const DiskNodeView &node) {
  std::vector<uint32_t> neighbors;
  neighbors.reserve(node.total_degree());
  if (node.nnbrs > 0) {
    neighbors.insert(neighbors.end(), node.nbrs, node.nbrs + node.nnbrs);
  }
  if (node.n_dense_nbrs > 0) {
    neighbors.insert(neighbors.end(), node.dense_nbrs, node.dense_nbrs + node.n_dense_nbrs);
  }
  return neighbors;
}

void BuildReorderFromLayouts(const std::vector<std::vector<uint32_t>> &layouts,
                             std::vector<uint64_t> *offsets,
                             std::vector<uint32_t> *ids) {
  offsets->assign(layouts.size() + 1, 0);
  ids->clear();
  uint64_t offset = 0;
  for (size_t page_id = 0; page_id < layouts.size(); ++page_id) {
    (*offsets)[page_id] = offset;
    ids->insert(ids->end(), layouts[page_id].begin(), layouts[page_id].end());
    offset += layouts[page_id].size();
  }
  (*offsets)[layouts.size()] = offset;
}

LayoutScanResult ScanNativeLayouts(const std::string &index_path,
                                   const SearchIndexMetadata &search_meta,
                                   const gorgeous_integration::DiskIndexMetadata &native_meta) {
  std::ifstream in(index_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open native index for layout scan");
  }

  LayoutScanResult result;
  result.layouts.resize(search_meta.num_pages);
  result.id_to_page.assign(search_meta.num_points, std::numeric_limits<uint32_t>::max());
  const size_t header_bytes = static_cast<size_t>(native_meta.dim) * sizeof(float);
  std::vector<char> page(search_meta.page_size, 0);
  for (uint32_t page_id = 0; page_id < search_meta.num_pages; ++page_id) {
    const uint64_t offset = static_cast<uint64_t>(page_id + 1) * search_meta.page_size;
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(page.data(), static_cast<std::streamsize>(page.size()));
    if (!in) {
      throw std::runtime_error("failed to read native page while scanning layouts");
    }

    size_t cursor = header_bytes;
    uint32_t layout_size = 0;
    std::memcpy(&layout_size, page.data() + cursor, sizeof(layout_size));
    cursor += sizeof(layout_size);
    if (layout_size == 0 || layout_size > native_meta.nodes_per_sector) {
      throw std::runtime_error("native page layout size exceeds metadata bounds");
    }
    const uint32_t *layout_ids = reinterpret_cast<const uint32_t *>(page.data() + cursor);
    std::vector<uint32_t> layout(layout_ids, layout_ids + layout_size);
    ValidatePageLayout(search_meta.num_points, layout, "native page layout contains node ids outside the dataset range");
    result.layouts[page_id] = layout;
    for (uint32_t node_id : layout) {
      if (result.id_to_page[node_id] == std::numeric_limits<uint32_t>::max()) {
        result.id_to_page[node_id] = page_id;
      }
    }
  }
  return result;
}

void ValidatePartitionSidecar(const PartitionSidecar &sidecar,
                              const SearchIndexMetadata &meta,
                              const std::vector<std::vector<uint32_t>> &layouts) {
  if (sidecar.num_points != meta.num_points) {
    throw std::runtime_error("partition sidecar point count does not match index metadata");
  }
  if (sidecar.num_pages != meta.num_pages) {
    throw std::runtime_error("partition sidecar page count does not match index metadata");
  }
  if (sidecar.layouts.size() != layouts.size()) {
    throw std::runtime_error("partition sidecar layout count does not match scanned pages");
  }
  for (uint32_t page_id = 0; page_id < sidecar.layouts.size(); ++page_id) {
    ValidatePageLayout(meta.num_points,
                       sidecar.layouts[page_id],
                       "partition sidecar contains node ids outside the dataset range");
    if (sidecar.layouts[page_id] != layouts[page_id]) {
      throw std::runtime_error("partition sidecar layout does not match native page layout");
    }
  }
  if (sidecar.id_to_page.size() != meta.num_points) {
    throw std::runtime_error("partition sidecar id-to-page size does not match point count");
  }
  for (uint32_t node_id = 0; node_id < sidecar.id_to_page.size(); ++node_id) {
    const uint32_t page_id = sidecar.id_to_page[node_id];
    if (page_id >= meta.num_pages) {
      throw std::runtime_error("partition sidecar id-to-page mapping exceeds page count");
    }
    const auto &layout = layouts[page_id];
    if (std::find(layout.begin(), layout.end(), node_id) == layout.end()) {
      throw std::runtime_error("partition sidecar primary page does not contain the mapped node");
    }
  }
}

void ValidateReorderSidecar(const ReorderSidecar &sidecar,
                            const SearchIndexMetadata &meta,
                            const std::vector<std::vector<uint32_t>> &layouts) {
  if (sidecar.num_pages != meta.num_pages) {
    throw std::runtime_error("reorder sidecar page count does not match index metadata");
  }
  if (sidecar.offsets.size() != static_cast<size_t>(meta.num_pages) + 1) {
    throw std::runtime_error("reorder sidecar offsets length is invalid");
  }
  if (!sidecar.offsets.empty() && sidecar.offsets.front() != 0) {
    throw std::runtime_error("reorder sidecar must start at offset zero");
  }
  if (!sidecar.offsets.empty() && sidecar.offsets.back() != sidecar.num_entries) {
    throw std::runtime_error("reorder sidecar entry count does not match offsets");
  }
  for (size_t i = 1; i < sidecar.offsets.size(); ++i) {
    if (sidecar.offsets[i] < sidecar.offsets[i - 1]) {
      throw std::runtime_error("reorder sidecar offsets must be non-decreasing");
    }
  }
  ValidateNeighborIds(sidecar.ids, meta.num_points, "reorder sidecar contains node ids outside the dataset range");
  for (uint32_t page_id = 0; page_id < meta.num_pages; ++page_id) {
    const uint64_t begin = sidecar.offsets[page_id];
    const uint64_t end = sidecar.offsets[page_id + 1];
    std::vector<uint32_t> layout(sidecar.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                 sidecar.ids.begin() + static_cast<std::ptrdiff_t>(end));
    if (layout != layouts[page_id]) {
      throw std::runtime_error("reorder sidecar layout does not match native page layout");
    }
  }
}

}  // namespace

uint32_t PageView::LayoutAt(uint32_t slot) const {
  if (slot >= layout_size) {
    throw std::runtime_error("page layout index out of range");
  }
  return layout_ids[slot];
}

float L2Distance(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("dimension mismatch in L2Distance");
  }
  float sum = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

float L2Distance(const std::vector<float> &lhs, const float *rhs, uint32_t dim) {
  if (lhs.size() != dim || rhs == nullptr) {
    throw std::runtime_error("dimension mismatch in L2Distance");
  }
  float sum = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

void GraphReplicatedLayoutBuilder::Build(const std::string &index_path,
                                         const std::string &approx_path,
                                         const std::vector<std::vector<float>> &vectors,
                                         const std::vector<std::vector<uint32_t>> &graph,
                                         const std::vector<std::vector<uint32_t>> &page_layouts,
                                         uint32_t entry_id) {
  if (vectors.empty()) {
    throw std::runtime_error("vectors must not be empty");
  }
  if (vectors.size() != graph.size() || vectors.size() != page_layouts.size()) {
    throw std::runtime_error("vectors, graph, and page_layouts must have the same size");
  }
  if (entry_id >= vectors.size()) {
    throw std::runtime_error("entry_id out of range");
  }

  const uint32_t dim = static_cast<uint32_t>(vectors.front().size());
  uint32_t max_degree = 0;
  uint32_t max_page_nodes = 0;
  for (size_t id = 0; id < vectors.size(); ++id) {
    if (vectors[id].size() != dim) {
      throw std::runtime_error("all vectors must share the same dimension");
    }
    ValidateNeighborIds(graph[id],
                        static_cast<uint32_t>(vectors.size()),
                        "graph contains neighbor ids outside the dataset range");
    max_degree = std::max<uint32_t>(max_degree, static_cast<uint32_t>(graph[id].size()));
    max_page_nodes = std::max<uint32_t>(max_page_nodes, static_cast<uint32_t>(page_layouts[id].size()));
    ValidatePageLayout(static_cast<uint32_t>(vectors.size()),
                       page_layouts[id],
                       "page layout contains node ids outside the dataset range");
  }

  GraphReplicatedMetadata meta{};
  meta.magic = kMagic;
  meta.version = kSupportedVersion;
  meta.dim = dim;
  meta.num_points = static_cast<uint32_t>(vectors.size());
  meta.num_pages = static_cast<uint32_t>(page_layouts.size());
  meta.max_degree = max_degree;
  meta.max_base_degree = max_degree;
  meta.max_page_nodes = max_page_nodes;
  meta.entry_id = entry_id;
  const uint64_t raw_page_bytes = MinRawPageBytes(meta);
  if (raw_page_bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("page size exceeds supported range");
  }
  meta.page_size = AlignUp(static_cast<uint32_t>(raw_page_bytes), kAlignment);

  std::ofstream approx_out(approx_path, std::ios::binary | std::ios::trunc);
  if (!approx_out) {
    throw std::runtime_error("failed to open approx output file");
  }
  WritePod(approx_out, meta.num_points);
  WritePod(approx_out, meta.dim);
  for (const auto &vec : vectors) {
    WriteBytes(approx_out, vec.data(), vec.size() * sizeof(float));
  }

  std::ofstream index_out(index_path, std::ios::binary | std::ios::trunc);
  if (!index_out) {
    throw std::runtime_error("failed to open index output file");
  }

  std::vector<char> header(meta.page_size, 0);
  std::memcpy(header.data(), &meta, sizeof(meta));
  WriteBytes(index_out, header.data(), header.size());

  for (uint32_t page_id = 0; page_id < meta.num_pages; ++page_id) {
    std::vector<char> page(meta.page_size, 0);
    size_t cursor = 0;

    const auto &layout = page_layouts[page_id];
    const uint32_t layout_size = static_cast<uint32_t>(layout.size());
    std::memcpy(page.data() + cursor, &layout_size, sizeof(layout_size));
    cursor += sizeof(layout_size);

    std::memcpy(page.data() + cursor, layout.data(), layout.size() * sizeof(uint32_t));
    cursor += max_page_nodes * sizeof(uint32_t);

    for (uint32_t slot = 0; slot < max_page_nodes; ++slot) {
      if (slot < layout_size) {
        std::memcpy(page.data() + cursor, vectors[layout[slot]].data(), dim * sizeof(float));
      }
      cursor += dim * sizeof(float);

      uint16_t degree = 0;
      uint16_t dense_degree = 0;
      if (slot < layout_size) {
        const auto &neighbors = graph[layout[slot]];
        degree = static_cast<uint16_t>(neighbors.size());
        std::memcpy(page.data() + cursor, &degree, sizeof(degree));
        std::memcpy(page.data() + cursor + sizeof(uint16_t), &dense_degree, sizeof(dense_degree));
        std::memcpy(page.data() + cursor + sizeof(uint32_t),
                    neighbors.data(),
                    neighbors.size() * sizeof(uint32_t));
      } else {
        std::memcpy(page.data() + cursor, &degree, sizeof(degree));
        std::memcpy(page.data() + cursor + sizeof(uint16_t), &dense_degree, sizeof(dense_degree));
      }
      cursor += sizeof(uint32_t) + max_degree * sizeof(uint32_t);
    }

    WriteBytes(index_out, page.data(), page.size());
  }
}

void GraphReplicatedIndex::Load(const std::string &index_path, const std::string &approx_path) {
  if (!HasGraphReplicatedMagic(index_path)) {
    throw std::runtime_error("index file is not a project graph replicated index");
  }

  index_path_ = index_path;
  std::ifstream index_in(index_path, std::ios::binary);
  if (!index_in) {
    throw std::runtime_error("failed to open graph replicated index");
  }
  index_in.read(reinterpret_cast<char *>(&metadata_), sizeof(metadata_));
  if (!index_in) {
    throw std::runtime_error("failed to read index metadata");
  }
  ValidateGraphReplicatedMetadata(metadata_);
  search_metadata_ = ToSearchMetadata(metadata_);
  const uint64_t index_file_size = GetFileSize(index_path, "failed to stat graph replicated index");
  if (index_file_size != ExpectedGraphReplicatedIndexFileSize(metadata_)) {
    throw std::runtime_error("graph replicated index file size does not match metadata");
  }

  vector_data_path_ = approx_path.empty() ? DefaultPipeannBaseDataPath(index_path) : approx_path;
  if (vector_data_path_.empty()) {
    throw std::runtime_error("graph replicated index requires a full-precision vector source");
  }
  exact_vectors_ = LoadApproxVectors(vector_data_path_, search_metadata_);
  approx_vectors_ = exact_vectors_;
  vector_data_file_size_ = GetFileSize(vector_data_path_, "failed to stat full-precision vector file");

  id_to_page_.clear();
  page_layouts_.clear();
  reorder_offsets_.clear();
  reorder_ids_.clear();
  const std::string partition_path =
      ResolveExistingSidecarPath(DefaultPartitionSidecarPath(index_path),
                                 DefaultGorgeousPartitionSidecarPath(index_path));
  if (std::filesystem::exists(partition_path)) {
    const PartitionSidecar sidecar = ReadPartitionSidecar(partition_path);
    if (sidecar.num_points != search_metadata_.num_points) {
      throw std::runtime_error("partition sidecar point count does not match index metadata");
    }
    if (sidecar.num_pages != search_metadata_.num_pages) {
      throw std::runtime_error("partition sidecar page count does not match index metadata");
    }
    if (sidecar.page_capacity < metadata_.max_page_nodes) {
      throw std::runtime_error("partition sidecar page capacity is smaller than index metadata");
    }
    for (uint32_t page_id = 0; page_id < sidecar.layouts.size(); ++page_id) {
      ValidatePageLayout(search_metadata_.num_points,
                         sidecar.layouts[page_id],
                         "partition sidecar contains node ids outside the dataset range");
    }
    for (uint32_t node_id = 0; node_id < sidecar.id_to_page.size(); ++node_id) {
      if (sidecar.id_to_page[node_id] >= search_metadata_.num_pages) {
        throw std::runtime_error("partition sidecar id-to-page mapping exceeds page count");
      }
    }
    id_to_page_ = sidecar.id_to_page;
    page_layouts_ = sidecar.layouts;
  }

  const std::string reorder_path =
      ResolveExistingSidecarPath(DefaultReorderSidecarPath(index_path),
                                 DefaultGorgeousReorderSidecarPath(index_path));
  if (std::filesystem::exists(reorder_path)) {
    const ReorderSidecar sidecar = ReadReorderSidecar(reorder_path);
    if (sidecar.num_pages != search_metadata_.num_pages) {
      throw std::runtime_error("reorder sidecar page count does not match index metadata");
    }
    if (sidecar.offsets.size() != static_cast<size_t>(search_metadata_.num_pages) + 1) {
      throw std::runtime_error("reorder sidecar offsets length is invalid");
    }
    if (!sidecar.offsets.empty() && sidecar.offsets.front() != 0) {
      throw std::runtime_error("reorder sidecar must start at offset zero");
    }
    if (!sidecar.offsets.empty() && sidecar.offsets.back() != sidecar.num_entries) {
      throw std::runtime_error("reorder sidecar entry count does not match offsets");
    }
    for (size_t i = 1; i < sidecar.offsets.size(); ++i) {
      if (sidecar.offsets[i] < sidecar.offsets[i - 1]) {
        throw std::runtime_error("reorder sidecar offsets must be non-decreasing");
      }
    }
    ValidateNeighborIds(sidecar.ids, search_metadata_.num_points, "reorder sidecar contains node ids outside the dataset range");
    reorder_offsets_ = sidecar.offsets;
    reorder_ids_ = sidecar.ids;

    if (!page_layouts_.empty()) {
      for (uint32_t page_id = 0; page_id < search_metadata_.num_pages; ++page_id) {
        const uint64_t begin = reorder_offsets_[page_id];
        const uint64_t end = reorder_offsets_[page_id + 1];
        std::vector<uint32_t> layout(reorder_ids_.begin() + static_cast<std::ptrdiff_t>(begin),
                                     reorder_ids_.begin() + static_cast<std::ptrdiff_t>(end));
        if (layout != page_layouts_[page_id]) {
          throw std::runtime_error("reorder sidecar layout does not match partition sidecar");
        }
      }
    } else {
      page_layouts_.resize(search_metadata_.num_pages);
      id_to_page_.assign(search_metadata_.num_points, std::numeric_limits<uint32_t>::max());
      for (uint32_t page_id = 0; page_id < search_metadata_.num_pages; ++page_id) {
        const uint64_t begin = reorder_offsets_[page_id];
        const uint64_t end = reorder_offsets_[page_id + 1];
        auto &layout = page_layouts_[page_id];
        layout.assign(reorder_ids_.begin() + static_cast<std::ptrdiff_t>(begin),
                      reorder_ids_.begin() + static_cast<std::ptrdiff_t>(end));
        ValidatePageLayout(search_metadata_.num_points,
                           layout,
                           "reorder sidecar contains node ids outside the dataset range");
        for (uint32_t node_id : layout) {
          if (id_to_page_[node_id] == std::numeric_limits<uint32_t>::max()) {
            id_to_page_[node_id] = page_id;
          }
        }
      }
      for (uint32_t node_id = 0; node_id < search_metadata_.num_points; ++node_id) {
        if (id_to_page_[node_id] == std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("reorder sidecar is missing a primary page for some node");
        }
      }
    }
  }

  if (id_to_page_.empty()) {
    id_to_page_.resize(search_metadata_.num_points);
    for (uint32_t node_id = 0; node_id < search_metadata_.num_points; ++node_id) {
      id_to_page_[node_id] = std::min(node_id, search_metadata_.num_pages - 1);
    }
  }

  if (page_layouts_.empty()) {
    page_layouts_.resize(search_metadata_.num_pages);
    for (uint32_t page_id = 0; page_id < search_metadata_.num_pages; ++page_id) {
      page_layouts_[page_id].push_back(page_id);
    }
  }
}

void NativeGorgeousIndex::Load(const std::string &index_path, const std::string &approx_path) {
  if (HasGraphReplicatedMagic(index_path)) {
    throw std::runtime_error("index file is a project graph replicated index");
  }

  index_path_ = index_path;
  native_metadata_ = gorgeous_integration::ReadDiskIndexMetadata(index_path);
  const uint64_t index_file_size = GetFileSize(index_path, "failed to stat native Gorgeous index");
  if (native_metadata_.file_size != 0 && native_metadata_.file_size != index_file_size) {
    throw std::runtime_error("native Gorgeous index file size does not match metadata");
  }

  const std::string partition_path =
      ResolveExistingSidecarPath(DefaultPartitionSidecarPath(index_path),
                                 DefaultGorgeousPartitionSidecarPath(index_path));
  const bool has_partition_sidecar = std::filesystem::exists(partition_path);
  PartitionSidecar partition_sidecar;
  if (has_partition_sidecar) {
    partition_sidecar = ReadPartitionSidecar(partition_path);
  }

  payload_info_ = {};
  payload_info_.page_region_start_offset = 0;
  uint64_t num_pages = 0;
  uint32_t page_size = 0;
  const bool has_full_precision_payload = native_metadata_.full_precision_payload_start_sector != 0 &&
                                          native_metadata_.full_precision_nodes_per_sector != 0 &&
                                          native_metadata_.full_precision_dim != 0;
  if (has_full_precision_payload) {
    const uint64_t payload_start_sector = native_metadata_.full_precision_payload_start_sector;
    const uint64_t payload_nodes_per_sector = native_metadata_.full_precision_nodes_per_sector;
    if (payload_start_sector <= 1) {
      throw std::runtime_error("native full-precision payload metadata is invalid");
    }
    const uint64_t payload_sector_count = RoundUpDiv(native_metadata_.num_points, payload_nodes_per_sector);
    const uint64_t total_sectors = payload_start_sector + payload_sector_count;
    if (total_sectors == 0 || index_file_size % total_sectors != 0) {
      throw std::runtime_error("native Gorgeous index size is not compatible with payload metadata");
    }
    page_size = static_cast<uint32_t>(index_file_size / total_sectors);
    num_pages = payload_start_sector - 1;
    payload_info_.has_full_precision_payload = true;
    payload_info_.full_precision_payload_start_offset = payload_start_sector * page_size;
    payload_info_.full_precision_payload_end_offset = index_file_size;
    payload_info_.full_precision_dim = static_cast<uint32_t>(native_metadata_.full_precision_dim);
    payload_info_.full_precision_nodes_per_sector = static_cast<uint32_t>(payload_nodes_per_sector);
  } else if (native_metadata_.page_region_sectors > 1) {
    const uint64_t total_sectors = native_metadata_.page_region_sectors;
    if (total_sectors == 0 || index_file_size % total_sectors != 0) {
      throw std::runtime_error("native Gorgeous index size is not compatible with embedded page geometry");
    }
    page_size = static_cast<uint32_t>(index_file_size / total_sectors);
    num_pages = total_sectors - 1;
  } else {
    if (index_file_size % kNativeSectorLen != 0 || index_file_size <= kNativeSectorLen) {
      throw std::runtime_error("native Gorgeous index size is not sector aligned");
    }
    page_size = kNativeSectorLen;
    num_pages = index_file_size / page_size - 1;
  }

  search_metadata_.dim = static_cast<uint32_t>(native_metadata_.dim);
  search_metadata_.num_points = static_cast<uint32_t>(native_metadata_.num_points);
  search_metadata_.num_pages = static_cast<uint32_t>(num_pages);
  search_metadata_.entry_id = static_cast<uint32_t>(native_metadata_.entry_point);
  search_metadata_.page_size = page_size;
  ValidateSearchMetadata(search_metadata_);

  payload_info_.page_region_start_offset = page_size;
  payload_info_.page_region_end_offset = static_cast<uint64_t>(search_metadata_.num_pages + 1) * page_size;
  if (!payload_info_.has_full_precision_payload) {
    payload_info_.full_precision_payload_start_offset = payload_info_.page_region_end_offset;
    payload_info_.full_precision_payload_end_offset = payload_info_.page_region_end_offset;
  }

  page_boundaries_.resize(static_cast<size_t>(search_metadata_.num_pages) + 1);
  for (uint32_t page_id = 0; page_id < search_metadata_.num_pages; ++page_id) {
    page_boundaries_[page_id] = static_cast<uint64_t>(page_id + 1) * search_metadata_.page_size;
  }
  page_boundaries_.back() = payload_info_.page_region_end_offset;

  approx_vector_stream_ = std::ifstream();
  exact_vector_stream_ = std::ifstream();
  approx_vector_cache_.clear();
  exact_vector_cache_.clear();
  approx_vector_cache_id_ = std::numeric_limits<uint32_t>::max();
  exact_vector_cache_id_ = std::numeric_limits<uint32_t>::max();

  exact_vectors_.clear();
  exact_vector_path_.clear();
  exact_vector_file_size_ = 0;
  if (payload_info_.has_full_precision_payload) {
    exact_vector_path_ = index_path_;
    exact_vector_file_size_ =
        payload_info_.full_precision_payload_end_offset - payload_info_.full_precision_payload_start_offset;
  } else {
    exact_vector_path_ = approx_path.empty() ? DefaultPipeannBaseDataPath(index_path) : approx_path;
    if (exact_vector_path_.empty()) {
      throw std::runtime_error("native Gorgeous index requires a full-precision vector source");
    }
    exact_vectors_ = LoadApproxVectors(exact_vector_path_, search_metadata_);
    exact_vector_file_size_ = GetFileSize(exact_vector_path_, "failed to stat full-precision vector file");
  }

  approx_vectors_.clear();
  approx_vector_path_.clear();
  approx_vector_file_size_ = 0;
  if (!approx_path.empty()) {
    approx_vector_path_ = approx_path;
    approx_vectors_ = LoadApproxVectors(approx_vector_path_, search_metadata_);
    approx_vector_file_size_ = GetFileSize(approx_vector_path_, "failed to stat approx vector file");
  } else if (payload_info_.has_full_precision_payload) {
    approx_vector_path_ = exact_vector_path_;
    approx_vector_file_size_ = exact_vector_file_size_;
  } else {
    approx_vectors_ = exact_vectors_;
    approx_vector_path_ = exact_vector_path_;
    approx_vector_file_size_ = exact_vector_file_size_;
  }

  const LayoutScanResult scanned = ScanNativeLayouts(index_path_, search_metadata_, native_metadata_);
  page_layouts_ = scanned.layouts;
  id_to_page_ = scanned.id_to_page;
  for (uint32_t node_id = 0; node_id < id_to_page_.size(); ++node_id) {
    if (id_to_page_[node_id] == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("native page scan is missing a node from all page layouts");
    }
  }

  if (has_partition_sidecar) {
    ValidatePartitionSidecar(partition_sidecar, search_metadata_, page_layouts_);
  }

  const std::string reorder_path =
      ResolveExistingSidecarPath(DefaultReorderSidecarPath(index_path),
                                 DefaultGorgeousReorderSidecarPath(index_path));
  if (std::filesystem::exists(reorder_path)) {
    const ReorderSidecar reorder_sidecar = ReadReorderSidecar(reorder_path);
    ValidateReorderSidecar(reorder_sidecar, search_metadata_, page_layouts_);
  }
  BuildReorderFromLayouts(page_layouts_, &reorder_offsets_, &reorder_ids_);
}

const std::vector<float> &GraphReplicatedIndex::approx_vector(uint32_t id) const {
  if (id >= approx_vectors_.size()) {
    throw std::runtime_error("approx_vector id out of range");
  }
  return approx_vectors_[id];
}

const std::vector<float> &GraphReplicatedIndex::exact_vector(uint32_t id) const {
  if (id >= exact_vectors_.size()) {
    throw std::runtime_error("exact_vector id out of range");
  }
  return exact_vectors_[id];
}

const std::vector<float> &NativeGorgeousIndex::approx_vector(uint32_t id) const {
  if (!approx_vectors_.empty()) {
    if (id >= approx_vectors_.size()) {
      throw std::runtime_error("approx_vector id out of range");
    }
    return approx_vectors_[id];
  }
  if (id >= search_metadata_.num_points) {
    throw std::runtime_error("approx_vector id out of range");
  }
  if (approx_vector_cache_id_ != id) {
    if (payload_info_.has_full_precision_payload && approx_vector_path_ == exact_vector_path_) {
      ReadPayloadVector(id, &approx_vector_cache_);
    } else {
      ReadBinVector(approx_vector_path_, &approx_vector_stream_, id, &approx_vector_cache_);
    }
    approx_vector_cache_id_ = id;
  }
  return approx_vector_cache_;
}

const std::vector<float> &NativeGorgeousIndex::exact_vector(uint32_t id) const {
  if (!exact_vectors_.empty()) {
    if (id >= exact_vectors_.size()) {
      throw std::runtime_error("exact_vector id out of range");
    }
    return exact_vectors_[id];
  }
  if (id >= search_metadata_.num_points) {
    throw std::runtime_error("exact_vector id out of range");
  }
  if (exact_vector_cache_id_ != id) {
    if (payload_info_.has_full_precision_payload) {
      ReadPayloadVector(id, &exact_vector_cache_);
    } else {
      ReadBinVector(exact_vector_path_, &exact_vector_stream_, id, &exact_vector_cache_);
    }
    exact_vector_cache_id_ = id;
  }
  return exact_vector_cache_;
}

uint64_t GraphReplicatedIndex::IndexFileSizeBytes() const {
  ValidateGraphReplicatedMetadata(metadata_);
  return ExpectedGraphReplicatedIndexFileSize(metadata_);
}

uint64_t GraphReplicatedIndex::ApproxFileSizeBytes() const {
  ValidateGraphReplicatedMetadata(metadata_);
  return vector_data_file_size_;
}

uint64_t NativeGorgeousIndex::IndexFileSizeBytes() const {
  ValidateSearchMetadata(search_metadata_);
  return GetFileSize(index_path_, "failed to stat native Gorgeous index");
}

uint64_t NativeGorgeousIndex::ApproxFileSizeBytes() const {
  ValidateSearchMetadata(search_metadata_);
  return approx_vector_file_size_;
}

void NativeGorgeousIndex::ReadPayloadVector(uint32_t id, std::vector<float> *out) const {
  if (out == nullptr) {
    throw std::runtime_error("payload vector output must not be null");
  }
  if (!payload_info_.has_full_precision_payload) {
    throw std::runtime_error("native Gorgeous index does not have a full-precision payload");
  }
  const uint64_t page_size = payload_info_.page_region_start_offset;
  const size_t vector_bytes = static_cast<size_t>(payload_info_.full_precision_dim) * sizeof(float);
  const uint64_t sector = id / payload_info_.full_precision_nodes_per_sector;
  const uint64_t slot = id % payload_info_.full_precision_nodes_per_sector;
  const uint64_t offset = payload_info_.full_precision_payload_start_offset +
                          sector * page_size +
                          slot * vector_bytes;
  if (offset + vector_bytes > payload_info_.full_precision_payload_end_offset) {
    throw std::runtime_error("native full-precision payload access exceeds payload bounds");
  }
  EnsureStreamOpen(index_path_, &exact_vector_stream_, "failed to open native index for payload vector access");
  exact_vector_stream_.clear();
  exact_vector_stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!exact_vector_stream_) {
    throw std::runtime_error("failed to seek native full-precision payload");
  }
  out->assign(payload_info_.full_precision_dim, 0.0f);
  exact_vector_stream_.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(vector_bytes));
  if (!exact_vector_stream_) {
    throw std::runtime_error("failed to read native full-precision payload");
  }
}

void NativeGorgeousIndex::ReadBinVector(const std::string &path,
                                        std::ifstream *stream,
                                        uint32_t id,
                                        std::vector<float> *out) const {
  if (stream == nullptr || out == nullptr) {
    throw std::runtime_error("bin vector access requires valid stream and output buffers");
  }
  EnsureStreamOpen(path, stream, "failed to open vector file for lazy access");
  stream->clear();
  const uint64_t offset = sizeof(uint32_t) * 2 +
                          static_cast<uint64_t>(id) * static_cast<uint64_t>(search_metadata_.dim) * sizeof(float);
  stream->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!*stream) {
    throw std::runtime_error("failed to seek vector file for lazy access");
  }
  out->assign(search_metadata_.dim, 0.0f);
  stream->read(reinterpret_cast<char *>(out->data()),
               static_cast<std::streamsize>(static_cast<size_t>(search_metadata_.dim) * sizeof(float)));
  if (!*stream) {
    throw std::runtime_error("failed to read vector file for lazy access");
  }
}

PageView GraphReplicatedIndex::ViewPage(uint32_t page_id, const std::vector<char> &page_bytes) const {
  return ViewPage(page_id, page_bytes.data(), page_bytes.size());
}

PageView GraphReplicatedIndex::ViewPage(uint32_t page_id,
                                        const char *page_bytes,
                                        size_t page_bytes_size) const {
  ValidateGraphReplicatedMetadata(metadata_);
  if (page_id >= search_metadata_.num_pages) {
    throw std::runtime_error("page id out of range");
  }
  if (page_bytes_size < search_metadata_.page_size) {
    throw std::runtime_error("page payload is smaller than configured page size");
  }

  PageView page;
  page.page_id = page_id;
  page.page_bytes = page_bytes;
  page.page_bytes_size = page_bytes_size;
  page.native_gorgeous = false;

  size_t cursor = 0;
  std::memcpy(&page.layout_size, page_bytes + cursor, sizeof(page.layout_size));
  cursor += sizeof(page.layout_size);
  if (page.layout_size == 0 || page.layout_size > metadata_.max_page_nodes) {
    throw std::runtime_error("page layout size exceeds metadata bounds");
  }

  page.layout_ids = reinterpret_cast<const uint32_t *>(page_bytes + cursor);
  std::vector<uint32_t> layout_copy(page.layout_ids, page.layout_ids + page.layout_size);
  ValidatePageLayout(search_metadata_.num_points, layout_copy, "page layout contains node ids outside the dataset range");
  page.target_id = page.layout_ids[0];
  cursor += static_cast<size_t>(metadata_.max_page_nodes) * sizeof(uint32_t);
  page.node_region = page_bytes + cursor;

  const size_t block_stride = sizeof(uint32_t) + static_cast<size_t>(metadata_.max_degree) * sizeof(uint32_t);
  if (cursor + static_cast<size_t>(metadata_.max_page_nodes) * block_stride > page_bytes_size) {
    throw std::runtime_error("page graph block exceeds configured page size");
  }

  for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
    const DiskNodeView node = ViewNode(page, slot);
    ValidateNeighborIds(node.nbrs,
                        node.nnbrs,
                        search_metadata_.num_points,
                        "page graph block contains neighbor ids outside the dataset range");
    ValidateNeighborIds(node.dense_nbrs,
                        node.n_dense_nbrs,
                        search_metadata_.num_points,
                        "page graph block contains neighbor ids outside the dataset range");
  }

  return page;
}

PageView NativeGorgeousIndex::ViewPage(uint32_t page_id, const std::vector<char> &page_bytes) const {
  return ViewPage(page_id, page_bytes.data(), page_bytes.size());
}

PageView NativeGorgeousIndex::ViewPage(uint32_t page_id,
                                       const char *page_bytes,
                                       size_t page_bytes_size) const {
  ValidateSearchMetadata(search_metadata_);
  if (page_id >= search_metadata_.num_pages) {
    throw std::runtime_error("page id out of range");
  }
  if (page_bytes_size < search_metadata_.page_size) {
    throw std::runtime_error("page payload is smaller than configured page size");
  }

  PageView page;
  page.page_id = page_id;
  page.page_bytes = page_bytes;
  page.page_bytes_size = page_bytes_size;
  page.native_gorgeous = true;
  page.target_coords = reinterpret_cast<const float *>(page_bytes);

  size_t cursor = static_cast<size_t>(native_metadata_.dim) * sizeof(float);
  std::memcpy(&page.layout_size, page_bytes + cursor, sizeof(page.layout_size));
  cursor += sizeof(page.layout_size);
  if (page.layout_size == 0 || page.layout_size > native_metadata_.nodes_per_sector) {
    throw std::runtime_error("native page layout size exceeds metadata bounds");
  }

  page.layout_ids = reinterpret_cast<const uint32_t *>(page_bytes + cursor);
  std::vector<uint32_t> layout_copy(page.layout_ids, page.layout_ids + page.layout_size);
  ValidatePageLayout(search_metadata_.num_points,
                     layout_copy,
                     "native page layout contains node ids outside the dataset range");
  page.target_id = page.layout_ids[0];
  cursor += static_cast<size_t>(native_metadata_.nodes_per_sector) * sizeof(uint32_t);
  page.node_region = page_bytes + cursor;

  const size_t graph_node_len =
      static_cast<size_t>(native_metadata_.max_node_len) - static_cast<size_t>(native_metadata_.dim) * sizeof(float);
  if (graph_node_len < sizeof(uint32_t)) {
    throw std::runtime_error("native graph block is smaller than degree header");
  }
  if (cursor + static_cast<size_t>(page.layout_size) * graph_node_len > page_bytes_size) {
    throw std::runtime_error("native page graph block exceeds configured page size");
  }

  for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
    const DiskNodeView node = ViewNode(page, slot);
    ValidateNeighborIds(node.nbrs,
                        node.nnbrs,
                        search_metadata_.num_points,
                        "native page graph block contains neighbor ids outside the dataset range");
    ValidateNeighborIds(node.dense_nbrs,
                        node.n_dense_nbrs,
                        search_metadata_.num_points,
                        "native page graph block contains dense neighbor ids outside the dataset range");
  }

  return page;
}

DiskNodeView GraphReplicatedIndex::ViewNode(const PageView &page, uint32_t slot) const {
  ValidateGraphReplicatedMetadata(metadata_);
  if (slot >= page.layout_size) {
    throw std::runtime_error("page node slot exceeds layout size");
  }

  DiskNodeView view;
  view.id = page.LayoutAt(slot);
  const pipeann::SSDIndexMetadata<float> disk_meta = BuildPageDiskNodeMetadata(metadata_);
  pipeann::DiskNode<float> node(const_cast<char *>(page.node_region), slot, disk_meta);
  const uint32_t max_dense_degree = metadata_.max_degree - metadata_.max_base_degree;
  if (node.nnbrs > metadata_.max_base_degree) {
    throw std::runtime_error("page graph block base degree exceeds metadata bounds");
  }
  if (node.n_dense_nbrs > max_dense_degree) {
    throw std::runtime_error("page graph block dense degree exceeds metadata bounds");
  }
  view.coords = node.coords;
  view.nnbrs = node.nnbrs;
  view.n_dense_nbrs = node.n_dense_nbrs;
  view.nbrs = node.nbrs;
  view.attrs = node.attrs;
  view.attr_size = static_cast<uint32_t>(disk_meta.attr_size);
  view.dense_nbrs = node.dense_nbrs;
  return view;
}

DiskNodeView NativeGorgeousIndex::ViewNode(const PageView &page, uint32_t slot) const {
  ValidateSearchMetadata(search_metadata_);
  if (slot >= page.layout_size) {
    throw std::runtime_error("page node slot exceeds layout size");
  }

  const pipeann::SSDIndexMetadata<float> disk_meta =
      BuildNativeDiskNodeMetadata(native_metadata_, page.layout_size);
  const size_t graph_node_len =
      static_cast<size_t>(disk_meta.max_node_len) - static_cast<size_t>(disk_meta.data_dim) * sizeof(float);
  const char *block = page.node_region + static_cast<size_t>(slot) * graph_node_len;
  const size_t page_prefix_bytes = static_cast<size_t>(page.node_region - page.page_bytes);
  const size_t block_offset = page_prefix_bytes + static_cast<size_t>(slot) * graph_node_len;
  if (block_offset + graph_node_len > page.page_bytes_size) {
    throw std::runtime_error("native Gorgeous page node block exceeds page bounds");
  }
  uint16_t nnbrs = 0;
  uint16_t n_dense_nbrs = 0;
  std::memcpy(&nnbrs, block, sizeof(nnbrs));
  std::memcpy(&n_dense_nbrs, block + sizeof(uint16_t), sizeof(n_dense_nbrs));
  if (nnbrs > disk_meta.range) {
    throw std::runtime_error("native Gorgeous page graph base degree exceeds metadata bounds");
  }
  const uint64_t dense_capacity = disk_meta.range_dense >= disk_meta.range ? disk_meta.range_dense - disk_meta.range : 0;
  if (n_dense_nbrs > dense_capacity) {
    throw std::runtime_error("native Gorgeous page graph dense degree exceeds metadata bounds");
  }

  DiskNodeView view;
  view.id = page.LayoutAt(slot);
  view.coords = slot == 0 ? page.target_coords : nullptr;
  view.nnbrs = nnbrs;
  view.n_dense_nbrs = n_dense_nbrs;
  view.nbrs = reinterpret_cast<const uint32_t *>(block + sizeof(uint32_t));
  const size_t attrs_offset = block_offset + sizeof(uint32_t) + static_cast<size_t>(disk_meta.range) * sizeof(uint32_t);
  const size_t dense_offset = attrs_offset + static_cast<size_t>(disk_meta.attr_size);
  const size_t dense_bytes = static_cast<size_t>(n_dense_nbrs) * sizeof(uint32_t);
  if (dense_offset + dense_bytes > page.page_bytes_size) {
    throw std::runtime_error("native Gorgeous attrs or dense-neighbor payload exceeds page bounds");
  }
  view.attrs = disk_meta.attr_size == 0 ? nullptr
                                        : static_cast<const void *>(page.page_bytes + attrs_offset);
  view.attr_size = static_cast<uint32_t>(disk_meta.attr_size);
  view.dense_nbrs = reinterpret_cast<const uint32_t *>(page.page_bytes + dense_offset);
  return view;
}

std::vector<uint32_t> GraphReplicatedIndex::CopyPageLayout(const PageView &page) const {
  return std::vector<uint32_t>(page.layout_ids, page.layout_ids + page.layout_size);
}

std::vector<uint32_t> NativeGorgeousIndex::CopyPageLayout(const PageView &page) const {
  return std::vector<uint32_t>(page.layout_ids, page.layout_ids + page.layout_size);
}

std::vector<uint32_t> GraphReplicatedIndex::CopyNeighbors(const DiskNodeView &node) const {
  return CopyNeighborsVector(node);
}

std::vector<uint32_t> NativeGorgeousIndex::CopyNeighbors(const DiskNodeView &node) const {
  return CopyNeighborsVector(node);
}

pipeann_integration::IORequest GraphReplicatedIndex::BuildPageReadRequest(uint32_t page_id, void *aligned_buf) const {
  if (aligned_buf == nullptr) {
    throw std::runtime_error("page read buffer must not be null");
  }
  const uint64_t offset = PageOffset(page_id);
  const uint64_t len = search_metadata_.page_size;
  return pipeann_integration::IORequest(offset, len, aligned_buf, offset, len, aligned_buf);
}

pipeann_integration::IORequest NativeGorgeousIndex::BuildPageReadRequest(uint32_t page_id, void *aligned_buf) const {
  if (aligned_buf == nullptr) {
    throw std::runtime_error("page read buffer must not be null");
  }
  const uint64_t offset = PageOffset(page_id);
  const uint64_t len = search_metadata_.page_size;
  return pipeann_integration::IORequest(offset, len, aligned_buf, offset, len, aligned_buf);
}

uint64_t GraphReplicatedIndex::PageOffset(uint32_t page_id) const {
  if (page_id >= search_metadata_.num_pages) {
    throw std::runtime_error("page id out of range");
  }
  return static_cast<uint64_t>(page_id + 1) * search_metadata_.page_size;
}

uint64_t NativeGorgeousIndex::PageOffset(uint32_t page_id) const {
  if (page_id >= search_metadata_.num_pages) {
    throw std::runtime_error("page id out of range");
  }
  return page_boundaries_[page_id];
}

uint32_t GraphReplicatedIndex::PageForNode(uint32_t node_id) const {
  if (node_id >= search_metadata_.num_points) {
    throw std::runtime_error("node id out of range");
  }
  if (!id_to_page_.empty()) {
    return id_to_page_[node_id];
  }
  if (node_id >= search_metadata_.num_pages) {
    throw std::runtime_error("node id cannot be mapped to a page");
  }
  return node_id;
}

uint32_t NativeGorgeousIndex::PageForNode(uint32_t node_id) const {
  if (node_id >= search_metadata_.num_points) {
    throw std::runtime_error("node id out of range");
  }
  if (id_to_page_.empty()) {
    throw std::runtime_error("native index does not have a node-to-page mapping");
  }
  return id_to_page_[node_id];
}

bool GraphReplicatedIndex::HasPartitionData() const {
  return !page_layouts_.empty();
}

bool GraphReplicatedIndex::HasReorderData() const {
  return !reorder_offsets_.empty();
}

bool NativeGorgeousIndex::HasPartitionData() const {
  return !page_layouts_.empty();
}

bool NativeGorgeousIndex::HasReorderData() const {
  return !reorder_offsets_.empty();
}

std::unique_ptr<IndexReader> LoadIndexReader(const std::string &index_path, const std::string &approx_path) {
  if (HasGraphReplicatedMagic(index_path)) {
    auto index = std::make_unique<GraphReplicatedIndex>();
    index->Load(index_path, approx_path);
    return index;
  }
  auto index = std::make_unique<NativeGorgeousIndex>();
  index->Load(index_path, approx_path);
  return index;
}

std::string DefaultPartitionSidecarPath(const std::string &index_path) {
  return index_path + ".partition";
}

std::string DefaultReorderSidecarPath(const std::string &index_path) {
  return index_path + ".reorder";
}

std::string DefaultGorgeousPartitionSidecarPath(const std::string &index_path) {
  return RemoveKnownIndexSuffix(index_path) + "_partition.bin";
}

std::string DefaultGorgeousReorderSidecarPath(const std::string &index_path) {
  return RemoveKnownIndexSuffix(index_path) + "_reorder.bin";
}

std::string DefaultPipeannBaseDataPath(const std::string &index_path) {
  return RemoveKnownIndexSuffix(index_path) + ".bin";
}

std::string DefaultPipeannPqPivotsPath(const std::string &index_path) {
  return RemoveKnownIndexSuffix(index_path) + "_pq_pivots.bin";
}

std::string DefaultPipeannPqCompressedPath(const std::string &index_path) {
  return RemoveKnownIndexSuffix(index_path) + "_pq_compressed.bin";
}

void WriteGorgeousPartitionSidecar(const std::string &path,
                                   uint32_t page_capacity,
                                   uint32_t num_points,
                                   const std::vector<std::vector<uint32_t>> &layouts,
                                   const std::vector<uint32_t> *id_to_page_override) {
  std::vector<uint32_t> id_to_page(num_points, std::numeric_limits<uint32_t>::max());
  for (uint32_t page_id = 0; page_id < layouts.size(); ++page_id) {
    ValidatePageLayout(num_points,
                       layouts[page_id],
                       "partition sidecar contains node ids outside the dataset range");
    if (id_to_page_override == nullptr) {
      for (uint32_t node_id : layouts[page_id]) {
        if (id_to_page[node_id] == std::numeric_limits<uint32_t>::max()) {
          id_to_page[node_id] = page_id;
        }
      }
    }
  }
  if (id_to_page_override != nullptr) {
    if (id_to_page_override->size() != num_points) {
      throw std::runtime_error("partition sidecar id_to_page override size does not match num_points");
    }
    id_to_page = *id_to_page_override;
    for (uint32_t node_id = 0; node_id < id_to_page.size(); ++node_id) {
      if (id_to_page[node_id] >= layouts.size()) {
        throw std::runtime_error("partition sidecar id_to_page override exceeds page count");
      }
    }
  }

  for (uint32_t node_id = 0; node_id < num_points; ++node_id) {
    if (id_to_page[node_id] == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("partition sidecar is missing a primary page for some node");
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open partition sidecar output");
  }
  WritePod<uint64_t>(out, page_capacity);
  WritePod<uint64_t>(out, layouts.size());
  WritePod<uint64_t>(out, num_points);
  for (const auto &layout : layouts) {
    const uint32_t size = static_cast<uint32_t>(layout.size());
    WritePod<uint32_t>(out, size);
    if (size > 0) {
      WriteBytes(out, layout.data(), static_cast<size_t>(size) * sizeof(uint32_t));
    }
  }
  if (!id_to_page.empty()) {
    WriteBytes(out, id_to_page.data(), id_to_page.size() * sizeof(uint32_t));
  }
}

void WriteGorgeousReorderSidecar(const std::string &path,
                                 const std::vector<std::vector<uint32_t>> &layouts) {
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> ids;
  BuildReorderFromLayouts(layouts, &offsets, &ids);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open reorder sidecar output");
  }
  WritePod<uint64_t>(out, layouts.size());
  WritePod<uint64_t>(out, ids.size());
  if (!offsets.empty()) {
    WriteBytes(out, offsets.data(), offsets.size() * sizeof(uint64_t));
  }
  if (!ids.empty()) {
    WriteBytes(out, ids.data(), ids.size() * sizeof(uint32_t));
  }
}

}  // namespace hybrid
