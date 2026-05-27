#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "integrations/linux_aligned_file_reader.h"
#include "integrations/disk_index_relayout.h"

namespace hybrid {

enum class IndexStorageFormat {
  kProjectGraphReplicated = 0,
  kGorgeousNative = 1,
};

struct GraphReplicatedMetadata {
  uint64_t magic = 0;
  uint32_t version = 4;
  uint32_t dim = 0;
  uint32_t num_points = 0;
  uint32_t num_pages = 0;
  uint32_t max_degree = 0;
  uint32_t max_base_degree = 0;
  uint32_t max_page_nodes = 0;
  uint32_t entry_id = 0;
  uint32_t page_size = 0;
};

struct SearchIndexMetadata {
  uint32_t dim = 0;
  uint32_t num_points = 0;
  uint32_t num_pages = 0;
  uint32_t entry_id = 0;
  uint32_t page_size = 0;
};

struct DiskNodeView {
  uint32_t id = 0;
  const float *coords = nullptr;
  uint16_t nnbrs = 0;
  uint16_t n_dense_nbrs = 0;
  const uint32_t *nbrs = nullptr;
  const void *attrs = nullptr;
  uint32_t attr_size = 0;
  const uint32_t *dense_nbrs = nullptr;

  uint32_t total_degree() const { return static_cast<uint32_t>(nnbrs) + static_cast<uint32_t>(n_dense_nbrs); }
  bool has_embedded_coords() const { return coords != nullptr; }
  bool has_attrs() const { return attrs != nullptr && attr_size != 0; }
};

struct PageView {
  uint32_t page_id = 0;
  uint32_t target_id = 0;
  uint32_t layout_size = 0;
  const uint32_t *layout_ids = nullptr;
  const float *target_coords = nullptr;
  const char *node_region = nullptr;
  const char *page_bytes = nullptr;
  size_t page_bytes_size = 0;
  bool native_gorgeous = false;

  uint32_t LayoutAt(uint32_t slot) const;
};

struct NativePayloadInfo {
  bool has_full_precision_payload = false;
  uint64_t page_region_start_offset = 0;
  uint64_t page_region_end_offset = 0;
  uint64_t full_precision_payload_start_offset = 0;
  uint64_t full_precision_payload_end_offset = 0;
  uint32_t full_precision_dim = 0;
  uint32_t full_precision_nodes_per_sector = 0;
};

class GraphReplicatedLayoutBuilder {
 public:
  static void Build(const std::string &index_path,
                    const std::string &approx_path,
                    const std::vector<std::vector<float>> &vectors,
                    const std::vector<std::vector<uint32_t>> &graph,
                    const std::vector<std::vector<uint32_t>> &page_layouts,
                    uint32_t entry_id);
};

class IndexReader {
 public:
  virtual ~IndexReader() = default;

  virtual IndexStorageFormat storage_format() const = 0;
  virtual const SearchIndexMetadata &search_metadata() const = 0;
  virtual const std::string &index_path() const = 0;
  virtual uint64_t IndexFileSizeBytes() const = 0;
  virtual uint64_t ApproxFileSizeBytes() const = 0;
  virtual const std::vector<float> &approx_vector(uint32_t id) const = 0;
  virtual const std::vector<float> &exact_vector(uint32_t id) const = 0;
  virtual PageView ViewPage(uint32_t page_id, const std::vector<char> &page_bytes) const = 0;
  virtual PageView ViewPage(uint32_t page_id, const char *page_bytes, size_t page_bytes_size) const = 0;
  virtual DiskNodeView ViewNode(const PageView &page, uint32_t slot) const = 0;
  virtual std::vector<uint32_t> CopyPageLayout(const PageView &page) const = 0;
  virtual std::vector<uint32_t> CopyNeighbors(const DiskNodeView &node) const = 0;
  virtual pipeann_integration::IORequest BuildPageReadRequest(uint32_t page_id, void *aligned_buf) const = 0;
  virtual uint64_t PageOffset(uint32_t page_id) const = 0;
  virtual uint32_t PageForNode(uint32_t node_id) const = 0;
  virtual bool HasPartitionData() const = 0;
  virtual bool HasReorderData() const = 0;
  virtual const std::vector<uint32_t> &id_to_page() const = 0;
  virtual const std::vector<std::vector<uint32_t>> &page_layouts() const = 0;
  virtual const std::vector<uint64_t> &reorder_offsets() const = 0;
  virtual const std::vector<uint32_t> &reorder_ids() const = 0;
};

class GraphReplicatedIndex : public IndexReader {
 public:
  void Load(const std::string &index_path, const std::string &approx_path);

  const GraphReplicatedMetadata &metadata() const { return metadata_; }
  IndexStorageFormat storage_format() const override { return IndexStorageFormat::kProjectGraphReplicated; }
  const SearchIndexMetadata &search_metadata() const override { return search_metadata_; }
  const std::string &index_path() const override { return index_path_; }
  uint64_t IndexFileSizeBytes() const override;
  uint64_t ApproxFileSizeBytes() const override;
  const std::vector<float> &approx_vector(uint32_t id) const override;
  const std::vector<float> &exact_vector(uint32_t id) const override;
  PageView ViewPage(uint32_t page_id, const std::vector<char> &page_bytes) const override;
  PageView ViewPage(uint32_t page_id, const char *page_bytes, size_t page_bytes_size) const override;
  DiskNodeView ViewNode(const PageView &page, uint32_t slot) const override;
  std::vector<uint32_t> CopyPageLayout(const PageView &page) const override;
  std::vector<uint32_t> CopyNeighbors(const DiskNodeView &node) const override;
  pipeann_integration::IORequest BuildPageReadRequest(uint32_t page_id, void *aligned_buf) const override;
  uint64_t PageOffset(uint32_t page_id) const override;
  uint32_t PageForNode(uint32_t node_id) const override;
  bool HasPartitionData() const override;
  bool HasReorderData() const override;
  const std::vector<uint32_t> &id_to_page() const override { return id_to_page_; }
  const std::vector<std::vector<uint32_t>> &page_layouts() const override { return page_layouts_; }
  const std::vector<uint64_t> &reorder_offsets() const override { return reorder_offsets_; }
  const std::vector<uint32_t> &reorder_ids() const override { return reorder_ids_; }

 private:
  SearchIndexMetadata search_metadata_{};
  GraphReplicatedMetadata metadata_{};
  std::string index_path_;
  std::string vector_data_path_;
  uint64_t vector_data_file_size_ = 0;
  std::vector<std::vector<float>> approx_vectors_;
  std::vector<std::vector<float>> exact_vectors_;
  std::vector<uint32_t> id_to_page_;
  std::vector<std::vector<uint32_t>> page_layouts_;
  std::vector<uint64_t> reorder_offsets_;
  std::vector<uint32_t> reorder_ids_;
};

class NativeGorgeousIndex : public IndexReader {
 public:
  void Load(const std::string &index_path, const std::string &approx_path);

  const gorgeous_integration::DiskIndexMetadata &native_metadata() const { return native_metadata_; }
  const NativePayloadInfo &payload_info() const { return payload_info_; }
  const std::vector<uint64_t> &page_boundaries() const { return page_boundaries_; }

  IndexStorageFormat storage_format() const override { return IndexStorageFormat::kGorgeousNative; }
  const SearchIndexMetadata &search_metadata() const override { return search_metadata_; }
  const std::string &index_path() const override { return index_path_; }
  uint64_t IndexFileSizeBytes() const override;
  uint64_t ApproxFileSizeBytes() const override;
  const std::vector<float> &approx_vector(uint32_t id) const override;
  const std::vector<float> &exact_vector(uint32_t id) const override;
  PageView ViewPage(uint32_t page_id, const std::vector<char> &page_bytes) const override;
  PageView ViewPage(uint32_t page_id, const char *page_bytes, size_t page_bytes_size) const override;
  DiskNodeView ViewNode(const PageView &page, uint32_t slot) const override;
  std::vector<uint32_t> CopyPageLayout(const PageView &page) const override;
  std::vector<uint32_t> CopyNeighbors(const DiskNodeView &node) const override;
  pipeann_integration::IORequest BuildPageReadRequest(uint32_t page_id, void *aligned_buf) const override;
  uint64_t PageOffset(uint32_t page_id) const override;
  uint32_t PageForNode(uint32_t node_id) const override;
  bool HasPartitionData() const override;
  bool HasReorderData() const override;
  const std::vector<uint32_t> &id_to_page() const override { return id_to_page_; }
  const std::vector<std::vector<uint32_t>> &page_layouts() const override { return page_layouts_; }
  const std::vector<uint64_t> &reorder_offsets() const override { return reorder_offsets_; }
  const std::vector<uint32_t> &reorder_ids() const override { return reorder_ids_; }

 private:
  void ReadPayloadVector(uint32_t id, std::vector<float> *out) const;
  void ReadBinVector(const std::string &path, std::ifstream *stream, uint32_t id, std::vector<float> *out) const;

  SearchIndexMetadata search_metadata_{};
  gorgeous_integration::DiskIndexMetadata native_metadata_{};
  NativePayloadInfo payload_info_{};
  std::string index_path_;
  std::string approx_vector_path_;
  std::string exact_vector_path_;
  uint64_t approx_vector_file_size_ = 0;
  uint64_t exact_vector_file_size_ = 0;
  std::vector<uint64_t> page_boundaries_;
  std::vector<std::vector<float>> approx_vectors_;
  std::vector<std::vector<float>> exact_vectors_;
  std::vector<uint32_t> id_to_page_;
  std::vector<std::vector<uint32_t>> page_layouts_;
  std::vector<uint64_t> reorder_offsets_;
  std::vector<uint32_t> reorder_ids_;
  mutable std::ifstream approx_vector_stream_;
  mutable std::ifstream exact_vector_stream_;
  mutable std::vector<float> approx_vector_cache_;
  mutable std::vector<float> exact_vector_cache_;
  mutable uint32_t approx_vector_cache_id_ = UINT32_MAX;
  mutable uint32_t exact_vector_cache_id_ = UINT32_MAX;
};

std::unique_ptr<IndexReader> LoadIndexReader(const std::string &index_path, const std::string &approx_path);

std::string DefaultPartitionSidecarPath(const std::string &index_path);
std::string DefaultReorderSidecarPath(const std::string &index_path);
std::string DefaultGorgeousPartitionSidecarPath(const std::string &index_path);
std::string DefaultGorgeousReorderSidecarPath(const std::string &index_path);
std::string DefaultPipeannBaseDataPath(const std::string &index_path);
std::string DefaultPipeannPqPivotsPath(const std::string &index_path);
std::string DefaultPipeannPqCompressedPath(const std::string &index_path);
void WriteGorgeousPartitionSidecar(const std::string &path,
                                   uint32_t page_capacity,
                                   uint32_t num_points,
                                   const std::vector<std::vector<uint32_t>> &layouts,
                                   const std::vector<uint32_t> *id_to_page = nullptr);
void WriteGorgeousReorderSidecar(const std::string &path,
                                 const std::vector<std::vector<uint32_t>> &layouts);

float L2Distance(const std::vector<float> &lhs, const std::vector<float> &rhs);
float L2Distance(const std::vector<float> &lhs, const float *rhs, uint32_t dim);

}  // namespace hybrid
