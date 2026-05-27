#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "gorgeous_layout.h"
#include "integrations/disk_index_relayout.h"
#include "integrations/linux_aligned_file_reader.h"
#include "filter/attribute.h"

namespace {

constexpr uint64_t kSectorLen = 4096;

uint64_t NodeOffset(uint32_t node_id, uint64_t nodes_per_sector, uint64_t max_node_len) {
  return (static_cast<uint64_t>(node_id) / nodes_per_sector + 1) * kSectorLen +
         (static_cast<uint64_t>(node_id) % nodes_per_sector) * max_node_len;
}

void WriteSyntheticDiskIndex(const std::filesystem::path &path) {
  constexpr uint64_t kNumPoints = 4;
  constexpr uint64_t kDim = 2;
  constexpr uint64_t kRange = 2;
  constexpr uint64_t kRangeDense = 3;
  constexpr uint32_t kAttrKey = 11;
  const std::vector<pipeann::Attributes> attrs = [] {
    std::vector<pipeann::Attributes> rows(4);
    rows[0].set(kAttrKey, {7});
    rows[1].set(kAttrKey, {1});
    rows[2].set(kAttrKey, {7});
    rows[3].set(kAttrKey, {2});
    return rows;
  }();
  uint64_t attr_size = 0;
  for (const auto &row : attrs) {
    attr_size = std::max<uint64_t>(attr_size, row.serialized_size());
  }
  const uint64_t kMaxNodeLen = kDim * sizeof(float) + sizeof(uint32_t) + kRange * sizeof(uint32_t) + attr_size +
                               (kRangeDense - kRange) * sizeof(uint32_t);
  constexpr uint64_t kNodesPerSector = 2;
  constexpr uint64_t kFileSize = kSectorLen + 2 * kSectorLen;

  std::vector<char> bytes(static_cast<size_t>(kFileSize), 0);
  const uint32_t nr = 9;
  const uint32_t nc = 1;
  std::memcpy(bytes.data(), &nr, sizeof(nr));
  std::memcpy(bytes.data() + sizeof(uint32_t), &nc, sizeof(nc));
  uint64_t meta[9] = {
      kNumPoints,
      kDim,
      0,
      kMaxNodeLen,
      kNodesPerSector,
      kNumPoints,
      attr_size,
      kRange,
      0,
  };
  std::memcpy(bytes.data() + 2 * sizeof(uint32_t), meta, sizeof(meta));

  for (uint32_t node_id = 0; node_id < kNumPoints; ++node_id) {
    const uint64_t offset = NodeOffset(node_id, kNodesPerSector, kMaxNodeLen);
    float coords[2] = {static_cast<float>(node_id), static_cast<float>(node_id) + 0.5f};
    std::memcpy(bytes.data() + offset, coords, sizeof(coords));
    uint16_t nnbrs = 2;
    uint16_t dense = node_id == 0 ? 1 : 0;
    std::memcpy(bytes.data() + offset + sizeof(coords), &nnbrs, sizeof(nnbrs));
    std::memcpy(bytes.data() + offset + sizeof(coords) + sizeof(nnbrs), &dense, sizeof(dense));
    uint32_t graph[2] = {
        static_cast<uint32_t>((node_id + 1) % kNumPoints),
        static_cast<uint32_t>((node_id + 2) % kNumPoints),
    };
    std::memcpy(bytes.data() + offset + sizeof(coords) + sizeof(uint32_t), graph, sizeof(graph));
    std::vector<char> attr_buf(static_cast<size_t>(attr_size), 0);
    attrs[node_id].serialize(attr_buf.data());
    std::memcpy(bytes.data() + offset + sizeof(coords) + sizeof(uint32_t) + kRange * sizeof(uint32_t),
                attr_buf.data(),
                attr_buf.size());
    if (dense != 0) {
      const uint32_t dense_neighbor = 2;
      std::memcpy(bytes.data() + offset + sizeof(coords) + sizeof(uint32_t) + kRange * sizeof(uint32_t) + attr_size,
                  &dense_neighbor,
                  sizeof(dense_neighbor));
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteApproxVectors(const std::filesystem::path &path) {
  const uint32_t num_points = 4;
  const uint32_t dim = 2;
  const float vectors[4][2] = {
      {0.0f, 0.5f},
      {1.0f, 1.5f},
      {2.0f, 2.5f},
      {3.0f, 3.5f},
  };
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  out.write(reinterpret_cast<const char *>(&num_points), sizeof(num_points));
  out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char *>(vectors), sizeof(vectors));
}

}  // namespace

int main() {
  const std::filesystem::path out_dir = std::filesystem::path("test_integration_modules");
  std::filesystem::create_directories(out_dir);

  const std::filesystem::path read_path = out_dir / "aio_pages.bin";
  {
    std::vector<char> payload(static_cast<size_t>(2 * kSectorLen), 0);
    std::fill(payload.begin(), payload.begin() + static_cast<std::ptrdiff_t>(kSectorLen), 'A');
    std::fill(payload.begin() + static_cast<std::ptrdiff_t>(kSectorLen), payload.end(), 'B');
    std::ofstream out(read_path, std::ios::binary | std::ios::trunc);
    assert(out);
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }

  hybrid::pipeann_integration::LinuxAlignedFileReader reader;
  reader.Open(read_path.string());
  void *ctx = reader.GetContext();

  auto *buf0 = reader.Allocate(kSectorLen, kSectorLen);
  auto *buf1 = reader.Allocate(kSectorLen, kSectorLen);
  std::vector<hybrid::pipeann_integration::IORequest> batch = {
      {0, kSectorLen, buf0, 128, 256, buf0},
      {kSectorLen, kSectorLen, buf1, kSectorLen + 64, 512, buf1},
  };
  reader.Read(batch, ctx);
  assert(batch[0].finished);
  assert(batch[1].finished);
  assert(batch[0].u_offset == 128);
  assert(batch[0].u_len == 256);
  assert(batch[0].base == buf0);
  assert(static_cast<char *>(buf0)[0] == 'A');
  assert(static_cast<char *>(buf1)[0] == 'B');

  auto *buf2 = reader.Allocate(kSectorLen, kSectorLen);
  hybrid::pipeann_integration::IORequest async_req{kSectorLen, kSectorLen, buf2, kSectorLen + 32, 128, buf2};
  reader.SendRead(async_req, ctx);
  for (int i = 0; i < 100 && !async_req.finished; ++i) {
    reader.Poll(ctx);
    if (!async_req.finished) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  assert(async_req.finished);
  assert(static_cast<char *>(buf2)[0] == 'B');

  reader.Free(buf0);
  reader.Free(buf1);
  reader.Free(buf2);
  reader.Close();

  const std::filesystem::path disk_index_path = out_dir / "synthetic_disk.index";
  const std::filesystem::path relayout_path = out_dir / "synthetic_graphrep.index";
  const std::filesystem::path approx_path = out_dir / "synthetic_vectors.bin";
  WriteSyntheticDiskIndex(disk_index_path);
  WriteApproxVectors(approx_path);

  const hybrid::gorgeous_integration::DiskIndexMetadata metadata =
      hybrid::gorgeous_integration::ReadDiskIndexMetadata(disk_index_path.string());
  assert(metadata.is_new_format);
  assert(metadata.num_points == 4);
  assert(metadata.dim == 2);
  assert(metadata.attr_size > 0);
  assert(metadata.range_dense == 3);
  assert(metadata.nodes_per_sector == 2);

  const std::vector<std::vector<uint32_t>> layouts = {
      {0, 1},
      {1, 2},
      {2, 0},
      {3, 0},
  };
  hybrid::gorgeous_integration::RelayoutDiskIndexToGraphReplica(
      disk_index_path.string(), layouts, relayout_path.string(), sizeof(float));

  assert(std::filesystem::exists(relayout_path));
  assert(std::filesystem::file_size(relayout_path) == 5 * kSectorLen);
  const hybrid::gorgeous_integration::DiskIndexMetadata relayout_metadata =
      hybrid::gorgeous_integration::ReadDiskIndexMetadata(relayout_path.string());
  assert(relayout_metadata.is_new_format);
  assert(relayout_metadata.file_size == 5 * kSectorLen);
  assert(relayout_metadata.page_region_sectors == 5);
  assert(relayout_metadata.full_precision_payload_start_sector == 0);

  std::ifstream relayout_in(relayout_path, std::ios::binary);
  assert(relayout_in);
  relayout_in.seekg(static_cast<std::streamoff>(kSectorLen), std::ios::beg);
  std::vector<char> page0(static_cast<size_t>(kSectorLen), 0);
  relayout_in.read(page0.data(), static_cast<std::streamsize>(page0.size()));
  assert(relayout_in);

  float coords[2] = {0.0f, 0.0f};
  std::memcpy(coords, page0.data(), sizeof(coords));
  assert(coords[0] == 0.0f);
  assert(coords[1] == 0.5f);

  uint32_t layout_size = 0;
  std::memcpy(&layout_size, page0.data() + 2 * sizeof(float), sizeof(layout_size));
  assert(layout_size == 2);

  uint32_t ids[2] = {0, 0};
  std::memcpy(ids, page0.data() + 2 * sizeof(float) + sizeof(uint32_t), sizeof(ids));
  assert(ids[0] == 0);
  assert(ids[1] == 1);

  hybrid::NativeGorgeousIndex native_index;
  native_index.Load(relayout_path.string(), approx_path.string());
  std::ifstream native_in(relayout_path, std::ios::binary);
  assert(native_in);
  std::vector<char> native_page(static_cast<size_t>(kSectorLen), 0);
  native_in.seekg(static_cast<std::streamoff>(native_index.PageOffset(0)), std::ios::beg);
  native_in.read(native_page.data(), static_cast<std::streamsize>(native_page.size()));
  assert(native_in);
  const hybrid::PageView page = native_index.ViewPage(0, native_page);
  const hybrid::DiskNodeView node0 = native_index.ViewNode(page, 0);
  assert(node0.nnbrs == 2);
  assert(node0.n_dense_nbrs == 1);
  assert(node0.nbrs[0] == 1);
  assert(node0.nbrs[1] == 2);
  assert(node0.dense_nbrs[0] == 2);
  assert(node0.has_attrs());
  const pipeann::Attributes target_attrs = pipeann::Attributes::deserialize(static_cast<const char *>(node0.attrs));
  assert(target_attrs.get(11)[0] == 7);
  const hybrid::DiskNodeView node1 = native_index.ViewNode(page, 1);
  assert(node1.nnbrs == 2);
  assert(node1.n_dense_nbrs == 0);
  assert(node1.has_attrs());
  const pipeann::Attributes neighbor_attrs = pipeann::Attributes::deserialize(static_cast<const char *>(node1.attrs));
  assert(neighbor_attrs.get(11)[0] == 1);

  const std::filesystem::path relayout_with_payload_path = out_dir / "synthetic_graphrep_with_payload.index";
  const std::vector<std::vector<float>> reorder_vectors = {
      {0.0f, 0.5f},
      {1.0f, 1.5f},
      {2.0f, 2.5f},
      {3.0f, 3.5f},
  };
  hybrid::gorgeous_integration::RelayoutDiskIndexToGraphReplica(
      disk_index_path.string(),
      layouts,
      relayout_with_payload_path.string(),
      sizeof(float),
      &reorder_vectors);
  const hybrid::gorgeous_integration::DiskIndexMetadata payload_metadata =
      hybrid::gorgeous_integration::ReadDiskIndexMetadata(relayout_with_payload_path.string());
  assert(payload_metadata.page_region_sectors == 5);
  assert(payload_metadata.full_precision_payload_start_sector == 5);
  assert(payload_metadata.full_precision_dim == 2);
  assert(payload_metadata.full_precision_nodes_per_sector > 0);
  hybrid::NativeGorgeousIndex payload_index;
  payload_index.Load(relayout_with_payload_path.string(), "");
  std::ifstream payload_in(relayout_with_payload_path, std::ios::binary);
  assert(payload_in);
  std::vector<char> payload_page(static_cast<size_t>(kSectorLen), 0);
  payload_in.seekg(static_cast<std::streamoff>(payload_index.PageOffset(0)), std::ios::beg);
  payload_in.read(payload_page.data(), static_cast<std::streamsize>(payload_page.size()));
  assert(payload_in);
  const hybrid::PageView payload_page_view = payload_index.ViewPage(0, payload_page);
  const hybrid::DiskNodeView payload_node1 = payload_index.ViewNode(payload_page_view, 1);
  assert(!payload_node1.has_embedded_coords());
  const std::vector<float> payload_exact = payload_index.exact_vector(1);
  assert(payload_exact[0] == 1.0f);
  assert(payload_exact[1] == 1.5f);

  return 0;
}
