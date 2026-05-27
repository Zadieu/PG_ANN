#include "integrations/pipeann_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "nbr/pq_nbr.h"
#include "nbr/dummy_nbr.h"
#include "ssd_index_defs.h"
#include "utils.h"
#include "utils/index_build_utils.h"

namespace hybrid::pipeann_integration {

namespace {

constexpr uint64_t kPipeannRefineSidecarMagic = 0x50495045464e4758ULL;

void WriteBytes(std::ofstream &out, const void *data, size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("failed to write PipeANN sidecar");
  }
}

void RewriteEntryPoint(const std::string &disk_index_path, uint32_t entry_id) {
  std::fstream io(disk_index_path, std::ios::in | std::ios::out | std::ios::binary);
  if (!io) {
    throw std::runtime_error("failed to reopen PipeANN disk index for entry-point rewrite");
  }
  constexpr std::streamoff kEntryOffset =
      static_cast<std::streamoff>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2);
  const uint64_t value = entry_id;
  io.seekp(kEntryOffset, std::ios::beg);
  io.write(reinterpret_cast<const char *>(&value), sizeof(value));
  if (!io) {
    throw std::runtime_error("failed to rewrite PipeANN entry point");
  }
}

pipeann::SSDIndexMetadata<float> LoadPipeannMetadata(const std::string &disk_index_path) {
  pipeann::SSDIndexMetadata<float> meta;
  meta.load_from_disk_index(disk_index_path);
  return meta;
}

}  // namespace

PipeannDiskBuildResult BuildPipeannDiskIndex(const std::string &base_bin_path,
                                             const std::string &output_prefix,
                                             const PipeannBuildParameters &params,
                                             const std::string &train_query_bin_path,
                                             std::optional<uint32_t> explicit_entry_id) {
  if (!file_exists(base_bin_path)) {
    throw std::runtime_error("PipeANN base bin file does not exist");
  }
  if (params.range == 0 || params.build_l == 0 || params.build_ram_budget_gb == 0) {
    throw std::runtime_error("PipeANN build parameters must be positive");
  }
  if (params.range <= params.r_ood) {
    throw std::runtime_error("PipeANN range must be greater than r_ood");
  }

  size_t num_points = 0;
  size_t dim = 0;
  pipeann::get_bin_metadata(base_bin_path, num_points, dim);
  if (num_points == 0 || dim == 0) {
    throw std::runtime_error("PipeANN base bin file must not be empty");
  }
  if (explicit_entry_id.has_value() && *explicit_entry_id >= num_points) {
    throw std::runtime_error("explicit entry_id is out of range for PipeANN build input");
  }

  pipeann::DummyNeighbor<float> nbr_handler(pipeann::Metric::L2);
  nbr_handler.npoints = num_points;

  const auto build_started = std::chrono::steady_clock::now();
  const bool ok = pipeann::build_disk_index<float, uint32_t>(base_bin_path.c_str(),
                                                             output_prefix.c_str(),
                                                             static_cast<uint16_t>(params.range),
                                                             params.build_l,
                                                             params.build_ram_budget_gb,
                                                             params.build_threads,
                                                             0,
                                                             pipeann::Metric::L2,
                                                             nullptr,
                                                             &nbr_handler,
                                                             nullptr,
                                                             static_cast<uint16_t>(std::max(params.range_dense,
                                                                                            params.range)),
                                                             0,
                                                             train_query_bin_path,
                                                             static_cast<uint16_t>(params.r_ood),
                                                             params.l_ood);
  if (!ok) {
    throw std::runtime_error("PipeANN build_disk_index failed");
  }

  PipeannDiskBuildResult result;
  result.disk_index_path = output_prefix + "_disk.index";
  result.tags_path = result.disk_index_path + ".tags";
  if (explicit_entry_id.has_value()) {
    RewriteEntryPoint(result.disk_index_path, *explicit_entry_id);
  }

  const pipeann::SSDIndexMetadata<float> meta = LoadPipeannMetadata(result.disk_index_path);
  result.num_points = static_cast<uint32_t>(meta.npoints);
  result.dim = static_cast<uint32_t>(meta.data_dim);
  result.entry_id = explicit_entry_id.has_value() ? *explicit_entry_id
                                                  : static_cast<uint32_t>(meta.entry_point);
  result.range = static_cast<uint32_t>(meta.range);
  result.range_dense = static_cast<uint32_t>(meta.range_dense);
  result.r_ood = static_cast<uint32_t>(meta.R_ood);
  result.stats.cpu_us =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - build_started)
                                .count());
  return result;
}

PipeannPQBuildResult BuildPipeannPQArtifacts(const std::string &base_bin_path,
                                             const std::string &output_prefix,
                                             uint32_t bytes_per_neighbor) {
  if (!file_exists(base_bin_path)) {
    throw std::runtime_error("PipeANN base bin file does not exist for PQ build");
  }
  if (bytes_per_neighbor == 0) {
    throw std::runtime_error("PipeANN PQ bytes_per_neighbor must be positive");
  }

  pipeann::PQNeighbor<float> pq_neighbor(pipeann::Metric::L2);
  pq_neighbor.build(output_prefix, base_bin_path, bytes_per_neighbor);

  PipeannPQBuildResult result;
  result.pivots_path = output_prefix + "_pq_pivots.bin";
  result.compressed_path = output_prefix + "_pq_compressed.bin";
  result.bytes_per_neighbor = bytes_per_neighbor;
  return result;
}

std::vector<std::vector<uint32_t>> LoadPipeannFlatGraph(const std::string &disk_index_path) {
  const pipeann::SSDIndexMetadata<float> meta = LoadPipeannMetadata(disk_index_path);
  std::ifstream in(disk_index_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open PipeANN disk index while loading flat graph");
  }

  const uint64_t unit_bytes = meta.io_size_dense();
  const uint64_t unit_nodes = meta.nodes_per_io();
  std::vector<char> buffer(static_cast<size_t>(unit_bytes), 0);
  std::vector<std::vector<uint32_t>> graph(meta.npoints);

  for (uint64_t loc = 0; loc < meta.npoints;) {
    const uint64_t cur_nodes = std::min<uint64_t>(unit_nodes, meta.npoints - loc);
    in.seekg(static_cast<std::streamoff>(meta.loc_sector_no(loc) * SECTOR_LEN), std::ios::beg);
    in.read(buffer.data(), static_cast<std::streamsize>(unit_bytes));
    if (!in) {
      throw std::runtime_error("failed to read PipeANN disk node sector");
    }

    for (uint64_t i = 0; i < cur_nodes; ++i) {
      const uint32_t node_id = static_cast<uint32_t>(loc + i);
      pipeann::DiskNode<float> node(buffer.data(), node_id, meta);
      auto &neighbors = graph[node_id];
      neighbors.assign(node.nbrs, node.nbrs + node.nnbrs);
      std::unordered_set<uint32_t> seen(neighbors.begin(), neighbors.end());
      for (uint16_t j = 0; j < node.n_dense_nbrs; ++j) {
        const uint32_t dense = node.dense_nbrs[j];
        if (seen.insert(dense).second) {
          neighbors.push_back(dense);
        }
      }
    }
    loc += cur_nodes;
  }

  return graph;
}

std::string DefaultPipeannRefineSidecarPath(const std::string &index_path) {
  return index_path + ".pipeann.refine";
}

void WritePipeannRefineSidecar(const std::string &path, const PipeannRefineSidecar &sidecar) {
  PipeannRefineSidecarHeader header = sidecar.header;
  header.magic = kPipeannRefineSidecarMagic;
  header.version = 1;
  if (header.num_points == 0) {
    header.num_points = static_cast<uint32_t>(sidecar.nodes.size());
  }
  if (header.num_points != sidecar.nodes.size()) {
    throw std::runtime_error("PipeANN refine sidecar header count does not match node payload");
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open PipeANN refine sidecar output file");
  }
  WriteBytes(out, &header, sizeof(header));

  for (const auto &node : sidecar.nodes) {
    const uint32_t count = static_cast<uint32_t>(node.refine_neighbors.size());
    if (count != node.ehs.size()) {
      throw std::runtime_error("PipeANN refine sidecar neighbor and EH counts do not match");
    }
    if (count > header.r_ood) {
      throw std::runtime_error("PipeANN refine sidecar node count exceeds configured R_ood");
    }
    WriteBytes(out, &count, sizeof(count));
    if (count > 0) {
      WriteBytes(out,
                 node.refine_neighbors.data(),
                 static_cast<size_t>(count) * sizeof(uint32_t));
      WriteBytes(out, node.ehs.data(), static_cast<size_t>(count) * sizeof(uint16_t));
    }
  }
}

PipeannRefineSidecar ReadPipeannRefineSidecar(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open PipeANN refine sidecar");
  }

  PipeannRefineSidecar sidecar;
  in.read(reinterpret_cast<char *>(&sidecar.header), sizeof(sidecar.header));
  if (!in) {
    throw std::runtime_error("failed to read PipeANN refine sidecar header");
  }
  if (sidecar.header.magic != kPipeannRefineSidecarMagic) {
    throw std::runtime_error("unexpected PipeANN refine sidecar magic");
  }
  if (sidecar.header.version != 1) {
    throw std::runtime_error("unsupported PipeANN refine sidecar version");
  }

  sidecar.nodes.resize(sidecar.header.num_points);
  for (uint32_t node = 0; node < sidecar.header.num_points; ++node) {
    uint32_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in) {
      throw std::runtime_error("failed to read PipeANN refine sidecar node count");
    }
    if (count > sidecar.header.r_ood) {
      throw std::runtime_error("PipeANN refine sidecar node count exceeds R_ood");
    }
    sidecar.nodes[node].refine_neighbors.resize(count);
    sidecar.nodes[node].ehs.resize(count);
    if (count > 0) {
      in.read(reinterpret_cast<char *>(sidecar.nodes[node].refine_neighbors.data()),
              static_cast<std::streamsize>(count * sizeof(uint32_t)));
      in.read(reinterpret_cast<char *>(sidecar.nodes[node].ehs.data()),
              static_cast<std::streamsize>(count * sizeof(uint16_t)));
      if (!in) {
        throw std::runtime_error("failed to read PipeANN refine sidecar node payload");
      }
    }
  }
  return sidecar;
}

}  // namespace hybrid::pipeann_integration
