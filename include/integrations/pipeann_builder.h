#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hybrid::pipeann_integration {

struct PipeannBuildParameters {
  uint32_t range = 4;
  uint32_t range_dense = 8;
  uint32_t r_ood = 0;
  uint32_t build_l = 64;
  uint32_t build_ram_budget_gb = 32;
  uint32_t build_threads = 1;
  uint32_t l_ood = 1500;
};

struct PipeannBuildStats {
  uint64_t n_cmps = 0;
  uint64_t n_prunes = 0;
  uint64_t n_refine_queries = 0;
  uint64_t n_refine_edges = 0;
  uint64_t n_dense_edges = 0;
  uint64_t cpu_us = 0;
};

struct PipeannDiskBuildResult {
  std::string disk_index_path;
  std::string tags_path;
  uint32_t num_points = 0;
  uint32_t dim = 0;
  uint32_t entry_id = 0;
  uint32_t range = 0;
  uint32_t range_dense = 0;
  uint32_t r_ood = 0;
  PipeannBuildStats stats{};
};

struct PipeannPQBuildResult {
  std::string pivots_path;
  std::string compressed_path;
  uint32_t bytes_per_neighbor = 0;
};

struct PipeannRefineSidecarHeader {
  uint64_t magic = 0;
  uint32_t version = 1;
  uint32_t num_points = 0;
  uint32_t entry_id = 0;
  uint32_t range = 0;
  uint32_t range_base = 0;
  uint32_t range_dense = 0;
  uint32_t r_ood = 0;
  uint32_t l_ood = 0;
  uint32_t train_query_count = 0;
  PipeannBuildStats stats{};
};

struct PipeannRefineSidecarNode {
  std::vector<uint32_t> refine_neighbors;
  std::vector<uint16_t> ehs;
};

struct PipeannRefineSidecar {
  PipeannRefineSidecarHeader header{};
  std::vector<PipeannRefineSidecarNode> nodes;
};

PipeannDiskBuildResult BuildPipeannDiskIndex(const std::string &base_bin_path,
                                             const std::string &output_prefix,
                                             const PipeannBuildParameters &params,
                                             const std::string &train_query_bin_path = "",
                                             std::optional<uint32_t> explicit_entry_id = std::nullopt);

PipeannPQBuildResult BuildPipeannPQArtifacts(const std::string &base_bin_path,
                                             const std::string &output_prefix,
                                             uint32_t bytes_per_neighbor);

std::vector<std::vector<uint32_t>> LoadPipeannFlatGraph(const std::string &disk_index_path);

std::string DefaultPipeannRefineSidecarPath(const std::string &index_path);
void WritePipeannRefineSidecar(const std::string &path,
                               const PipeannRefineSidecar &sidecar);
PipeannRefineSidecar ReadPipeannRefineSidecar(const std::string &path);

}  // namespace hybrid::pipeann_integration
