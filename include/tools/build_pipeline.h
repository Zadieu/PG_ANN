#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "integrations/pipeann_builder.h"

namespace hybrid {

enum class VectorInputMode {
  kToy = 0,
  kText = 1,
  kFvecs = 2,
  kBvecs = 3,
  kBin = 4,
};

enum class BuildOutputMode {
  kProjectCompatible = 0,
  kGorgeousNative = 1,
};

struct BuildConfig {
  VectorInputMode input_mode = VectorInputMode::kToy;
  std::string input_path;
  VectorInputMode train_query_mode = VectorInputMode::kBin;
  std::string train_query_path;
  std::string output_dir = "build_data";
  std::string dataset_name = "toy";
  BuildOutputMode output_mode = BuildOutputMode::kGorgeousNative;
  uint32_t degree = 4;
  uint32_t dense_degree = 0;
  uint32_t r_ood = 0;
  uint32_t build_l = 64;
  uint32_t build_ram_budget_gb = 32;
  uint32_t build_threads = 1;
  uint32_t build_candidates = 128;
  float build_alpha = 1.2f;
  uint32_t l_ood = 1500;
  uint32_t page_nodes = 4;
  uint32_t partition_scale = 0;
  uint32_t partition_ldg_times = 4;
  bool use_explicit_entry_id = false;
  uint32_t entry_id = 0;
  uint32_t pq_subspaces = 3;
  uint32_t pq_centroids = 4;
  uint32_t pq_iterations = 6;
  uint32_t toy_points = 12;
};

struct BuildArtifacts {
  BuildOutputMode output_mode = BuildOutputMode::kGorgeousNative;
  std::string workflow_prefix;
  std::string pipeann_base_data_path;
  std::string pipeann_train_query_path;
  std::string pipeann_index_prefix;
  std::string gorgeous_partition_bin_path;
  std::string gorgeous_relayout_index_path;
  std::string raw_disk_index_path;
  std::string relayout_index_path;
  std::string index_path;
  std::string partition_path;
  std::string reorder_path;
  std::string pipeann_refine_sidecar_path;
  std::string pipeann_refine_manifest_path;
  std::string pipeann_refine_nodes_path;
  std::string approx_path;
  std::string pq_codebook_path;
  std::string pq_codes_path;
  bool has_project_compatible_export = false;
  uint32_t num_points = 0;
  uint32_t dim = 0;
  pipeann_integration::PipeannBuildStats pipeann_build_stats{};
};

struct ReplicatedLayoutResult {
  std::vector<std::vector<uint32_t>> layouts;
  std::vector<uint32_t> id_to_page;
  uint32_t page_capacity = 0;
  uint32_t num_points = 0;
  uint32_t num_pages = 0;
  uint32_t n_primary_partition = 0;
  uint32_t effective_scale_factor = 0;
};

std::vector<std::vector<float>> LoadTextVectors(const std::string &path);
std::vector<std::vector<float>> LoadFvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBinVectors(const std::string &path);
std::vector<std::vector<float>> GenerateToyVectors(uint32_t num_points);
std::vector<std::vector<uint32_t>> BuildKnnGraph(const std::vector<std::vector<float>> &vectors,
                                                 uint32_t degree);
ReplicatedLayoutResult BuildReplicatedLayoutResult(
    const std::vector<std::vector<uint32_t>> &graph,
    uint32_t page_nodes,
    uint32_t partition_scale = 0,
    uint32_t partition_ldg_times = 4);
std::vector<std::vector<uint32_t>> BuildReplicatedLayouts(
    const std::vector<std::vector<uint32_t>> &graph,
    uint32_t page_nodes,
    uint32_t partition_scale = 0,
    uint32_t partition_ldg_times = 4);
BuildArtifacts RunBuildPipeline(const BuildConfig &config);

}  // namespace hybrid
