#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "gorgeous_layout.h"
#include "integrations/disk_index_relayout.h"
#include "integrations/pipeann_builder.h"
#include "pipeline_search.h"
#include "quant/approx_distance.h"
#include "tools/build_pipeline.h"

int main() {
  const std::filesystem::path out_dir = std::filesystem::path("test_build_data");
  std::filesystem::create_directories(out_dir);

  const std::filesystem::path vectors_path = out_dir / "vectors.txt";
  {
    std::ofstream out(vectors_path);
    assert(out);
    out << "1.0 0.0 0.0\n";
    out << "0.8 0.2 0.1\n";
    out << "0.0 1.0 0.2\n";
    out << "-0.9 0.0 0.3\n";
    out << "0.0 -1.0 0.4\n";
    out << "0.7 -0.1 0.0\n";
    out << "0.6 0.6 0.1\n";
    out << "-0.6 0.7 0.2\n";
    out << "-0.8 -0.2 0.3\n";
    out << "-0.2 -0.8 0.4\n";
    out << "0.5 -0.7 0.1\n";
    out << "0.9 -0.3 0.0\n";
  }

  const std::filesystem::path train_queries_path = out_dir / "train_queries.txt";
  {
    std::ofstream out(train_queries_path);
    assert(out);
    out << "1.0 0.05 0.0\n";
    out << "0.75 0.15 0.05\n";
    out << "0.05 0.95 0.2\n";
  }

  hybrid::BuildConfig config;
  config.input_mode = hybrid::VectorInputMode::kText;
  config.input_path = vectors_path.string();
  config.train_query_mode = hybrid::VectorInputMode::kText;
  config.train_query_path = train_queries_path.string();
  config.output_dir = out_dir.string();
  config.dataset_name = "text_sample";
  config.degree = 3;
  config.r_ood = 1;
  config.build_threads = 2;
  config.page_nodes = 3;
  config.entry_id = 0;
  config.pq_subspaces = 3;
  config.pq_centroids = 2;
  config.pq_iterations = 4;

  const hybrid::BuildArtifacts artifacts = hybrid::RunBuildPipeline(config);
  assert(artifacts.output_mode == hybrid::BuildOutputMode::kGorgeousNative);
  assert(artifacts.num_points == 12);
  assert(artifacts.dim == 3);
  assert(std::filesystem::exists(artifacts.raw_disk_index_path));
  assert(std::filesystem::exists(artifacts.relayout_index_path));
  assert(std::filesystem::exists(artifacts.index_path));
  assert(std::filesystem::exists(artifacts.pipeann_base_data_path));
  assert(std::filesystem::exists(artifacts.pipeann_train_query_path));
  assert(std::filesystem::exists(artifacts.partition_path));
  assert(std::filesystem::exists(artifacts.reorder_path));
  assert(artifacts.workflow_prefix == (out_dir / "text_sample").string());
  assert(artifacts.pipeann_index_prefix == artifacts.workflow_prefix);
  assert(artifacts.gorgeous_partition_bin_path == (out_dir / "text_sample_partition.bin").string());
  assert(artifacts.gorgeous_relayout_index_path == (out_dir / "text_sample_graph_relayout.index").string());
  assert(artifacts.gorgeous_relayout_index_path == artifacts.relayout_index_path);
  assert(!artifacts.has_project_compatible_export);
  assert(artifacts.pipeann_refine_sidecar_path.empty());
  assert(artifacts.pipeann_refine_manifest_path.empty());
  assert(artifacts.pipeann_refine_nodes_path.empty());
  assert(std::filesystem::exists(artifacts.approx_path));
  assert(std::filesystem::exists(artifacts.pq_codebook_path));
  assert(std::filesystem::exists(artifacts.pq_codes_path));

  const hybrid::gorgeous_integration::DiskIndexMetadata raw_metadata =
      hybrid::gorgeous_integration::ReadDiskIndexMetadata(artifacts.raw_disk_index_path);
  assert(raw_metadata.is_new_format);
  assert(raw_metadata.num_points == 12);
  assert(raw_metadata.dim == 3);
  assert(raw_metadata.nodes_per_sector > 0);
  assert(raw_metadata.range > 0);
  assert(raw_metadata.range <= config.degree);
  assert(raw_metadata.r_ood <= config.r_ood);
  assert(raw_metadata.range_dense >= raw_metadata.range);
  assert(raw_metadata.normal_node_len ==
         raw_metadata.max_node_len - (raw_metadata.range_dense - raw_metadata.range) * sizeof(uint32_t));

  {
    const uint64_t node_offset =
        4096 + (0 / raw_metadata.nodes_per_sector) * 4096 + (0 % raw_metadata.nodes_per_sector) * raw_metadata.max_node_len;
    std::ifstream raw_in(artifacts.raw_disk_index_path, std::ios::binary);
    assert(raw_in);
    raw_in.seekg(static_cast<std::streamoff>(node_offset + raw_metadata.dim * sizeof(float)), std::ios::beg);
    uint16_t nnbrs = 0;
    uint16_t n_dense_nbrs = 0;
    raw_in.read(reinterpret_cast<char *>(&nnbrs), sizeof(nnbrs));
    raw_in.read(reinterpret_cast<char *>(&n_dense_nbrs), sizeof(n_dense_nbrs));
    assert(raw_in);
    assert(nnbrs > 0);
    assert(nnbrs <= raw_metadata.range);
    assert(n_dense_nbrs <= raw_metadata.range_dense - raw_metadata.range);

    std::vector<uint32_t> nbrs(nnbrs, 0);
    std::vector<uint32_t> dense_nbrs(n_dense_nbrs, 0);
    raw_in.read(reinterpret_cast<char *>(nbrs.data()), static_cast<std::streamsize>(nbrs.size() * sizeof(uint32_t)));
    raw_in.seekg(static_cast<std::streamoff>(node_offset + raw_metadata.dim * sizeof(float) +
                                             static_cast<uint64_t>(1 + raw_metadata.range) * sizeof(uint32_t)),
                 std::ios::beg);
    raw_in.read(reinterpret_cast<char *>(dense_nbrs.data()),
                static_cast<std::streamsize>(dense_nbrs.size() * sizeof(uint32_t)));
    assert(raw_in || dense_nbrs.empty());

    std::unordered_set<uint32_t> seen_one_hop;
    const size_t base_degree = static_cast<size_t>(nnbrs) - std::min<size_t>(nnbrs, config.r_ood);
    for (uint32_t id : nbrs) {
      assert(id < raw_metadata.num_points);
      assert(id != 0);
      assert(seen_one_hop.insert(id).second);
    }
    std::unordered_set<uint32_t> seen_base;
    for (size_t i = 0; i < base_degree; ++i) {
      assert(seen_base.insert(nbrs[i]).second);
    }
    bool has_ood_tail = false;
    for (size_t i = base_degree; i < nbrs.size(); ++i) {
      assert(seen_base.find(nbrs[i]) == seen_base.end());
      has_ood_tail = true;
    }
    for (uint32_t id : dense_nbrs) {
      assert(id < raw_metadata.num_points);
      assert(id != 0);
      assert(seen_one_hop.find(id) == seen_one_hop.end());
    }
    assert(has_ood_tail || dense_nbrs.size() <= static_cast<size_t>(raw_metadata.range_dense - raw_metadata.range));
  }

  hybrid::NativeGorgeousIndex index;
  index.Load(artifacts.index_path, artifacts.approx_path);
  assert(index.search_metadata().num_points == 12);
  assert(index.search_metadata().num_pages == 12);
  assert(index.search_metadata().dim == 3);
  assert(index.native_metadata().num_points == 12);
  assert(index.native_metadata().dim == 3);
  assert(index.native_metadata().nodes_per_sector >= 1);
  assert(index.native_metadata().max_node_len > 0);
  assert(index.HasPartitionData());
  assert(index.id_to_page().size() == 12);
  assert(index.page_layouts().size() == 12);
  assert(index.PageForNode(0) == 0);
  assert(index.page_boundaries().size() == 13);
  assert(index.page_boundaries().front() == index.search_metadata().page_size);
  assert(index.page_boundaries().back() == index.IndexFileSizeBytes());

  assert(artifacts.index_path == artifacts.relayout_index_path);
  assert(artifacts.approx_path == artifacts.pipeann_base_data_path);
  assert(artifacts.partition_path == hybrid::DefaultGorgeousPartitionSidecarPath(artifacts.index_path));
  assert(artifacts.reorder_path == hybrid::DefaultGorgeousReorderSidecarPath(artifacts.index_path));
  const hybrid::gorgeous_integration::DiskIndexMetadata native_metadata =
      hybrid::gorgeous_integration::ReadDiskIndexMetadata(artifacts.index_path);
  assert(native_metadata.is_new_format);
  assert(native_metadata.num_points == artifacts.num_points);
  assert(native_metadata.dim == artifacts.dim);
  assert(native_metadata.file_size == std::filesystem::file_size(artifacts.index_path));
  assert(native_metadata.page_region_sectors == static_cast<uint64_t>(index.search_metadata().num_pages) + 1);
  assert(native_metadata.full_precision_payload_start_sector == 0);

  const std::filesystem::path removed_partition_path = artifacts.partition_path + ".bak";
  const std::filesystem::path removed_reorder_path = artifacts.reorder_path + ".bak";
  std::filesystem::rename(artifacts.partition_path, removed_partition_path);
  std::filesystem::rename(artifacts.reorder_path, removed_reorder_path);
  hybrid::NativeGorgeousIndex sidecarless_index;
  sidecarless_index.Load(artifacts.index_path, artifacts.approx_path);
  assert(sidecarless_index.search_metadata().num_pages == index.search_metadata().num_pages);
  assert(sidecarless_index.page_layouts() == index.page_layouts());
  assert(sidecarless_index.id_to_page() == index.id_to_page());
  std::filesystem::rename(removed_partition_path, artifacts.partition_path);
  std::filesystem::rename(removed_reorder_path, artifacts.reorder_path);

  hybrid::BuildConfig compat_config = config;
  compat_config.dataset_name = "text_sample_compat";
  compat_config.output_mode = hybrid::BuildOutputMode::kProjectCompatible;
  const hybrid::BuildArtifacts compat_artifacts = hybrid::RunBuildPipeline(compat_config);
  assert(compat_artifacts.output_mode == hybrid::BuildOutputMode::kProjectCompatible);
  assert(std::filesystem::exists(compat_artifacts.index_path));
  assert(std::filesystem::exists(compat_artifacts.relayout_index_path));
  assert(compat_artifacts.has_project_compatible_export);
  assert(compat_artifacts.index_path != compat_artifacts.relayout_index_path);
  assert(compat_artifacts.partition_path == hybrid::DefaultPartitionSidecarPath(compat_artifacts.index_path));
  assert(compat_artifacts.reorder_path == hybrid::DefaultReorderSidecarPath(compat_artifacts.index_path));

  {
    const std::vector<std::vector<uint32_t>> partition_graph = {
        {1, 2}, {0, 2, 3}, {0, 1, 4}, {1, 4, 5}, {2, 3, 5}, {3, 4}};
    const std::vector<std::vector<uint32_t>> partition_layouts =
        hybrid::BuildReplicatedLayouts(partition_graph, 3, 1, 2);
    assert(partition_layouts.size() == 2);
    size_t total_entries = 0;
    std::unordered_set<uint32_t> unique_nodes;
    for (const auto &layout : partition_layouts) {
      assert(!layout.empty());
      assert(layout.size() <= 3);
      total_entries += layout.size();
      for (uint32_t node_id : layout) {
        assert(unique_nodes.insert(node_id).second);
      }
    }
    assert(total_entries == partition_graph.size());
    assert(unique_nodes.size() == partition_graph.size());
  }

  {
    std::ifstream graphrep_in(compat_artifacts.index_path, std::ios::binary);
    assert(graphrep_in);
    hybrid::GraphReplicatedIndex compat_index;
    compat_index.Load(compat_artifacts.index_path, compat_artifacts.approx_path);
    graphrep_in.seekg(static_cast<std::streamoff>(compat_index.PageOffset(0) + sizeof(uint32_t) +
                                                  compat_index.metadata().max_page_nodes * sizeof(uint32_t)),
                      std::ios::beg);
    std::vector<float> first_coords(raw_metadata.dim, 0.0f);
    graphrep_in.read(reinterpret_cast<char *>(first_coords.data()),
                     static_cast<std::streamsize>(first_coords.size() * sizeof(float)));
    uint16_t nnbrs = 0;
    uint16_t n_dense_nbrs = 0;
    graphrep_in.read(reinterpret_cast<char *>(&nnbrs), sizeof(nnbrs));
    graphrep_in.read(reinterpret_cast<char *>(&n_dense_nbrs), sizeof(n_dense_nbrs));
    assert(graphrep_in);
    assert(first_coords == std::vector<float>({1.0f, 0.0f, 0.0f}));
    assert(nnbrs <= compat_index.metadata().max_base_degree);
    assert(n_dense_nbrs <= compat_index.metadata().max_degree - compat_index.metadata().max_base_degree);
  }

  hybrid::PipelinedGraphReplicatedSearcher searcher(index);
  hybrid::SearchConfig search_config;
  search_config.top_k = 3;
  search_config.beam_width = 2;
  search_config.l_search = 5;

  auto pq_distance = std::make_unique<hybrid::ProductQuantizationDistanceComputer>(index);
  pq_distance->Load(artifacts.pq_codebook_path, artifacts.pq_codes_path);
  hybrid::SearchStats stats;
  const std::vector<float> query = {1.0f, 0.1f, 0.0f};
  const std::vector<hybrid::SearchResult> results =
      searcher.Search(query, search_config, std::move(pq_distance), &stats);

  assert(!results.empty());
  assert(results.front().id < artifacts.num_points);
  assert(stats.approx_distance_evals >= stats.exact_distance_evals);

  return 0;
}
