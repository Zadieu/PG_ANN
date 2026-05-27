#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "filter/attribute.h"
#include "filter/selector.h"
#include "gorgeous_layout.h"
#include "integrations/disk_index_relayout.h"
#include "integrations/pipeann_builder.h"
#include "pipeline_search.h"

namespace {

std::vector<std::vector<float>> BuildVectors() {
  std::vector<std::vector<float>> vectors;
  constexpr size_t kPoints = 12;
  constexpr float kPi = 3.14159265358979323846f;
  for (size_t i = 0; i < kPoints; ++i) {
    const float angle = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kPoints);
    vectors.push_back({std::cos(angle), std::sin(angle), static_cast<float>(i) / 10.0f});
  }
  return vectors;
}

std::vector<std::vector<uint32_t>> BuildKnnGraph(const std::vector<std::vector<float>> &vectors, uint32_t degree) {
  std::vector<std::vector<uint32_t>> graph(vectors.size());
  for (uint32_t id = 0; id < vectors.size(); ++id) {
    std::vector<std::pair<float, uint32_t>> neighbors;
    for (uint32_t other = 0; other < vectors.size(); ++other) {
      if (id == other) {
        continue;
      }
      neighbors.push_back({hybrid::L2Distance(vectors[id], vectors[other]), other});
    }
    std::sort(neighbors.begin(), neighbors.end());
    for (uint32_t i = 0; i < degree && i < neighbors.size(); ++i) {
      graph[id].push_back(neighbors[i].second);
    }
  }
  return graph;
}

std::vector<std::vector<uint32_t>> BuildReplicatedLayouts(const std::vector<std::vector<uint32_t>> &graph,
                                                          uint32_t page_nodes) {
  std::vector<std::vector<uint32_t>> layouts(graph.size());
  for (uint32_t id = 0; id < graph.size(); ++id) {
    layouts[id].push_back(id);
    for (uint32_t neighbor : graph[id]) {
      if (layouts[id].size() >= page_nodes) {
        break;
      }
      layouts[id].push_back(neighbor);
    }
  }
  return layouts;
}

bool Throws(const std::function<void()> &fn) {
  try {
    fn();
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

void WriteUint32At(const std::string &path, uint64_t offset, uint32_t value) {
  std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
  assert(out);
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
  out.flush();
  assert(out.good());
}

uint64_t NodeOffset(uint32_t node_id, uint64_t nodes_per_sector, uint64_t max_node_len) {
  constexpr uint64_t kSectorLen = 4096;
  return (static_cast<uint64_t>(node_id) / nodes_per_sector + 1) * kSectorLen +
         (static_cast<uint64_t>(node_id) % nodes_per_sector) * max_node_len;
}

void WriteFilterDiskIndex(const std::filesystem::path &path) {
  constexpr uint64_t kSectorLen = 4096;
  constexpr uint64_t kNumPoints = 4;
  constexpr uint64_t kDim = 2;
  constexpr uint64_t kRange = 2;
  constexpr uint64_t kRangeDense = 3;
  constexpr uint32_t kAttrKey = 11;
  std::vector<pipeann::Attributes> attrs(kNumPoints);
  attrs[0].set(kAttrKey, {7});
  attrs[1].set(kAttrKey, {1});
  attrs[2].set(kAttrKey, {7});
  attrs[3].set(kAttrKey, {2});

  uint64_t attr_size = 0;
  for (const auto &row : attrs) {
    attr_size = std::max<uint64_t>(attr_size, row.serialized_size());
  }
  const uint64_t max_node_len = kDim * sizeof(float) + sizeof(uint32_t) + kRange * sizeof(uint32_t) + attr_size +
                                (kRangeDense - kRange) * sizeof(uint32_t);
  const uint64_t nodes_per_sector = 2;
  const uint64_t file_size = kSectorLen + 2 * kSectorLen;
  std::vector<char> bytes(static_cast<size_t>(file_size), 0);

  const uint32_t nr = 9;
  const uint32_t nc = 1;
  std::memcpy(bytes.data(), &nr, sizeof(nr));
  std::memcpy(bytes.data() + sizeof(uint32_t), &nc, sizeof(nc));
  uint64_t meta[9] = {
      kNumPoints,
      kDim,
      0,
      max_node_len,
      nodes_per_sector,
      kNumPoints,
      attr_size,
      kRange,
      0,
  };
  std::memcpy(bytes.data() + 2 * sizeof(uint32_t), meta, sizeof(meta));

  const float vectors[4][2] = {
      {0.0f, 0.0f},
      {1.0f, 0.0f},
      {2.0f, 0.0f},
      {3.0f, 0.0f},
  };
  for (uint32_t node_id = 0; node_id < kNumPoints; ++node_id) {
    const uint64_t offset = NodeOffset(node_id, nodes_per_sector, max_node_len);
    std::memcpy(bytes.data() + offset, vectors[node_id], sizeof(vectors[node_id]));
    const uint16_t nnbrs = 2;
    const uint16_t n_dense = node_id == 0 ? 1 : 0;
    std::memcpy(bytes.data() + offset + sizeof(vectors[node_id]), &nnbrs, sizeof(nnbrs));
    std::memcpy(bytes.data() + offset + sizeof(vectors[node_id]) + sizeof(nnbrs), &n_dense, sizeof(n_dense));
    uint32_t base_neighbors[2] = {
        static_cast<uint32_t>((node_id + 1) % kNumPoints),
        static_cast<uint32_t>((node_id + 2) % kNumPoints),
    };
    std::memcpy(bytes.data() + offset + sizeof(vectors[node_id]) + sizeof(uint32_t),
                base_neighbors,
                sizeof(base_neighbors));
    std::vector<char> attr_buf(static_cast<size_t>(attr_size), 0);
    attrs[node_id].serialize(attr_buf.data());
    std::memcpy(bytes.data() + offset + sizeof(vectors[node_id]) + sizeof(uint32_t) + kRange * sizeof(uint32_t),
                attr_buf.data(),
                attr_buf.size());
    if (n_dense != 0) {
      const uint32_t dense_neighbor = 2;
      std::memcpy(bytes.data() + offset + sizeof(vectors[node_id]) + sizeof(uint32_t) +
                      kRange * sizeof(uint32_t) + attr_size,
                  &dense_neighbor,
                  sizeof(dense_neighbor));
    }
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteFilterApprox(const std::filesystem::path &path) {
  const uint32_t num_points = 4;
  const uint32_t dim = 2;
  const float vectors[4][2] = {
      {0.0f, 0.0f},
      {1.0f, 0.0f},
      {2.0f, 0.0f},
      {3.0f, 0.0f},
  };
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  assert(out);
  out.write(reinterpret_cast<const char *>(&num_points), sizeof(num_points));
  out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char *>(vectors), sizeof(vectors));
}

}  // namespace

int main() {
  const std::vector<std::vector<float>> vectors = BuildVectors();
  const std::vector<std::vector<uint32_t>> graph = BuildKnnGraph(vectors, 4);
  const std::vector<std::vector<uint32_t>> layouts = BuildReplicatedLayouts(graph, 4);

  const std::filesystem::path out_dir = std::filesystem::path("test_data");
  std::filesystem::create_directories(out_dir);

  const std::string index_path = (out_dir / "toy.graphrep").string();
  const std::string approx_path = (out_dir / "toy.approx").string();
  hybrid::GraphReplicatedLayoutBuilder::Build(index_path, approx_path, vectors, graph, layouts, 0);
  const hybrid::pipeann_integration::PipeannPQBuildResult pq_artifacts =
      hybrid::pipeann_integration::BuildPipeannPQArtifacts(approx_path, (out_dir / "toy").string(), 3);

  hybrid::GraphReplicatedIndex index;
  index.Load(index_path, approx_path);

  hybrid::PipelinedGraphReplicatedSearcher searcher(index);
  hybrid::SearchConfig config;
  config.top_k = 4;
  config.beam_width = 3;
  config.l_search = 8;

  std::vector<float> query = {0.95f, 0.15f, 0.0f};
  hybrid::SearchStats stats;
  std::cout << "stage=default_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> results = searcher.Search(query, config, &stats);
  hybrid::SearchStats full_stats;
  std::cout << "stage=full_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> full_results =
      searcher.Search(query, config, hybrid::ApproxDistanceKind::kFullPrecision, &full_stats);

  hybrid::ProductQuantizationDistanceComputer pq_distance(index);
  pq_distance.Load(pq_artifacts.pivots_path, pq_artifacts.compressed_path);
  pq_distance.BeginQuery(query);
  assert(pq_distance.ready());
  assert(std::string(pq_distance.name()) == "pipeann_pq");
  assert(pq_distance.metadata().num_subspaces == 3);
  assert(std::isfinite(pq_distance.Distance(0)));
  assert(std::isfinite(pq_distance.Distance(6)));

  hybrid::SearchStats pq_stats;
  std::cout << "stage=explicit_pq_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> pq_results =
      searcher.Search(query,
                      config,
                      hybrid::ApproxDistanceKind::kProductQuantization,
                      &pq_stats,
                      pq_artifacts.pivots_path,
                      pq_artifacts.compressed_path);

  auto legacy_pq_search_distance = std::make_unique<hybrid::ProductQuantizationDistanceComputer>(index);
  legacy_pq_search_distance->Load(pq_artifacts.pivots_path, pq_artifacts.compressed_path);
  hybrid::SearchStats legacy_pq_stats;
  std::cout << "stage=legacy_pq_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> legacy_pq_results =
      searcher.Search(query, config, std::move(legacy_pq_search_distance), &legacy_pq_stats);

  hybrid::SearchConfig warm_config = config;
  warm_config.mem_l = 4;
  hybrid::SearchStats warm_stats;
  std::cout << "stage=warm_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> warm_results = searcher.Search(query, warm_config, &warm_stats);

  hybrid::SearchConfig range_config = config;
  range_config.mem_l = 1;
  range_config.range_partial = 0.0f;
  hybrid::SearchStats range_stats;
  std::cout << "stage=range_search\n" << std::flush;
  const std::vector<hybrid::SearchResult> range_results = searcher.Search(query, range_config, &range_stats);

  assert(!results.empty());
  assert(results.front().id == 0);
  assert(results.size() == 4);
  assert(stats.async_reads >= stats.pages_completed);
  assert(stats.approx_distance_evals >= stats.exact_distance_evals);
  assert(stats.exact_distance_evals >= 4);
  assert(!full_results.empty());
  assert(full_results.front().id == 0);
  assert(!pq_results.empty());
  assert(pq_results.front().id == results.front().id);
  assert(pq_stats.approx_distance_evals >= pq_stats.exact_distance_evals);
  assert(!legacy_pq_results.empty());
  assert(legacy_pq_results.front().id == results.front().id);
  assert(legacy_pq_stats.approx_distance_evals >= legacy_pq_stats.exact_distance_evals);
  assert(!warm_results.empty());
  assert(warm_results.front().id == results.front().id);
  assert(warm_stats.approx_distance_evals >= stats.approx_distance_evals);
  assert(!range_results.empty());
  assert(range_stats.range_stop);

  assert(index.IndexFileSizeBytes() == std::filesystem::file_size(index_path));
  assert(index.ApproxFileSizeBytes() == std::filesystem::file_size(approx_path));

  const std::string bad_layout_path = (out_dir / "bad_layout.graphrep").string();
  const std::string bad_layout_approx_path = (out_dir / "bad_layout.approx").string();
  hybrid::GraphReplicatedLayoutBuilder::Build(bad_layout_path, bad_layout_approx_path, vectors, graph, layouts, 0);
  const uint64_t first_page_offset = 2 * static_cast<uint64_t>(index.metadata().page_size);
  const uint64_t layout_size_offset = first_page_offset;
  WriteUint32At(bad_layout_path, layout_size_offset, index.metadata().max_page_nodes + 1);
  assert(Throws([&]() {
    hybrid::GraphReplicatedIndex corrupted_index;
    corrupted_index.Load(bad_layout_path, bad_layout_approx_path);
    std::ifstream in(bad_layout_path, std::ios::binary);
    assert(in);
    in.seekg(static_cast<std::streamoff>(corrupted_index.PageOffset(1)));
    std::vector<char> page_bytes(corrupted_index.metadata().page_size);
    in.read(page_bytes.data(), static_cast<std::streamsize>(page_bytes.size()));
    assert(in.good());
    static_cast<void>(corrupted_index.ViewPage(1, page_bytes));
  }));

  const std::string truncated_index_path = (out_dir / "truncated.graphrep").string();
  const std::string truncated_approx_path = (out_dir / "truncated.approx").string();
  hybrid::GraphReplicatedLayoutBuilder::Build(
      truncated_index_path, truncated_approx_path, vectors, graph, layouts, 0);
  std::filesystem::resize_file(truncated_index_path, std::filesystem::file_size(truncated_index_path) - 1);
  assert(Throws([&]() {
    hybrid::GraphReplicatedIndex truncated_index;
    truncated_index.Load(truncated_index_path, truncated_approx_path);
  }));

  std::vector<std::vector<uint32_t>> duplicate_layouts = layouts;
  duplicate_layouts[0].push_back(duplicate_layouts[0][1]);
  assert(Throws([&]() {
    hybrid::GraphReplicatedLayoutBuilder::Build(
        (out_dir / "duplicate.graphrep").string(),
        (out_dir / "duplicate.approx").string(),
        vectors,
        graph,
        duplicate_layouts,
        0);
  }));

  const hybrid::pipeann_integration::PipeannPQBuildResult bad_pq_artifacts =
      hybrid::pipeann_integration::BuildPipeannPQArtifacts(approx_path, (out_dir / "bad").string(), 3);
  WriteUint32At(bad_pq_artifacts.compressed_path, sizeof(uint32_t), static_cast<uint32_t>(vectors.size() + 1));
  assert(Throws([&]() {
    hybrid::ProductQuantizationDistanceComputer bad_pq(index);
    bad_pq.Load(bad_pq_artifacts.pivots_path, bad_pq_artifacts.compressed_path);
  }));

  const std::filesystem::path filter_raw_path = out_dir / "filter_raw.index";
  const std::filesystem::path filter_native_path = out_dir / "filter_native.gorgeous";
  const std::filesystem::path filter_approx_path = out_dir / "filter.approx";
  WriteFilterDiskIndex(filter_raw_path);
  WriteFilterApprox(filter_approx_path);
  const std::vector<std::vector<uint32_t>> filter_layouts = {
      {0, 1, 2},
      {1, 2, 3},
      {2, 3, 0},
      {3, 0, 1},
  };
  hybrid::gorgeous_integration::RelayoutDiskIndexToGraphReplica(
      filter_raw_path.string(), filter_layouts, filter_native_path.string(), sizeof(float));

  hybrid::NativeGorgeousIndex filter_index;
  filter_index.Load(filter_native_path.string(), filter_approx_path.string());
  hybrid::PipelinedGraphReplicatedSearcher filter_searcher(filter_index);
  hybrid::SearchConfig filter_config;
  filter_config.top_k = 2;
  filter_config.beam_width = 2;
  filter_config.l_search = 2;
  filter_config.l_pool = 4;

  const std::filesystem::path label_index_path = out_dir / "labels.attr";
  pipeann::save_attr_index_from_rows({{7}, {1}, {7}, {2}}, label_index_path.string(), "label");
  std::unique_ptr<pipeann::AttrIndex> label_index(pipeann::load_attr_index_from_file(label_index_path.string(),
                                                                                      "label",
                                                                                      filter_index.search_metadata().num_points));
  pipeann::LabelOrSelector selector(11, 11, label_index.get());
  pipeann::Attributes query_attrs;
  query_attrs.set(11, {7});
  hybrid::FilterSearchOptions post_filter;
  post_filter.mode = hybrid::FilterSearchMode::kPostFilter;
  post_filter.selector = &selector;
  post_filter.query_attrs = &query_attrs;
  post_filter.l_max = 4;

  const std::vector<float> filter_query = {2.05f, 0.0f};
  hybrid::SearchStats post_filter_stats;
  std::cout << "stage=post_filter\n" << std::flush;
  const std::vector<hybrid::SearchResult> post_results =
      filter_searcher.PostFilterSearch(filter_query, filter_config, &selector, query_attrs, post_filter.l_max, &post_filter_stats);
  assert(post_results.size() == 2);
  assert(post_results[0].id == 2);
  assert(post_results[1].id == 0);
  assert(post_filter_stats.selected_filter_mode == hybrid::FilterSearchMode::kPostFilter);
  assert(post_filter_stats.filter_reads[static_cast<size_t>(hybrid::FilterSearchMode::kPostFilter)] > 0);
  assert(post_filter_stats.filter_accessed_vectors[static_cast<size_t>(hybrid::FilterSearchMode::kPostFilter)] > 0);

  hybrid::FilterSearchOptions in_filter = post_filter;
  in_filter.mode = hybrid::FilterSearchMode::kInFilter;
  hybrid::SearchStats in_filter_stats;
  std::cout << "stage=in_filter\n" << std::flush;
  const std::vector<hybrid::SearchResult> in_results =
      filter_searcher.InFilterSearch(filter_query, filter_config, &selector, query_attrs, in_filter.l_max, &in_filter_stats);
  assert(in_results.size() == 2);
  assert(in_results[0].id == 2);
  assert(in_filter_stats.selected_filter_mode == hybrid::FilterSearchMode::kInFilter);
  assert(in_filter_stats.filter_reads[static_cast<size_t>(hybrid::FilterSearchMode::kInFilter)] > 0);
  assert(in_filter_stats.filter_io_us[static_cast<size_t>(hybrid::FilterSearchMode::kInFilter)] > 0);

  hybrid::FilterSearchOptions pre_filter = post_filter;
  pre_filter.mode = hybrid::FilterSearchMode::kPreFilter;
  hybrid::SearchStats pre_filter_stats;
  std::cout << "stage=pre_filter\n" << std::flush;
  const std::vector<hybrid::SearchResult> pre_results =
      filter_searcher.PreFilterSearch(filter_query,
                                      filter_config,
                                      &selector,
                                      query_attrs,
                                      pre_filter.l_max,
                                      hybrid::ApproxDistanceKind::kFullPrecision,
                                      &pre_filter_stats);
  assert(pre_results.size() == 2);
  assert(pre_results[0].id == 2);
  assert(pre_results[1].id == 0);
  assert(pre_filter_stats.selected_filter_mode == hybrid::FilterSearchMode::kPreFilter);
  assert(pre_filter_stats.filter_reads[static_cast<size_t>(hybrid::FilterSearchMode::kPreFilter)] > 0);
  assert(pre_filter_stats.filter_io_us1[static_cast<size_t>(hybrid::FilterSearchMode::kPreFilter)] > 0);

  hybrid::FilterSearchOptions auto_filter = post_filter;
  auto_filter.mode = hybrid::FilterSearchMode::kAuto;
  hybrid::SearchStats auto_filter_stats;
  std::cout << "stage=auto_filter\n" << std::flush;
  const std::vector<hybrid::SearchResult> auto_results =
      filter_searcher.AutoFilterSearch(filter_query,
                                       filter_config,
                                       &selector,
                                       query_attrs,
                                       auto_filter.l_max,
                                       hybrid::ApproxDistanceKind::kFullPrecision,
                                       &auto_filter_stats);
  assert(!auto_results.empty());
  assert(auto_results[0].id == 2);
  assert(auto_filter_stats.selected_filter_mode != hybrid::FilterSearchMode::kAuto);
  assert(auto_filter_stats.estimated_filter_reads[static_cast<size_t>(hybrid::FilterSearchMode::kPreFilter)] > 0);
  assert(auto_filter_stats.estimated_filter_cmps[static_cast<size_t>(hybrid::FilterSearchMode::kPostFilter)] > 0);

  hybrid::SearchConfig range_api_config = filter_config;
  range_api_config.l_search = 3;
  std::cout << "stage=range_api\n" << std::flush;
  const std::vector<hybrid::SearchResult> range_api_results =
      filter_searcher.RangeSearch(filter_query, range_api_config, 0.1f, hybrid::ApproxDistanceKind::kFullPrecision, nullptr);
  assert(range_api_results.size() == 1);
  assert(range_api_results[0].id == 2);

  std::cout << "pipeline_search_test passed\n";
  return 0;
}
