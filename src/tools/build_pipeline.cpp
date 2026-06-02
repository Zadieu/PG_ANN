#include "tools/build_pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "integrations/disk_index_relayout.h"
#include "integrations/gorgeous_original.h"
#include "integrations/pipeann_builder.h"

namespace hybrid {

namespace {

constexpr uint64_t kIntegratedSectorLen = 4096;

std::string DefaultWorkflowBaseDataPath(const std::filesystem::path &base) {
  return base.string() + ".bin";
}

std::string DefaultWorkflowTrainQueryPath(const std::filesystem::path &base) {
  return base.string() + "_train.bin";
}

std::string DefaultWorkflowGorgeousPartitionPath(const std::filesystem::path &base) {
  return base.string() + "_partition.bin";
}

std::string DefaultWorkflowGorgeousRelayoutPath(const std::filesystem::path &base) {
  return base.string() + "_graph_relayout.index";
}

void ValidateConfig(const BuildConfig &config) {
  if (config.output_dir.empty() || config.dataset_name.empty()) {
    throw std::runtime_error("output_dir and dataset_name must not be empty");
  }
  if (config.degree == 0 || config.page_nodes == 0) {
    throw std::runtime_error("degree and page_nodes must be positive");
  }
  if (config.partition_ldg_times == 0) {
    throw std::runtime_error("partition_ldg_times must be positive");
  }
  if (config.dense_degree != 0 && config.dense_degree < config.degree) {
    throw std::runtime_error("dense_degree must be >= degree when provided");
  }
  if (config.r_ood > config.degree) {
    throw std::runtime_error("r_ood must be <= degree");
  }
  if (config.degree == 0 || config.degree == config.r_ood) {
    throw std::runtime_error("degree must be greater than r_ood");
  }
  if (config.build_l == 0 || config.build_candidates == 0) {
    throw std::runtime_error("PipeANN-style builder parameters must be positive");
  }
  if (config.build_ram_budget_gb == 0) {
    throw std::runtime_error("build_ram_budget_gb must be positive");
  }
  if (config.build_threads == 0) {
    throw std::runtime_error("build_threads must be positive");
  }
  if (config.build_l > config.build_candidates) {
    throw std::runtime_error("build_l must be <= build_candidates");
  }
  if (config.build_alpha < 1.0f) {
    throw std::runtime_error("build_alpha must be >= 1.0");
  }
  if (config.pq_subspaces == 0 || config.pq_centroids == 0 || config.pq_iterations == 0) {
    throw std::runtime_error("PQ parameters must be positive");
  }
  if (config.input_path.empty()) {
    throw std::runtime_error("input_path must be provided in text input mode");
  }
  if (config.l_ood == 0) {
    throw std::runtime_error("l_ood must be positive");
  }
}

float SquaredL2Distance(const std::vector<float> &lhs, const std::vector<float> &rhs) {
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

std::vector<std::vector<float>> LoadVectors(const BuildConfig &config) {
  switch (config.input_mode) {
    case VectorInputMode::kText:
      return LoadTextVectors(config.input_path);
    case VectorInputMode::kFvecs:
      return LoadFvecsVectors(config.input_path);
    case VectorInputMode::kBvecs:
      return LoadBvecsVectors(config.input_path);
    case VectorInputMode::kBin:
      return LoadBinVectors(config.input_path);
  }
  throw std::runtime_error("unsupported vector input mode");
}

std::vector<std::vector<float>> LoadVectorsByMode(VectorInputMode mode, const std::string &path) {
  switch (mode) {
    case VectorInputMode::kText:
      return LoadTextVectors(path);
    case VectorInputMode::kFvecs:
      return LoadFvecsVectors(path);
    case VectorInputMode::kBvecs:
      return LoadBvecsVectors(path);
    case VectorInputMode::kBin:
      return LoadBinVectors(path);
  }
  throw std::runtime_error("unsupported vector input mode");
}

void CopyFileOrThrow(const std::string &from, const std::string &to) {
  const std::filesystem::path from_path(from);
  const std::filesystem::path to_path(to);
  std::error_code ec;
  std::filesystem::copy_file(from_path, to_path, std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    throw std::runtime_error("failed to copy file from " + from + " to " + to);
  }
}

template <typename T>
T ReadBinary(std::ifstream &in, const char *what) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error(what);
  }
  return value;
}

void ValidateNonEmptyVectors(const std::vector<std::vector<float>> &vectors, const char *what) {
  if (vectors.empty()) {
    throw std::runtime_error(what);
  }
}

void WriteBytes(std::ofstream &out, const void *data, size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("failed to write output file");
  }
}

void WriteBinVectors(const std::string &path, const std::vector<std::vector<float>> &vectors) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open PipeANN build input bin");
  }
  const uint32_t num_points = static_cast<uint32_t>(vectors.size());
  const uint32_t dim = static_cast<uint32_t>(vectors.front().size());
  WriteBytes(out, &num_points, sizeof(num_points));
  WriteBytes(out, &dim, sizeof(dim));
  for (const auto &vec : vectors) {
    if (vec.size() != dim) {
      throw std::runtime_error("PipeANN build input vectors have inconsistent dimensions");
    }
    WriteBytes(out, vec.data(), vec.size() * sizeof(float));
  }
}

}  // namespace

std::vector<std::vector<float>> LoadTextVectors(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open text vector file");
  }

  std::vector<std::vector<float>> vectors;
  std::string line;
  size_t dim = 0;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::vector<float> values;
    float value = 0.0f;
    while (row >> value) {
      values.push_back(value);
    }
    if (values.empty()) {
      continue;
    }
    if (dim == 0) {
      dim = values.size();
    } else if (values.size() != dim) {
      throw std::runtime_error("text vector file contains inconsistent dimensions");
    }
    vectors.push_back(std::move(values));
  }
  if (vectors.empty()) {
    throw std::runtime_error("text vector file does not contain any vectors");
  }
  return vectors;
}

std::vector<std::vector<float>> LoadFvecsVectors(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open fvecs vector file");
  }

  std::vector<std::vector<float>> vectors;
  while (in.peek() != std::ifstream::traits_type::eof()) {
    const uint32_t dim = ReadBinary<uint32_t>(in, "failed to read fvecs dimension");
    if (dim == 0) {
      throw std::runtime_error("fvecs contains zero-dimensional vector");
    }
    std::vector<float> values(dim);
    in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(dim * sizeof(float)));
    if (!in) {
      throw std::runtime_error("failed to read fvecs payload");
    }
    if (!vectors.empty() && values.size() != vectors.front().size()) {
      throw std::runtime_error("fvecs file contains inconsistent dimensions");
    }
    vectors.push_back(std::move(values));
  }
  ValidateNonEmptyVectors(vectors, "fvecs file does not contain any vectors");
  return vectors;
}

std::vector<std::vector<float>> LoadBvecsVectors(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open bvecs vector file");
  }

  std::vector<std::vector<float>> vectors;
  while (in.peek() != std::ifstream::traits_type::eof()) {
    const uint32_t dim = ReadBinary<uint32_t>(in, "failed to read bvecs dimension");
    if (dim == 0) {
      throw std::runtime_error("bvecs contains zero-dimensional vector");
    }
    std::vector<uint8_t> bytes(dim);
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(dim));
    if (!in) {
      throw std::runtime_error("failed to read bvecs payload");
    }
    std::vector<float> values(dim, 0.0f);
    for (uint32_t i = 0; i < dim; ++i) {
      values[i] = static_cast<float>(bytes[i]);
    }
    if (!vectors.empty() && values.size() != vectors.front().size()) {
      throw std::runtime_error("bvecs file contains inconsistent dimensions");
    }
    vectors.push_back(std::move(values));
  }
  ValidateNonEmptyVectors(vectors, "bvecs file does not contain any vectors");
  return vectors;
}

std::vector<std::vector<float>> LoadBinVectors(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open bin vector file");
  }

  const uint32_t num_points = ReadBinary<uint32_t>(in, "failed to read bin vector count");
  const uint32_t dim = ReadBinary<uint32_t>(in, "failed to read bin vector dimension");
  if (num_points == 0 || dim == 0) {
    throw std::runtime_error("bin vector file must describe a non-empty dataset");
  }
  std::vector<std::vector<float>> vectors(num_points, std::vector<float>(dim, 0.0f));
  for (uint32_t i = 0; i < num_points; ++i) {
    in.read(reinterpret_cast<char *>(vectors[i].data()), static_cast<std::streamsize>(dim * sizeof(float)));
    if (!in) {
      throw std::runtime_error("failed to read bin vector payload");
    }
  }
  char extra = 0;
  if (in.read(&extra, 1)) {
    throw std::runtime_error("bin vector file has unexpected trailing bytes");
  }
  return vectors;
}

std::vector<std::vector<uint32_t>> BuildKnnGraph(const std::vector<std::vector<float>> &vectors, uint32_t degree) {
  if (vectors.empty()) {
    throw std::runtime_error("vectors must not be empty");
  }
  std::vector<std::vector<uint32_t>> graph(vectors.size());
  for (uint32_t id = 0; id < vectors.size(); ++id) {
    std::vector<std::pair<float, uint32_t>> neighbors;
    neighbors.reserve(vectors.size() - 1);
    for (uint32_t other = 0; other < vectors.size(); ++other) {
      if (id == other) {
        continue;
      }
      neighbors.push_back({SquaredL2Distance(vectors[id], vectors[other]), other});
    }
    std::sort(neighbors.begin(), neighbors.end());
    for (uint32_t i = 0; i < degree && i < neighbors.size(); ++i) {
      graph[id].push_back(neighbors[i].second);
    }
  }
  return graph;
}

ReplicatedLayoutResult BuildReplicatedLayoutResult(const std::vector<std::vector<uint32_t>> &graph,
                                                   uint32_t page_nodes,
                                                   uint32_t partition_scale,
                                                   uint32_t partition_ldg_times) {
  if (graph.empty()) {
    throw std::runtime_error("graph must not be empty");
  }
  if (page_nodes == 0) {
    throw std::runtime_error("page_nodes must be positive");
  }
  if (partition_ldg_times == 0) {
    throw std::runtime_error("partition_ldg_times must be positive");
  }
  if (page_nodes == 1) {
    std::vector<std::vector<uint32_t>> singleton_layouts(graph.size(), std::vector<uint32_t>(1, 0));
    std::vector<uint32_t> id_to_page(graph.size(), 0);
    for (uint32_t node_id = 0; node_id < singleton_layouts.size(); ++node_id) {
      singleton_layouts[node_id][0] = node_id;
      id_to_page[node_id] = node_id;
    }
    ReplicatedLayoutResult result;
    result.layouts = std::move(singleton_layouts);
    result.id_to_page = std::move(id_to_page);
    result.page_capacity = 1;
    result.num_points = static_cast<uint32_t>(graph.size());
    result.num_pages = static_cast<uint32_t>(graph.size());
    result.n_primary_partition = static_cast<uint32_t>(graph.size());
    result.effective_scale_factor = partition_scale == 0 ? 1 : partition_scale;
    return result;
  }

  struct GraphPartitionerState {
    const std::vector<std::vector<uint32_t>> &direct_graph;
    std::vector<std::vector<uint32_t>> reverse_graph;
    const std::vector<std::vector<uint32_t>> &full_graph;
    uint32_t capacity = 0;
    uint32_t requested_scale_factor = 0;
    uint32_t effective_scale_factor = 0;
    uint32_t ldg_times = 0;
    uint32_t nd = 0;
    uint32_t partition_number = 0;
    uint32_t n_primary_partition = 0;
    std::vector<std::vector<uint32_t>> partitions;
    std::vector<uint32_t> id2pid;
    std::vector<std::vector<uint32_t>> aux_id2pid;
    std::vector<uint32_t> store_cnt;
    std::vector<bool> filled_nodes;
    std::deque<uint32_t> free_q;

    GraphPartitionerState(const std::vector<std::vector<uint32_t>> &graph_ref,
                          uint32_t page_nodes_ref,
                          uint32_t scale_factor_ref,
                          uint32_t ldg_times_ref)
        : direct_graph(graph_ref),
          reverse_graph(graph_ref.size()),
          full_graph(graph_ref),
          capacity(page_nodes_ref),
          requested_scale_factor(scale_factor_ref),
          effective_scale_factor(scale_factor_ref == 0 ? page_nodes_ref : scale_factor_ref),
          ldg_times(ldg_times_ref),
          nd(static_cast<uint32_t>(graph_ref.size())) {
      for (uint32_t node_id = 0; node_id < graph_ref.size(); ++node_id) {
        for (uint32_t neighbor : graph_ref[node_id]) {
          if (neighbor >= graph_ref.size()) {
            throw std::runtime_error("graph contains node ids outside the dataset range");
          }
          reverse_graph[neighbor].push_back(node_id);
        }
      }
    }

    uint32_t RoundUpPartitions(uint64_t value) const {
      return static_cast<uint32_t>((value + capacity - 1) / capacity);
    }

    bool InsertUnique(std::vector<uint32_t> &partition,
                      std::unordered_set<uint32_t> &seen,
                      uint32_t node_id) const {
      if (partition.size() == capacity) {
        return false;
      }
      if (!seen.insert(node_id).second) {
        return false;
      }
      partition.push_back(node_id);
      return true;
    }

    void Setup() {
      if (effective_scale_factor != 1) {
        partition_number = RoundUpPartitions(static_cast<uint64_t>(nd) * effective_scale_factor);
        n_primary_partition = (capacity * partition_number - nd) / (capacity - 1);
      } else {
        partition_number = RoundUpPartitions(nd);
        n_primary_partition = 0;
      }
      partitions.assign(partition_number, {});
      id2pid.assign(nd, std::numeric_limits<uint32_t>::max());
      aux_id2pid.assign(nd, {});
      store_cnt.assign(nd, 0);
      filled_nodes.assign(nd, false);
    }

    void LightGraphReplicatedPartition() {
      for (uint32_t node_id = 0; node_id < nd; ++node_id) {
        partitions[node_id].reserve(capacity);
        partitions[node_id].push_back(node_id);
        id2pid[node_id] = node_id;
        filled_nodes[node_id] = true;
      }

      for (uint32_t pid = 0; pid < nd; ++pid) {
        std::vector<uint32_t> candidate_neighbors;
        candidate_neighbors.reserve(64);
        std::unordered_set<uint32_t> seen(partitions[pid].begin(), partitions[pid].end());
        for (size_t slot = 0; slot < partitions[pid].size(); ++slot) {
          for (uint32_t neighbor : full_graph[partitions[pid][slot]]) {
            candidate_neighbors.push_back(neighbor);
          }
          auto rng = std::default_random_engine{};
          std::shuffle(candidate_neighbors.begin(), candidate_neighbors.end(), rng);
          for (uint32_t neighbor : candidate_neighbors) {
            if (partitions[pid].size() == capacity) {
              break;
            }
            InsertUnique(partitions[pid], seen, neighbor);
          }
          if (partitions[pid].size() == capacity) {
            break;
          }
        }
      }
    }

    void ScaledGraphPartitionPrimary() {
      for (uint32_t init_pid = 0; init_pid < n_primary_partition; ++init_pid) {
        store_cnt[init_pid]++;
        partitions[init_pid].push_back(init_pid);
        id2pid[init_pid] = init_pid;
        filled_nodes[init_pid] = true;
      }

      for (uint32_t pid = 0; pid < n_primary_partition; ++pid) {
        std::unordered_set<uint32_t> seen(partitions[pid].begin(), partitions[pid].end());
        for (size_t slot = 0; slot < partitions[pid].size(); ++slot) {
          std::vector<std::pair<uint32_t, uint32_t>> candidate_neighbors;
          for (uint32_t neighbor : full_graph[partitions[pid][slot]]) {
            candidate_neighbors.push_back({neighbor, store_cnt[neighbor]});
          }
          std::sort(candidate_neighbors.begin(),
                    candidate_neighbors.end(),
                    [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
          for (const auto &[neighbor, _] : candidate_neighbors) {
            if (partitions[pid].size() == capacity) {
              break;
            }
            if (InsertUnique(partitions[pid], seen, neighbor)) {
              aux_id2pid[neighbor].push_back(pid);
              store_cnt[neighbor]++;
            }
          }
        }
      }
    }

    void InitializeSecondaryPartitions() {
      std::unordered_set<uint32_t> vis;
      uint32_t pid = n_primary_partition;
      for (uint32_t node_id = n_primary_partition; node_id < nd; ++node_id) {
        if (vis.count(node_id) != 0) {
          continue;
        }
        while (pid < partition_number && partitions[pid].size() == capacity) {
          ++pid;
        }
        if (pid >= partition_number) {
          break;
        }
        std::unordered_set<uint32_t> seen(partitions[pid].begin(), partitions[pid].end());
        vis.insert(node_id);
        InsertUnique(partitions[pid], seen, node_id);
        id2pid[node_id] = pid;
        for (uint32_t neighbor : full_graph[node_id]) {
          if (vis.count(neighbor) != 0 || filled_nodes[neighbor]) {
            continue;
          }
          if (partitions[pid].size() == capacity) {
            ++pid;
            break;
          }
          if (InsertUnique(partitions[pid], seen, neighbor)) {
            id2pid[neighbor] = pid;
            vis.insert(neighbor);
          }
        }
      }
    }

    uint32_t GetUnfilledPartition() {
      while (!free_q.empty()) {
        const uint32_t pid = free_q.front();
        free_q.pop_front();
        if (pid < partition_number && partitions[pid].size() < capacity) {
          return pid;
        }
      }
      for (uint32_t pid = n_primary_partition; pid < partition_number; ++pid) {
        if (partitions[pid].size() < capacity) {
          return pid;
        }
      }
      throw std::runtime_error("graph_partitioner could not find an unfilled partition");
    }

    uint32_t SelectPartition(uint32_t node_id) {
      float max_score = 0.0f;
      uint32_t selected = std::numeric_limits<uint32_t>::max();
      std::unordered_map<uint32_t, uint32_t> counts;
      for (uint32_t neighbor : direct_graph[node_id]) {
        const uint32_t pid = id2pid[neighbor];
        if (pid != std::numeric_limits<uint32_t>::max()) {
          counts[pid]++;
        }
      }
      for (uint32_t neighbor : reverse_graph[node_id]) {
        const uint32_t pid = id2pid[neighbor];
        if (pid != std::numeric_limits<uint32_t>::max()) {
          counts[pid]++;
        }
      }
      for (const auto &[pid, count] : counts) {
        const double size = static_cast<double>(partitions[pid].size());
        const float score = static_cast<float>(count * (1.0 - size / capacity));
        if (score > max_score && partitions[pid].size() < capacity) {
          max_score = score;
          selected = pid;
        }
      }
      if (selected == std::numeric_limits<uint32_t>::max()) {
        selected = GetUnfilledPartition();
      }
      return selected;
    }

    void Sync(uint32_t node_id) {
      uint32_t pid = SelectPartition(node_id);
      while (partitions[pid].size() == capacity) {
        pid = GetUnfilledPartition();
      }
      partitions[pid].push_back(node_id);
      id2pid[node_id] = pid;
      if (partitions[pid].size() < capacity) {
        free_q.push_back(pid);
      }
    }

    void GraphPartitionLdg(uint32_t round) {
      free_q.clear();
      for (uint32_t pid = n_primary_partition; pid < partition_number; ++pid) {
        partitions[pid].clear();
        free_q.push_back(pid);
      }

      std::vector<uint32_t> stream;
      stream.reserve(nd - n_primary_partition);
      for (uint32_t node_id = n_primary_partition; node_id < nd; ++node_id) {
        stream.push_back(node_id);
      }
      std::mt19937 rng(42 + round);
      std::shuffle(stream.begin(), stream.end(), rng);
      for (uint32_t node_id : stream) {
        Sync(node_id);
      }
    }

    ReplicatedLayoutResult Build() {
      Setup();
      if (requested_scale_factor == 0 || n_primary_partition == nd) {
        LightGraphReplicatedPartition();
      } else {
        if (effective_scale_factor != 1) {
          ScaledGraphPartitionPrimary();
        }
        InitializeSecondaryPartitions();
        for (uint32_t round = 0; round < ldg_times; ++round) {
          GraphPartitionLdg(round);
        }
      }

      for (const auto &partition : partitions) {
        if (partition.empty()) {
          throw std::runtime_error("graph_partitioner produced an empty page layout");
        }
      }
      for (uint32_t node_id = 0; node_id < nd; ++node_id) {
        if (id2pid[node_id] == std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("graph_partitioner did not assign a primary partition to every node");
        }
      }

      ReplicatedLayoutResult result;
      result.layouts = partitions;
      result.id_to_page = id2pid;
      result.page_capacity = capacity;
      result.num_points = nd;
      result.num_pages = partition_number;
      result.n_primary_partition = n_primary_partition;
      result.effective_scale_factor = effective_scale_factor;
      return result;
    }
  };

  GraphPartitionerState partitioner(graph, page_nodes, partition_scale, partition_ldg_times);
  return partitioner.Build();
}

std::vector<std::vector<uint32_t>> BuildReplicatedLayouts(const std::vector<std::vector<uint32_t>> &graph,
                                                          uint32_t page_nodes,
                                                          uint32_t partition_scale,
                                                          uint32_t partition_ldg_times) {
  return BuildReplicatedLayoutResult(graph, page_nodes, partition_scale, partition_ldg_times).layouts;
}

BuildArtifacts RunBuildPipeline(const BuildConfig &config) {
  ValidateConfig(config);
  const std::vector<std::vector<float>> vectors = LoadVectors(config);
  std::vector<std::vector<float>> train_queries;
  if (!config.train_query_path.empty()) {
    train_queries = LoadVectorsByMode(config.train_query_mode, config.train_query_path);
    if (train_queries.empty()) {
      throw std::runtime_error("train query file does not contain any vectors");
    }
    if (train_queries.front().size() != vectors.front().size()) {
      throw std::runtime_error("train query dimension does not match base vectors");
    }
  }
  const uint32_t dim = static_cast<uint32_t>(vectors.front().size());
  if (config.use_explicit_entry_id && config.entry_id >= vectors.size()) {
    throw std::runtime_error("entry_id out of range");
  }
  if (dim % config.pq_subspaces != 0) {
    throw std::runtime_error("vector dimension must be divisible by pq_subspaces");
  }

  std::filesystem::create_directories(config.output_dir);
  const std::filesystem::path base = std::filesystem::path(config.output_dir) / config.dataset_name;
  const std::string builder_base_bin = DefaultWorkflowBaseDataPath(base);
  const std::string builder_train_bin = DefaultWorkflowTrainQueryPath(base);

  WriteBinVectors(builder_base_bin, vectors);
  if (!train_queries.empty()) {
    WriteBinVectors(builder_train_bin, train_queries);
  }

  pipeann_integration::PipeannBuildParameters builder_params;
  builder_params.range = config.degree;
  builder_params.r_ood = config.r_ood;
  builder_params.build_l = config.build_l;
  builder_params.build_ram_budget_gb = config.build_ram_budget_gb;
  builder_params.build_threads = config.build_threads;
  builder_params.range_dense = config.dense_degree == 0 ? config.degree * 2 : config.dense_degree;
  builder_params.l_ood = config.l_ood;
  const pipeann_integration::PipeannDiskBuildResult build_result =
      pipeann_integration::BuildPipeannDiskIndex(builder_base_bin,
                                                 base.string(),
                                                 builder_params,
                                                 train_queries.empty() ? std::string() : builder_train_bin,
                                                 config.use_explicit_entry_id
                                                     ? std::optional<uint32_t>(config.entry_id)
                                                     : std::nullopt);
  const uint32_t entry_id = build_result.entry_id;
  const std::string gorgeous_partition_path = DefaultWorkflowGorgeousPartitionPath(base);
  const std::string gorgeous_relayout_path = DefaultWorkflowGorgeousRelayoutPath(base);
  const std::string pipeann_equal_layout_path = build_result.disk_index_path + ".equal";
  const std::string pipeann_gorgeous_layout_path = build_result.disk_index_path + ".gorgeous";
  const std::vector<std::vector<uint32_t>> flattened_graph =
      pipeann_integration::LoadPipeannFlatGraph(build_result.disk_index_path);
  const gorgeous_integration::GorgeousOriginalPartitionResult gorgeous_result =
      gorgeous_integration::BuildGorgeousGraphReplicaIndex(vectors,
                                                           flattened_graph,
                                                           entry_id,
                                                           gorgeous_partition_path,
                                                           gorgeous_relayout_path,
                                                           config.partition_scale,
                                                           config.partition_ldg_times,
                                                           kIntegratedSectorLen,
                                                           kIntegratedSectorLen);
  const pipeann_integration::PipeannPQBuildResult pq_result =
      pipeann_integration::BuildPipeannPQArtifacts(builder_base_bin, base.string(), config.pq_subspaces);

  CopyFileOrThrow(build_result.disk_index_path, pipeann_equal_layout_path);
  CopyFileOrThrow(gorgeous_relayout_path, pipeann_gorgeous_layout_path);

  BuildArtifacts artifacts;
  artifacts.workflow_prefix = base.string();
  artifacts.pipeann_base_data_path = builder_base_bin;
  artifacts.pipeann_train_query_path = train_queries.empty() ? std::string() : builder_train_bin;
  artifacts.pipeann_index_prefix = base.string();
  artifacts.gorgeous_partition_bin_path = gorgeous_partition_path;
  artifacts.gorgeous_relayout_index_path = gorgeous_relayout_path;
  artifacts.raw_disk_index_path = build_result.disk_index_path;
  artifacts.pipeann_equal_layout_path = pipeann_equal_layout_path;
  artifacts.pipeann_gorgeous_layout_path = pipeann_gorgeous_layout_path;
  artifacts.approx_path = builder_base_bin;
  artifacts.pq_codebook_path = pq_result.pivots_path;
  artifacts.pq_codes_path = pq_result.compressed_path;
  artifacts.num_points = build_result.num_points;
  artifacts.dim = build_result.dim;
  artifacts.pipeann_build_stats = build_result.stats;
  artifacts.pipeann_refine_sidecar_path.clear();
  artifacts.pipeann_refine_manifest_path.clear();
  artifacts.pipeann_refine_nodes_path.clear();
  std::filesystem::remove(pipeann_integration::DefaultPipeannRefineSidecarPath(artifacts.raw_disk_index_path));
  std::filesystem::remove(artifacts.raw_disk_index_path + ".pipeann.refine.txt");
  std::filesystem::remove(artifacts.raw_disk_index_path + ".pipeann.refine.nodes.tsv");
  return artifacts;
}

}  // namespace hybrid
