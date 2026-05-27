#include "quant/approx_distance.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "nbr/pq_table.h"

namespace hybrid {

namespace {

constexpr uint64_t kProductQuantizationMagic = 0x4859425051303031ULL;
constexpr uint32_t kProductQuantizationVersion = 1;

template <typename T>
void WriteBinary(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!out) {
    throw std::runtime_error("failed to write PQ file");
  }
}

void WriteBinary(std::ofstream &out, const void *data, size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("failed to write PQ file");
  }
}

template <typename T>
T ReadBinary(std::ifstream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("failed to read PQ file");
  }
  return value;
}

uint64_t CheckedMul(uint64_t lhs, uint64_t rhs, const char *what) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw std::runtime_error(what);
  }
  return lhs * rhs;
}

uint64_t CheckedAdd(uint64_t lhs, uint64_t rhs, const char *what) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    throw std::runtime_error(what);
  }
  return lhs + rhs;
}

void ValidateTrainingInput(const std::vector<std::vector<float>> &vectors,
                           uint32_t num_subspaces,
                           uint32_t centroids_per_subspace,
                           uint32_t num_iterations) {
  if (vectors.empty()) {
    throw std::runtime_error("PQ training vectors must not be empty");
  }
  if (num_subspaces == 0 || centroids_per_subspace == 0 || num_iterations == 0) {
    throw std::runtime_error("PQ training parameters must be positive");
  }
  const size_t dim = vectors.front().size();
  if (dim == 0 || dim % num_subspaces != 0) {
    throw std::runtime_error("PQ requires a non-zero dimension divisible by num_subspaces");
  }
  for (const auto &vector : vectors) {
    if (vector.size() != dim) {
      throw std::runtime_error("all PQ training vectors must have the same dimension");
    }
  }
}

float SubspaceDistanceToCodebook(const std::vector<float> &vector,
                                 size_t vector_offset,
                                 const std::vector<float> &codebooks,
                                 size_t codebook_offset,
                                 uint32_t subspace_dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < subspace_dim; ++i) {
    const float diff = vector[vector_offset + i] - codebooks[codebook_offset + i];
    sum += diff * diff;
  }
  return sum;
}

size_t CodebookOffset(const ProductQuantizationMetadata &metadata,
                      uint32_t subspace,
                      uint32_t centroid) {
  const uint64_t offset =
      (static_cast<uint64_t>(subspace) * metadata.centroids_per_subspace + centroid) *
      metadata.subspace_dim;
  return static_cast<size_t>(offset);
}

void ReadPayload(std::ifstream &in, void *data, size_t size) {
  in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("failed to read PQ file payload");
  }
}

}  // namespace

FullPrecisionDistanceComputer::FullPrecisionDistanceComputer(const IndexReader &index)
    : index_(index) {
}

void FullPrecisionDistanceComputer::BeginQuery(const std::vector<float> &query) {
  if (query.size() != index_.search_metadata().dim) {
    throw std::runtime_error("query dimension does not match index dimension");
  }
  query_ = query;
}

float FullPrecisionDistanceComputer::Distance(uint32_t id) const {
  if (query_.empty()) {
    throw std::runtime_error("BeginQuery must be called before Distance");
  }
  return L2Distance(query_, index_.approx_vector(id));
}

const char *FullPrecisionDistanceComputer::name() const {
  return "full_precision";
}

PipeannProductQuantization::PipeannProductQuantization(const IndexReader &index)
    : index_(index) {
}

void ProductQuantizationEncoder::Build(const std::string &codebook_path,
                                       const std::string &codes_path,
                                       const std::vector<std::vector<float>> &vectors,
                                       uint32_t num_subspaces,
                                       uint32_t centroids_per_subspace,
                                       uint32_t num_iterations) {
  ValidateTrainingInput(vectors, num_subspaces, centroids_per_subspace, num_iterations);

  ProductQuantizationMetadata metadata{};
  metadata.magic = kProductQuantizationMagic;
  metadata.version = kProductQuantizationVersion;
  metadata.dim = static_cast<uint32_t>(vectors.front().size());
  metadata.num_points = static_cast<uint32_t>(vectors.size());
  metadata.num_subspaces = num_subspaces;
  metadata.subspace_dim = metadata.dim / num_subspaces;
  metadata.centroids_per_subspace = centroids_per_subspace;

  const size_t codebook_values = static_cast<size_t>(metadata.num_subspaces) *
                                 metadata.centroids_per_subspace * metadata.subspace_dim;
  std::vector<float> codebooks(codebook_values, 0.0f);
  std::vector<uint32_t> codes(static_cast<size_t>(metadata.num_points) * metadata.num_subspaces, 0);

  for (uint32_t subspace = 0; subspace < metadata.num_subspaces; ++subspace) {
    const size_t vector_offset = static_cast<size_t>(subspace) * metadata.subspace_dim;
    for (uint32_t centroid = 0; centroid < metadata.centroids_per_subspace; ++centroid) {
      const size_t sample_index = (static_cast<size_t>(centroid) * metadata.num_points) /
                                  metadata.centroids_per_subspace;
      const size_t centroid_offset = CodebookOffset(metadata, subspace, centroid);
      std::copy_n(vectors[sample_index].begin() + static_cast<std::ptrdiff_t>(vector_offset),
                  metadata.subspace_dim,
                  codebooks.begin() + static_cast<std::ptrdiff_t>(centroid_offset));
    }

    std::vector<float> sums(static_cast<size_t>(metadata.centroids_per_subspace) * metadata.subspace_dim, 0.0f);
    std::vector<uint32_t> counts(metadata.centroids_per_subspace, 0);
    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
      std::fill(sums.begin(), sums.end(), 0.0f);
      std::fill(counts.begin(), counts.end(), 0);
      for (uint32_t point_id = 0; point_id < metadata.num_points; ++point_id) {
        uint32_t best_centroid = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (uint32_t centroid = 0; centroid < metadata.centroids_per_subspace; ++centroid) {
          const size_t centroid_offset = CodebookOffset(metadata, subspace, centroid);
          const float distance = SubspaceDistanceToCodebook(
              vectors[point_id], vector_offset, codebooks, centroid_offset, metadata.subspace_dim);
          if (distance < best_distance) {
            best_distance = distance;
            best_centroid = centroid;
          }
        }
        codes[static_cast<size_t>(point_id) * metadata.num_subspaces + subspace] = best_centroid;
        ++counts[best_centroid];
        const size_t sum_offset = static_cast<size_t>(best_centroid) * metadata.subspace_dim;
        for (uint32_t i = 0; i < metadata.subspace_dim; ++i) {
          sums[sum_offset + i] += vectors[point_id][vector_offset + i];
        }
      }

      for (uint32_t centroid = 0; centroid < metadata.centroids_per_subspace; ++centroid) {
        if (counts[centroid] == 0) {
          continue;
        }
        const size_t centroid_offset = CodebookOffset(metadata, subspace, centroid);
        const size_t sum_offset = static_cast<size_t>(centroid) * metadata.subspace_dim;
        for (uint32_t i = 0; i < metadata.subspace_dim; ++i) {
          codebooks[centroid_offset + i] = sums[sum_offset + i] / static_cast<float>(counts[centroid]);
        }
      }
    }

    for (uint32_t point_id = 0; point_id < metadata.num_points; ++point_id) {
      uint32_t best_centroid = 0;
      float best_distance = std::numeric_limits<float>::max();
      for (uint32_t centroid = 0; centroid < metadata.centroids_per_subspace; ++centroid) {
        const size_t centroid_offset = CodebookOffset(metadata, subspace, centroid);
        const float distance = SubspaceDistanceToCodebook(
            vectors[point_id], vector_offset, codebooks, centroid_offset, metadata.subspace_dim);
        if (distance < best_distance) {
          best_distance = distance;
          best_centroid = centroid;
        }
      }
      codes[static_cast<size_t>(point_id) * metadata.num_subspaces + subspace] = best_centroid;
    }
  }

  std::ofstream codebook_out(codebook_path, std::ios::binary | std::ios::trunc);
  if (!codebook_out) {
    throw std::runtime_error("failed to open PQ codebook output file");
  }
  WriteBinary(codebook_out, metadata);
  WriteBinary(codebook_out, codebooks.data(), codebooks.size() * sizeof(float));

  std::ofstream codes_out(codes_path, std::ios::binary | std::ios::trunc);
  if (!codes_out) {
    throw std::runtime_error("failed to open PQ codes output file");
  }
  WriteBinary(codes_out, metadata);
  WriteBinary(codes_out, codes.data(), codes.size() * sizeof(uint32_t));
}

void PipeannProductQuantization::Load(const std::string &codebook_path,
                                      const std::string &codes_path) {
  codebook_path_ = codebook_path.empty() ? DefaultPipeannPqPivotsPath(index_.index_path()) : codebook_path;
  codes_path_ = codes_path.empty() ? DefaultPipeannPqCompressedPath(index_.index_path()) : codes_path;
  ready_ = false;
  codebooks_.clear();
  centroid_.clear();
  chunk_offsets_.clear();
  codes_.clear();

  if (!std::filesystem::exists(codebook_path_)) {
    throw std::runtime_error("failed to open PipeANN PQ pivots file");
  }
  std::ifstream codes_in(codes_path_, std::ios::binary);
  if (!codes_in) {
    throw std::runtime_error("failed to open PipeANN PQ compressed file");
  }

  const uint32_t num_points = ReadBinary<uint32_t>(codes_in);
  const uint32_t num_chunks = ReadBinary<uint32_t>(codes_in);
  if (num_points != index_.search_metadata().num_points) {
    throw std::runtime_error("PipeANN PQ compressed point count does not match loaded index");
  }
  if (num_chunks == 0) {
    throw std::runtime_error("PipeANN PQ compressed metadata is invalid");
  }
  const uint64_t expected_codes_size =
      CheckedAdd(sizeof(uint32_t) * 2,
                 CheckedMul(num_points, num_chunks, "PipeANN PQ compressed size overflow"),
                 "PipeANN PQ compressed size overflow");
  if (std::filesystem::file_size(codes_path_) != expected_codes_size) {
    throw std::runtime_error("PipeANN PQ compressed file size does not match metadata");
  }

  pipeann::FixedChunkPQTable<float> native_pq_table(pipeann::Metric::L2);
  native_pq_table.load_pq_centroid_bin(codebook_path_.c_str(), num_chunks, 0);
  if (native_pq_table.get_dim() != index_.search_metadata().dim) {
    throw std::runtime_error("PipeANN PQ pivot dimension does not match loaded index");
  }
  if (native_pq_table.n_chunks != num_chunks) {
    throw std::runtime_error("PipeANN PQ pivot chunk count does not match compressed metadata");
  }
  if (index_.search_metadata().dim % num_chunks != 0) {
    throw std::runtime_error("PipeANN PQ chunk layout is not evenly divisible");
  }

  metadata_.magic = kProductQuantizationMagic;
  metadata_.version = kProductQuantizationVersion;
  metadata_.dim = static_cast<uint32_t>(native_pq_table.get_dim());
  metadata_.num_points = num_points;
  metadata_.num_subspaces = num_chunks;
  metadata_.subspace_dim = metadata_.dim / metadata_.num_subspaces;
  metadata_.centroids_per_subspace = 256;
  codebooks_.assign(native_pq_table.tables,
                    native_pq_table.tables +
                        static_cast<std::ptrdiff_t>(metadata_.centroids_per_subspace) * metadata_.dim);
  centroid_.assign(native_pq_table.centroid,
                   native_pq_table.centroid + static_cast<std::ptrdiff_t>(metadata_.dim));
  chunk_offsets_.assign(native_pq_table.chunk_offsets,
                        native_pq_table.chunk_offsets + static_cast<std::ptrdiff_t>(num_chunks + 1));

  codes_.resize(static_cast<size_t>(metadata_.num_points) * metadata_.num_subspaces);
  ReadPayload(codes_in, codes_.data(), codes_.size() * sizeof(uint8_t));
  ready_ = true;
}

void PipeannProductQuantization::InitializeQuery(const std::vector<float> &query,
                                                 std::vector<float> *query_distance_table) const {
  if (query_distance_table == nullptr) {
    throw std::runtime_error("PQ query distance table output must not be null");
  }
  if (!ready_) {
    throw std::runtime_error("PipeANN PQ must be loaded before InitializeQuery");
  }
  if (query.size() != metadata_.dim) {
    throw std::runtime_error("query dimension does not match PQ dimension");
  }

  query_distance_table->assign(static_cast<size_t>(metadata_.num_subspaces) * metadata_.centroids_per_subspace, 0.0f);
  for (uint32_t chunk = 0; chunk < metadata_.num_subspaces; ++chunk) {
    const uint32_t begin = chunk_offsets_[chunk];
    const uint32_t end = chunk_offsets_[chunk + 1];
    float *chunk_distances =
        query_distance_table->data() + static_cast<size_t>(chunk) * metadata_.centroids_per_subspace;
    for (uint32_t dim = begin; dim < end; ++dim) {
      const float centered_query = query[dim] - centroid_[dim];
      for (uint32_t centroid_id = 0; centroid_id < metadata_.centroids_per_subspace; ++centroid_id) {
        const float diff =
            codebooks_[static_cast<size_t>(centroid_id) * metadata_.dim + dim] - centered_query;
        chunk_distances[centroid_id] += diff * diff;
      }
    }
  }
}

float PipeannProductQuantization::Distance(const std::vector<float> &query_distance_table, uint32_t id) const {
  if (!ready_) {
    throw std::runtime_error("PipeANN PQ must be loaded before Distance");
  }
  if (query_distance_table.empty()) {
    throw std::runtime_error("InitializeQuery must be called before PQ Distance");
  }
  if (query_distance_table.size() !=
      static_cast<size_t>(metadata_.num_subspaces) * metadata_.centroids_per_subspace) {
    throw std::runtime_error("PQ query distance table shape does not match loaded metadata");
  }
  if (id >= metadata_.num_points) {
    throw std::runtime_error("PQ distance id out of range");
  }

  float distance = 0.0f;
  const size_t code_offset = static_cast<size_t>(id) * metadata_.num_subspaces;
  for (uint32_t subspace = 0; subspace < metadata_.num_subspaces; ++subspace) {
    const uint32_t centroid = static_cast<uint32_t>(codes_[code_offset + subspace]);
    const size_t table_offset =
        static_cast<size_t>(subspace) * metadata_.centroids_per_subspace + centroid;
    distance += query_distance_table[table_offset];
  }
  return distance;
}

void PipeannProductQuantization::DistanceBatch(const std::vector<float> &query_distance_table,
                                               const uint32_t *ids,
                                               size_t count,
                                               float *distances) const {
  if (ids == nullptr || distances == nullptr) {
    throw std::runtime_error("PQ distance batch buffers must not be null");
  }
  for (size_t i = 0; i < count; ++i) {
    distances[i] = Distance(query_distance_table, ids[i]);
  }
}

ProductQuantizationDistanceComputer::ProductQuantizationDistanceComputer(const IndexReader &index)
    : fallback_(index), pq_(index) {
}

void ProductQuantizationDistanceComputer::Load(const std::string &codebook_path,
                                               const std::string &codes_path) {
  pq_.Load(codebook_path, codes_path);
  query_distance_table_.clear();
}

void ProductQuantizationDistanceComputer::BeginQuery(const std::vector<float> &query) {
  if (!pq_.ready()) {
    fallback_.BeginQuery(query);
    return;
  }
  pq_.InitializeQuery(query, &query_distance_table_);
}

float ProductQuantizationDistanceComputer::Distance(uint32_t id) const {
  if (!pq_.ready()) {
    return fallback_.Distance(id);
  }
  return pq_.Distance(query_distance_table_, id);
}

const char *ProductQuantizationDistanceComputer::name() const {
  return pq_.ready() ? "pipeann_pq" : "pipeann_pq_fallback";
}

std::unique_ptr<IApproximateDistanceComputer> CreateApproximateDistanceComputer(
    ApproxDistanceKind kind,
    const IndexReader &index) {
  switch (kind) {
    case ApproxDistanceKind::kFullPrecision:
      return std::make_unique<FullPrecisionDistanceComputer>(index);
    case ApproxDistanceKind::kProductQuantization:
      return std::make_unique<ProductQuantizationDistanceComputer>(index);
  }
  throw std::runtime_error("unsupported approximate distance kind");
}

}  // namespace hybrid
