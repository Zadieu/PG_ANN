#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gorgeous_layout.h"

namespace hybrid {

enum class ApproxDistanceKind {
  kFullPrecision = 0,
  kProductQuantization = 1,
};

class IApproximateDistanceComputer {
 public:
  virtual ~IApproximateDistanceComputer() = default;

  virtual void BeginQuery(const std::vector<float> &query) = 0;
  virtual float Distance(uint32_t id) const = 0;
  virtual const char *name() const = 0;
};

class FullPrecisionDistanceComputer : public IApproximateDistanceComputer {
 public:
  explicit FullPrecisionDistanceComputer(const IndexReader &index);

  void BeginQuery(const std::vector<float> &query) override;
  float Distance(uint32_t id) const override;
  const char *name() const override;

 private:
  const IndexReader &index_;
  std::vector<float> query_;
};

struct ProductQuantizationMetadata {
  uint64_t magic = 0;
  uint32_t version = 1;
  uint32_t dim = 0;
  uint32_t num_points = 0;
  uint32_t num_subspaces = 0;
  uint32_t subspace_dim = 0;
  uint32_t centroids_per_subspace = 0;
};

class ProductQuantizationEncoder {
 public:
  static void Build(const std::string &codebook_path,
                    const std::string &codes_path,
                    const std::vector<std::vector<float>> &vectors,
                    uint32_t num_subspaces,
                    uint32_t centroids_per_subspace,
                    uint32_t num_iterations = 6);
};

class PipeannProductQuantization {
 public:
  explicit PipeannProductQuantization(const IndexReader &index);

  void Load(const std::string &codebook_path, const std::string &codes_path);
  bool ready() const { return ready_; }
  const ProductQuantizationMetadata &metadata() const { return metadata_; }
  const std::string &codebook_path() const { return codebook_path_; }
  const std::string &codes_path() const { return codes_path_; }

  void InitializeQuery(const std::vector<float> &query, std::vector<float> *query_distance_table) const;
  float Distance(const std::vector<float> &query_distance_table, uint32_t id) const;
  void DistanceBatch(const std::vector<float> &query_distance_table,
                     const uint32_t *ids,
                     size_t count,
                     float *distances) const;

 private:
  const IndexReader &index_;
  bool ready_ = false;
  std::string codebook_path_;
  std::string codes_path_;
  ProductQuantizationMetadata metadata_{};
  std::vector<float> codebooks_;
  std::vector<float> centroid_;
  std::vector<uint32_t> chunk_offsets_;
  std::vector<uint8_t> codes_;
};

class ProductQuantizationDistanceComputer : public IApproximateDistanceComputer {
 public:
  explicit ProductQuantizationDistanceComputer(const IndexReader &index);

  void Load(const std::string &codebook_path, const std::string &codes_path);
  bool ready() const { return pq_.ready(); }
  const ProductQuantizationMetadata &metadata() const { return pq_.metadata(); }

  void BeginQuery(const std::vector<float> &query) override;
  float Distance(uint32_t id) const override;
  const char *name() const override;

 private:
  FullPrecisionDistanceComputer fallback_;
  PipeannProductQuantization pq_;
  std::vector<float> query_distance_table_;
};

std::unique_ptr<IApproximateDistanceComputer> CreateApproximateDistanceComputer(
    ApproxDistanceKind kind,
    const IndexReader &index);

}  // namespace hybrid
