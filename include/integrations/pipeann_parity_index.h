#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "integrations/pipeann_layout_activate.h"
#include "ssd_index.h"
#include "utils.h"

class AlignedFileReader;
class LinuxAlignedFileReader;

namespace pipeann {
template <typename T>
class AbstractNeighbor;
}

namespace hybrid::pipeann_parity {

struct PipeannParityIndexConfig {
  std::string index_prefix;
  pipeann::Metric metric = pipeann::Metric::L2;
  uint32_t max_threads = 8;
  PipeannLayout layout = PipeannLayout::kGorgeous;
  std::string gp_partition_path;
};

class PipeannParityIndex {
 public:
  explicit PipeannParityIndex(PipeannParityIndexConfig config);

  void Load();

  pipeann::SSDIndex<float> &index();
  const pipeann::SSDIndex<float> &index() const;
  const PipeannParityIndexConfig &config() const;
  bool loaded() const;

 private:
  PipeannParityIndexConfig config_;
  std::shared_ptr<AlignedFileReader> reader_;
  std::unique_ptr<pipeann::AbstractNeighbor<float>> nbr_;
  std::unique_ptr<pipeann::SSDIndex<float>> index_;
};

}  // namespace hybrid::pipeann_parity
