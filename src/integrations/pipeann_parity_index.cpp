#include "integrations/pipeann_parity_index.h"

#include <memory>
#include <stdexcept>

#include "linux_aligned_file_reader.h"
#include "nbr/pq_nbr.h"

namespace hybrid::pipeann_parity {

PipeannParityIndex::PipeannParityIndex(PipeannParityIndexConfig config)
    : config_(std::move(config)) {}

void PipeannParityIndex::Load() {
  if (config_.index_prefix.empty()) {
    throw std::runtime_error("PipeANN parity index_prefix must not be empty");
  }
  if (config_.max_threads == 0) {
    throw std::runtime_error("PipeANN parity max_threads must be positive");
  }

  ActivatePipeannLayout(config_.index_prefix, config_.layout, config_.gp_partition_path);

  index_.reset();
  nbr_.reset();
  reader_.reset();

  pipeann::IndexBuildParameters params;
  params.max_nthreads = config_.max_threads;

  reader_ = std::make_shared<LinuxAlignedFileReader>();
  nbr_ = std::make_unique<pipeann::PQNeighbor<float>>(config_.metric);
  index_ = std::make_unique<pipeann::SSDIndex<float>>(config_.metric, reader_, nbr_.get(), true, &params);
  index_->load(config_.index_prefix.c_str(), false);
}

pipeann::SSDIndex<float> &PipeannParityIndex::index() {
  if (index_ == nullptr) {
    throw std::runtime_error("PipeANN parity index is not loaded");
  }
  return *index_;
}

const pipeann::SSDIndex<float> &PipeannParityIndex::index() const {
  if (index_ == nullptr) {
    throw std::runtime_error("PipeANN parity index is not loaded");
  }
  return *index_;
}

const PipeannParityIndexConfig &PipeannParityIndex::config() const {
  return config_;
}

bool PipeannParityIndex::loaded() const {
  return index_ != nullptr;
}

}  // namespace hybrid::pipeann_parity
