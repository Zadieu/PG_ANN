#include "integrations/pipeann_layout_activate.h"

#include <filesystem>
#include <stdexcept>

namespace hybrid::pipeann_parity {

namespace fs = std::filesystem;

namespace {

std::string DiskIndexPath(const std::string &index_prefix) {
  return index_prefix + "_disk.index";
}

std::string EqualLayoutPath(const std::string &index_prefix) {
  return DiskIndexPath(index_prefix) + ".equal";
}

std::string GorgeousLayoutPath(const std::string &index_prefix) {
  return DiskIndexPath(index_prefix) + ".gorgeous";
}

std::string PartitionPath(const std::string &index_prefix) {
  return index_prefix + "_partition.bin";
}

void CopyFileOrThrow(const std::string &from, const std::string &to) {
  const fs::path from_path = fs::path(from).lexically_normal();
  const fs::path to_path = fs::path(to).lexically_normal();
  if (from_path == to_path) {
    return;
  }
  std::error_code same_file_ec;
  if (fs::exists(from_path) && fs::exists(to_path) && fs::equivalent(from_path, to_path, same_file_ec)) {
    return;
  }
  std::error_code ec;
  const bool copied = fs::copy_file(from_path, to_path, fs::copy_options::overwrite_existing, ec);
  if (!copied || ec) {
    throw std::runtime_error("failed to copy file from " + from + " to " + to);
  }
}

}  // namespace

void ActivatePipeannLayout(const std::string &index_prefix,
                           PipeannLayout layout,
                           const std::string &gp_file_path) {
  const std::string disk_index_path = DiskIndexPath(index_prefix);
  const std::string equal_layout_path = EqualLayoutPath(index_prefix);
  const std::string gorgeous_layout_path = GorgeousLayoutPath(index_prefix);
  const std::string partition_path = PartitionPath(index_prefix);

  switch (layout) {
    case PipeannLayout::kEqual: {
      if (!fs::exists(equal_layout_path)) {
        throw std::runtime_error("missing PipeANN equal layout file: " + equal_layout_path);
      }
      CopyFileOrThrow(equal_layout_path, disk_index_path);
      std::error_code ec;
      fs::remove(partition_path, ec);
      return;
    }
    case PipeannLayout::kGorgeous: {
      if (!fs::exists(gorgeous_layout_path)) {
        throw std::runtime_error("missing PipeANN gorgeous layout file: " + gorgeous_layout_path);
      }
      if (gp_file_path.empty()) {
        throw std::runtime_error("gp_partition_path is required for gorgeous layout activation");
      }
      if (!fs::exists(gp_file_path)) {
        throw std::runtime_error("missing GP partition file: " + gp_file_path);
      }
      CopyFileOrThrow(gorgeous_layout_path, disk_index_path);
      CopyFileOrThrow(gp_file_path, partition_path);
      return;
    }
  }

  throw std::runtime_error("unsupported PipeANN layout");
}

}  // namespace hybrid::pipeann_parity
