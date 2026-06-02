#pragma once

#include <string>

namespace hybrid::pipeann_parity {

enum class PipeannLayout {
  kEqual = 0,
  kGorgeous = 1,
};

void ActivatePipeannLayout(const std::string &index_prefix,
                           PipeannLayout layout,
                           const std::string &gp_file_path = {});

}  // namespace hybrid::pipeann_parity
