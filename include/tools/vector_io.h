#pragma once

#include <string>
#include <vector>

namespace hybrid {

std::vector<std::vector<float>> LoadTextVectors(const std::string &path);
std::vector<std::vector<float>> LoadFvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBvecsVectors(const std::string &path);
std::vector<std::vector<float>> LoadBinVectors(const std::string &path);

}  // namespace hybrid
