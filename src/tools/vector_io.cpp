#include "tools/vector_io.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hybrid {

namespace {

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
  ValidateNonEmptyVectors(vectors, "text vector file does not contain any vectors");
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
  std::vector<float> flat(static_cast<size_t>(num_points) * dim);
  in.read(reinterpret_cast<char *>(flat.data()),
          static_cast<std::streamsize>(flat.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("failed to read bin vector payload");
  }
  char extra = 0;
  if (in.read(&extra, 1)) {
    throw std::runtime_error("bin vector file has unexpected trailing bytes");
  }

  std::vector<std::vector<float>> vectors(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    vectors[i].assign(flat.begin() + static_cast<size_t>(i) * dim, flat.begin() + static_cast<size_t>(i + 1) * dim);
  }
  return vectors;
}

}  // namespace hybrid
