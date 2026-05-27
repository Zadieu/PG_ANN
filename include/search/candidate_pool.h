#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hybrid {

struct Neighbor {
  uint32_t id = 0;
  float distance = 0.0f;
  bool flag = true;
  bool visited = false;
  bool is_member = false;
  bool exact = false;
};

bool operator<(const Neighbor &lhs, const Neighbor &rhs);
bool operator>(const Neighbor &lhs, const Neighbor &rhs);

Neighbor *FindNeighbor(std::vector<Neighbor> *neighbors, uint32_t id);
const Neighbor *FindNeighbor(const std::vector<Neighbor> &neighbors, uint32_t id);
size_t InsertIntoPool(std::vector<Neighbor> *neighbors, size_t limit, const Neighbor &neighbor);

}  // namespace hybrid
