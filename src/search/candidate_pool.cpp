#include "search/candidate_pool.h"

#include <algorithm>

namespace hybrid {

namespace {

void SortNeighbors(std::vector<Neighbor> *neighbors) {
  std::sort(neighbors->begin(), neighbors->end(), [](const Neighbor &lhs, const Neighbor &rhs) {
    return lhs < rhs;
  });
}

}  // namespace

bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
  if (lhs.distance == rhs.distance) {
    return lhs.id < rhs.id;
  }
  return lhs.distance < rhs.distance;
}

bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
  return rhs < lhs;
}

Neighbor *FindNeighbor(std::vector<Neighbor> *neighbors, uint32_t id) {
  for (auto &neighbor : *neighbors) {
    if (neighbor.id == id) {
      return &neighbor;
    }
  }
  return nullptr;
}

const Neighbor *FindNeighbor(const std::vector<Neighbor> &neighbors, uint32_t id) {
  for (const auto &neighbor : neighbors) {
    if (neighbor.id == id) {
      return &neighbor;
    }
  }
  return nullptr;
}

size_t InsertIntoPool(std::vector<Neighbor> *neighbors, size_t limit, const Neighbor &neighbor) {
  Neighbor *existing = FindNeighbor(neighbors, neighbor.id);
  if (existing != nullptr) {
    if (neighbor.distance < existing->distance) {
      existing->distance = neighbor.distance;
      SortNeighbors(neighbors);
    }
    for (size_t i = 0; i < neighbors->size(); ++i) {
      if ((*neighbors)[i].id == neighbor.id) {
        return i;
      }
    }
    return neighbors->size();
  }

  if (neighbors->size() >= limit && !neighbors->empty() && !(neighbor < neighbors->back())) {
    return neighbors->size();
  }

  neighbors->push_back(neighbor);
  SortNeighbors(neighbors);
  if (neighbors->size() > limit) {
    neighbors->pop_back();
  }
  for (size_t i = 0; i < neighbors->size(); ++i) {
    if ((*neighbors)[i].id == neighbor.id) {
      return i;
    }
  }
  return neighbors->size();
}

}  // namespace hybrid
