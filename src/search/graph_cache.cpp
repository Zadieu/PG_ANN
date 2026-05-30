#include "search/graph_cache.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace hybrid {
namespace {

uint64_t EstimateNodeBytes(const CachedGraphNode &node) {
  return sizeof(CachedGraphNode) +
         static_cast<uint64_t>(node.neighbors.size() + node.dense_neighbors.size()) * sizeof(uint32_t);
}

uint32_t FindNodeSlot(const PageView &page, uint32_t id) {
  for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
    if (page.LayoutAt(slot) == id) {
      return slot;
    }
  }
  return std::numeric_limits<uint32_t>::max();
}

void EnqueueNeighbors(const CachedGraphNode &node, std::vector<uint8_t> *queued, std::deque<uint32_t> *frontier) {
  const auto enqueue = [&](uint32_t id) {
    if (id >= queued->size() || queued->at(id) != 0) {
      return;
    }
    queued->at(id) = 1;
    frontier->push_back(id);
  };
  for (uint32_t id : node.neighbors) {
    enqueue(id);
  }
  for (uint32_t id : node.dense_neighbors) {
    enqueue(id);
  }
}

CachedGraphNode CopyCachedNode(const DiskNodeView &node) {
  CachedGraphNode cached;
  cached.id = node.id;
  if (node.nbrs != nullptr && node.nnbrs != 0) {
    cached.neighbors.assign(node.nbrs, node.nbrs + node.nnbrs);
  }
  if (node.dense_nbrs != nullptr && node.n_dense_nbrs != 0) {
    cached.dense_neighbors.assign(node.dense_nbrs, node.dense_nbrs + node.n_dense_nbrs);
  }
  return cached;
}

}  // namespace

uint64_t CachedGraphNode::PayloadBytes() const {
  return EstimateNodeBytes(*this);
}

bool GraphAdjacencyCache::TryInsert(CachedGraphNode cached) {
  const uint64_t node_bytes = cached.PayloadBytes();
  if (resident_bytes_ + node_bytes > budget_bytes_) {
    return false;
  }
  resident_bytes_ += node_bytes;
  nodes_.emplace(cached.id, std::move(cached));
  return true;
}

GraphAdjacencyCache GraphAdjacencyCache::Build(const IndexReader &index,
                                               uint64_t budget_bytes,
                                               GraphCacheBuildPolicy policy) {
  switch (policy) {
    case GraphCacheBuildPolicy::kEntryBfs:
      return BuildEntryBfs(index, budget_bytes);
    case GraphCacheBuildPolicy::kPageLayout:
      return BuildPageLayout(index, budget_bytes);
  }
  throw std::runtime_error("unsupported graph cache build policy");
}

GraphAdjacencyCache GraphAdjacencyCache::BuildEntryBfs(const IndexReader &index, uint64_t budget_bytes) {
  GraphAdjacencyCache cache;
  cache.policy_ = GraphCacheBuildPolicy::kEntryBfs;
  cache.budget_bytes_ = budget_bytes;
  if (budget_bytes == 0 || index.search_metadata().num_points == 0) {
    return cache;
  }

  std::ifstream in(index.index_path(), std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open index file while building graph adjacency cache");
  }

  std::vector<uint8_t> queued(index.search_metadata().num_points, 0);
  std::deque<uint32_t> frontier;
  const uint32_t entry_id = index.search_metadata().entry_id;
  if (entry_id >= queued.size()) {
    throw std::runtime_error("graph cache entry id is out of range");
  }
  queued[entry_id] = 1;
  frontier.push_back(entry_id);

  std::vector<char> page_bytes(index.search_metadata().page_size);
  while (!frontier.empty()) {
    const uint32_t seed_id = frontier.front();
    frontier.pop_front();
    if (seed_id >= index.search_metadata().num_points || cache.nodes_.find(seed_id) != cache.nodes_.end()) {
      continue;
    }

    const uint32_t page_id = index.PageForNode(seed_id);
    in.seekg(static_cast<std::streamoff>(index.PageOffset(page_id)), std::ios::beg);
    in.read(page_bytes.data(), static_cast<std::streamsize>(page_bytes.size()));
    if (!in) {
      throw std::runtime_error("failed to read index page while building graph adjacency cache");
    }
    ++cache.build_page_reads_;

    const PageView page = index.ViewPage(page_id, page_bytes);
    const uint32_t first_slot = FindNodeSlot(page, seed_id);
    if (first_slot == std::numeric_limits<uint32_t>::max()) {
      continue;
    }

    std::vector<uint32_t> slots;
    slots.reserve(page.layout_size);
    slots.push_back(first_slot);
    for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
      if (slot != first_slot) {
        slots.push_back(slot);
      }
    }

    for (uint32_t slot : slots) {
      const DiskNodeView node = index.ViewNode(page, slot);
      if (node.id >= index.search_metadata().num_points || cache.nodes_.find(node.id) != cache.nodes_.end()) {
        continue;
      }

      CachedGraphNode cached = CopyCachedNode(node);
      if (!cache.TryInsert(CachedGraphNode(cached))) {
        return cache;
      }

      EnqueueNeighbors(cached, &queued, &frontier);
      if (cache.resident_bytes_ >= budget_bytes) {
        return cache;
      }
    }
  }

  return cache;
}

GraphAdjacencyCache GraphAdjacencyCache::BuildPageLayout(const IndexReader &index, uint64_t budget_bytes) {
  GraphAdjacencyCache cache;
  cache.policy_ = GraphCacheBuildPolicy::kPageLayout;
  cache.budget_bytes_ = budget_bytes;
  if (budget_bytes == 0 || index.search_metadata().num_points == 0) {
    return cache;
  }

  std::ifstream in(index.index_path(), std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open index file while building graph adjacency cache");
  }

  std::vector<char> page_bytes(index.search_metadata().page_size);
  for (uint32_t page_id = 0; page_id < index.search_metadata().num_pages; ++page_id) {
    in.seekg(static_cast<std::streamoff>(index.PageOffset(page_id)), std::ios::beg);
    in.read(page_bytes.data(), static_cast<std::streamsize>(page_bytes.size()));
    if (!in) {
      throw std::runtime_error("failed to read index page while building graph adjacency cache");
    }
    ++cache.build_page_reads_;

    const PageView page = index.ViewPage(page_id, page_bytes);
    for (uint32_t slot = 0; slot < page.layout_size; ++slot) {
      const DiskNodeView node = index.ViewNode(page, slot);
      if (node.id >= index.search_metadata().num_points || cache.nodes_.find(node.id) != cache.nodes_.end()) {
        continue;
      }
      if (!cache.TryInsert(CopyCachedNode(node))) {
        return cache;
      }
      if (cache.resident_bytes_ >= budget_bytes) {
        return cache;
      }
    }
  }
  return cache;
}

const CachedGraphNode *GraphAdjacencyCache::Find(uint32_t id) const {
  const auto it = nodes_.find(id);
  return it == nodes_.end() ? nullptr : &it->second;
}

const char *GraphCacheBuildPolicyName(GraphCacheBuildPolicy policy) {
  switch (policy) {
    case GraphCacheBuildPolicy::kEntryBfs:
      return "entry_bfs";
    case GraphCacheBuildPolicy::kPageLayout:
      return "page_layout";
  }
  throw std::runtime_error("unsupported graph cache build policy");
}

GraphCacheBuildPolicy ParseGraphCacheBuildPolicy(const std::string &text) {
  if (text == "entry_bfs") {
    return GraphCacheBuildPolicy::kEntryBfs;
  }
  if (text == "page_layout") {
    return GraphCacheBuildPolicy::kPageLayout;
  }
  throw std::runtime_error("unsupported graph cache policy: " + text);
}

}  // namespace hybrid
