#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gorgeous_layout.h"

namespace hybrid {

enum class GraphCacheBuildPolicy {
  kEntryBfs = 0,
  kPageLayout = 1,
};

struct CachedGraphNode {
  uint32_t id = 0;
  std::vector<uint32_t> neighbors;
  std::vector<uint32_t> dense_neighbors;

  uint64_t PayloadBytes() const;
};

class GraphAdjacencyCache {
 public:
  static GraphAdjacencyCache Build(const IndexReader &index, uint64_t budget_bytes, GraphCacheBuildPolicy policy);
  static GraphAdjacencyCache BuildEntryBfs(const IndexReader &index, uint64_t budget_bytes);
  static GraphAdjacencyCache BuildPageLayout(const IndexReader &index, uint64_t budget_bytes);

  const CachedGraphNode *Find(uint32_t id) const;
  GraphCacheBuildPolicy policy() const { return policy_; }
  uint64_t budget_bytes() const { return budget_bytes_; }
  uint64_t resident_bytes() const { return resident_bytes_; }
  uint64_t build_page_reads() const { return build_page_reads_; }
  size_t entries() const { return nodes_.size(); }
  bool empty() const { return nodes_.empty(); }

 private:
  bool TryInsert(CachedGraphNode cached);

  GraphCacheBuildPolicy policy_ = GraphCacheBuildPolicy::kEntryBfs;
  uint64_t budget_bytes_ = 0;
  uint64_t resident_bytes_ = 0;
  uint64_t build_page_reads_ = 0;
  std::unordered_map<uint32_t, CachedGraphNode> nodes_;
};

const char *GraphCacheBuildPolicyName(GraphCacheBuildPolicy policy);
GraphCacheBuildPolicy ParseGraphCacheBuildPolicy(const std::string &text);

}  // namespace hybrid
