#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gorgeous_layout.h"
#include "quant/approx_distance.h"
#include "search/graph_cache.h"

namespace pipeann {
struct Attributes;
struct Selector;
}  // namespace pipeann

namespace hybrid {

struct SearchConfig {
  uint32_t top_k = 5;
  uint32_t beam_width = 4;
  uint32_t l_search = 12;
  uint32_t l_pool = 0;
  uint32_t mem_l = 0;
  uint64_t graph_cache_budget_bytes = 0;
  GraphCacheBuildPolicy graph_cache_policy = GraphCacheBuildPolicy::kEntryBfs;
  uint32_t refine_k = 0;
  float refine_ratio = 0.0f;
  bool defer_exact_until_refinement = false;
  uint32_t scheduler_policy_limit = 0;
  enum class SchedulerPolicy {
    kConservative = 0,
    kBounded = 1,
    kAggressive = 2,
  } scheduler_policy = SchedulerPolicy::kConservative;
  enum class DynamicBeamPolicy {
    kAdaptive = 0,
    kFixed = 1,
  } dynamic_beam_policy = DynamicBeamPolicy::kAdaptive;
  float range_partial = std::numeric_limits<float>::infinity();
};

const char *SchedulerPolicyName(SearchConfig::SchedulerPolicy policy);
SearchConfig::SchedulerPolicy ParseSchedulerPolicy(const std::string &text);
const char *DynamicBeamPolicyName(SearchConfig::DynamicBeamPolicy policy);
SearchConfig::DynamicBeamPolicy ParseDynamicBeamPolicy(const std::string &text);

enum class FilterSearchMode {
  kAuto = 0,
  kPreFilter = 1,
  kPostFilter = 2,
  kInFilter = 3,
};

struct SearchStats {
  uint64_t async_reads = 0;
  uint64_t pages_completed = 0;
  uint64_t resident_expansions = 0;
  uint64_t bytes_read = 0;
  uint64_t page_resident_hits = 0;
  uint64_t graph_replicated_hits = 0;
  uint64_t graph_cache_hits = 0;
  uint64_t graph_cache_misses = 0;
  uint64_t graph_cache_expansions = 0;
  uint64_t graph_cache_avoided_reads = 0;
  uint64_t graph_cache_resident_bytes = 0;
  uint64_t graph_cache_entries = 0;
  uint64_t graph_cache_build_page_reads = 0;
  uint64_t exact_from_page = 0;
  uint64_t exact_from_payload = 0;
  uint64_t refinement_candidates = 0;
  uint64_t refinement_reads = 0;
  uint64_t approximate_candidates = 0;
  uint64_t refinement_bound = 0;
  uint64_t refinement_already_exact = 0;
  uint64_t refinement_exactified = 0;
  uint64_t deferred_exact_candidates = 0;
  uint64_t read_hits_in_pool = 0;
  uint64_t read_waste_out_of_pool = 0;
  uint64_t max_inflight_reads = 0;
  uint64_t max_beam_width = 0;
  uint64_t beam_width_increases = 0;
  uint64_t scheduler_policy_limit = 0;
  uint64_t scheduler_pending_max = 0;
  uint64_t scheduler_ready_unexpanded_max = 0;
  uint64_t scheduler_limit_hits = 0;
  uint64_t poll_calls = 0;
  uint64_t drain_calls = 0;
  uint64_t exact_distance_evals = 0;
  uint64_t approx_distance_evals = 0;
  uint64_t n_ios = 0;
  uint64_t n_cmps = 0;
  uint64_t n_hops = 0;
  uint64_t cpu_us = 0;
  uint64_t io_us = 0;
  uint64_t cpu_us1 = 0;
  uint64_t cpu_us2 = 0;
  uint64_t io_us1 = 0;
  uint64_t total_us = 0;
  bool range_stop = false;
  FilterSearchMode selected_filter_mode = FilterSearchMode::kAuto;
  uint64_t estimated_filter_reads[4] = {};
  uint64_t estimated_filter_cmps[4] = {};
  uint64_t filter_reads[4] = {};
  uint64_t filter_accessed_vectors[4] = {};
  uint64_t filter_false_positives[4] = {};
  uint64_t filter_io_us[4] = {};
  uint64_t filter_cpu_us[4] = {};
  uint64_t filter_io_us1[4] = {};
};

struct SearchResult {
  uint32_t id = 0;
  float distance = 0.0f;
};

struct FilterSearchOptions {
  FilterSearchMode mode = FilterSearchMode::kAuto;
  pipeann::Selector *selector = nullptr;
  const pipeann::Attributes *query_attrs = nullptr;
  uint32_t l_max = 0;
};

class PipelinedGraphReplicatedSearcher {
 public:
  explicit PipelinedGraphReplicatedSearcher(const IndexReader &index);

  std::vector<SearchResult> PipeSearch(const std::vector<float> &query,
                                       const SearchConfig &config,
                                       SearchStats *stats = nullptr) const;
  std::vector<SearchResult> PipeSearch(const std::vector<float> &query,
                                       const SearchConfig &config,
                                       ApproxDistanceKind approx_kind,
                                       SearchStats *stats,
                                       const std::string &pq_codebook_path = {},
                                       const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> Search(const std::vector<float> &query,
                                   const SearchConfig &config,
                                   SearchStats *stats = nullptr) const;
  std::vector<SearchResult> Search(const std::vector<float> &query,
                                   const SearchConfig &config,
                                   ApproxDistanceKind approx_kind,
                                   SearchStats *stats,
                                   const std::string &pq_codebook_path = {},
                                   const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> Search(const std::vector<float> &query,
                                   const SearchConfig &config,
                                   std::unique_ptr<IApproximateDistanceComputer> approx_distance,
                                   SearchStats *stats) const;
  std::vector<SearchResult> RangeSearch(const std::vector<float> &query,
                                        const SearchConfig &config,
                                        float range,
                                        SearchStats *stats = nullptr) const;
  std::vector<SearchResult> RangeSearch(const std::vector<float> &query,
                                        const SearchConfig &config,
                                        float range,
                                        ApproxDistanceKind approx_kind,
                                        SearchStats *stats,
                                        const std::string &pq_codebook_path = {},
                                        const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> FilterSearch(const std::vector<float> &query,
                                         const SearchConfig &config,
                                         const FilterSearchOptions &filter,
                                         SearchStats *stats = nullptr) const;
  std::vector<SearchResult> FilterSearch(const std::vector<float> &query,
                                         const SearchConfig &config,
                                         const FilterSearchOptions &filter,
                                         ApproxDistanceKind approx_kind,
                                         SearchStats *stats,
                                         const std::string &pq_codebook_path = {},
                                         const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> PreFilterSearch(const std::vector<float> &query,
                                            const SearchConfig &config,
                                            pipeann::Selector *selector,
                                            const pipeann::Attributes &query_attrs,
                                            uint32_t l_max = 0,
                                            SearchStats *stats = nullptr) const;
  std::vector<SearchResult> PreFilterSearch(const std::vector<float> &query,
                                            const SearchConfig &config,
                                            pipeann::Selector *selector,
                                            const pipeann::Attributes &query_attrs,
                                            uint32_t l_max,
                                            ApproxDistanceKind approx_kind,
                                            SearchStats *stats,
                                            const std::string &pq_codebook_path = {},
                                            const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> PostFilterSearch(const std::vector<float> &query,
                                             const SearchConfig &config,
                                             pipeann::Selector *selector,
                                             const pipeann::Attributes &query_attrs,
                                             uint32_t l_max = 0,
                                             SearchStats *stats = nullptr) const;
  std::vector<SearchResult> PostFilterSearch(const std::vector<float> &query,
                                             const SearchConfig &config,
                                             pipeann::Selector *selector,
                                             const pipeann::Attributes &query_attrs,
                                             uint32_t l_max,
                                             ApproxDistanceKind approx_kind,
                                             SearchStats *stats,
                                             const std::string &pq_codebook_path = {},
                                             const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> InFilterSearch(const std::vector<float> &query,
                                           const SearchConfig &config,
                                           pipeann::Selector *selector,
                                           const pipeann::Attributes &query_attrs,
                                           uint32_t l_max = 0,
                                           SearchStats *stats = nullptr) const;
  std::vector<SearchResult> InFilterSearch(const std::vector<float> &query,
                                           const SearchConfig &config,
                                           pipeann::Selector *selector,
                                           const pipeann::Attributes &query_attrs,
                                           uint32_t l_max,
                                           ApproxDistanceKind approx_kind,
                                           SearchStats *stats,
                                           const std::string &pq_codebook_path = {},
                                           const std::string &pq_codes_path = {}) const;
  std::vector<SearchResult> AutoFilterSearch(const std::vector<float> &query,
                                             const SearchConfig &config,
                                             pipeann::Selector *selector,
                                             const pipeann::Attributes &query_attrs,
                                             uint32_t l_max = 0,
                                             SearchStats *stats = nullptr) const;
  std::vector<SearchResult> AutoFilterSearch(const std::vector<float> &query,
                                             const SearchConfig &config,
                                             pipeann::Selector *selector,
                                             const pipeann::Attributes &query_attrs,
                                             uint32_t l_max,
                                             ApproxDistanceKind approx_kind,
                                             SearchStats *stats,
                                             const std::string &pq_codebook_path = {},
                                             const std::string &pq_codes_path = {}) const;

 private:
  const GraphAdjacencyCache *GraphCacheForConfig(const SearchConfig &config) const;

  const IndexReader &index_;
  mutable std::unique_ptr<GraphAdjacencyCache> graph_cache_;
  mutable uint64_t graph_cache_budget_bytes_ = std::numeric_limits<uint64_t>::max();
  mutable GraphCacheBuildPolicy graph_cache_policy_ = GraphCacheBuildPolicy::kEntryBfs;
};

}  // namespace hybrid
