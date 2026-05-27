#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gorgeous_layout.h"
#include "io/page_reader.h"
#include "pipeline_search.h"
#include "quant/approx_distance.h"
#include "search/candidate_pool.h"

namespace hybrid {

class SearchSession {
 public:
  SearchSession(const IndexReader &index,
                const std::vector<float> &query,
                const SearchConfig &config,
                SearchStats *stats,
                std::unique_ptr<IPageReader> page_reader,
                ApproxDistanceKind approx_kind,
                uint32_t l_pool = 0,
                bool use_dense_neighbors = false,
                bool range_mode = false,
                std::function<bool(uint32_t)> is_member_approx = {},
                std::function<bool(uint32_t, const DiskNodeView &)> is_member = {},
                const std::string &pq_codebook_path = {},
                const std::string &pq_codes_path = {});
  SearchSession(const IndexReader &index,
                const std::vector<float> &query,
                const SearchConfig &config,
                SearchStats *stats,
                std::unique_ptr<IPageReader> page_reader,
                std::unique_ptr<IApproximateDistanceComputer> approx_distance,
                uint32_t l_pool = 0,
                bool use_dense_neighbors = false,
                bool range_mode = false,
                std::function<bool(uint32_t)> is_member_approx = {},
                std::function<bool(uint32_t, const DiskNodeView &)> is_member = {});

  std::vector<SearchResult> Run();

 private:
  static constexpr uint32_t kMaxRequestSlots = 128;

  enum class ApproxExecutionMode {
    kPipeannPq,
    kFullPrecision,
    kExternal,
  };

  struct BufferedNode {
    uint32_t request_slot = 0;
    uint32_t page_id = 0;
    uint32_t layout_index = 0;
  };

  struct InflightRead {
    uint32_t candidate_id = 0;
    uint32_t page_id = 0;
    uint32_t request_slot = 0;
  };

  struct RequestSlot {
    uint32_t page_id = std::numeric_limits<uint32_t>::max();
    uint32_t candidate_id = std::numeric_limits<uint32_t>::max();
    bool in_flight = false;
    bool completed = false;
    bool resident = false;
  };

  struct QueryBufferState {
    std::vector<char> sector_scratch;
    std::vector<PageReadRequest> reqs;
    std::vector<RequestSlot> request_slots;
    std::vector<uint8_t> visited;
    std::vector<uint8_t> page_visited;
    std::vector<uint32_t> neighbor_ids;
    std::vector<float> neighbor_distances;
    std::unordered_map<uint32_t, BufferedNode> id_buf_map;
    uint32_t sector_idx = 0;

    void Reset(uint32_t page_size, uint32_t num_points, uint32_t num_pages) {
      sector_idx = 0;
      sector_scratch.assign(static_cast<size_t>(kMaxRequestSlots) * page_size, 0);
      reqs.assign(kMaxRequestSlots, PageReadRequest{});
      request_slots.assign(kMaxRequestSlots, RequestSlot{});
      visited.assign(num_points, 0);
      page_visited.assign(num_pages, 0);
      neighbor_ids.clear();
      neighbor_distances.clear();
      id_buf_map.clear();
    }

    char *SlotBuffer(uint32_t slot, uint32_t page_size) {
      return sector_scratch.data() + static_cast<size_t>(slot) * page_size;
    }
  };

  struct PollOutcome {
    size_t registered = 0;
    size_t n_in = 0;
    size_t n_out = 0;
  };

  void SeedWarmStart();
  size_t ScheduleBestReadRequests(size_t max_reads);
  PollOutcome PollCompletedPages(bool require_all);
  bool ProcessBufferedCandidate();
  size_t CalcBestNode();
  void DrainAndRefine();
  bool ShouldTerminate();
  bool SubmitNeighborRead(Neighbor *neighbor);
  uint32_t AcquireRequestSlot();
  void ReleaseRequestSlot(uint32_t request_slot);
  void EvictSlot(uint32_t request_slot);
  PollOutcome ConsumeCompletedInflight(bool require_all);
  void StoreCompletedPages(const std::vector<PageReadCompletion> &completed_pages);
  void RegisterPage(uint32_t request_slot, uint32_t page_id);
  void MaybeExactify(Neighbor *candidate);
  float CurrentRetsetBound() const;
  void ComputeApproximateDistances(const uint32_t *ids, size_t count, float *distances);
  float ApproxDistance(uint32_t id);
  void VisitExpandedNodeNeighbors(const DiskNodeView &node);
  std::vector<SearchResult> CollectResults() const;
  size_t retset_limit() const;

  const IndexReader &index_;
  const std::vector<float> &query_;
  SearchConfig config_{};
  SearchStats *stats_ = nullptr;
  std::unique_ptr<IPageReader> page_reader_;
  ApproxExecutionMode approx_mode_ = ApproxExecutionMode::kPipeannPq;
  FullPrecisionDistanceComputer full_precision_distance_;
  PipeannProductQuantization pipeann_pq_;
  std::vector<float> pq_query_distance_table_;
  std::unique_ptr<IApproximateDistanceComputer> external_approx_distance_;
  std::vector<Neighbor> retset_;
  std::vector<Neighbor> full_retset_;
  std::deque<InflightRead> on_flight_ios_;
  QueryBufferState query_buffer_;
  std::vector<PageReadCompletion> completed_pages_scratch_;
  std::unordered_map<uint32_t, uint32_t> completed_by_slot_;
  uint32_t l_pool_ = 0;
  uint32_t current_beam_width_ = 0;
  uint32_t max_marker_seen_ = 0;
  uint32_t beam_hits_ = 0;
  uint32_t beam_total_ = 0;
  bool use_dense_neighbors_ = false;
  bool range_mode_ = false;
  std::function<bool(uint32_t)> is_member_approx_;
  std::function<bool(uint32_t, const DiskNodeView &)> is_member_;
};

}  // namespace hybrid
