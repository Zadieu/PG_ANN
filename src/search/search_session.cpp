#include "search/search_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace hybrid {

SearchSession::SearchSession(const IndexReader &index,
                             const std::vector<float> &query,
                             const SearchConfig &config,
                             SearchStats *stats,
                             std::unique_ptr<IPageReader> page_reader,
                             ApproxDistanceKind approx_kind,
                             uint32_t l_pool,
                             bool use_dense_neighbors,
                             bool range_mode,
                             std::function<bool(uint32_t)> is_member_approx,
                             std::function<bool(uint32_t, const DiskNodeView &)> is_member,
                             const std::string &pq_codebook_path,
                             const std::string &pq_codes_path,
                             const GraphAdjacencyCache *graph_cache)
    : index_(index),
      query_(query),
      config_(config),
      stats_(stats),
      page_reader_(std::move(page_reader)),
      graph_cache_(graph_cache),
      approx_mode_(approx_kind == ApproxDistanceKind::kProductQuantization ? ApproxExecutionMode::kPipeannPq
                                                                           : ApproxExecutionMode::kFullPrecision),
      full_precision_distance_(index),
      pipeann_pq_(index),
      l_pool_(l_pool == 0 ? config.l_search : l_pool),
      use_dense_neighbors_(use_dense_neighbors),
      range_mode_(range_mode),
      is_member_approx_(std::move(is_member_approx)),
      is_member_(std::move(is_member)) {
  if (!is_member_approx_) {
    is_member_approx_ = [](uint32_t) { return true; };
  }
  if (!is_member_) {
    is_member_ = [](uint32_t, const DiskNodeView &) { return true; };
  }
  if (approx_mode_ == ApproxExecutionMode::kPipeannPq) {
    pipeann_pq_.Load(pq_codebook_path, pq_codes_path);
    pipeann_pq_.InitializeQuery(query_, &pq_query_distance_table_);
  } else {
    full_precision_distance_.BeginQuery(query_);
  }
  const uint32_t entry_id = index_.search_metadata().entry_id;
  current_beam_width_ = config_.dynamic_beam_policy == SearchConfig::DynamicBeamPolicy::kFixed
                            ? config_.beam_width
                            : std::min<uint32_t>(config_.beam_width, 4);
  completed_pages_scratch_.reserve(config_.beam_width);
  query_buffer_.Reset(index_.search_metadata().page_size,
                      index_.search_metadata().num_points,
                      index_.search_metadata().num_pages);
  SeedWarmStart();
  if (!retset_.empty()) {
    return;
  }

  Neighbor entry;
  entry.id = entry_id;
  entry.distance = ApproxDistance(entry_id);
  query_buffer_.visited.at(entry_id) = 1;
  InsertIntoPool(&retset_, retset_limit(), entry);
}

SearchSession::SearchSession(const IndexReader &index,
                             const std::vector<float> &query,
                             const SearchConfig &config,
                             SearchStats *stats,
                             std::unique_ptr<IPageReader> page_reader,
                             std::unique_ptr<IApproximateDistanceComputer> approx_distance,
                             uint32_t l_pool,
                             bool use_dense_neighbors,
                             bool range_mode,
                             std::function<bool(uint32_t)> is_member_approx,
                             std::function<bool(uint32_t, const DiskNodeView &)> is_member,
                             const GraphAdjacencyCache *graph_cache)
    : index_(index),
      query_(query),
      config_(config),
      stats_(stats),
      page_reader_(std::move(page_reader)),
      graph_cache_(graph_cache),
      approx_mode_(ApproxExecutionMode::kExternal),
      full_precision_distance_(index),
      pipeann_pq_(index),
      external_approx_distance_(std::move(approx_distance)),
      l_pool_(l_pool == 0 ? config.l_search : l_pool),
      use_dense_neighbors_(use_dense_neighbors),
      range_mode_(range_mode),
      is_member_approx_(std::move(is_member_approx)),
      is_member_(std::move(is_member)) {
  if (external_approx_distance_ == nullptr) {
    throw std::runtime_error("approximate distance computer must not be null");
  }
  if (!is_member_approx_) {
    is_member_approx_ = [](uint32_t) { return true; };
  }
  if (!is_member_) {
    is_member_ = [](uint32_t, const DiskNodeView &) { return true; };
  }
  external_approx_distance_->BeginQuery(query_);
  const uint32_t entry_id = index_.search_metadata().entry_id;
  current_beam_width_ = config_.dynamic_beam_policy == SearchConfig::DynamicBeamPolicy::kFixed
                            ? config_.beam_width
                            : std::min<uint32_t>(config_.beam_width, 4);
  completed_pages_scratch_.reserve(config_.beam_width);
  query_buffer_.Reset(index_.search_metadata().page_size,
                      index_.search_metadata().num_points,
                      index_.search_metadata().num_pages);
  SeedWarmStart();
  if (!retset_.empty()) {
    return;
  }

  Neighbor entry;
  entry.id = entry_id;
  entry.distance = ApproxDistance(entry_id);
  query_buffer_.visited.at(entry_id) = 1;
  InsertIntoPool(&retset_, retset_limit(), entry);
}

std::vector<SearchResult> SearchSession::Run() {
  const auto query_start = std::chrono::steady_clock::now();
  if (stats_ != nullptr) {
    stats_->range_stop = false;
    stats_->max_beam_width = std::max<uint64_t>(stats_->max_beam_width, current_beam_width_);
    stats_->scheduler_policy_limit =
        std::max<uint64_t>(stats_->scheduler_policy_limit, EffectiveSchedulerPolicyLimit());
    if (graph_cache_ != nullptr) {
      stats_->graph_cache_resident_bytes = graph_cache_->resident_bytes();
      stats_->graph_cache_entries = graph_cache_->entries();
      stats_->graph_cache_build_page_reads = graph_cache_->build_page_reads();
    }
  }

  if (page_reader_->NumInflight() < current_beam_width_) {
    ScheduleBestReadRequests(current_beam_width_ - page_reader_->NumInflight());
  }

  size_t marker = retset_.size();
  while (!ShouldTerminate()) {
    const PollOutcome poll = PollCompletedPages(false);
    if (page_reader_->NumInflight() < current_beam_width_) {
      ScheduleBestReadRequests(1);
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    marker = CalcBestNode();
    if (stats_ != nullptr) {
      stats_->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - cpu_start)
                            .count();
    }
    max_marker_seen_ =
        std::max<uint32_t>(max_marker_seen_, static_cast<uint32_t>(std::min(marker, static_cast<size_t>(UINT32_MAX))));


    if (config_.dynamic_beam_policy == SearchConfig::DynamicBeamPolicy::kAdaptive &&
        max_marker_seen_ >= 5 && poll.n_in + poll.n_out > 0) {
      beam_hits_ += static_cast<uint32_t>(poll.n_in);
      beam_total_ += static_cast<uint32_t>(poll.n_in + poll.n_out);
      constexpr double kWasteThreshold = 0.1;
      const double waste_ratio =
          beam_total_ == 0 ? 1.0 : static_cast<double>(beam_total_ - beam_hits_) / static_cast<double>(beam_total_);
      if (waste_ratio <= kWasteThreshold) {
        const uint32_t previous_beam_width = current_beam_width_;
        current_beam_width_ = std::min<uint32_t>(config_.beam_width, std::max<uint32_t>(4, current_beam_width_ + 1));
        if (stats_ != nullptr && current_beam_width_ > previous_beam_width) {
          ++stats_->beam_width_increases;
          stats_->max_beam_width = std::max<uint64_t>(stats_->max_beam_width, current_beam_width_);
        }
      }
    }

    if (page_reader_->NumInflight() == 0 && marker >= retset_.size()) {
      break;
    }
    if (poll.registered == 0 && marker >= retset_.size() && page_reader_->NumInflight() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  while (page_reader_->NumInflight() > 0) {
    PollCompletedPages(true);
  }
  const auto cpu_start = std::chrono::steady_clock::now();
  while (!ShouldTerminate() && ProcessReadyCandidate()) {
  }
  if (stats_ != nullptr) {
    stats_->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - cpu_start)
                          .count();
  }

  DrainAndRefine();
  if (stats_ != nullptr) {
    stats_->total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - query_start).count();
  }
  return CollectResults();
}

void SearchSession::SeedWarmStart() {
  const uint32_t warm_start = std::min<uint32_t>(config_.mem_l, index_.search_metadata().num_points);
  const uint32_t warm_limit = std::min<uint32_t>(warm_start, l_pool_);
  if (warm_start == 0) {
    return;
  }

  std::vector<Neighbor> warm_candidates;
  warm_candidates.reserve(warm_start);
  for (uint32_t id = 0; id < index_.search_metadata().num_points; ++id) {
    Neighbor seed;
    seed.id = id;
    seed.distance = ApproxDistance(id);
    InsertIntoPool(&warm_candidates, warm_start, seed);
  }

  for (const Neighbor &seed : warm_candidates) {
    query_buffer_.visited.at(seed.id) = 1;
    InsertIntoPool(&retset_, warm_limit, seed);
  }
}

size_t SearchSession::ScheduleBestReadRequests(size_t max_reads) {
  size_t scheduled = 0;
  size_t marker = 0;
  while (marker < retset_.size() &&
         scheduled < max_reads &&
         page_reader_->NumInflight() < current_beam_width_) {
    if (ShouldThrottleSubmissions()) {
      break;
    }
    Neighbor *candidate = nullptr;
    while (marker < retset_.size()) {
      Neighbor *next = &retset_[marker];
      if (!next->flag || query_buffer_.id_buf_map.find(next->id) != query_buffer_.id_buf_map.end()) {
        next->flag = false;
        ++marker;
        continue;
      }
      if (graph_cache_ != nullptr && graph_cache_->Find(next->id) != nullptr) {
        ++marker;
        continue;
      }
      const uint32_t next_page_id = index_.PageForNode(next->id);
      if (query_buffer_.page_visited.at(next_page_id) != 0) {
        next->flag = false;
        ++marker;
        continue;
      }
      candidate = next;
      ++marker;
      break;
    }
    if (candidate == nullptr) {
      break;
    }
    if (!SubmitNeighborRead(candidate, true)) {
      break;
    }
    ++scheduled;
  }
  return scheduled;
}

SearchSession::PollOutcome SearchSession::PollCompletedPages(bool require_all) {
  completed_pages_scratch_.clear();
  const auto io_start = std::chrono::steady_clock::now();
  if (stats_ != nullptr) {
    if (require_all) {
      ++stats_->drain_calls;
    } else {
      ++stats_->poll_calls;
    }
  }
  if (require_all) {
    page_reader_->Drain(&completed_pages_scratch_);
  } else {
    page_reader_->Poll(&completed_pages_scratch_);
  }
  if (stats_ != nullptr) {
    stats_->io_us += std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - io_start)
                         .count();
  }
  StoreCompletedPages(completed_pages_scratch_);
  return ConsumeCompletedInflight(require_all);
}

size_t SearchSession::CalcBestNode() {
  ProcessReadyCandidate();

  for (size_t i = 0; i < retset_.size(); ++i) {
    const Neighbor *candidate = &retset_[i];
    if (candidate->visited || !candidate->flag) {
      continue;
    }
    if (query_buffer_.id_buf_map.find(candidate->id) != query_buffer_.id_buf_map.end()) {
      continue;
    }
    if (graph_cache_ != nullptr && graph_cache_->Find(candidate->id) != nullptr) {
      continue;
    }
    return i;
  }
  return retset_.size();
}

bool SearchSession::ProcessReadyCandidate() {
  for (size_t i = 0; i < retset_.size(); ++i) {
    Neighbor *candidate = &retset_[i];
    if (candidate->visited) {
      continue;
    }
    auto buffered = query_buffer_.id_buf_map.find(candidate->id);
    if (buffered != query_buffer_.id_buf_map.end()) {
      const BufferedNode &buffered_node = buffered->second;
      if (buffered_node.request_slot >= query_buffer_.request_slots.size()) {
        throw std::runtime_error("buffered node points to an invalid request slot");
      }
      const RequestSlot &slot = query_buffer_.request_slots[buffered_node.request_slot];
      if (!slot.resident || slot.page_id != buffered_node.page_id) {
        throw std::runtime_error("buffered node points to a non-resident page slot");
      }

      candidate->flag = false;
      candidate->visited = true;
      if (config_.defer_exact_until_refinement) {
        if (stats_ != nullptr && !candidate->exact) {
          ++stats_->deferred_exact_candidates;
        }
      } else {
        MaybeExactify(candidate);
      }
      const char *slot_buf = query_buffer_.SlotBuffer(buffered_node.request_slot, index_.search_metadata().page_size);
      const PageView page = index_.ViewPage(buffered_node.page_id, slot_buf, index_.search_metadata().page_size);
      if (stats_ != nullptr) {
        ++stats_->resident_expansions;
        ++stats_->page_resident_hits;
        if (buffered_node.origin_candidate_id != std::numeric_limits<uint32_t>::max() &&
            buffered_node.origin_candidate_id != candidate->id) {
          ++stats_->graph_replicated_hits;
        }
        ++stats_->n_hops;
      }
      const DiskNodeView node = index_.ViewNode(page, buffered_node.layout_index);
      VisitExpandedNodeNeighbors(node);
      return true;
    }

    const CachedGraphNode *cached_node = graph_cache_ == nullptr ? nullptr : graph_cache_->Find(candidate->id);
    if (cached_node != nullptr) {
      candidate->flag = false;
      candidate->visited = true;
      if (stats_ != nullptr) {
        ++stats_->graph_cache_hits;
        ++stats_->graph_cache_expansions;
        ++stats_->graph_cache_avoided_reads;
        if (!candidate->exact) {
          ++stats_->deferred_exact_candidates;
        }
        ++stats_->n_hops;
      }
      VisitCachedNodeNeighbors(*cached_node);
      return true;
    }
  }
  return false;
}

void SearchSession::DrainAndRefine() {
  PollCompletedPages(true);

  const size_t refine_bound = RefinementBound();
  if (stats_ != nullptr) {
    stats_->approximate_candidates += retset_.size();
    stats_->refinement_candidates += refine_bound;
    stats_->refinement_bound += refine_bound;
  }
  for (size_t i = 0; i < refine_bound; ++i) {
    Neighbor *candidate = &retset_[i];
    if (candidate->exact) {
      if (stats_ != nullptr) {
        ++stats_->refinement_already_exact;
      }
      continue;
    }
    if (!candidate->exact && query_buffer_.id_buf_map.find(candidate->id) == query_buffer_.id_buf_map.end()) {
      if (SubmitNeighborRead(candidate) && stats_ != nullptr) {
        ++stats_->refinement_reads;
      }
    }
  }

  PollCompletedPages(true);

  const auto cpu_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < refine_bound; ++i) {
    if (MaybeExactify(&retset_[i]) && stats_ != nullptr) {
      ++stats_->refinement_exactified;
    }
  }
  if (stats_ != nullptr) {
    stats_->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - cpu_start)
                          .count();
  }
}

bool SearchSession::ShouldTerminate() {
  constexpr float kRangeEarlyStopFactor = 2.0f;
  const float approx_range_partial = config_.range_partial * kRangeEarlyStopFactor;
  uint32_t member_count = 0;
  int first_unvisited = -1;
  bool range_crossed = false;
  for (size_t i = 0; i < retset_.size(); ++i) {
    member_count +=
        (retset_[i].is_member || (config_.defer_exact_until_refinement && retset_[i].visited)) ? 1U : 0U;
    if (!retset_[i].visited) {
      first_unvisited = static_cast<int>(i);
      break;
    }
    if (retset_[i].distance > approx_range_partial) {
      range_crossed = true;
      break;
    }
  }
  if (range_crossed && stats_ != nullptr) {
    stats_->range_stop = true;
  }
  return member_count >= config_.l_search || first_unvisited == -1 || range_crossed;
}

bool SearchSession::ShouldThrottleSubmissions() {
  const size_t limit = EffectiveSchedulerPolicyLimit();
  if (limit == 0) {
    return false;
  }
  RecordSchedulerPendingStats();
  if (SchedulerPendingCount() < limit) {
    return false;
  }
  if (stats_ != nullptr) {
    ++stats_->scheduler_limit_hits;
  }
  return true;
}

bool SearchSession::SubmitNeighborRead(Neighbor *neighbor, bool count_graph_cache_miss) {
  if (neighbor == nullptr) {
    return false;
  }
  const uint32_t page_id = index_.PageForNode(neighbor->id);
  if (query_buffer_.page_visited.at(page_id) != 0) {
    return false;
  }
  const uint32_t request_slot = AcquireRequestSlot();
  char *slot_buf = query_buffer_.SlotBuffer(request_slot, index_.search_metadata().page_size);
  PageReadRequest &request = query_buffer_.reqs[request_slot];
  request.page_id = page_id;
  request.request_slot = request_slot;
  request.io = index_.BuildPageReadRequest(page_id, slot_buf);
  request.io.finished = false;
  if (!page_reader_->Submit(&request)) {
    ReleaseRequestSlot(request_slot);
    return false;
  }
  RequestSlot &slot = query_buffer_.request_slots[request_slot];
  slot.page_id = page_id;
  slot.candidate_id = neighbor->id;
  slot.in_flight = true;
  slot.completed = false;
  slot.resident = false;
  query_buffer_.page_visited[page_id] = 1;
  neighbor->flag = false;
  on_flight_ios_.push_back({neighbor->id, page_id, request_slot});
  if (stats_ != nullptr) {
    constexpr double kSectorLen = 4096.0;
    ++stats_->async_reads;
    if (count_graph_cache_miss && graph_cache_ != nullptr) {
      ++stats_->graph_cache_misses;
    }
    stats_->bytes_read += request.io.len;
    stats_->n_ios += static_cast<uint64_t>(
        std::max<double>(1.0, static_cast<double>(index_.search_metadata().page_size) / kSectorLen));
    stats_->max_inflight_reads = std::max<uint64_t>(stats_->max_inflight_reads, page_reader_->NumInflight());
  }
  RecordSchedulerPendingStats();
  return true;
}

uint32_t SearchSession::AcquireRequestSlot() {
  for (uint32_t attempt = 0; attempt < kMaxRequestSlots; ++attempt) {
    const uint32_t slot = (query_buffer_.sector_idx + attempt) % kMaxRequestSlots;
    const RequestSlot &state = query_buffer_.request_slots[slot];
    if (state.in_flight || state.completed) {
      continue;
    }
    EvictSlot(slot);
    query_buffer_.sector_idx = (slot + 1) % kMaxRequestSlots;
    return slot;
  }
  throw std::runtime_error("no free request slot available in query buffer");
}

void SearchSession::ReleaseRequestSlot(uint32_t request_slot) {
  if (request_slot >= query_buffer_.request_slots.size()) {
    throw std::runtime_error("request slot out of range");
  }
  query_buffer_.request_slots[request_slot] = RequestSlot{};
}

void SearchSession::EvictSlot(uint32_t request_slot) {
  if (request_slot >= query_buffer_.request_slots.size()) {
    throw std::runtime_error("request slot out of range");
  }
  RequestSlot &slot = query_buffer_.request_slots[request_slot];
  if (slot.in_flight || slot.completed) {
    throw std::runtime_error("cannot evict a slot that is still inflight");
  }
  if (slot.resident) {
    const char *slot_buf = query_buffer_.SlotBuffer(request_slot, index_.search_metadata().page_size);
    const PageView page = index_.ViewPage(slot.page_id, slot_buf, index_.search_metadata().page_size);
    for (uint32_t i = 0; i < page.layout_size; ++i) {
      const uint32_t node_id = page.LayoutAt(i);
      auto buffered = query_buffer_.id_buf_map.find(node_id);
      if (buffered != query_buffer_.id_buf_map.end() &&
          buffered->second.request_slot == request_slot &&
          buffered->second.page_id == slot.page_id) {
        query_buffer_.id_buf_map.erase(buffered);
      }
      Neighbor *candidate = FindNeighbor(&retset_, node_id);
      if (candidate != nullptr && !candidate->visited) {
        candidate->flag = true;
      }
    }
    query_buffer_.page_visited.at(slot.page_id) = 0;
  }
  query_buffer_.request_slots[request_slot] = RequestSlot{};
}

SearchSession::PollOutcome SearchSession::ConsumeCompletedInflight(bool require_all) {
  PollOutcome outcome;
  const float retset_bound = CurrentRetsetBound();
  while (!on_flight_ios_.empty()) {
    const InflightRead &front = on_flight_ios_.front();
    auto ready = completed_by_slot_.find(front.request_slot);
    if (ready == completed_by_slot_.end()) {
      if (require_all) {
        throw std::runtime_error("missing completed page while draining inflight queue");
      }
      break;
    }
    if (const Neighbor *candidate = FindNeighbor(retset_, front.candidate_id); candidate != nullptr) {
      if (candidate->distance <= retset_bound) {
        ++outcome.n_in;
        if (stats_ != nullptr) {
          ++stats_->read_hits_in_pool;
        }
      } else {
        ++outcome.n_out;
        if (stats_ != nullptr) {
          ++stats_->read_waste_out_of_pool;
        }
      }
    }
    completed_by_slot_.erase(ready);
    RequestSlot &slot = query_buffer_.request_slots[front.request_slot];
    if (!slot.in_flight || !slot.completed || slot.page_id != front.page_id) {
      throw std::runtime_error("completed inflight queue is out of sync with request slots");
    }
    RegisterPage(front.request_slot, front.page_id);
    slot.in_flight = false;
    slot.completed = false;
    slot.resident = true;
    slot.candidate_id = std::numeric_limits<uint32_t>::max();
    on_flight_ios_.pop_front();
    ++outcome.registered;
    if (stats_ != nullptr) {
      ++stats_->pages_completed;
    }
  }
  return outcome;
}

void SearchSession::StoreCompletedPages(const std::vector<PageReadCompletion> &completed_pages) {
  for (const auto &completed : completed_pages) {
    if (completed.request_slot >= query_buffer_.request_slots.size()) {
      throw std::runtime_error("completed request slot exceeds query buffer capacity");
    }
    RequestSlot &slot = query_buffer_.request_slots[completed.request_slot];
    if (!slot.in_flight || slot.page_id != completed.page_id) {
      throw std::runtime_error("completed page does not match the inflight request slot");
    }
    slot.completed = true;
    completed_by_slot_[completed.request_slot] = completed.page_id;
  }
}

void SearchSession::RegisterPage(uint32_t request_slot, uint32_t page_id) {
  const char *slot_buf = query_buffer_.SlotBuffer(request_slot, index_.search_metadata().page_size);
  const PageView page = index_.ViewPage(page_id, slot_buf, index_.search_metadata().page_size);
  const RequestSlot &slot = query_buffer_.request_slots.at(request_slot);
  for (uint32_t i = 0; i < page.layout_size; ++i) {
    query_buffer_.id_buf_map[page.LayoutAt(i)] = BufferedNode{request_slot, page_id, i, slot.candidate_id};
  }
}

bool SearchSession::MaybeExactify(Neighbor *candidate) {
  if (candidate->exact) {
    return false;
  }
  if (query_buffer_.id_buf_map.find(candidate->id) == query_buffer_.id_buf_map.end()) {
    return false;
  }
  const BufferedNode &buffered = query_buffer_.id_buf_map.at(candidate->id);
  if (buffered.request_slot >= query_buffer_.request_slots.size()) {
    throw std::runtime_error("exact compare requested for an invalid request slot");
  }
  const RequestSlot &slot = query_buffer_.request_slots[buffered.request_slot];
  if (!slot.resident || slot.page_id != buffered.page_id) {
    return false;
  }
  const char *slot_buf = query_buffer_.SlotBuffer(buffered.request_slot, index_.search_metadata().page_size);
  const PageView page = index_.ViewPage(buffered.page_id, slot_buf, index_.search_metadata().page_size);
  const DiskNodeView node = index_.ViewNode(page, buffered.layout_index);
  const bool exact_from_page = node.has_embedded_coords();
  const float exact_distance = exact_from_page ? L2Distance(query_, node.coords, index_.search_metadata().dim)
                                               : L2Distance(query_, index_.exact_vector(candidate->id));
  const bool is_member = is_member_(candidate->id, node);
  candidate->exact = true;
  candidate->is_member = is_member;
  if (!is_member) {
    return true;
  }
  Neighbor exact_neighbor;
  exact_neighbor.id = candidate->id;
  exact_neighbor.distance = exact_distance;
  exact_neighbor.is_member = true;
  exact_neighbor.exact = true;
  InsertIntoPool(&full_retset_, index_.search_metadata().num_points, exact_neighbor);
  if (stats_ != nullptr) {
    ++stats_->exact_distance_evals;
    if (exact_from_page) {
      ++stats_->exact_from_page;
    } else {
      ++stats_->exact_from_payload;
    }
  }
  return true;
}

size_t SearchSession::RefinementBound() const {
  size_t requested = l_pool_;
  if (config_.refine_k != 0) {
    requested = config_.refine_k;
  } else if (config_.refine_ratio > 0.0f) {
    requested = static_cast<size_t>(
        std::ceil(static_cast<double>(l_pool_) * static_cast<double>(config_.refine_ratio)));
  }
  const size_t minimum_results = range_mode_ ? config_.l_search : config_.top_k;
  requested = std::max(requested, minimum_results);
  return std::min(requested, retset_.size());
}

size_t SearchSession::ReadyUnexpandedCount() const {
  size_t count = 0;
  for (const Neighbor &candidate : retset_) {
    if (candidate.visited || !candidate.flag) {
      continue;
    }
    if (query_buffer_.id_buf_map.find(candidate.id) != query_buffer_.id_buf_map.end()) {
      ++count;
      continue;
    }
    if (graph_cache_ != nullptr && graph_cache_->Find(candidate.id) != nullptr) {
      ++count;
    }
  }
  return count;
}

size_t SearchSession::SchedulerPendingCount() const {
  return page_reader_->NumInflight() + ReadyUnexpandedCount();
}

size_t SearchSession::EffectiveSchedulerPolicyLimit() const {
  if (config_.scheduler_policy_limit != 0) {
    return config_.scheduler_policy_limit;
  }
  size_t multiplier = 1;
  switch (config_.scheduler_policy) {
    case SearchConfig::SchedulerPolicy::kConservative:
      multiplier = 1;
      break;
    case SearchConfig::SchedulerPolicy::kBounded:
      multiplier = 2;
      break;
    case SearchConfig::SchedulerPolicy::kAggressive:
      multiplier = 4;
      break;
  }
  return std::max<size_t>(1, static_cast<size_t>(config_.beam_width) * multiplier);
}

void SearchSession::RecordSchedulerPendingStats() {
  if (stats_ == nullptr) {
    return;
  }
  const size_t ready = ReadyUnexpandedCount();
  const size_t pending = page_reader_->NumInflight() + ready;
  stats_->scheduler_policy_limit =
      std::max<uint64_t>(stats_->scheduler_policy_limit, EffectiveSchedulerPolicyLimit());
  stats_->scheduler_ready_unexpanded_max =
      std::max<uint64_t>(stats_->scheduler_ready_unexpanded_max, ready);
  stats_->scheduler_pending_max = std::max<uint64_t>(stats_->scheduler_pending_max, pending);
}

float SearchSession::CurrentRetsetBound() const {
  if (retset_.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  const size_t bound_index = std::min<size_t>(l_pool_, retset_.size()) - 1;
  return retset_[bound_index].distance;
}

void SearchSession::ComputeApproximateDistances(const uint32_t *ids, size_t count, float *distances) {
  if (count == 0) {
    return;
  }
  if (ids == nullptr || distances == nullptr) {
    throw std::runtime_error("approximate distance buffers must not be null");
  }
  if (stats_ != nullptr) {
    stats_->approx_distance_evals += count;
    stats_->n_cmps += count;
  }
  switch (approx_mode_) {
    case ApproxExecutionMode::kPipeannPq:
      pipeann_pq_.DistanceBatch(pq_query_distance_table_, ids, count, distances);
      return;
    case ApproxExecutionMode::kFullPrecision:
      for (size_t i = 0; i < count; ++i) {
        distances[i] = full_precision_distance_.Distance(ids[i]);
      }
      return;
    case ApproxExecutionMode::kExternal:
      for (size_t i = 0; i < count; ++i) {
        distances[i] = external_approx_distance_->Distance(ids[i]);
      }
      return;
  }
  throw std::runtime_error("unsupported approximate execution mode");
}

float SearchSession::ApproxDistance(uint32_t id) {
  float distance = 0.0f;
  ComputeApproximateDistances(&id, 1, &distance);
  return distance;
}

void SearchSession::VisitExpandedNodeNeighbors(const DiskNodeView &node) {
  query_buffer_.neighbor_ids.clear();
  query_buffer_.neighbor_ids.reserve(node.nnbrs);

  const auto collect_neighbor = [&](uint32_t neighbor_id) {
    if (neighbor_id >= query_buffer_.visited.size()) {
      throw std::runtime_error("graph neighbor id out of range");
    }
    if (query_buffer_.visited[neighbor_id] != 0) {
      return;
    }
    query_buffer_.visited[neighbor_id] = 1;
    query_buffer_.neighbor_ids.push_back(neighbor_id);
  };

  for (uint16_t i = 0; i < node.nnbrs && query_buffer_.neighbor_ids.size() < node.nnbrs; ++i) {
    if (!is_member_approx_(node.nbrs[i])) {
      continue;
    }
    collect_neighbor(node.nbrs[i]);
  }
  if (use_dense_neighbors_) {
    for (uint16_t i = 0; i < node.n_dense_nbrs && query_buffer_.neighbor_ids.size() < node.nnbrs; ++i) {
      if (!is_member_approx_(node.dense_nbrs[i])) {
        continue;
      }
      collect_neighbor(node.dense_nbrs[i]);
    }
    for (uint16_t i = 0; i < node.nnbrs && query_buffer_.neighbor_ids.size() < node.nnbrs; ++i) {
      collect_neighbor(node.nbrs[i]);
    }
  }
  if (query_buffer_.neighbor_ids.empty()) {
    return;
  }

  query_buffer_.neighbor_distances.resize(query_buffer_.neighbor_ids.size());
  ComputeApproximateDistances(query_buffer_.neighbor_ids.data(),
                              query_buffer_.neighbor_ids.size(),
                              query_buffer_.neighbor_distances.data());
  for (size_t i = 0; i < query_buffer_.neighbor_ids.size(); ++i) {
    Neighbor neighbor;
    neighbor.id = query_buffer_.neighbor_ids[i];
    neighbor.distance = query_buffer_.neighbor_distances[i];
    InsertIntoPool(&retset_, retset_limit(), neighbor);
  }
}

void SearchSession::VisitCachedNodeNeighbors(const CachedGraphNode &node) {
  query_buffer_.neighbor_ids.clear();
  query_buffer_.neighbor_ids.reserve(node.neighbors.size());

  const auto collect_neighbor = [&](uint32_t neighbor_id, bool require_approx_member) {
    if (neighbor_id >= query_buffer_.visited.size()) {
      throw std::runtime_error("cached graph neighbor id out of range");
    }
    if (query_buffer_.visited[neighbor_id] != 0) {
      return;
    }
    if (require_approx_member && !is_member_approx_(neighbor_id)) {
      return;
    }
    query_buffer_.visited[neighbor_id] = 1;
    query_buffer_.neighbor_ids.push_back(neighbor_id);
  };

  const size_t target_count = node.neighbors.size();
  for (uint32_t neighbor_id : node.neighbors) {
    if (query_buffer_.neighbor_ids.size() >= target_count) {
      break;
    }
    collect_neighbor(neighbor_id, true);
  }
  if (use_dense_neighbors_) {
    for (uint32_t neighbor_id : node.dense_neighbors) {
      if (query_buffer_.neighbor_ids.size() >= target_count) {
        break;
      }
      collect_neighbor(neighbor_id, true);
    }
    for (uint32_t neighbor_id : node.neighbors) {
      if (query_buffer_.neighbor_ids.size() >= target_count) {
        break;
      }
      collect_neighbor(neighbor_id, false);
    }
  }
  if (query_buffer_.neighbor_ids.empty()) {
    return;
  }

  query_buffer_.neighbor_distances.resize(query_buffer_.neighbor_ids.size());
  ComputeApproximateDistances(query_buffer_.neighbor_ids.data(),
                              query_buffer_.neighbor_ids.size(),
                              query_buffer_.neighbor_distances.data());
  for (size_t i = 0; i < query_buffer_.neighbor_ids.size(); ++i) {
    Neighbor neighbor;
    neighbor.id = query_buffer_.neighbor_ids[i];
    neighbor.distance = query_buffer_.neighbor_distances[i];
    InsertIntoPool(&retset_, retset_limit(), neighbor);
  }
}

std::vector<SearchResult> SearchSession::CollectResults() const {
  std::unordered_map<uint32_t, float> best_by_id;
  for (const Neighbor &neighbor : full_retset_) {
    auto it = best_by_id.find(neighbor.id);
    if (it == best_by_id.end() || neighbor.distance < it->second) {
      best_by_id[neighbor.id] = neighbor.distance;
    }
  }

  std::vector<SearchResult> results;
  results.reserve(best_by_id.size());
  for (const auto &entry : best_by_id) {
    SearchResult result;
    result.id = entry.first;
    result.distance = entry.second;
    results.push_back(result);
  }
  std::sort(results.begin(), results.end(), [](const SearchResult &lhs, const SearchResult &rhs) {
    if (lhs.distance == rhs.distance) {
      return lhs.id < rhs.id;
    }
    return lhs.distance < rhs.distance;
  });
  if (range_mode_) {
    results.erase(std::remove_if(results.begin(),
                                 results.end(),
                                 [&](const SearchResult &result) { return result.distance > config_.range_partial; }),
                  results.end());
    if (results.size() > config_.l_search) {
      results.resize(config_.l_search);
    }
  } else if (results.size() > config_.top_k) {
    results.resize(config_.top_k);
  }
  return results;
}

size_t SearchSession::retset_limit() const {
  size_t refine_capacity = l_pool_;
  if (config_.refine_k != 0) {
    refine_capacity = std::max(refine_capacity, static_cast<size_t>(config_.refine_k));
  } else if (config_.refine_ratio > 0.0f) {
    refine_capacity = std::max(
        refine_capacity,
        static_cast<size_t>(std::ceil(static_cast<double>(l_pool_) * static_cast<double>(config_.refine_ratio))));
  }
  const size_t pool_capacity = static_cast<size_t>(l_pool_) * 8;
  const size_t beam_capacity = static_cast<size_t>(config_.beam_width) * 8;
  const size_t refinement_capacity = refine_capacity * 2;
  return std::max(std::max(pool_capacity, beam_capacity), refinement_capacity);
}

}  // namespace hybrid
