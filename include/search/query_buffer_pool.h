#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "io/page_reader.h"

namespace hybrid {

struct QueryBufferedNode {
  uint32_t request_slot = 0;
  uint32_t page_id = 0;
  uint32_t layout_index = 0;
  uint32_t origin_candidate_id = std::numeric_limits<uint32_t>::max();
};

struct QueryRequestSlot {
  uint32_t page_id = std::numeric_limits<uint32_t>::max();
  uint32_t candidate_id = std::numeric_limits<uint32_t>::max();
  bool in_flight = false;
  bool completed = false;
  bool resident = false;
};

struct QueryBufferState {
  static constexpr uint32_t kRequestSlotCount = 256;

  uint32_t page_size = 0;
  uint32_t sector_idx = 0;
  std::vector<char> sector_scratch;
  std::vector<PageReadRequest> reqs;
  std::vector<QueryRequestSlot> request_slots;
  std::vector<uint8_t> visited;
  std::vector<uint8_t> page_visited;
  std::vector<uint32_t> neighbor_ids;
  std::vector<float> neighbor_distances;
  std::unordered_map<uint32_t, QueryBufferedNode> id_buf_map;

  void EnsureCapacity(uint32_t requested_page_size, uint32_t num_points, uint32_t num_pages) {
    page_size = requested_page_size;
    const size_t scratch_bytes = static_cast<size_t>(kRequestSlotCount) * static_cast<size_t>(page_size);
    if (sector_scratch.size() != scratch_bytes) {
      sector_scratch.resize(scratch_bytes);
    }
    if (reqs.size() != kRequestSlotCount) {
      reqs.resize(kRequestSlotCount);
    }
    if (request_slots.size() != kRequestSlotCount) {
      request_slots.resize(kRequestSlotCount);
    }
    if (visited.size() != num_points) {
      visited.resize(num_points);
    }
    if (page_visited.size() != num_pages) {
      page_visited.resize(num_pages);
    }
    if (id_buf_map.bucket_count() < num_points) {
      id_buf_map.reserve(num_points);
    }
  }

  void ResetForQuery() {
    sector_idx = 0;
    std::fill(reqs.begin(), reqs.end(), PageReadRequest{});
    std::fill(request_slots.begin(), request_slots.end(), QueryRequestSlot{});
    std::fill(visited.begin(), visited.end(), uint8_t{0});
    std::fill(page_visited.begin(), page_visited.end(), uint8_t{0});
    neighbor_ids.clear();
    neighbor_distances.clear();
    id_buf_map.clear();
  }

  char *SlotBuffer(uint32_t slot) {
    if (slot >= kRequestSlotCount) {
      throw std::runtime_error("query buffer slot out of range");
    }
    return sector_scratch.data() + static_cast<size_t>(slot) * static_cast<size_t>(page_size);
  }

  const char *SlotBuffer(uint32_t slot) const {
    if (slot >= kRequestSlotCount) {
      throw std::runtime_error("query buffer slot out of range");
    }
    return sector_scratch.data() + static_cast<size_t>(slot) * static_cast<size_t>(page_size);
  }
};

class QueryBufferPool;

class QueryBufferLease {
 public:
  QueryBufferLease() = default;
  QueryBufferLease(QueryBufferPool *pool, QueryBufferState *state);
  QueryBufferLease(const QueryBufferLease &) = delete;
  QueryBufferLease &operator=(const QueryBufferLease &) = delete;
  QueryBufferLease(QueryBufferLease &&other) noexcept;
  QueryBufferLease &operator=(QueryBufferLease &&other) noexcept;
  ~QueryBufferLease();

  QueryBufferState *get() const { return state_; }
  QueryBufferState &operator*() const { return *state_; }
  QueryBufferState *operator->() const { return state_; }
  explicit operator bool() const { return state_ != nullptr; }

 private:
  void Release();

  QueryBufferPool *pool_ = nullptr;
  QueryBufferState *state_ = nullptr;
};

class QueryBufferPool {
 public:
  void InitBuffers(uint32_t buffer_count, uint32_t page_size, uint32_t num_points, uint32_t num_pages) {
    page_size_ = page_size;
    num_points_ = num_points;
    num_pages_ = num_pages;
    buffers_.clear();
    available_.clear();
    buffers_.resize(std::max<uint32_t>(buffer_count, 1u));
    available_.reserve(buffers_.size());
    for (QueryBufferState &buffer : buffers_) {
      buffer.EnsureCapacity(page_size_, num_points_, num_pages_);
      available_.push_back(&buffer);
    }
  }

  QueryBufferLease Acquire() {
    if (page_size_ == 0) {
      throw std::runtime_error("query buffer pool must be initialized before acquire");
    }
    if (available_.empty()) {
      buffers_.emplace_back();
      QueryBufferState &buffer = buffers_.back();
      buffer.EnsureCapacity(page_size_, num_points_, num_pages_);
      buffer.ResetForQuery();
      return QueryBufferLease(this, &buffer);
    }
    QueryBufferState *buffer = available_.back();
    available_.pop_back();
    buffer->EnsureCapacity(page_size_, num_points_, num_pages_);
    buffer->ResetForQuery();
    return QueryBufferLease(this, buffer);
  }

 private:
  friend class QueryBufferLease;

  void Release(QueryBufferState *buffer) {
    if (buffer == nullptr) {
      return;
    }
    available_.push_back(buffer);
  }

  uint32_t page_size_ = 0;
  uint32_t num_points_ = 0;
  uint32_t num_pages_ = 0;
  std::deque<QueryBufferState> buffers_;
  std::vector<QueryBufferState *> available_;
};

inline QueryBufferLease::QueryBufferLease(QueryBufferPool *pool, QueryBufferState *state)
    : pool_(pool), state_(state) {
}

inline QueryBufferLease::QueryBufferLease(QueryBufferLease &&other) noexcept
    : pool_(other.pool_), state_(other.state_) {
  other.pool_ = nullptr;
  other.state_ = nullptr;
}

inline QueryBufferLease &QueryBufferLease::operator=(QueryBufferLease &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  Release();
  pool_ = other.pool_;
  state_ = other.state_;
  other.pool_ = nullptr;
  other.state_ = nullptr;
  return *this;
}

inline QueryBufferLease::~QueryBufferLease() {
  Release();
}

inline void QueryBufferLease::Release() {
  if (pool_ != nullptr && state_ != nullptr) {
    pool_->Release(state_);
  }
  pool_ = nullptr;
  state_ = nullptr;
}

}  // namespace hybrid
