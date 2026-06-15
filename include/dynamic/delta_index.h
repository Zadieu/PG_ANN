// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <immintrin.h>

#include "distance.h"
#include "utils.h"

namespace diskann {

  struct DeltaIndexConfig {
    _u64 max_points = 0;
    _u32 fsync_every = 100;
    _u32 fsync_interval_ms = 100;
    _u64 delete_filter_slack = 0;
  };

  enum class DeltaOpStatus {
    OK = 0,
    DUPLICATE_TAG,
    NOT_FOUND,
    FULL,
    BAD_DIM,
    IO_ERROR
  };

  template<typename T>
  class DeltaIndex {
   public:
    struct SearchResult {
      _u64 id;
      float distance;
    };

    DeltaIndex(_u64 data_dim, _u64 aligned_dim, _u64 base_num_points,
               DeltaIndexConfig config);
    ~DeltaIndex();

    void recover(const std::string& wal_path);
    void flush();

    DeltaOpStatus insert(_u64 tag, const std::vector<T>& vector);
    DeltaOpStatus insert_auto(const std::vector<T>& vector, _u64& assigned_tag);
    DeltaOpStatus erase(_u64 tag);

    std::vector<SearchResult> search(const T* query, _u64 topk,
                                     const Distance<T>& distance) const;

    bool is_deleted_base(_u64 id) const;
    bool contains_live_tag(_u64 tag) const;
    bool has_base_deletes() const;

    _u64 live_size() const;
    _u64 deleted_base_size() const;
    _u64 max_points() const {
      return config_.max_points;
    }
    _u64 base_num_points() const {
      return base_num_points_;
    }
    _u64 delete_filter_slack() const {
      return config_.delete_filter_slack;
    }

    void apply_text_ops_file(const std::string& ops_path);

   private:
    struct Entry {
      _u64 tag = 0;
      T* values = nullptr;
      bool deleted = false;

      Entry() = default;
      Entry(const Entry&) = delete;
      Entry& operator=(const Entry&) = delete;
      Entry(Entry&& other) noexcept;
      Entry& operator=(Entry&& other) noexcept;
      ~Entry();
    };

    DeltaOpStatus insert_impl(_u64 tag, const std::vector<T>& vector,
                              bool write_wal, bool check_capacity);
    DeltaOpStatus erase_impl(_u64 tag, bool write_wal);

    bool append_wal_record(_u16 op, _u64 tag, const T* vector);
    bool open_wal_for_append();
    bool write_all(const char* data, size_t bytes);
    void maybe_fsync();
    void fsync_now();
    _u64 next_auto_tag_locked() const;
    bool live_base_tag_locked(_u64 tag) const;

    _u64 data_dim_;
    _u64 aligned_dim_;
    _u64 base_num_points_;
    DeltaIndexConfig config_;

    std::string wal_path_;
    int wal_fd_ = -1;
    _u64 next_seq_ = 0;
    _u32 pending_since_fsync_ = 0;
    std::chrono::steady_clock::time_point last_fsync_;

    std::vector<Entry> entries_;
    std::unordered_map<_u64, size_t> live_delta_pos_;
    std::unordered_set<_u64> deleted_base_;

    mutable std::shared_timed_mutex mutex_;
  };

  const char* delta_op_status_to_string(DeltaOpStatus status);

}  // namespace diskann
