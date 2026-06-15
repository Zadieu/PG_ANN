// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <immintrin.h>

#include "deco_index.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include <omp.h>

#include "ann_exception.h"
#include "logger.h"
#include "timer.h"

namespace diskann {

  namespace {
    template<typename T>
    struct DeltaMergeCandidate {
      _u64 id;
      float distance;
    };

    template<typename T>
    void merge_base_and_delta_results(
        const T* query, _u64 query_num, _u64 query_aligned_dim,
        _u64 output_k, _u64 base_k, const std::vector<_u64>& base_indices,
        const std::vector<float>& base_distances, std::vector<_u64>& out_indices,
        std::vector<float>& out_distances, DeltaIndex<T>& delta,
        const Distance<T>& distance, QueryStats* stats) {
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 qi = 0; qi < static_cast<_s64>(query_num); qi++) {
        Timer merge_timer;
        std::vector<DeltaMergeCandidate<T>> candidates;
        candidates.reserve(static_cast<size_t>(base_k + output_k));
        std::unordered_set<_u64> seen;
        seen.reserve(static_cast<size_t>(base_k + output_k));

        const _u64 base_offset = static_cast<_u64>(qi) * base_k;
        for (_u64 j = 0; j < base_k; j++) {
          const _u64 id = base_indices[base_offset + j];
          if (id == static_cast<_u64>(INF) || delta.is_deleted_base(id)) {
            continue;
          }
          if (seen.insert(id).second) {
            candidates.push_back({id, base_distances[base_offset + j]});
          }
        }

        const _u64 delta_topk =
            std::max<_u64>(output_k, delta.delete_filter_slack());
        auto delta_results =
            delta.search(query + static_cast<_u64>(qi) * query_aligned_dim,
                         delta_topk, distance);
        for (const auto& result : delta_results) {
          if (seen.insert(result.id).second) {
            candidates.push_back({result.id, result.distance});
          }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const DeltaMergeCandidate<T>& lhs,
                     const DeltaMergeCandidate<T>& rhs) {
                    if (lhs.distance != rhs.distance) {
                      return lhs.distance < rhs.distance;
                    }
                    return lhs.id < rhs.id;
                  });

        const _u64 out_offset = static_cast<_u64>(qi) * output_k;
        for (_u64 j = 0; j < output_k; j++) {
          if (j < candidates.size()) {
            out_indices[out_offset + j] = candidates[j].id;
            out_distances[out_offset + j] = candidates[j].distance;
          } else {
            out_indices[out_offset + j] = static_cast<_u64>(INF);
            out_distances[out_offset + j] =
                std::numeric_limits<float>::max();
          }
        }

        if (stats != nullptr) {
          const float merge_us = static_cast<float>(merge_timer.elapsed());
          stats[qi].postprocess_us += merge_us;
          stats[qi].total_us += merge_us;
          const _u64 live_delta = delta.live_size();
          stats[qi].n_ext_cmps += static_cast<unsigned>(
              std::min<_u64>(live_delta,
                             static_cast<_u64>(
                                 std::numeric_limits<unsigned>::max())));
        }
      }
    }
  }  // namespace

  template<typename T>
  void DecoIndex<T>::load_delta_wal(const std::string& wal_path,
                                    _u64 max_delta_points,
                                    _u32 fsync_every,
                                    _u32 fsync_interval_ms,
                                    _u64 delete_filter_slack) {
    if (wal_path.empty()) {
      throw ANNException("--delta_wal_path is required when Delta is enabled",
                         -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (data_dim == 0 || aligned_dim == 0 || num_points == 0) {
      throw ANNException("Delta WAL must be loaded after base DecoIndex::load",
                         -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    DeltaIndexConfig config;
    if (max_delta_points == 0) {
      max_delta_points = num_points / 100;
      if (max_delta_points == 0) {
        max_delta_points = 1;
      }
      max_delta_points = std::min<_u64>(max_delta_points, 10000);
    }
    config.max_points = max_delta_points;
    config.fsync_every = fsync_every;
    config.fsync_interval_ms = fsync_interval_ms;
    config.delete_filter_slack = delete_filter_slack;

    delta_index_ = std::unique_ptr<DeltaIndex<T>>(
        new DeltaIndex<T>(data_dim, aligned_dim, num_points, config));
    delta_index_->recover(wal_path);

    diskann::cout << "Delta overlay enabled. wal=" << wal_path
                  << ", max_delta_points=" << config.max_points
                  << ", live_delta=" << delta_index_->live_size()
                  << ", base_tombstones="
                  << delta_index_->deleted_base_size()
                  << ", fsync_every=" << config.fsync_every
                  << ", fsync_interval_ms=" << config.fsync_interval_ms
                  << std::endl;
  }

  template<typename T>
  void DecoIndex<T>::apply_delta_ops_file(const std::string& ops_path) {
    if (ops_path.empty()) {
      return;
    }
    if (delta_index_ == nullptr) {
      throw ANNException("Delta ops require load_delta_wal first", -1,
                         __FUNCSIG__, __FILE__, __LINE__);
    }
    delta_index_->apply_text_ops_file(ops_path);
    if (delta_index_->max_points() > 0 &&
        delta_index_->live_size() >= delta_index_->max_points()) {
      diskann::cout << "Delta overlay reached max_delta_points="
                    << delta_index_->max_points()
                    << ". Offline merge/rebuild is recommended."
                    << std::endl;
    }
    if (delta_index_->deleted_base_size() > num_points / 50) {
      diskann::cout << "Base tombstones exceed 2% of base index. "
                    << "Offline merge/rebuild is recommended." << std::endl;
    }
  }

  template<typename T>
  DeltaOpStatus DecoIndex<T>::delta_insert(_u64 tag,
                                           const std::vector<T>& vector) {
    if (delta_index_ == nullptr) {
      return DeltaOpStatus::IO_ERROR;
    }
    return delta_index_->insert(tag, vector);
  }

  template<typename T>
  DeltaOpStatus DecoIndex<T>::delta_insert_auto(
      const std::vector<T>& vector, _u64& assigned_tag) {
    if (delta_index_ == nullptr) {
      return DeltaOpStatus::IO_ERROR;
    }
    return delta_index_->insert_auto(vector, assigned_tag);
  }

  template<typename T>
  DeltaOpStatus DecoIndex<T>::delta_erase(_u64 tag) {
    if (delta_index_ == nullptr) {
      return DeltaOpStatus::IO_ERROR;
    }
    return delta_index_->erase(tag);
  }

  template<typename T>
  void DecoIndex<T>::page_search_with_delta(
      const T *query, const _u64 query_num, const _u64 query_aligned_dim,
      const _u64 k_search, const _u32 mem_L, const _u64 l_search,
      std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit, const float pq_filter_ratio,
      const float emb_search_ratio, QueryStats *stats) {
    if (delta_index_ == nullptr ||
        (delta_index_->live_size() == 0 &&
         delta_index_->deleted_base_size() == 0)) {
      page_search(query, query_num, query_aligned_dim, k_search, mem_L,
                  l_search, indices_vec, distances_vec, beam_width, io_limit,
                  pq_filter_ratio, emb_search_ratio, stats);
      return;
    }

    _u64 slack = delta_index_->delete_filter_slack();
    if (slack == 0) {
      slack = std::max<_u64>(32, 2 * k_search);
    }
    const _u64 base_k =
        delta_index_->has_base_deletes()
            ? std::min<_u64>(l_search, k_search + slack)
            : k_search;
    if (delta_index_->has_base_deletes() && base_k < k_search + slack) {
      diskann::cout << "Delta delete_filter_slack clipped by L. L="
                    << l_search << ", K=" << k_search
                    << ", requested_slack=" << slack << std::endl;
    }

    std::vector<_u64> base_indices(query_num * base_k);
    std::vector<float> base_distances(query_num * base_k);
    page_search(query, query_num, query_aligned_dim, base_k, mem_L, l_search,
                base_indices, base_distances, beam_width, io_limit,
                pq_filter_ratio, emb_search_ratio, stats);

    merge_base_and_delta_results(query, query_num, query_aligned_dim, k_search,
                                 base_k, base_indices, base_distances,
                                 indices_vec, distances_vec, *delta_index_,
                                 *dist_cmp, stats);
  }

  template<typename T>
  void DecoIndex<T>::page_search_dup_graph_with_delta(
      const T *query, const _u64 query_num, const _u64 query_aligned_dim,
      const _u64 k_search, const _u32 mem_L, const _u64 l_search,
      std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit, const float pq_filter_ratio,
      const float emb_search_ratio, QueryStats *stats) {
    if (delta_index_ == nullptr ||
        (delta_index_->live_size() == 0 &&
         delta_index_->deleted_base_size() == 0)) {
      page_search_dup_graph(query, query_num, query_aligned_dim, k_search,
                            mem_L, l_search, indices_vec, distances_vec,
                            beam_width, io_limit, pq_filter_ratio,
                            emb_search_ratio, stats);
      return;
    }

    _u64 slack = delta_index_->delete_filter_slack();
    if (slack == 0) {
      slack = std::max<_u64>(32, 2 * k_search);
    }
    const _u64 base_k =
        delta_index_->has_base_deletes()
            ? std::min<_u64>(l_search, k_search + slack)
            : k_search;
    if (delta_index_->has_base_deletes() && base_k < k_search + slack) {
      diskann::cout << "Delta delete_filter_slack clipped by L. L="
                    << l_search << ", K=" << k_search
                    << ", requested_slack=" << slack << std::endl;
    }

    std::vector<_u64> base_indices(query_num * base_k);
    std::vector<float> base_distances(query_num * base_k);
    page_search_dup_graph(query, query_num, query_aligned_dim, base_k, mem_L,
                          l_search, base_indices, base_distances, beam_width,
                          io_limit, pq_filter_ratio, emb_search_ratio, stats);

    merge_base_and_delta_results(query, query_num, query_aligned_dim, k_search,
                                 base_k, base_indices, base_distances,
                                 indices_vec, distances_vec, *delta_index_,
                                 *dist_cmp, stats);
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;

}  // namespace diskann
