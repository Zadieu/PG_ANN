#include <immintrin.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include "logger.h"
#include "percentile_stats.h"
#include "deco_index.h"
#include "timer.h"

namespace diskann {

  struct CachedGraphNode {
    unsigned id = 0;
    unsigned cache_pos = INF;
    std::vector<unsigned> nbrs;
  };

  enum class PipeCandidateState : _u8 {
    kInPool = 0,
    kCacheReady = 1,
    kGraphIoSubmitted = 2,
    kGraphPageReady = 3,
    kExpanded = 4,
    kStale = 5
  };

  enum class PipePageState : _u8 {
    kUnseen = 0,
    kGraphIoSubmitted = 1,
    kGraphPageReady = 2,
    kExpanded = 3
  };

  template<typename T>
  void DecoIndex<T>::page_search_dup_graph(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim,
      const _u64 k_search, const _u32 mem_L, const _u64 l_search,
      std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit, const float pq_filter_ratio,
      const float emb_search_ratio, QueryStats* stats_ptr) {
    if (beam_width > MAX_N_SECTOR_READS) {
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    uint32_t query_dim =
        metric == diskann::Metric::INNER_PRODUCT ? this->data_dim - 1
                                                 : this->data_dim;

    const char *pipeline_env = std::getenv("GORGEOUS_PIPELINED_GRAPH_IO");
    const bool use_pipelined_graph_io =
        pipeline_env != nullptr && std::strcmp(pipeline_env, "0") != 0 &&
        std::strcmp(pipeline_env, "false") != 0;
    const char *pipeann_scheduler_env =
        std::getenv("GORGEOUS_PIPEANN_STATE_SCHEDULER");
    const bool use_pipeann_state_scheduler =
        pipeann_scheduler_env != nullptr &&
        std::strcmp(pipeann_scheduler_env, "0") != 0 &&
        std::strcmp(pipeann_scheduler_env, "false") != 0;
    const char *pipeann_scheduler_window_env =
        std::getenv("GORGEOUS_PIPEANN_SCHEDULER_WINDOW");
    const bool use_adaptive_pipeann_scheduler_window =
        pipeann_scheduler_window_env == nullptr ||
        std::strcmp(pipeann_scheduler_window_env, "0") == 0 ||
        std::strcmp(pipeann_scheduler_window_env, "auto") == 0;
    const _u64 pipeann_scheduler_window =
        use_adaptive_pipeann_scheduler_window
            ? 0
            : std::strtoull(pipeann_scheduler_window_env, nullptr, 10);
    const char *pipeann_dynamic_pipe_env =
        std::getenv("GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH");
    const bool use_pipeann_dynamic_pipe_width =
        use_pipeann_state_scheduler && pipeann_dynamic_pipe_env != nullptr &&
        std::strcmp(pipeann_dynamic_pipe_env, "0") != 0 &&
        std::strcmp(pipeann_dynamic_pipe_env, "false") != 0;
    const char *pipeann_pipe_start_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_START");
    const bool use_auto_pipeann_pipe_start =
        pipeann_pipe_start_env == nullptr ||
        std::strcmp(pipeann_pipe_start_env, "0") == 0 ||
        std::strcmp(pipeann_pipe_start_env, "auto") == 0;
    const _u64 pipeann_pipe_start =
        use_auto_pipeann_pipe_start
            ? 0
            : std::max<_u64>(
                  1, std::strtoull(pipeann_pipe_start_env, nullptr, 10));
    const char *pipeann_l_aware_start_env =
        std::getenv("GORGEOUS_PIPEANN_L_AWARE_PIPE_START");
    const bool use_pipeann_l_aware_pipe_start =
        use_pipeann_dynamic_pipe_width &&
        pipeann_l_aware_start_env != nullptr &&
        std::strcmp(pipeann_l_aware_start_env, "0") != 0 &&
        std::strcmp(pipeann_l_aware_start_env, "false") != 0;
    const char *pipeann_l_aware_low_env =
        std::getenv("GORGEOUS_PIPEANN_L_AWARE_LOW_L");
    const _u64 pipeann_l_aware_low_l =
        pipeann_l_aware_low_env == nullptr ||
                std::strcmp(pipeann_l_aware_low_env, "0") == 0
            ? beam_width * 2
            : std::max<_u64>(
                  1, std::strtoull(pipeann_l_aware_low_env, nullptr, 10));
    const char *pipeann_l_aware_high_env =
        std::getenv("GORGEOUS_PIPEANN_L_AWARE_HIGH_L");
    const _u64 pipeann_l_aware_high_l =
        pipeann_l_aware_high_env == nullptr ||
                std::strcmp(pipeann_l_aware_high_env, "0") == 0
            ? beam_width * 3
            : std::max<_u64>(
                  pipeann_l_aware_low_l + 1,
                  std::strtoull(pipeann_l_aware_high_env, nullptr, 10));
    const char *pipeann_pipe_min_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_MIN");
    const bool use_auto_pipeann_pipe_min =
        pipeann_pipe_min_env == nullptr ||
        std::strcmp(pipeann_pipe_min_env, "0") == 0 ||
        std::strcmp(pipeann_pipe_min_env, "auto") == 0;
    const _u64 pipeann_pipe_min =
        use_auto_pipeann_pipe_min
            ? 0
            : std::max<_u64>(
                  1, std::strtoull(pipeann_pipe_min_env, nullptr, 10));
    const char *pipeann_pipe_feedback_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_FEEDBACK");
    const bool use_pipeann_pipe_feedback =
        use_pipeann_dynamic_pipe_width &&
        (pipeann_pipe_feedback_env == nullptr ||
         (std::strcmp(pipeann_pipe_feedback_env, "0") != 0 &&
          std::strcmp(pipeann_pipe_feedback_env, "false") != 0));
    const char *pipeann_pipe_waste_threshold_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD");
    const float pipeann_pipe_waste_threshold =
        pipeann_pipe_waste_threshold_env == nullptr
            ? 0.10f
            : std::strtof(pipeann_pipe_waste_threshold_env, nullptr);
    const char *pipeann_pipe_min_marker_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_MIN_MARKER");
    const _u32 pipeann_pipe_min_marker =
        pipeann_pipe_min_marker_env == nullptr
            ? 5
            : static_cast<_u32>(std::strtoull(pipeann_pipe_min_marker_env,
                                              nullptr, 10));
    const bool track_pipeann_states = use_pipeann_state_scheduler;
    std::atomic_int cur_task = 0;
    pool->runTask([&, this](int tid) {
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      const T *query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, bool> &exact_visited =
          *(query_scratch->exact_visited);
      tsl::robin_map<_u32, std::vector<unsigned>> loaded_nbrs;
      loaded_nbrs.reserve(2048);

      CircleQueue<char*> sector_buffers(MAX_N_SECTOR_READS);
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      std::vector<std::pair<unsigned, char*>> cached_id_bufs(
          MAX_N_SECTOR_READS);
      CircleQueue<CachedGraphNode> cached_node(MAX_N_SECTOR_READS);

      while (true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }

        Timer query_timer, tmp_timer, part_timer;
        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0;

        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;
        const bool use_effective_pipeann_state_scheduler =
            use_pipeann_state_scheduler;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        query_scratch->reset();
        loaded_nbrs.clear();

        pq_table.populate_chunk_distances(query_float, pq_dists);

        std::vector<Neighbor> retset(l_search + 1);
        unsigned cur_list_size = 0;
        std::vector<Neighbor> full_retset;
        full_retset.reserve(4096);
        tsl::robin_map<unsigned, PipeCandidateState> candidate_state;
        tsl::robin_map<unsigned, PipePageState> page_state;
        if (track_pipeann_states) {
          candidate_state.reserve(l_search * 2);
          page_state.reserve(l_search * 2);
        }

        auto set_candidate_state = [&](unsigned id, PipeCandidateState state) {
          if (track_pipeann_states) {
            candidate_state[id] = state;
          }
        };
        auto set_page_state = [&](unsigned pid, PipePageState state) {
          if (track_pipeann_states) {
            page_state[pid] = state;
          }
        };
        auto get_candidate_state = [&](unsigned id) -> PipeCandidateState {
          if (!track_pipeann_states) {
            return PipeCandidateState::kInPool;
          }
          auto state_iter = candidate_state.find(id);
          if (state_iter == candidate_state.end()) {
            return PipeCandidateState::kInPool;
          }
          return state_iter->second;
        };
        auto candidate_ready_for_graph_dispatch =
            [&](const Neighbor& candidate, const _u32 rank,
                const _u32 rank_limit) -> bool {
          if (!candidate.flag) {
            return false;
          }
          if (!use_effective_pipeann_state_scheduler) {
            return true;
          }
          if (rank >= rank_limit) {
            return false;
          }
          return get_candidate_state(candidate.id) ==
                 PipeCandidateState::kInPool;
        };

        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](
                                    const unsigned *ids, const _u64 n_ids,
                                    float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                                         pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks,
                                       pq_dists, dists_out);
        };

        auto compute_exact_dists_and_push =
            [&](const char* node_buf, const unsigned id) -> float {
          tmp_timer.reset();
          float cur_expanded_dist =
              dist_cmp->compare(query, (T*)node_buf, (unsigned) aligned_dim);
          if (stats != nullptr) {
            stats->n_ext_cmps++;
          }
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          exact_visited.insert({id, true});
          return cur_expanded_dist;
        };

        auto add_to_retset = [&](const int nbor_id, const float nbor_dist,
                                 const bool flag) -> unsigned {
          if (cur_list_size == l_search &&
              nbor_dist >= retset[cur_list_size - 1].distance) {
            return INF;
          }
          Neighbor nn(nbor_id, nbor_dist, flag);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          set_candidate_state(nbor_id, PipeCandidateState::kInPool);
          if (cur_list_size < l_search) {
            ++cur_list_size;
          }
          return r;
        };

        auto push_neighbor_list = [&](const unsigned *node_nbrs,
                                      const unsigned nnbrs,
                                      const unsigned node_id) {
          if (dynamic_graph_cache_size > 0) {
            maybe_cache_graph_node(node_id, node_nbrs, nnbrs);
          }
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              add_to_retset(nbr_buf[m], dist_scratch[m], true);
            }
          }
        };

        auto compute_and_push_nbrs_target_update =
            [&](const char *node_buf, const unsigned node_id) {
          unsigned *node_nbrs = (unsigned*)node_buf;
          unsigned nnbrs = *(node_nbrs++);
          if (dynamic_graph_cache_size > 0) {
            maybe_cache_graph_node(node_id, node_nbrs, nnbrs);
          }
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            std::vector<unsigned> expand_nb_ids;
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              const auto r = add_to_retset(nbor_id, nbor_dist, true);
              if (cur_list_size > 0 &&
                  dist_scratch[m] <
                      retset[cur_list_size - 1].distance * pq_filter_ratio &&
                  loaded_nbrs.find(nbor_id) != loaded_nbrs.end()) {
                expand_nb_ids.push_back(nbor_id);
                if (stats != nullptr) {
                  stats->pipe_rep_adj_hits++;
                  stats->pipe_rep_adj_skipped++;
                }
                if (r < cur_list_size) {
                  retset[r].flag = false;
                }
              }
            }
            for (unsigned expand_id : expand_nb_ids) {
              const auto loaded_iter = loaded_nbrs.find(expand_id);
              if (loaded_iter != loaded_nbrs.end()) {
                push_neighbor_list(loaded_iter->second.data(),
                                   static_cast<unsigned>(loaded_iter->second.size()),
                                   expand_id);
              }
              set_candidate_state(expand_id, PipeCandidateState::kExpanded);
            }
          }
        };

        auto compute_and_add_to_retset = [&](const unsigned *node_ids,
                                             const _u64 n_ids) {
          compute_pq_dists(node_ids, n_ids, dist_scratch);
          for (_u64 i = 0; i < n_ids; ++i) {
            retset[cur_list_size].id = node_ids[i];
            retset[cur_list_size].distance = dist_scratch[i];
            retset[cur_list_size++].flag = true;
            visited.insert(node_ids[i]);
            set_candidate_state(node_ids[i], PipeCandidateState::kInPool);
          }
        };

        part_timer.reset();
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(),
                                       mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(),
                                    std::min((unsigned)mem_L,
                                             (unsigned)l_search));
        } else {
          compute_and_add_to_retset(&medoids[0], 1);
        }

        std::sort(retset.begin(), retset.begin() + cur_list_size);

        if (stats != nullptr) {
          stats->preprocess_us += (double) part_timer.elapsed();
        }
        unsigned num_ios = 0;
        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;
        _u32 n_io_in_q = 0;
        _u32 n_cached_in_q = 0;
        _u32 n_proc_in_q = 0;
        _u32 pipeann_current_pipe_width = static_cast<_u32>(beam_width);
        double refine_io_wait_us = 0.0;
        double refine_exact_us = 0.0;
        _u32 pipeann_min_pipe_width = static_cast<_u32>(beam_width);
        _u32 pipeann_feedback_in_pool = 0;
        _u32 pipeann_feedback_total = 0;
        _u32 pipeann_max_marker = 0;
        if (use_pipeann_dynamic_pipe_width &&
            use_effective_pipeann_state_scheduler) {
          _u64 initial_pipe_width = pipeann_pipe_start;
          if (initial_pipe_width == 0) {
            if (use_pipeann_l_aware_pipe_start) {
              const _u64 low_width =
                  std::max<_u64>(1, std::min<_u64>(beam_width, beam_width / 2));
              if (l_search <= pipeann_l_aware_low_l) {
                initial_pipe_width = low_width;
              } else if (l_search >= pipeann_l_aware_high_l) {
                initial_pipe_width = beam_width;
              } else {
                const double ratio =
                    static_cast<double>(l_search - pipeann_l_aware_low_l) /
                    static_cast<double>(pipeann_l_aware_high_l -
                                        pipeann_l_aware_low_l);
                initial_pipe_width = static_cast<_u64>(std::ceil(
                    static_cast<double>(low_width) +
                    ratio * static_cast<double>(beam_width - low_width)));
              }
            } else {
              initial_pipe_width = std::min<_u64>(4, beam_width);
            }
          }
          pipeann_current_pipe_width = static_cast<_u32>(
              std::max<_u64>(1, std::min<_u64>(beam_width,
                                               initial_pipe_width)));
          _u64 min_pipe_width = pipeann_pipe_min;
          if (min_pipe_width == 0) {
            min_pipe_width = initial_pipe_width;
          }
          pipeann_min_pipe_width = static_cast<_u32>(std::max<_u64>(
              1, std::min<_u64>(pipeann_current_pipe_width, min_pipe_width)));
        }
        std::vector<char*> graph_free_sector_bufs;
        if (use_effective_pipeann_state_scheduler) {
          graph_free_sector_bufs.reserve(MAX_N_SECTOR_READS);
          for (_u32 i = 0; i < MAX_N_SECTOR_READS; ++i) {
            graph_free_sector_bufs.push_back(sector_scratch +
                                             i * GR_SECTOR_LEN);
          }
        }
        const int refine_fid = gc_index_fid;

        auto pipeann_scheduler_rank_limit = [&]() -> _u32 {
          if (!use_effective_pipeann_state_scheduler) {
            return cur_list_size;
          }
          if (use_adaptive_pipeann_scheduler_window) {
            return static_cast<_u32>(
                std::min<_u64>(cur_list_size, l_search));
          }
          const _u64 tight_rank_window =
              std::max<_u64>(k_search, beam_width * 4);
          const _u64 recall_rank_window =
              std::max<_u64>(k_search, beam_width * 8);
          const _u64 adaptive_rank_window =
              l_search <= tight_rank_window ? tight_rank_window
                                            : recall_rank_window;
          const _u64 configured_rank_window =
              use_adaptive_pipeann_scheduler_window
                  ? adaptive_rank_window
                  : pipeann_scheduler_window;
          return static_cast<_u32>(std::min<_u64>(
              cur_list_size,
              std::min<_u64>(l_search, configured_rank_window)));
        };

        auto pipeann_stable_prefix = [&]() -> _u32 {
          _u32 stable_prefix = 0;
          const _u32 scan_limit =
              static_cast<_u32>(std::min<_u64>(cur_list_size, l_search));
          for (_u32 rank = 0; rank < scan_limit; ++rank) {
            const PipeCandidateState state =
                get_candidate_state(retset[rank].id);
            if ((retset[rank].flag &&
                 state == PipeCandidateState::kInPool) ||
                state == PipeCandidateState::kStale) {
              break;
            }
            stable_prefix++;
          }
          return stable_prefix;
        };

        auto record_pipeann_graph_io_feedback =
            [&](const std::shared_ptr<FrontierNode>& fn) {
          if (!use_effective_pipeann_state_scheduler) {
            return;
          }
          const float retset_tail_distance =
              cur_list_size > 0
                  ? retset[cur_list_size - 1].distance
                  : std::numeric_limits<float>::max();
          const bool in_pool = fn->distance <= retset_tail_distance;
          if (stats != nullptr) {
            if (in_pool) {
              stats->pipe_graph_useful++;
            } else {
              stats->pipe_graph_wasted++;
            }
          }
          if (!use_pipeann_pipe_feedback) {
            return;
          }
          pipeann_feedback_total++;
          if (in_pool) {
            pipeann_feedback_in_pool++;
          }
          pipeann_max_marker =
              std::max(pipeann_max_marker, pipeann_stable_prefix());
          if (pipeann_max_marker < pipeann_pipe_min_marker ||
              pipeann_feedback_total == 0) {
            return;
          }
          const float waste_ratio =
              static_cast<float>(pipeann_feedback_total -
                                 pipeann_feedback_in_pool) /
              static_cast<float>(pipeann_feedback_total);
          const bool should_increase =
              waste_ratio <= pipeann_pipe_waste_threshold &&
              pipeann_current_pipe_width < beam_width;
          if (should_increase) {
            pipeann_current_pipe_width++;
            pipeann_current_pipe_width = std::max<_u32>(
                pipeann_current_pipe_width, pipeann_min_pipe_width);
            pipeann_current_pipe_width = std::min<_u32>(
                pipeann_current_pipe_width, static_cast<_u32>(beam_width));
            if (stats != nullptr) {
              stats->pipe_width_increases++;
            }
          }
        };

        while (num_ios < io_limit || n_io_in_q > 0 || n_proc_in_q > 0 ||
               n_cached_in_q > 0) {
          if (n_proc_in_q > 0) {
            part_timer.reset();
            char *sector_buf = sector_buffers.get();
            auto sec_iter = sec_buf2ftr.find(sector_buf);
            if (sec_iter == sec_buf2ftr.end()) {
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }

            auto fn = sec_iter->second;
            const _u32 exact_id = fn->id;
            set_candidate_state(exact_id, PipeCandidateState::kGraphPageReady);
            set_page_state(fn->pid, PipePageState::kGraphPageReady);
            compute_exact_dists_and_push(sector_buf, exact_id);

            char *node_buf =
                sector_buf + emb_node_len +
                sizeof(unsigned) * (1 + n_gc_node_per_sector);
            unsigned *p_layout = (unsigned*)(sector_buf + emb_node_len);
            unsigned p_size = *(p_layout++);
            for (unsigned j = 1; j < p_size; j++) {
              if (node_in_mem_pos(p_layout[j]) == INF) {
                char *nnbr_buf = node_buf + j * graph_node_len;
                unsigned *nnbrs_ptr = (unsigned*)nnbr_buf;
                const unsigned nnbrs = *(nnbrs_ptr++);
                loaded_nbrs[p_layout[j]] =
                    std::vector<unsigned>(nnbrs_ptr, nnbrs_ptr + nnbrs);
              }
            }
            compute_and_push_nbrs_target_update(node_buf, exact_id);
            if (stats != nullptr) {
              stats->disk_proc_us += (double) part_timer.elapsed();
            }

            sec_buf2ftr.erase(sec_iter);
            set_candidate_state(exact_id, PipeCandidateState::kExpanded);
            set_page_state(fn->pid, PipePageState::kExpanded);
            if (use_effective_pipeann_state_scheduler) {
              graph_free_sector_bufs.push_back(sector_buf);
            }
            n_proc_in_q--;
          }

          while (n_cached_in_q > 0) {
            part_timer.reset();
            CachedGraphNode cn = cached_node.get();

            unsigned nbors_size = 0;
            if (cn.cache_pos != INF) {
              for (unsigned m = 0; m < mem_graph_[cn.cache_pos].size(); ++m) {
                if (visited.insert(mem_graph_[cn.cache_pos][m]).second) {
                  nbr_buf[nbors_size++] = mem_graph_[cn.cache_pos][m];
                }
              }
            } else {
              for (unsigned m = 0; m < cn.nbrs.size(); ++m) {
                if (visited.insert(cn.nbrs[m]).second) {
                  nbr_buf[nbors_size++] = cn.nbrs[m];
                }
              }
            }
            compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_size;
            }
            for (unsigned m = 0; m < nbors_size; ++m) {
              add_to_retset(nbr_buf[m], dist_scratch[m], true);
            }
            set_candidate_state(cn.id, PipeCandidateState::kExpanded);
            n_cached_in_q--;
            if (stats != nullptr) {
              stats->cache_proc_us += (double) part_timer.elapsed();
            }
          }

          if (n_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) {
              min_r = 1;
            }
            part_timer.reset();
            int n_read_blks =
                io_manager->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
            for (int i = n_read_blks - 1; i >= 0; i--) {
              auto graph_iter = sec_buf2ftr.find(tmp_bufs[i]);
              if (graph_iter != sec_buf2ftr.end()) {
                sector_buffers.push(tmp_bufs[i]);
                set_candidate_state(graph_iter->second->id,
                                    PipeCandidateState::kGraphPageReady);
                set_page_state(graph_iter->second->pid,
                               PipePageState::kGraphPageReady);
                record_pipeann_graph_io_feedback(graph_iter->second);
                n_io_in_q--;
                n_proc_in_q++;
                continue;
              }
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }
            if (stats != nullptr) {
              stats->read_disk_us += (double) part_timer.elapsed();
            }
          }

          pipeann_max_marker =
              std::max(pipeann_max_marker, pipeann_stable_prefix());
          const _u64 graph_pipe_width =
              use_effective_pipeann_state_scheduler
                  ? static_cast<_u64>(pipeann_current_pipe_width)
                  : beam_width;
          const _u32 active_disk_buffers = n_io_in_q;
          const _u32 remaining_io_budget =
              num_ios < io_limit ? static_cast<_u32>(io_limit - num_ios) : 0;
          if (use_effective_pipeann_state_scheduler) {
            const bool pipe_slots_full =
                active_disk_buffers >= graph_pipe_width ||
                remaining_io_budget == 0;
            if (stats != nullptr) {
              stats->pipe_width_sum += static_cast<unsigned>(graph_pipe_width);
              stats->pipe_width_samples++;
              stats->pipe_width_max = std::max<unsigned>(
                  stats->pipe_width_max,
                  static_cast<unsigned>(graph_pipe_width));
              if (pipe_slots_full) {
                stats->pipe_slot_full++;
              } else {
                stats->pipe_slot_empty++;
              }
            }
          }

          const bool refill_graph_slots =
              use_pipelined_graph_io || use_effective_pipeann_state_scheduler;
          _u32 dispatch_budget =
              refill_graph_slots
                  ? static_cast<_u32>(std::min<_u64>(
                        graph_pipe_width -
                            std::min<_u64>(active_disk_buffers,
                                           graph_pipe_width),
                        remaining_io_budget))
                  : static_cast<_u32>(
                        std::min<_u64>(beam_width, remaining_io_budget));
          if (use_effective_pipeann_state_scheduler) {
            dispatch_budget = static_cast<_u32>(std::min<_u64>(
                dispatch_budget, graph_free_sector_bufs.size()));
            // PipeANN issues at most one new graph I/O after it has ready work,
            // instead of immediately refilling every free slot after a burst of
            // completions. This keeps later requests informed by fresher
            // neighbor expansions and reduces speculative graph reads.
            if (dispatch_budget > 1 &&
                (n_proc_in_q > 0 || n_cached_in_q > 0)) {
              dispatch_budget = 1;
            }
          }
          const bool should_dispatch =
              use_effective_pipeann_state_scheduler
                  ? (dispatch_budget > 0 && n_cached_in_q == 0)
                  : (use_pipelined_graph_io
                         ? (dispatch_budget > 0 && n_cached_in_q == 0 &&
                            n_proc_in_q == 0)
                         : (n_io_in_q == 0 && n_cached_in_q == 0 &&
                            n_proc_in_q < beam_width / 2));
          if (should_dispatch) {
            part_timer.reset();
            frontier_read_reqs.clear();
            _u32 marker = 0;
            _u32 num_seen = 0;
            _u32 disk_seen = 0;
            const _u32 scheduler_rank_limit = pipeann_scheduler_rank_limit();

            while (marker < scheduler_rank_limit && num_seen < beam_width &&
                   disk_seen < dispatch_budget) {
              if (candidate_ready_for_graph_dispatch(
                      retset[marker], marker, scheduler_rank_limit)) {
                const unsigned id = retset[marker].id;
                unsigned mem_pos = node_in_mem_pos(id);
                if (mem_pos != INF) {
                  CachedGraphNode cn;
                  cn.id = id;
                  if (dynamic_graph_cache_size == 0 ||
                      mem_pos < dynamic_graph_cache_start) {
                    cn.cache_pos = mem_pos;
                    cached_node.push(cn);
                    set_candidate_state(id, PipeCandidateState::kCacheReady);
                    num_seen++;
                    n_cached_in_q++;
                    if (stats != nullptr) {
                      stats->n_cache_hits++;
                    }
                  } else if (copy_mem_graph_neighbors(id, mem_pos, cn.nbrs)) {
                    cached_node.push(cn);
                    set_candidate_state(id, PipeCandidateState::kCacheReady);
                    num_seen++;
                    n_cached_in_q++;
                    if (stats != nullptr) {
                      stats->n_cache_hits++;
                    }
                  } else {
                    mem_pos = INF;
                  }
                }
                if (mem_pos == INF && loaded_nbrs.find(id) != loaded_nbrs.end()) {
                  CachedGraphNode cn;
                  cn.id = id;
                  const auto loaded_iter = loaded_nbrs.find(id);
                  cn.nbrs = loaded_iter->second;
                  cached_node.push(cn);
                  set_candidate_state(id, PipeCandidateState::kCacheReady);
                  num_seen++;
                  n_cached_in_q++;
                  if (stats != nullptr) {
                    stats->pipe_rep_adj_hits++;
                    stats->pipe_rep_adj_skipped++;
                  }
                } else if (mem_pos == INF) {
                  if (page_visited.insert(id).second) {
                    num_seen++;
                    disk_seen++;
                    auto fn = std::make_shared<FrontierNode>(
                        id, id, gc_index_fid, retset[marker].distance);
                    frontier.push_back(fn);
                    set_candidate_state(id,
                                        PipeCandidateState::kGraphIoSubmitted);
                    set_page_state(id, PipePageState::kGraphIoSubmitted);
                  } else {
                    set_candidate_state(id, PipeCandidateState::kStale);
                    if (stats != nullptr) {
                      stats->pipe_stale_candidates++;
                    }
                  }
                }
                retset[marker].flag = false;
              }
              marker++;
            }

            if (stats != nullptr) {
              stats->dispatch_us += (double) part_timer.elapsed();
            }

            if (ftr_id < frontier.size()) {
              part_timer.reset();
              if (stats != nullptr) {
                stats->n_hops++;
              }
              n_io_in_q += frontier.size() - ftr_id;
              while (ftr_id < frontier.size()) {
                char *sector_buf = nullptr;
                if (use_effective_pipeann_state_scheduler) {
                  sector_buf = graph_free_sector_bufs.back();
                  graph_free_sector_bufs.pop_back();
                } else {
                  sector_buf = sector_scratch +
                               sector_scratch_idx * GR_SECTOR_LEN;
                  sector_scratch_idx =
                      (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                }
                auto offset =
                    (static_cast<_u64>(frontier[ftr_id]->pid)) * GR_SECTOR_LEN;
                offset += GR_SECTOR_LEN;
                sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                set_candidate_state(frontier[ftr_id]->id,
                                    PipeCandidateState::kGraphIoSubmitted);
                set_page_state(frontier[ftr_id]->pid,
                               PipePageState::kGraphIoSubmitted);
                frontier_read_reqs.push_back(
                    AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                if (stats != nullptr) {
                  stats->n_ios++;
                  if (use_effective_pipeann_state_scheduler) {
                    stats->pipe_graph_submitted++;
                  }
                }
                num_ios++;
                ftr_id++;
              }
              io_manager->submit_read_reqs(frontier_read_reqs, gc_index_fid,
                                           ctx);
              if (stats != nullptr) {
                stats->read_disk_us += (double) part_timer.elapsed();
              }
            }

            if (n_io_in_q == 0 && n_proc_in_q == 0 && n_cached_in_q == 0) {
              break;
            }
          }
        }
        part_timer.reset();

        frontier.clear();

        Timer refine_timer;
        _u32 l_idx = 0;
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) {
          embedding_search_L = k_search;
        }
        tsl::robin_set<_u32> refine_page_visited;
        refine_page_visited.reserve(embedding_search_L);
        auto process_refine_page = [&](char *sector_buf, const unsigned pid) {
          Timer exact_timer;
          compute_exact_dists_and_push(sector_buf, pid);
          refine_exact_us += (double) exact_timer.elapsed();
        };
        while (l_idx < embedding_search_L) {
          frontier_read_reqs.clear();
          cached_id_bufs.clear();
          tsl::robin_map<char*, unsigned> sec_buf2pid;
          for (_u32 ord_idx = l_idx;
               l_idx - ord_idx < MAX_N_SECTOR_READS &&
               l_idx < embedding_search_L; l_idx++) {
            const unsigned id = retset[l_idx].id;
            if (exact_visited.find(id) != exact_visited.end()) {
              continue;
            }
            const unsigned pid = id;
            if (refine_page_visited.find(pid) == refine_page_visited.end()) {
              char* cached_emb_buf = get_mem_emb_addr(id);
              if (cached_emb_buf != nullptr) {
                cached_id_bufs.push_back(std::make_pair(id, cached_emb_buf));
              } else {
                auto sector_buf =
                    sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx =
                    (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset =
                    (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN;
                frontier_read_reqs.push_back(
                    AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                refine_page_visited.insert(pid);
                sec_buf2pid.insert({sector_buf, pid});
              }
            }
          }
          int n_ops = 0;
          if (!frontier_read_reqs.empty()) {
            n_ops =
                io_manager->submit_read_reqs(frontier_read_reqs,
                                             refine_fid, ctx);
            if (stats != nullptr) {
              stats->n_emb_ios += n_ops;
            }
          }
          if (!cached_id_bufs.empty()) {
            for (_u64 i = 0; i < cached_id_bufs.size(); i++) {
              _mm_prefetch((char *) cached_id_bufs[i].second, _MM_HINT_T0);
              Timer exact_timer;
              compute_exact_dists_and_push(cached_id_bufs[i].second,
                                           cached_id_bufs[i].first);
              refine_exact_us += (double) exact_timer.elapsed();
            }
          }
          while (n_ops > 0) {
            Timer io_wait_timer;
            int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
            refine_io_wait_us += (double) io_wait_timer.elapsed();
            n_ops -= n_read_blks;
            for (int i = 0; i < n_read_blks; i++) {
              auto sector_buf = tmp_bufs[i];
              auto pid = sec_buf2pid[sector_buf];
              process_refine_page(sector_buf, pid);
            }
          }
        }
        const double refine_us = (double) refine_timer.elapsed();

        frontier_read_reqs.clear();
        visited.clear();
        page_visited.clear();
        exact_visited.clear();

        Timer sort_timer;
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
            continue;
          }
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              distances[t] = (-distances[t]);
              if (max_base_norm != 0) {
                distances[t] *= (max_base_norm * query_norm);
              }
            }
          }
          t++;
        }

        if (t < k_search) {
          diskann::cerr << "The number of unique ids is less than topk"
                        << std::endl;
          exit(1);
        }

        const double sort_us = (double) sort_timer.elapsed();
        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) part_timer.elapsed();
          stats->refine_us = (float) refine_us;
          stats->sort_us = (float) sort_us;
          stats->refine_io_wait_us = (float) refine_io_wait_us;
          stats->refine_exact_us = (float) refine_exact_us;
        }
      }
    });
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
} // namespace diskann
