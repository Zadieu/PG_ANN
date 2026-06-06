#include <immintrin.h>
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

  // data could be parse streamingly from queue.
  template<typename T>
  void DecoIndex<T>::page_search(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit,
      const float pq_filter_ratio, const float emb_search_ratio, QueryStats* stats_ptr) {
    // here are global checks / global data init.
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;
    const char *pipeline_env = std::getenv("GORGEOUS_PIPELINED_GRAPH_IO");
    const bool use_pipelined_graph_io =
        pipeline_env != nullptr && std::strcmp(pipeline_env, "0") != 0 && std::strcmp(pipeline_env, "false") != 0;
    const char *refine_pipeline_env = std::getenv("GORGEOUS_PIPELINED_REFINE_IO");
    const bool use_pipelined_refine_io = refine_pipeline_env != nullptr &&
        std::strcmp(refine_pipeline_env, "0") != 0 && std::strcmp(refine_pipeline_env, "false") != 0;
    const char *early_refine_env = std::getenv("GORGEOUS_EARLY_REFINE_PREFETCH");
    const bool use_early_refine_prefetch = early_refine_env != nullptr &&
        std::strcmp(early_refine_env, "0") != 0 && std::strcmp(early_refine_env, "false") != 0;
    const char *early_refine_depth_env = std::getenv("GORGEOUS_EARLY_REFINE_DEPTH");
    const _u64 early_refine_depth =
        early_refine_depth_env == nullptr ? 2 : std::max<_u64>(1, std::strtoull(early_refine_depth_env, nullptr, 10));
    const char *early_refine_max_io_env = std::getenv("GORGEOUS_EARLY_REFINE_MAX_IO");
    const _u32 early_refine_max_io = early_refine_max_io_env == nullptr
        ? 2
        : static_cast<_u32>(std::max<_u64>(1, std::strtoull(early_refine_max_io_env, nullptr, 10)));
    const char *pipeann_state_env = std::getenv("GORGEOUS_PIPEANN_STATE_MACHINE");
    const bool use_pipeann_state_machine = pipeann_state_env != nullptr &&
        std::strcmp(pipeann_state_env, "0") != 0 && std::strcmp(pipeann_state_env, "false") != 0;
    const char *pipeann_scheduler_env = std::getenv("GORGEOUS_PIPEANN_STATE_SCHEDULER");
    const bool use_pipeann_state_scheduler = pipeann_scheduler_env != nullptr &&
        std::strcmp(pipeann_scheduler_env, "0") != 0 && std::strcmp(pipeann_scheduler_env, "false") != 0;
    const char *pipeann_scheduler_window_env = std::getenv("GORGEOUS_PIPEANN_SCHEDULER_WINDOW");
    const bool use_adaptive_pipeann_scheduler_window =
        pipeann_scheduler_window_env == nullptr ||
        std::strcmp(pipeann_scheduler_window_env, "0") == 0 ||
        std::strcmp(pipeann_scheduler_window_env, "auto") == 0;
    const _u64 pipeann_scheduler_window = use_adaptive_pipeann_scheduler_window
        ? 0
        : std::strtoull(pipeann_scheduler_window_env, nullptr, 10);
    const char *pipeann_dynamic_pipe_env = std::getenv("GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH");
    const bool use_pipeann_dynamic_pipe_width = use_pipeann_state_scheduler &&
        pipeann_dynamic_pipe_env != nullptr &&
        std::strcmp(pipeann_dynamic_pipe_env, "0") != 0 &&
        std::strcmp(pipeann_dynamic_pipe_env, "false") != 0;
    const char *pipeann_pipe_start_env = std::getenv("GORGEOUS_PIPEANN_PIPE_START");
    const bool use_auto_pipeann_pipe_start =
        pipeann_pipe_start_env == nullptr ||
        std::strcmp(pipeann_pipe_start_env, "0") == 0 ||
        std::strcmp(pipeann_pipe_start_env, "auto") == 0;
    const _u64 pipeann_pipe_start = use_auto_pipeann_pipe_start
        ? 0
        : std::max<_u64>(1, std::strtoull(pipeann_pipe_start_env, nullptr, 10));
    const char *pipeann_pipe_min_env = std::getenv("GORGEOUS_PIPEANN_PIPE_MIN");
    const bool use_auto_pipeann_pipe_min =
        pipeann_pipe_min_env == nullptr ||
        std::strcmp(pipeann_pipe_min_env, "0") == 0 ||
        std::strcmp(pipeann_pipe_min_env, "auto") == 0;
    const _u64 pipeann_pipe_min = use_auto_pipeann_pipe_min
        ? 0
        : std::max<_u64>(1, std::strtoull(pipeann_pipe_min_env, nullptr, 10));
    const char *pipeann_pipe_feedback_env = std::getenv("GORGEOUS_PIPEANN_PIPE_FEEDBACK");
    const bool use_pipeann_pipe_feedback = use_pipeann_dynamic_pipe_width &&
        (pipeann_pipe_feedback_env == nullptr ||
         (std::strcmp(pipeann_pipe_feedback_env, "0") != 0 &&
          std::strcmp(pipeann_pipe_feedback_env, "false") != 0));
    const char *pipeann_pipe_waste_threshold_env = std::getenv("GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD");
    const float pipeann_pipe_waste_threshold = pipeann_pipe_waste_threshold_env == nullptr
        ? 0.10f
        : std::strtof(pipeann_pipe_waste_threshold_env, nullptr);
    const char *pipeann_resource_aware_env = std::getenv("GORGEOUS_PIPEANN_RESOURCE_AWARE");
    const bool use_pipeann_resource_aware = use_pipeann_pipe_feedback &&
        pipeann_resource_aware_env != nullptr &&
        std::strcmp(pipeann_resource_aware_env, "0") != 0 &&
        std::strcmp(pipeann_resource_aware_env, "false") != 0;
    const char *pipeann_pipe_down_waste_threshold_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_DOWN_WASTE_THRESHOLD");
    const float pipeann_pipe_down_waste_threshold =
        pipeann_pipe_down_waste_threshold_env == nullptr
            ? 0.35f
            : std::strtof(pipeann_pipe_down_waste_threshold_env, nullptr);
    const char *pipeann_pipe_feedback_window_env =
        std::getenv("GORGEOUS_PIPEANN_PIPE_FEEDBACK_WINDOW");
    const _u32 pipeann_pipe_feedback_window =
        pipeann_pipe_feedback_window_env == nullptr
            ? 8
            : static_cast<_u32>(std::max<_u64>(
                  1, std::strtoull(pipeann_pipe_feedback_window_env, nullptr, 10)));
    const char *pipeann_pipe_min_marker_env = std::getenv("GORGEOUS_PIPEANN_PIPE_MIN_MARKER");
    const _u32 pipeann_pipe_min_marker = pipeann_pipe_min_marker_env == nullptr
        ? 5
        : static_cast<_u32>(std::strtoull(pipeann_pipe_min_marker_env, nullptr, 10));
    const char *pipeann_resource_threads_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_THREAD_THRESHOLD");
    const _u32 pipeann_resource_thread_threshold =
        pipeann_resource_threads_env == nullptr
            ? 8
            : static_cast<_u32>(std::max<_u64>(
                  1, std::strtoull(pipeann_resource_threads_env, nullptr, 10)));
    const char *pipeann_resource_mem_l_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_MEM_L_THRESHOLD");
    const _u32 pipeann_resource_mem_l_threshold =
        pipeann_resource_mem_l_env == nullptr
            ? 1
            : static_cast<_u32>(std::strtoull(pipeann_resource_mem_l_env, nullptr, 10));
    const bool pipeann_resource_pressure =
        use_pipeann_resource_aware &&
        mem_L >= pipeann_resource_mem_l_threshold &&
        max_nthreads >= pipeann_resource_thread_threshold;
    const char *pipeann_resource_adaptive_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_ADAPTIVE");
    const bool use_pipeann_resource_adaptive =
        pipeann_resource_pressure &&
        (pipeann_resource_adaptive_env == nullptr ||
         (std::strcmp(pipeann_resource_adaptive_env, "0") != 0 &&
          std::strcmp(pipeann_resource_adaptive_env, "false") != 0));
    const bool pipeann_high_thread_pressure = max_nthreads >= 64;
    const char *pipeann_resource_window_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_WINDOW");
    const _u32 pipeann_resource_window =
        pipeann_resource_window_env == nullptr
            ? (pipeann_high_thread_pressure ? 16 : 32)
            : static_cast<_u32>(std::max<_u64>(
                  1, std::strtoull(pipeann_resource_window_env, nullptr, 10)));
    const char *pipeann_resource_probe_window_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_PROBE_WINDOW");
    const _u32 pipeann_resource_probe_window =
        pipeann_resource_probe_window_env == nullptr
            ? (pipeann_high_thread_pressure ? 2 : 8)
            : static_cast<_u32>(std::max<_u64>(
                  1, std::strtoull(pipeann_resource_probe_window_env, nullptr, 10)));
    const char *pipeann_resource_cooldown_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_COOLDOWN");
    const _u32 pipeann_resource_cooldown =
        pipeann_resource_cooldown_env == nullptr
            ? (pipeann_high_thread_pressure ? 32 * pipeann_resource_window
                                            : 8 * pipeann_resource_window)
            : static_cast<_u32>(std::strtoull(pipeann_resource_cooldown_env, nullptr, 10));
    const char *pipeann_resource_qps_gain_target_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_QPS_GAIN_TARGET");
    const float pipeann_resource_qps_gain_target =
        pipeann_resource_qps_gain_target_env == nullptr
            ? (pipeann_high_thread_pressure ? 1.02f : 1.00f)
            : std::strtof(pipeann_resource_qps_gain_target_env, nullptr);
    const char *pipeann_resource_qps_close_ratio_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_QPS_CLOSE_RATIO");
    const float pipeann_resource_qps_close_ratio =
        pipeann_resource_qps_close_ratio_env == nullptr
            ? (pipeann_high_thread_pressure ? 1.00f : 0.97f)
            : std::strtof(pipeann_resource_qps_close_ratio_env, nullptr);
    const char *pipeann_resource_stale_limit_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_STALE_LIMIT");
    const float pipeann_resource_stale_limit =
        pipeann_resource_stale_limit_env == nullptr
            ? (pipeann_high_thread_pressure ? 0.01f : 0.10f)
            : std::strtof(pipeann_resource_stale_limit_env, nullptr);
    const char *pipeann_resource_refresh_on_windows_env =
        std::getenv("GORGEOUS_PIPEANN_RESOURCE_REFRESH_ON_WINDOWS");
    const _u32 pipeann_resource_refresh_on_windows =
        pipeann_resource_refresh_on_windows_env == nullptr
            ? (pipeann_high_thread_pressure ? 1 : 4)
            : static_cast<_u32>(std::max<_u64>(
                  1, std::strtoull(pipeann_resource_refresh_on_windows_env, nullptr, 10)));
    const bool track_pipeann_states = use_pipeann_state_machine || use_pipeann_state_scheduler;

    std::mutex pipeann_resource_controller_mutex;
    bool pipeann_controller_has_recent_off_window = false;
    bool pipeann_controller_pipeline_enabled = !use_pipeann_resource_adaptive;
    bool pipeann_controller_probe = false;
    _u32 pipeann_controller_probe_remaining = 0;
    _u32 pipeann_controller_cooldown_remaining = 0;
    _u32 pipeann_controller_off_count = 0;
    _u32 pipeann_controller_on_count = 0;
    _u32 pipeann_controller_on_windows_since_refresh = 0;
    double pipeann_controller_on_stale_sum = 0.0;
    double pipeann_controller_recent_off_qps = 0.0;
    Timer pipeann_controller_off_timer;
    Timer pipeann_controller_on_timer;

    // atomic pointer to query.
    std::atomic_int cur_task = 0;
    // parallel for
    pool->runTask([&, this](int tid) {
      // thread local data init
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      // these pointers can init earlier
      const T *    query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      // sector scratch
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      // query <-> neighbor list
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      // visited map/set
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);  // for refine phase
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, bool>& exact_visited = *(query_scratch->exact_visited);
      tsl::robin_map<_u32, char*> page2sec_buf;
      page2sec_buf.reserve(1024);

      // pre-allocated data field for searching
      std::vector<std::pair<_u32, const char*>> vis_cand;
      vis_cand.reserve(20);   // at most researve 12 is enough
      // this is a ring queue for storing sector buffers ptr.
      // when read done, push a sector_buf to here, wait for execute
      CircleQueue<char*> sector_buffers(MAX_N_SECTOR_READS);
      // pre-allocated buffer, will clear up each iter/step.
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<int> read_fids(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      std::vector<std::pair<unsigned, char*>> cached_id_bufs(MAX_N_SECTOR_READS);
      CircleQueue<CachedGraphNode> cached_node(MAX_N_SECTOR_READS);
      char *early_refine_scratch = nullptr;
      std::vector<char*> early_refine_free_bufs;
      if (use_early_refine_prefetch) {
        diskann::alloc_aligned((void **) &early_refine_scratch,
                               early_refine_depth * (_u64) GR_SECTOR_LEN, 4096);
        early_refine_free_bufs.reserve(early_refine_depth);
        for (_u64 i = 0; i < early_refine_depth; i++) {
          early_refine_free_bufs.push_back(early_refine_scratch + i * GR_SECTOR_LEN);
        }
      }

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        // record the frontier node, also used to record search path.
        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0; // frontier id

        // get the current query pointers
        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;
        bool use_effective_pipeann_state_scheduler = use_pipeann_state_scheduler;
        if (use_pipeann_resource_adaptive) {
          std::lock_guard<std::mutex> guard(pipeann_resource_controller_mutex);
          if (!pipeann_controller_has_recent_off_window) {
            use_effective_pipeann_state_scheduler = false;
          } else if (pipeann_controller_pipeline_enabled) {
            if (pipeann_controller_on_windows_since_refresh >= pipeann_resource_refresh_on_windows) {
              pipeann_controller_pipeline_enabled = false;
              pipeann_controller_has_recent_off_window = false;
              pipeann_controller_off_count = 0;
              pipeann_controller_off_timer.reset();
              pipeann_controller_on_windows_since_refresh = 0;
              use_effective_pipeann_state_scheduler = false;
            } else {
              use_effective_pipeann_state_scheduler = true;
            }
          } else if (pipeann_controller_probe && pipeann_controller_probe_remaining > 0) {
            pipeann_controller_probe_remaining--;
            use_effective_pipeann_state_scheduler = true;
          } else if (pipeann_controller_cooldown_remaining > 0) {
            pipeann_controller_cooldown_remaining--;
            use_effective_pipeann_state_scheduler = false;
          } else {
            pipeann_controller_probe = true;
            pipeann_controller_probe_remaining = pipeann_resource_probe_window;
            pipeann_controller_on_count = 0;
            pipeann_controller_on_stale_sum = 0.0;
            pipeann_controller_on_timer.reset();
            pipeann_controller_probe_remaining--;
            use_effective_pipeann_state_scheduler = true;
          }
        }

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        // if inner product, we also normalize the query and set the last coordinate
        // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        // reset query
        query_scratch->reset();

        // query <-> PQ chunk centers distances
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
        auto candidate_ready_for_graph_dispatch = [&](const Neighbor& candidate,
                                                      const _u32 rank,
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
          return get_candidate_state(candidate.id) == PipeCandidateState::kInPool;
        };
        // lambda to batch compute query<-> node distances in PQ space
        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                                const _u64 n_ids,
                                                                float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
        };

        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          tmp_timer.reset();
          float cur_expanded_dist = dist_cmp->compare(query, (T*)node_buf,
                                                (unsigned) aligned_dim);
          if (stats != nullptr) {
            stats->n_ext_cmps++;
          }
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          exact_visited.insert({id, true});
          return cur_expanded_dist;
        };
        auto process_exact_page_for_visited = [&](char *sector_buf, unsigned pid) {
          for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
            const unsigned id = gp_layout_[pid][j];
            if (visited.find(id) != visited.end() && exact_visited.find(id) == exact_visited.end()) {
              char *node_buf = sector_buf + j * max_node_len;
              _mm_prefetch((char *) node_buf, _MM_HINT_T0);
              compute_exact_dists_and_push(node_buf, id);
            }
          }
        };

        auto add_to_retset = [&](const int nbor_id, const float nbor_dist, const bool flag) {
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            return;
          }
          Neighbor nn(nbor_id, nbor_dist, flag);
          // Return position in sorted list where nn inserted
          InsertIntoPool(retset.data(), cur_list_size, nn);
          set_candidate_state(nbor_id, PipeCandidateState::kInPool);
          if (cur_list_size < l_search) ++cur_list_size;
        };

        auto compute_and_push_nbrs = [&](const char *node_buf, const unsigned node_id) {
          unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
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
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
          }
        };

        auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
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
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L, (unsigned)l_search));
        }  else {
          // we only have one medoid.
          compute_and_add_to_retset(&medoids[0], 1);
        }

        std::sort(retset.begin(), retset.begin() + cur_list_size);

        if (stats != nullptr) {
          stats->preprocess_us += (double) part_timer.elapsed();
        }
        unsigned num_ios = 0;

        // map unfinished sector_buf to the frontier node.
        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;
        tsl::robin_map<char*, unsigned> early_refine_buf2pid;
        tsl::robin_set<unsigned> early_refine_seen_pages;

        // these data are count seperately
        _u32 n_io_in_q = 0; // how many io left
        _u32 n_cached_in_q = 0; // how many proc left
        _u32 n_proc_in_q = 0; // how many proc left
        _u32 n_early_refine_io_in_q = 0;
        _u32 n_early_refine_submitted = 0;
        _u32 pipeann_current_pipe_width = static_cast<_u32>(beam_width);
        _u32 pipeann_min_pipe_width = static_cast<_u32>(beam_width);
        _u32 pipeann_feedback_in_pool = 0;
        _u32 pipeann_feedback_total = 0;
        _u32 pipeann_feedback_window_in_pool = 0;
        _u32 pipeann_feedback_window_total = 0;
        _u32 pipeann_max_marker = 0;
        if (use_pipeann_dynamic_pipe_width && use_effective_pipeann_state_scheduler) {
          _u64 initial_pipe_width = pipeann_pipe_start;
          if (initial_pipe_width == 0) {
            initial_pipe_width = std::min<_u64>(4, beam_width);
          }
          pipeann_current_pipe_width = static_cast<_u32>(
              std::max<_u64>(1, std::min<_u64>(beam_width, initial_pipe_width)));
          _u64 min_pipe_width = pipeann_pipe_min;
          if (min_pipe_width == 0) {
            min_pipe_width = use_pipeann_resource_aware ? 1 : initial_pipe_width;
          }
          pipeann_min_pipe_width = static_cast<_u32>(
              std::max<_u64>(1, std::min<_u64>(pipeann_current_pipe_width, min_pipe_width)));
        }
        std::vector<char*> graph_free_sector_bufs;
        if (use_effective_pipeann_state_scheduler) {
          graph_free_sector_bufs.reserve(MAX_N_SECTOR_READS);
          for (_u32 i = 0; i < MAX_N_SECTOR_READS; ++i) {
            graph_free_sector_bufs.push_back(sector_scratch + i * GR_SECTOR_LEN);
          }
        }

        auto pipeann_scheduler_rank_limit = [&]() -> _u32 {
          if (!use_effective_pipeann_state_scheduler) {
            return cur_list_size;
          }
          const _u64 tight_rank_window = std::max<_u64>(k_search, beam_width * 4);
          const _u64 recall_rank_window = std::max<_u64>(k_search, beam_width * 8);
          const _u64 adaptive_rank_window =
              l_search <= tight_rank_window ? tight_rank_window : recall_rank_window;
          const _u64 configured_rank_window =
              use_adaptive_pipeann_scheduler_window ? adaptive_rank_window : pipeann_scheduler_window;
          return static_cast<_u32>(
              std::min<_u64>(cur_list_size, std::min<_u64>(l_search, configured_rank_window)));
        };

        auto pipeann_stable_prefix = [&]() -> _u32 {
          _u32 stable_prefix = 0;
          const _u32 scan_limit = static_cast<_u32>(std::min<_u64>(cur_list_size, l_search));
          for (_u32 rank = 0; rank < scan_limit; ++rank) {
            const PipeCandidateState state = get_candidate_state(retset[rank].id);
            if ((retset[rank].flag && state == PipeCandidateState::kInPool) ||
                state == PipeCandidateState::kStale) {
              break;
            }
            stable_prefix++;
          }
          return stable_prefix;
        };

        auto record_pipeann_graph_io_feedback = [&](const std::shared_ptr<FrontierNode>& fn) {
          if (!use_effective_pipeann_state_scheduler) {
            return;
          }
          const float retset_tail_distance = cur_list_size > 0
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
          pipeann_feedback_window_total++;
          if (in_pool) {
            pipeann_feedback_in_pool++;
            pipeann_feedback_window_in_pool++;
          }
          pipeann_max_marker = std::max(pipeann_max_marker, pipeann_stable_prefix());
          if (pipeann_max_marker < pipeann_pipe_min_marker ||
              pipeann_feedback_window_total < pipeann_pipe_feedback_window) {
            return;
          }
          const float waste_ratio =
              static_cast<float>(pipeann_feedback_window_total - pipeann_feedback_window_in_pool) /
              static_cast<float>(pipeann_feedback_window_total);
          const bool should_decrease =
              use_pipeann_resource_aware &&
              waste_ratio >= pipeann_pipe_down_waste_threshold &&
              pipeann_current_pipe_width > pipeann_min_pipe_width;
          const bool should_increase =
              (!pipeann_resource_pressure || pipeann_max_marker >= pipeann_pipe_min_marker * 2) &&
              waste_ratio <= pipeann_pipe_waste_threshold &&
              pipeann_current_pipe_width < beam_width;
          if (should_decrease) {
            pipeann_current_pipe_width--;
            pipeann_current_pipe_width = std::max<_u32>(pipeann_current_pipe_width, pipeann_min_pipe_width);
            if (stats != nullptr) {
              stats->pipe_width_decreases++;
            }
          } else if (should_increase) {
            pipeann_current_pipe_width++;
            pipeann_current_pipe_width = std::max<_u32>(pipeann_current_pipe_width, pipeann_min_pipe_width);
            pipeann_current_pipe_width = std::min<_u32>(pipeann_current_pipe_width, static_cast<_u32>(beam_width));
            if (stats != nullptr) {
              stats->pipe_width_increases++;
            }
          }
          pipeann_feedback_window_total = 0;
          pipeann_feedback_window_in_pool = 0;
        };

        auto submit_early_refine_reads = [&]() {
          if (!use_early_refine_prefetch || early_refine_free_bufs.empty() ||
              n_early_refine_submitted >= early_refine_max_io ||
              n_io_in_q > 0 || n_proc_in_q == 0) {
            return;
          }
          const _u32 early_refine_l = std::max<_u32>(
              (_u32) k_search, (_u32) (cur_list_size * emb_search_ratio));
          const _u32 scan_l = std::min<_u32>(cur_list_size, early_refine_l);
          frontier_read_reqs.clear();
          read_fids.clear();
          for (_u32 i = 0; i < scan_l && !early_refine_free_bufs.empty() &&
                           n_early_refine_submitted + frontier_read_reqs.size() < early_refine_max_io; i++) {
            const unsigned id = retset[i].id;
            if (retset[i].flag || exact_visited.find(id) != exact_visited.end()) {
              continue;
            }
            auto pid = id2page_[id];
            if (page_visited.find(pid) != page_visited.end() ||
                early_refine_seen_pages.find(pid) != early_refine_seen_pages.end()) {
              continue;
            }
            char* cached_emb_buf = get_mem_emb_addr(id);
            if (cached_emb_buf != nullptr) {
              _mm_prefetch((char *) cached_emb_buf, _MM_HINT_T0);
              compute_exact_dists_and_push(cached_emb_buf, id);
              continue;
            }
            char *sector_buf = early_refine_free_bufs.back();
            early_refine_free_bufs.pop_back();
            auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN;
            frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
            read_fids.push_back(index_fid);
            early_refine_buf2pid.insert({sector_buf, pid});
            early_refine_seen_pages.insert(pid);
          }
          if (!frontier_read_reqs.empty()) {
            const int submitted = io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
            n_early_refine_io_in_q += submitted;
            n_early_refine_submitted += submitted;
            if (stats != nullptr) {
              stats->n_emb_ios += submitted;
            }
          }
        };

        while (num_ios < io_limit || n_io_in_q > 0 || n_proc_in_q > 0 ||
               n_cached_in_q > 0 || n_early_refine_io_in_q > 0) {

          if (n_proc_in_q > 0) {
            part_timer.reset();
            auto sector_buf = sector_buffers.get();
            if (sec_buf2ftr.find(sector_buf) == sec_buf2ftr.end()) {
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }

            auto fn = sec_buf2ftr[sector_buf];
            const _u32 exact_id = fn->id;
            const _u32 pid = fn->pid;
            set_candidate_state(exact_id, PipeCandidateState::kGraphPageReady);
            set_page_state(pid, PipePageState::kGraphPageReady);
            unsigned p_size = gp_layout_[pid].size();
            unsigned* p_layout = gp_layout_[pid].data();
            // calculate the approx. dist. in the page
            compute_pq_dists(p_layout, p_size, dist_scratch);
            if (stats != nullptr) stats->n_cmps += p_size;

            unsigned cand_size = 0;
            for (unsigned j = 0; j < p_size; ++j) {
              unsigned id = p_layout[j];
              if (id != exact_id && dist_scratch[j] >= retset[cur_list_size - 1].distance * pq_filter_ratio) {
                  continue;
              } else {
                // replace only the other nodes
                auto fn = std::make_shared<FrontierNode>(id, pid, index_fid, dist_scratch[j]);
                frontier.push_back(fn);
                ftr_id++;
                if (visited.insert(id).second) {
                  add_to_retset(id, dist_scratch[j], false);
                }
              }
              char *node_buf = sector_buf + j * max_node_len;
              if (exact_visited.find(id) == exact_visited.end()) {
                compute_exact_dists_and_push(node_buf, id);
              }
              vis_cand[cand_size++] = std::make_pair(id, node_buf);
            }
            for (unsigned j = 0; j < cand_size; ++j) {
              compute_and_push_nbrs(vis_cand[j].second, vis_cand[j].first);
            }
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

            sec_buf2ftr.erase(sector_buf);
            set_candidate_state(exact_id, PipeCandidateState::kExpanded);
            set_page_state(pid, PipePageState::kExpanded);
            if (use_effective_pipeann_state_scheduler) {
              graph_free_sector_bufs.push_back(sector_buf);
            }
            n_proc_in_q--;
          }

          if (n_io_in_q + n_early_refine_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) min_r = 1;
            part_timer.reset();
            int n_read_blks = io_manager->get_events(
                ctx, min_r, n_io_in_q + n_early_refine_io_in_q, tmp_bufs);
            for (int i = n_read_blks - 1; i >= 0; i--) {
              auto graph_iter = sec_buf2ftr.find(tmp_bufs[i]);
              if (graph_iter != sec_buf2ftr.end()) {
                sector_buffers.push(tmp_bufs[i]);
                set_candidate_state(graph_iter->second->id, PipeCandidateState::kGraphPageReady);
                set_page_state(graph_iter->second->pid, PipePageState::kGraphPageReady);
                record_pipeann_graph_io_feedback(graph_iter->second);
                n_io_in_q--;
                n_proc_in_q++;
                continue;
              }
              auto refine_iter = early_refine_buf2pid.find(tmp_bufs[i]);
              if (refine_iter != early_refine_buf2pid.end()) {
                process_exact_page_for_visited(tmp_bufs[i], refine_iter->second);
                early_refine_buf2pid.erase(refine_iter);
                early_refine_free_bufs.push_back(tmp_bufs[i]);
                n_early_refine_io_in_q--;
                continue;
              }
            }
            if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
          }

          // calculate in memory node.
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
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
            set_candidate_state(cn.id, PipeCandidateState::kExpanded);
            n_cached_in_q--;
            if (stats != nullptr) stats->cache_proc_us += (double) part_timer.elapsed();
          }

          if (n_io_in_q > 0 || n_proc_in_q > 0 || n_cached_in_q > 0) {
            submit_early_refine_reads();
          }

          pipeann_max_marker = std::max(pipeann_max_marker, pipeann_stable_prefix());
          const _u64 graph_pipe_width = use_effective_pipeann_state_scheduler
              ? static_cast<_u64>(pipeann_current_pipe_width)
              : beam_width;
          const _u32 active_disk_buffers = n_io_in_q + n_proc_in_q;
          const _u32 remaining_io_budget =
              num_ios < io_limit ? static_cast<_u32>(io_limit - num_ios) : 0;
          if (use_effective_pipeann_state_scheduler) {
            const bool pipe_slots_full =
                active_disk_buffers >= graph_pipe_width || remaining_io_budget == 0;
            if (stats != nullptr) {
              stats->pipe_width_sum += static_cast<unsigned>(graph_pipe_width);
              stats->pipe_width_samples++;
              stats->pipe_width_max = std::max<unsigned>(stats->pipe_width_max,
                                                         static_cast<unsigned>(graph_pipe_width));
              if (pipe_slots_full) {
                stats->pipe_slot_full++;
              } else {
                stats->pipe_slot_empty++;
              }
            }
          }
          const bool refill_graph_slots = use_pipelined_graph_io || use_effective_pipeann_state_scheduler;
          _u32 dispatch_budget = refill_graph_slots
                                           ? static_cast<_u32>(std::min<_u64>(
                                                 graph_pipe_width - std::min<_u64>(active_disk_buffers, graph_pipe_width),
                                                 remaining_io_budget))
                                           : static_cast<_u32>(std::min<_u64>(beam_width, remaining_io_budget));
          if (use_effective_pipeann_state_scheduler) {
            dispatch_budget = static_cast<_u32>(
                std::min<_u64>(dispatch_budget, graph_free_sector_bufs.size()));
          }
          const bool should_dispatch =
              use_effective_pipeann_state_scheduler
                  ? (dispatch_budget > 0 && n_cached_in_q == 0)
                  : (use_pipelined_graph_io
                         ? (dispatch_budget > 0 && n_cached_in_q == 0 && n_proc_in_q == 0)
                         : (n_io_in_q == 0 && n_cached_in_q == 0 && n_proc_in_q < beam_width / 2));
          if (should_dispatch) {
            part_timer.reset();
            // clear iteration state
            frontier_read_reqs.clear();
            read_fids.clear();
            _u32 marker = 0;
            _u32 num_seen = 0;
            _u32 disk_seen = 0;
            const _u32 scheduler_rank_limit = pipeann_scheduler_rank_limit();

            // PipeANN-style refill: scan the current best rank window and fill
            // only the free disk slots with candidates that are still in-pool.
            while (marker < scheduler_rank_limit && num_seen < beam_width && disk_seen < dispatch_budget) {
              if (candidate_ready_for_graph_dispatch(retset[marker], marker, scheduler_rank_limit)) {
                unsigned mem_pos = node_in_mem_pos(retset[marker].id);
                if (mem_pos != INF) {
                  CachedGraphNode cn;
                  cn.id = retset[marker].id;
                  if (dynamic_graph_cache_size == 0 || mem_pos < dynamic_graph_cache_start) {
                    cn.cache_pos = mem_pos;
                    cached_node.push(cn);
                    set_candidate_state(retset[marker].id, PipeCandidateState::kCacheReady);
                    num_seen++;
                    n_cached_in_q++;
                    if (stats != nullptr) {
                      stats->n_cache_hits++;
                    }
                  } else if (copy_mem_graph_neighbors(retset[marker].id, mem_pos, cn.nbrs)) {
                    cached_node.push(cn);
                    set_candidate_state(retset[marker].id, PipeCandidateState::kCacheReady);
                    num_seen++;
                    n_cached_in_q++;
                    if (stats != nullptr) {
                      stats->n_cache_hits++;
                    }
                  } else {
                    mem_pos = INF;
                  }
                }
                if (mem_pos == INF) {
                  auto pid = id2page_[retset[marker].id];
                  if (page_visited.insert(pid).second) {
                    num_seen++;
                    disk_seen++;
                    auto fn = std::make_shared<FrontierNode>(retset[marker].id, pid, index_fid,
                                                        retset[marker].distance);
                    frontier.push_back(fn);
                    set_candidate_state(retset[marker].id, PipeCandidateState::kGraphIoSubmitted);
                    set_page_state(pid, PipePageState::kGraphIoSubmitted);
                  } else {
                    set_candidate_state(retset[marker].id, PipeCandidateState::kStale);
                    if (stats != nullptr) {
                      stats->pipe_stale_candidates++;
                    }
                  }
                }
                retset[marker].flag = false;
              }
              marker++;
            }
            if (use_effective_pipeann_state_scheduler && scheduler_rank_limit < cur_list_size) {
              for (_u32 stale_rank = scheduler_rank_limit; stale_rank < cur_list_size; ++stale_rank) {
                if (retset[stale_rank].flag &&
                    get_candidate_state(retset[stale_rank].id) == PipeCandidateState::kInPool) {
                  retset[stale_rank].flag = false;
                  set_candidate_state(retset[stale_rank].id, PipeCandidateState::kStale);
                  if (stats != nullptr) {
                    stats->pipe_stale_candidates++;
                  }
                }
              }
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

            // read nhoods of frontier ids
            if (ftr_id < frontier.size()) {
              part_timer.reset();
              if (stats != nullptr) stats->n_hops++;
              n_io_in_q += frontier.size() - ftr_id;
              while(ftr_id < frontier.size()) {
                char *sector_buf = nullptr;
                if (use_effective_pipeann_state_scheduler) {
                  sector_buf = graph_free_sector_bufs.back();
                  graph_free_sector_bufs.pop_back();
                } else {
                  sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                  sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                }
                auto offset = (static_cast<_u64>(frontier[ftr_id]->pid)) * GR_SECTOR_LEN;
                offset += GR_SECTOR_LEN; // one page for metadata
                sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                set_candidate_state(frontier[ftr_id]->id, PipeCandidateState::kGraphIoSubmitted);
                set_page_state(frontier[ftr_id]->pid, PipePageState::kGraphIoSubmitted);
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                read_fids.push_back(frontier[ftr_id]->fid); // equals to 0
                if (stats != nullptr) {
                  stats->n_ios++;
                  if (use_effective_pipeann_state_scheduler) {
                    stats->pipe_graph_submitted++;
                  }
                }
                num_ios++;
                ftr_id++;
              }
              io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
              if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            }
            if (n_io_in_q > 0 || n_proc_in_q > 0 || n_cached_in_q > 0) {
              submit_early_refine_reads();
            }
            if (n_io_in_q == 0 && n_proc_in_q == 0 &&
                n_cached_in_q == 0 && n_early_refine_io_in_q == 0) break;
          }

        }
        part_timer.reset();

        // deperated here!
        frontier.clear();

        // done traversal, start read exact embedding.
        _u32 l_idx = 0;
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) embedding_search_L = k_search;
        auto process_refine_page = [&](char *sector_buf, unsigned pid) {
          process_exact_page_for_visited(sector_buf, pid);
        };
        if (!use_pipelined_refine_io) {
          while (l_idx < embedding_search_L) {
            frontier_read_reqs.clear();
            cached_id_bufs.clear();
            read_fids.clear();
            // page visited don't need to be clear.
            tsl::robin_map<char*, unsigned> sec_buf2pid;
            for (_u32 ord_idx = l_idx; l_idx - ord_idx < MAX_N_SECTOR_READS && l_idx < embedding_search_L; l_idx++) {
              if (exact_visited.find(retset[l_idx].id) != exact_visited.end()) {
                continue;
              }
              auto pid = id2page_[retset[l_idx].id];
              if (page_visited.find(pid) == page_visited.end()) {
                char* cached_emb_buf = get_mem_emb_addr(retset[l_idx].id);
                if (cached_emb_buf != nullptr) {  // find data in vector cache
                  cached_id_bufs.push_back(std::make_pair(retset[l_idx].id, cached_emb_buf));
                } else {
                  auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                  sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                  auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN; // one page for metadata
                  frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                  read_fids.push_back(index_fid);
                  page_visited.insert(pid);
                  sec_buf2pid.insert({sector_buf, pid});
                }
              }
            }
            int n_ops = 0;
            if (frontier_read_reqs.size() != 0) {
              n_ops = io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
              if (stats != nullptr) {
                stats-> n_emb_ios += n_ops;
              }
            }
            // pipeline disk read and calculate cached node.
            if (cached_id_bufs.size() != 0) {
              for (_u64 i = 0; i < cached_id_bufs.size(); i++) {
                _mm_prefetch((char *) cached_id_bufs[i].second, _MM_HINT_T0);
                compute_exact_dists_and_push(cached_id_bufs[i].second, cached_id_bufs[i].first);
              }
            }
            while (n_ops > 0) {
              int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
              n_ops -= n_read_blks;
              for (int i = 0; i < n_read_blks; i++) {
                auto sector_buf = tmp_bufs[i];
                process_refine_page(sector_buf, sec_buf2pid[sector_buf]);
              }
            }
          }
        } else {
          tsl::robin_map<char*, unsigned> sec_buf2pid;
          _u32 cached_refine_idx = 0;
          int n_refine_io_in_q = 0;
          const _u64 refine_io_depth = MAX_N_SECTOR_READS;

          auto submit_refine_reads = [&]() {
            frontier_read_reqs.clear();
            read_fids.clear();
            while (l_idx < embedding_search_L &&
                   static_cast<_u64>(n_refine_io_in_q + frontier_read_reqs.size()) < refine_io_depth) {
              const unsigned id = retset[l_idx].id;
              l_idx++;
              if (exact_visited.find(id) != exact_visited.end()) {
                continue;
              }
              auto pid = id2page_[id];
              if (page_visited.find(pid) != page_visited.end()) {
                continue;
              }
              char* cached_emb_buf = get_mem_emb_addr(id);
              if (cached_emb_buf != nullptr) {
                cached_id_bufs.push_back(std::make_pair(id, cached_emb_buf));
                continue;
              }
              auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
              sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
              auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN; // one page for metadata
              frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
              read_fids.push_back(index_fid);
              page_visited.insert(pid);
              sec_buf2pid.insert({sector_buf, pid});
            }
            if (!frontier_read_reqs.empty()) {
              int submitted = io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
              n_refine_io_in_q += submitted;
              if (stats != nullptr) {
                stats-> n_emb_ios += submitted;
              }
            }
          };

          cached_id_bufs.clear();
          submit_refine_reads();
          while (l_idx < embedding_search_L || n_refine_io_in_q > 0 || cached_refine_idx < cached_id_bufs.size()) {
            while (cached_refine_idx < cached_id_bufs.size()) {
              _mm_prefetch((char *) cached_id_bufs[cached_refine_idx].second, _MM_HINT_T0);
              compute_exact_dists_and_push(cached_id_bufs[cached_refine_idx].second,
                                           cached_id_bufs[cached_refine_idx].first);
              cached_refine_idx++;
            }

            if (n_refine_io_in_q > 0) {
              const bool no_submit_capacity = static_cast<_u64>(n_refine_io_in_q) >= refine_io_depth;
              int min_r = (l_idx >= embedding_search_L || no_submit_capacity) ? 1 : 0;
              int n_read_blks = io_manager->get_events(ctx, min_r, n_refine_io_in_q, tmp_bufs);
              n_refine_io_in_q -= n_read_blks;
              for (int i = 0; i < n_read_blks; i++) {
                auto sector_buf = tmp_bufs[i];
                process_refine_page(sector_buf, sec_buf2pid[sector_buf]);
                sec_buf2pid.erase(sector_buf);
              }
            }
            submit_refine_reads();
          }
        }

        // clear the data.
        frontier_read_reqs.clear();
        read_fids.clear();
        if (use_pipeann_state_machine) {
          _u32 unfinished_candidates = 0;
          for (const auto& kv : candidate_state) {
            if (kv.second == PipeCandidateState::kGraphIoSubmitted ||
                kv.second == PipeCandidateState::kGraphPageReady ||
                kv.second == PipeCandidateState::kCacheReady) {
              unfinished_candidates++;
            }
          }
          if (unfinished_candidates > 0 && stats != nullptr) {
            stats->n_hops += 0;
          }
        }
        visited.clear();
        page_visited.clear();

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        // copy k_search values
        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
            continue;
          }
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              // flip the sign to convert min to max
              distances[t] = (-distances[t]);
              // rescale to revert back to original norms (cancelling the effect of
              // base and query pre-processing)
              if (max_base_norm != 0)
                distances[t] *= (max_base_norm * query_norm);
            }
          }
          t++;
        }

        if (t < k_search) {
          diskann::cerr << "The number of unique ids is less than topk" << std::endl;
          exit(1);
        }

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) part_timer.elapsed();
        }
        if (use_pipeann_resource_adaptive && stats != nullptr) {
          std::lock_guard<std::mutex> guard(pipeann_resource_controller_mutex);
          if (use_effective_pipeann_state_scheduler) {
            pipeann_controller_on_stale_sum += static_cast<double>(stats->pipe_stale_candidates);
            pipeann_controller_on_count++;
            const _u32 target =
                pipeann_controller_probe ? pipeann_resource_probe_window : pipeann_resource_window;
            if (pipeann_controller_on_count >= target && target > 0) {
              const double elapsed_us =
                  static_cast<double>(std::max<long long>(1, pipeann_controller_on_timer.elapsed()));
              const double probe_qps =
                  (static_cast<double>(pipeann_controller_on_count) * 1000000.0) / elapsed_us;
              const double avg_stale =
                  pipeann_controller_on_stale_sum / static_cast<double>(pipeann_controller_on_count);
              const bool stale_ok =
                  avg_stale <= static_cast<double>(pipeann_resource_stale_limit);
              const bool qps_ok =
                  pipeann_controller_has_recent_off_window &&
                  probe_qps >= pipeann_controller_recent_off_qps *
                                   static_cast<double>(pipeann_resource_qps_gain_target);
              const bool force_close =
                  pipeann_controller_has_recent_off_window &&
                  probe_qps < pipeann_controller_recent_off_qps *
                                  static_cast<double>(pipeann_resource_qps_close_ratio);
              const bool keep_pipeline = stale_ok && qps_ok;
              pipeann_controller_pipeline_enabled = keep_pipeline;
              pipeann_controller_probe = false;
              pipeann_controller_probe_remaining = 0;
              pipeann_controller_on_windows_since_refresh++;
              if (!keep_pipeline || force_close) {
                pipeann_controller_cooldown_remaining = pipeann_resource_cooldown;
              }
              if (pipeann_controller_on_windows_since_refresh >= pipeann_resource_refresh_on_windows) {
                pipeann_controller_pipeline_enabled = false;
                pipeann_controller_cooldown_remaining = std::max<_u32>(1, pipeann_resource_window);
                pipeann_controller_has_recent_off_window = false;
                pipeann_controller_off_count = 0;
                pipeann_controller_off_timer.reset();
                pipeann_controller_on_windows_since_refresh = 0;
              }
              pipeann_controller_on_count = 0;
              pipeann_controller_on_stale_sum = 0.0;
              pipeann_controller_on_timer.reset();
            }
          } else {
            pipeann_controller_off_count++;
            if (pipeann_controller_off_count >= pipeann_resource_window) {
              const double elapsed_us =
                  static_cast<double>(std::max<long long>(1, pipeann_controller_off_timer.elapsed()));
              const double avg_qps =
                  (static_cast<double>(pipeann_controller_off_count) * 1000000.0) / elapsed_us;
              pipeann_controller_recent_off_qps = avg_qps;
              pipeann_controller_has_recent_off_window = true;
              pipeann_controller_off_count = 0;
              pipeann_controller_off_timer.reset();
              pipeann_controller_on_windows_since_refresh = 0;
            }
          }
        }
      }
      if (early_refine_scratch != nullptr) {
        diskann::aligned_free((void *) early_refine_scratch);
      }
    });
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
} // namespace diskann
