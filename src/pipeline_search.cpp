#include "pipeline_search.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <queue>
#include <stdexcept>
#include "linux_aligned_file_reader.h"
#include "filter/attribute.h"
#include "filter/selector.h"

#include "io/page_reader.h"
#include "search/search_session.h"

namespace hybrid {
namespace {

constexpr size_t kFilterModeStatsCount = 4;

uint32_t EffectiveLPool(const SearchConfig &config) {
  return config.l_pool == 0 ? config.l_search : config.l_pool;
}

size_t FilterModeIndex(FilterSearchMode mode) {
  const size_t index = static_cast<size_t>(mode);
  if (index >= kFilterModeStatsCount) {
    throw std::runtime_error("unsupported filter search mode");
  }
  return index;
}

void RecordFilterVectorAccess(SearchStats *stats, FilterSearchMode mode, bool is_member) {
  if (stats == nullptr) {
    return;
  }
  const size_t index = FilterModeIndex(mode);
  ++stats->filter_accessed_vectors[index];
  stats->filter_false_positives[index] += is_member ? 0 : 1;
}

ApproxDistanceKind DefaultApproxKind(const IndexReader &index) {
  const std::string default_pq_pivots = DefaultPipeannPqPivotsPath(index.index_path());
  const std::string default_pq_compressed = DefaultPipeannPqCompressedPath(index.index_path());
  return (std::filesystem::exists(default_pq_pivots) && std::filesystem::exists(default_pq_compressed))
             ? ApproxDistanceKind::kProductQuantization
             : ApproxDistanceKind::kFullPrecision;
}

void ValidateSearchInputs(const IndexReader &index, const std::vector<float> &query, const SearchConfig &config) {
  if (query.size() != index.search_metadata().dim) {
    throw std::runtime_error("query dimension does not match index dimension");
  }
  if (config.top_k == 0 || config.beam_width == 0 || config.l_search == 0) {
    throw std::runtime_error("search config requires top_k > 0, beam_width > 0, and l_search > 0");
  }
}

void ValidateFilterableIndex(const IndexReader &index) {
  const auto *native = dynamic_cast<const NativeGorgeousIndex *>(&index);
  if (native == nullptr || native->native_metadata().attr_size == 0) {
    throw std::runtime_error("filter search requires a native Gorgeous index with embedded attributes");
  }
}

bool VerifyNodeMember(pipeann::Selector *selector,
                      const pipeann::Attributes &query_attrs,
                      uint32_t id,
                      const DiskNodeView &node) {
  if (selector == nullptr) {
    return true;
  }
  if (!node.has_attrs()) {
    return false;
  }
  const auto target_attrs = pipeann::Attributes::deserialize(static_cast<const char *>(node.attrs));
  return selector->is_member(id, query_attrs, target_attrs);
}

std::vector<SearchResult> RerankPrefilterCandidates(const IndexReader &index,
                                                    const std::vector<float> &query,
                                                    const std::vector<uint32_t> &candidate_ids,
                                                    pipeann::Selector *selector,
                                                    const pipeann::Attributes &query_attrs,
                                                    uint32_t top_k,
                                                    SearchStats *stats) {
  std::ifstream in(index.index_path(), std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open index file for prefilter rerank");
  }

  const auto io_start = std::chrono::steady_clock::now();
  std::vector<SearchResult> results;
  results.reserve(candidate_ids.size());
  std::vector<char> page_bytes(index.search_metadata().page_size);
  uint64_t page_reads = 0;
  for (uint32_t id : candidate_ids) {
    const uint32_t page_id = index.PageForNode(id);
    in.seekg(static_cast<std::streamoff>(index.PageOffset(page_id)), std::ios::beg);
    in.read(page_bytes.data(), static_cast<std::streamsize>(page_bytes.size()));
    if (!in) {
      throw std::runtime_error("failed to read index page during prefilter rerank");
    }
    ++page_reads;
    const PageView page = index.ViewPage(page_id, page_bytes);
    uint32_t slot = page.layout_size;
    for (uint32_t i = 0; i < page.layout_size; ++i) {
      if (page.LayoutAt(i) == id) {
        slot = i;
        break;
      }
    }
    if (slot >= page.layout_size) {
      continue;
    }
    const DiskNodeView node = index.ViewNode(page, slot);
    const bool is_member = VerifyNodeMember(selector, query_attrs, id, node);
    RecordFilterVectorAccess(stats, FilterSearchMode::kPreFilter, is_member);
    if (!is_member) {
      continue;
    }
    const float exact_distance =
        node.has_embedded_coords() ? L2Distance(query, node.coords, index.search_metadata().dim)
                                   : L2Distance(query, index.exact_vector(id));
    results.push_back({id, exact_distance});
  }

  std::sort(results.begin(), results.end(), [](const SearchResult &lhs, const SearchResult &rhs) {
    if (lhs.distance == rhs.distance) {
      return lhs.id < rhs.id;
    }
    return lhs.distance < rhs.distance;
  });
  if (results.size() > top_k) {
    results.resize(top_k);
  }
  if (stats != nullptr) {
    const size_t index = FilterModeIndex(FilterSearchMode::kPreFilter);
    stats->filter_reads[index] += page_reads;
    stats->filter_io_us1[index] += std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - io_start)
                                       .count();
  }
  return results;
}

std::vector<SearchResult> RunPrefilterSearch(const IndexReader &index,
                                             const std::vector<float> &query,
                                             const SearchConfig &config,
                                             pipeann::Selector *selector,
                                             const pipeann::Attributes &query_attrs,
                                             ApproxDistanceKind approx_kind,
                                             const std::string &pq_codebook_path,
                                             const std::string &pq_codes_path,
                                             SearchStats *stats) {
  const size_t mode_index = FilterModeIndex(FilterSearchMode::kPreFilter);
  LinuxAlignedFileReader reader;
  reader.open(index.index_path(), false, false);
  const auto io_start = std::chrono::steady_clock::now();
  const std::vector<uint32_t> filtered_ids = selector->pre_filter(query_attrs, &reader);
  if (stats != nullptr) {
    stats->selected_filter_mode = FilterSearchMode::kPreFilter;
    stats->filter_io_us[mode_index] += std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - io_start)
                                           .count();
  }
  if (filtered_ids.empty()) {
    return {};
  }

  const uint32_t limit = std::max<uint32_t>(EffectiveLPool(config), config.l_search);
  using Candidate = std::pair<float, uint32_t>;
  auto worse_first = [](const Candidate &lhs, const Candidate &rhs) { return lhs.first < rhs.first; };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(worse_first)> pq(worse_first);
  const auto cpu_start = std::chrono::steady_clock::now();

  if (approx_kind == ApproxDistanceKind::kProductQuantization) {
    PipeannProductQuantization pipeann_pq(index);
    pipeann_pq.Load(pq_codebook_path, pq_codes_path);
    std::vector<float> query_table;
    pipeann_pq.InitializeQuery(query, &query_table);
    std::vector<float> distances(filtered_ids.size());
    pipeann_pq.DistanceBatch(query_table, filtered_ids.data(), filtered_ids.size(), distances.data());
    for (size_t i = 0; i < filtered_ids.size(); ++i) {
      pq.push({distances[i], filtered_ids[i]});
      if (pq.size() > limit) {
        pq.pop();
      }
    }
  } else {
    FullPrecisionDistanceComputer full(index);
    full.BeginQuery(query);
    for (uint32_t id : filtered_ids) {
      pq.push({full.Distance(id), id});
      if (pq.size() > limit) {
        pq.pop();
      }
    }
  }
  if (stats != nullptr) {
    stats->filter_cpu_us[mode_index] += std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - cpu_start)
                                            .count();
  }

  std::vector<uint32_t> candidates(pq.size());
  for (size_t i = pq.size(); i > 0; --i) {
    candidates[i - 1] = pq.top().second;
    pq.pop();
  }
  return RerankPrefilterCandidates(index, query, candidates, selector, query_attrs, config.top_k, stats);
}

struct FilterCost {
  uint64_t ssd_reads = std::numeric_limits<uint64_t>::max();
  uint64_t dist_cmp = std::numeric_limits<uint64_t>::max();

  double total_cost() const {
    constexpr double kSSDReadWeight = 10.0;
    constexpr double kDistCmpWeight = 1.0;
    return static_cast<double>(ssd_reads) * kSSDReadWeight + static_cast<double>(dist_cmp) * kDistCmpWeight;
  }
};

struct FilterPlan {
  FilterSearchMode selected_mode = FilterSearchMode::kAuto;
  FilterCost prefilter;
  FilterCost postfilter;
  FilterCost infilter;
  uint32_t l_max = 0;
};

void RecordFilterPlan(SearchStats *stats, const FilterPlan &plan) {
  if (stats == nullptr) {
    return;
  }
  stats->selected_filter_mode = plan.selected_mode;
  stats->estimated_filter_reads[FilterModeIndex(FilterSearchMode::kPreFilter)] = plan.prefilter.ssd_reads;
  stats->estimated_filter_reads[FilterModeIndex(FilterSearchMode::kPostFilter)] = plan.postfilter.ssd_reads;
  stats->estimated_filter_reads[FilterModeIndex(FilterSearchMode::kInFilter)] = plan.infilter.ssd_reads;
  stats->estimated_filter_cmps[FilterModeIndex(FilterSearchMode::kPreFilter)] = plan.prefilter.dist_cmp;
  stats->estimated_filter_cmps[FilterModeIndex(FilterSearchMode::kPostFilter)] = plan.postfilter.dist_cmp;
  stats->estimated_filter_cmps[FilterModeIndex(FilterSearchMode::kInFilter)] = plan.infilter.dist_cmp;
}

FilterPlan BuildFilterPlan(const IndexReader &index,
                           pipeann::Selector *selector,
                           const pipeann::Attributes &query_attrs,
                           const SearchConfig &config,
                           const FilterSearchOptions &filter) {
  FilterPlan plan;
  plan.l_max = filter.l_max == 0 ? EffectiveLPool(config) : filter.l_max;
  if (filter.mode != FilterSearchMode::kAuto) {
    plan.selected_mode = filter.mode;
  } else {
    plan.selected_mode = FilterSearchMode::kAuto;
  }

  const double selectivity = std::max(selector->estimate_selectivity(query_attrs), std::numeric_limits<double>::epsilon());
  const double precision_in = std::max(selector->estimate_precision(query_attrs), std::numeric_limits<double>::epsilon());
  const uint64_t l_search = config.l_search;

  plan.prefilter.ssd_reads = selector->estimate_prefilter_reads(query_attrs) + plan.l_max;
  plan.prefilter.dist_cmp = static_cast<uint64_t>(selectivity * index.search_metadata().num_points);

  constexpr double kGamma = 0.05;
  const auto *native = dynamic_cast<const NativeGorgeousIndex *>(&index);
  const double range = native == nullptr ? 0.0 : static_cast<double>(native->native_metadata().range);
  const double range_dense = native == nullptr ? 0.0 : static_cast<double>(native->native_metadata().range_dense);
  double in_filter_l_eq = static_cast<double>(l_search);
  const double srd_over_p = precision_in == 0.0 ? 0.0 : selectivity * range_dense / precision_in;
  if (range_dense > 0.0) {
    if (srd_over_p <= range) {
      in_filter_l_eq = l_search / selectivity * range / range_dense;
    } else {
      in_filter_l_eq = l_search / precision_in;
    }
  }
  plan.infilter.ssd_reads = selector->estimate_infilter_reads(query_attrs) + static_cast<uint64_t>(in_filter_l_eq);
  plan.infilter.dist_cmp = static_cast<uint64_t>(in_filter_l_eq * (range + kGamma * range_dense));

  plan.postfilter.ssd_reads = static_cast<uint64_t>(l_search / selectivity);
  plan.postfilter.dist_cmp = static_cast<uint64_t>(range * l_search / selectivity);

  if (plan.selected_mode != FilterSearchMode::kAuto) {
    return plan;
  }
  if (plan.prefilter.total_cost() <= plan.infilter.total_cost() &&
      plan.prefilter.total_cost() <= plan.postfilter.total_cost()) {
    plan.selected_mode = FilterSearchMode::kPreFilter;
    return plan;
  }
  if (plan.infilter.total_cost() <= plan.postfilter.total_cost()) {
    plan.selected_mode = FilterSearchMode::kInFilter;
    return plan;
  }
  plan.selected_mode = FilterSearchMode::kPostFilter;
  return plan;
}

}  // namespace


PipelinedGraphReplicatedSearcher::PipelinedGraphReplicatedSearcher(const IndexReader &index)
    : index_(index) {
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PipeSearch(const std::vector<float> &query,
                                                                       const SearchConfig &config,
                                                                       SearchStats *stats) const {
  return Search(query, config, stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PipeSearch(const std::vector<float> &query,
                                                                       const SearchConfig &config,
                                                                       ApproxDistanceKind approx_kind,
                                                                       SearchStats *stats,
                                                                       const std::string &pq_codebook_path,
                                                                       const std::string &pq_codes_path) const {
  return Search(query, config, approx_kind, stats, pq_codebook_path, pq_codes_path);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::Search(const std::vector<float> &query,
                                                                   const SearchConfig &config,
                                                                   SearchStats *stats) const {
  return Search(query, config, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::Search(const std::vector<float> &query,
                                                                   const SearchConfig &config,
                                                                   ApproxDistanceKind approx_kind,
                                                                   SearchStats *stats,
                                                                   const std::string &pq_codebook_path,
                                                                   const std::string &pq_codes_path) const {
  ValidateSearchInputs(index_, query, config);

  SearchSession session(index_,
                        query,
                        config,
                        stats,
                        CreateBestEffortPageReader(index_),
                        approx_kind,
                        EffectiveLPool(config),
                        false,
                        false,
                        {},
                        {},
                        pq_codebook_path,
                        pq_codes_path);
  return session.Run();
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::Search(
    const std::vector<float> &query,
    const SearchConfig &config,
    std::unique_ptr<IApproximateDistanceComputer> approx_distance,
    SearchStats *stats) const {
  ValidateSearchInputs(index_, query, config);

  SearchSession session(index_,
                        query,
                        config,
                        stats,
                        CreateBestEffortPageReader(index_),
                        std::move(approx_distance),
                        EffectiveLPool(config));
  return session.Run();
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::RangeSearch(const std::vector<float> &query,
                                                                        const SearchConfig &config,
                                                                        float range,
                                                                        SearchStats *stats) const {
  return RangeSearch(query, config, range, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::RangeSearch(const std::vector<float> &query,
                                                                        const SearchConfig &config,
                                                                        float range,
                                                                        ApproxDistanceKind approx_kind,
                                                                        SearchStats *stats,
                                                                        const std::string &pq_codebook_path,
                                                                        const std::string &pq_codes_path) const {
  ValidateSearchInputs(index_, query, config);
  SearchConfig range_config = config;
  range_config.range_partial = range;
  SearchSession session(index_,
                        query,
                        range_config,
                        stats,
                        CreateBestEffortPageReader(index_),
                        approx_kind,
                        EffectiveLPool(config),
                        false,
                        true,
                        {},
                        {},
                        pq_codebook_path,
                        pq_codes_path);
  return session.Run();
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::FilterSearch(const std::vector<float> &query,
                                                                         const SearchConfig &config,
                                                                         const FilterSearchOptions &filter,
                                                                         SearchStats *stats) const {
  return FilterSearch(query, config, filter, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::FilterSearch(const std::vector<float> &query,
                                                                         const SearchConfig &config,
                                                                         const FilterSearchOptions &filter,
                                                                         ApproxDistanceKind approx_kind,
                                                                         SearchStats *stats,
                                                                         const std::string &pq_codebook_path,
                                                                         const std::string &pq_codes_path) const {
  ValidateSearchInputs(index_, query, config);
  if (filter.selector == nullptr || filter.query_attrs == nullptr) {
    throw std::runtime_error("filter search requires both selector and query attributes");
  }
  ValidateFilterableIndex(index_);

  LinuxAlignedFileReader io_reader;
  io_reader.open(index_.index_path(), false, false);
  std::unique_ptr<pipeann::Selector> selector(filter.selector->copy());
  const pipeann::Attributes &query_attrs = *filter.query_attrs;
  const FilterPlan plan = BuildFilterPlan(index_, selector.get(), query_attrs, config, filter);
  RecordFilterPlan(stats, plan);
  if (plan.selected_mode == FilterSearchMode::kPreFilter) {
    if (stats != nullptr) {
      stats->filter_reads[FilterModeIndex(FilterSearchMode::kPreFilter)] += selector->estimate_prefilter_reads(query_attrs);
    }
    return RunPrefilterSearch(index_,
                              query,
                              config,
                              selector.get(),
                              query_attrs,
                              approx_kind,
                              pq_codebook_path,
                              pq_codes_path,
                              stats);
  }

  if (plan.selected_mode == FilterSearchMode::kInFilter) {
    const auto io_start = std::chrono::steady_clock::now();
    selector->prepare_in_filter(query_attrs, &io_reader);
    if (stats != nullptr) {
      const size_t mode_index = FilterModeIndex(FilterSearchMode::kInFilter);
      stats->filter_reads[mode_index] += selector->estimate_infilter_reads(query_attrs);
      stats->filter_io_us[mode_index] += std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - io_start)
                                             .count();
    }
  }
  const bool use_dense_neighbors = plan.selected_mode == FilterSearchMode::kInFilter;
  const auto approx_member = [selector_ptr = selector.get(), &query_attrs, use_dense_neighbors](uint32_t id) {
    return !use_dense_neighbors || selector_ptr->is_member_approx(id, query_attrs);
  };
  const auto verify_member = [selector_ptr = selector.get(), &query_attrs, stats, mode = plan.selected_mode](
                                 uint32_t id, const DiskNodeView &node) {
    const bool is_member = VerifyNodeMember(selector_ptr, query_attrs, id, node);
    RecordFilterVectorAccess(stats, mode, is_member);
    return is_member;
  };

  SearchSession session(index_,
                        query,
                        config,
                        stats,
                        CreateBestEffortPageReader(index_),
                        approx_kind,
                        plan.l_max,
                        use_dense_neighbors,
                        false,
                        approx_member,
                        verify_member,
                        pq_codebook_path,
                        pq_codes_path);
  std::vector<SearchResult> results = session.Run();
  if (stats != nullptr &&
      (plan.selected_mode == FilterSearchMode::kPostFilter || plan.selected_mode == FilterSearchMode::kInFilter)) {
    stats->filter_reads[FilterModeIndex(plan.selected_mode)] += stats->n_ios;
  }
  return results;
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PreFilterSearch(const std::vector<float> &query,
                                                                             const SearchConfig &config,
                                                                             pipeann::Selector *selector,
                                                                             const pipeann::Attributes &query_attrs,
                                                                             uint32_t l_max,
                                                                             SearchStats *stats) const {
  return PreFilterSearch(query, config, selector, query_attrs, l_max, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PreFilterSearch(
    const std::vector<float> &query,
    const SearchConfig &config,
    pipeann::Selector *selector,
    const pipeann::Attributes &query_attrs,
    uint32_t l_max,
    ApproxDistanceKind approx_kind,
    SearchStats *stats,
    const std::string &pq_codebook_path,
    const std::string &pq_codes_path) const {
  FilterSearchOptions filter;
  filter.mode = FilterSearchMode::kPreFilter;
  filter.selector = selector;
  filter.query_attrs = &query_attrs;
  filter.l_max = l_max;
  return FilterSearch(query, config, filter, approx_kind, stats, pq_codebook_path, pq_codes_path);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PostFilterSearch(const std::vector<float> &query,
                                                                              const SearchConfig &config,
                                                                              pipeann::Selector *selector,
                                                                              const pipeann::Attributes &query_attrs,
                                                                              uint32_t l_max,
                                                                              SearchStats *stats) const {
  return PostFilterSearch(query, config, selector, query_attrs, l_max, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::PostFilterSearch(
    const std::vector<float> &query,
    const SearchConfig &config,
    pipeann::Selector *selector,
    const pipeann::Attributes &query_attrs,
    uint32_t l_max,
    ApproxDistanceKind approx_kind,
    SearchStats *stats,
    const std::string &pq_codebook_path,
    const std::string &pq_codes_path) const {
  FilterSearchOptions filter;
  filter.mode = FilterSearchMode::kPostFilter;
  filter.selector = selector;
  filter.query_attrs = &query_attrs;
  filter.l_max = l_max;
  return FilterSearch(query, config, filter, approx_kind, stats, pq_codebook_path, pq_codes_path);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::InFilterSearch(const std::vector<float> &query,
                                                                            const SearchConfig &config,
                                                                            pipeann::Selector *selector,
                                                                            const pipeann::Attributes &query_attrs,
                                                                            uint32_t l_max,
                                                                            SearchStats *stats) const {
  return InFilterSearch(query, config, selector, query_attrs, l_max, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::InFilterSearch(
    const std::vector<float> &query,
    const SearchConfig &config,
    pipeann::Selector *selector,
    const pipeann::Attributes &query_attrs,
    uint32_t l_max,
    ApproxDistanceKind approx_kind,
    SearchStats *stats,
    const std::string &pq_codebook_path,
    const std::string &pq_codes_path) const {
  FilterSearchOptions filter;
  filter.mode = FilterSearchMode::kInFilter;
  filter.selector = selector;
  filter.query_attrs = &query_attrs;
  filter.l_max = l_max;
  return FilterSearch(query, config, filter, approx_kind, stats, pq_codebook_path, pq_codes_path);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::AutoFilterSearch(const std::vector<float> &query,
                                                                              const SearchConfig &config,
                                                                              pipeann::Selector *selector,
                                                                              const pipeann::Attributes &query_attrs,
                                                                              uint32_t l_max,
                                                                              SearchStats *stats) const {
  return AutoFilterSearch(query, config, selector, query_attrs, l_max, DefaultApproxKind(index_), stats);
}

std::vector<SearchResult> PipelinedGraphReplicatedSearcher::AutoFilterSearch(
    const std::vector<float> &query,
    const SearchConfig &config,
    pipeann::Selector *selector,
    const pipeann::Attributes &query_attrs,
    uint32_t l_max,
    ApproxDistanceKind approx_kind,
    SearchStats *stats,
    const std::string &pq_codebook_path,
    const std::string &pq_codes_path) const {
  FilterSearchOptions filter;
  filter.mode = FilterSearchMode::kAuto;
  filter.selector = selector;
  filter.query_attrs = &query_attrs;
  filter.l_max = l_max;
  return FilterSearch(query, config, filter, approx_kind, stats, pq_codebook_path, pq_codes_path);
}

}  // namespace hybrid
