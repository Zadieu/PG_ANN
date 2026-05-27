#include <exception>
#include <iostream>
#include <string>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_inspect [options]\n"
      << "  --index PATH\n"
      << "  --approx PATH (optional full-precision data override)\n"
      << "  --page_id N\n"
      << "  --pq_codebook PATH (optional PQ pivots override)\n"
      << "  --pq_codes PATH (optional PQ compressed override)\n"
      << "  --help\n";
}

uint32_t ParseUint32(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint32_t>(std::stoul(value));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + name);
  }
}

hybrid::InspectToolConfig ParseArgs(int argc, char **argv) {
  hybrid::InspectToolConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--help") {
      PrintUsage();
      std::exit(0);
    }
    if (arg == "--index") {
      config.index_path = need_value("--index");
      continue;
    }
    if (arg == "--approx") {
      config.approx_path = need_value("--approx");
      continue;
    }
    if (arg == "--page_id") {
      config.inspect_page = true;
      config.page_id = ParseUint32("--page_id", need_value("--page_id"));
      continue;
    }
    if (arg == "--pq_codebook") {
      config.pq_codebook_path = need_value("--pq_codebook");
      continue;
    }
    if (arg == "--pq_codes") {
      config.pq_codes_path = need_value("--pq_codes");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (config.index_path.empty()) {
    throw std::runtime_error("--index is required");
  }
  if (config.pq_codebook_path.empty() != config.pq_codes_path.empty()) {
    throw std::runtime_error("--pq_codebook and --pq_codes must be provided together");
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const hybrid::InspectToolConfig config = ParseArgs(argc, argv);
    const hybrid::InspectToolSummary summary = hybrid::RunInspectTool(config);
    std::cout << "Inspect completed\n";
    std::cout << "  points=" << summary.index_metadata.num_points << '\n';
    std::cout << "  pages=" << summary.index_metadata.num_pages << '\n';
    std::cout << "  dim=" << summary.index_metadata.dim << '\n';
    std::cout << "  entry_id=" << summary.index_metadata.entry_id << '\n';
    std::cout << "  page_size=" << summary.index_metadata.page_size << '\n';
    std::cout << "  storage_format="
              << (summary.storage_format == hybrid::IndexStorageFormat::kGorgeousNative ? "gorgeous_native"
                                                                                        : "project_graphrep")
              << '\n';
    if (summary.has_project_metadata) {
      std::cout << "  max_degree=" << summary.project_metadata.max_degree << '\n';
      std::cout << "  max_base_degree=" << summary.project_metadata.max_base_degree << '\n';
      std::cout << "  max_page_nodes=" << summary.project_metadata.max_page_nodes << '\n';
      std::cout << "  project_version=" << summary.project_metadata.version << '\n';
    }
    if (summary.has_native_metadata) {
      std::cout << "  native_nodes_per_sector=" << summary.native_metadata.nodes_per_sector << '\n';
      std::cout << "  native_max_node_len=" << summary.native_metadata.max_node_len << '\n';
      std::cout << "  native_range=" << summary.native_metadata.range << '\n';
      std::cout << "  native_range_dense=" << summary.native_metadata.range_dense << '\n';
      std::cout << "  native_r_ood=" << summary.native_metadata.r_ood << '\n';
      std::cout << "  native_page_boundaries=";
      for (size_t i = 0; i < summary.native_page_boundaries.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << summary.native_page_boundaries[i];
      }
      std::cout << '\n';
      std::cout << "  native_full_precision_payload="
                << (summary.native_payload.has_full_precision_payload ? 1 : 0) << '\n';
      if (summary.native_payload.has_full_precision_payload) {
        std::cout << "  native_full_precision_start_offset="
                  << summary.native_payload.full_precision_payload_start_offset << '\n';
        std::cout << "  native_full_precision_end_offset="
                  << summary.native_payload.full_precision_payload_end_offset << '\n';
        std::cout << "  native_full_precision_dim=" << summary.native_payload.full_precision_dim << '\n';
        std::cout << "  native_full_precision_nodes_per_sector="
                  << summary.native_payload.full_precision_nodes_per_sector << '\n';
      }
    }
    std::cout << "  index_file_size=" << summary.index_file_size << '\n';
    std::cout << "  full_data_file_size=" << summary.approx_file_size << '\n';
    std::cout << "  has_partition=" << (summary.has_partition ? 1 : 0) << '\n';
    std::cout << "  has_reorder=" << (summary.has_reorder ? 1 : 0) << '\n';
    if (summary.has_reorder) {
      std::cout << "  reorder_entries=" << summary.reorder_entries << '\n';
    }
    if (summary.has_pipeann_refine) {
      std::cout << "  pipeann_refine_num_points=" << summary.pipeann_refine_header.num_points << '\n';
      std::cout << "  pipeann_refine_range=" << summary.pipeann_refine_header.range << '\n';
      std::cout << "  pipeann_refine_range_base=" << summary.pipeann_refine_header.range_base << '\n';
      std::cout << "  pipeann_refine_range_dense=" << summary.pipeann_refine_header.range_dense << '\n';
      std::cout << "  pipeann_refine_r_ood=" << summary.pipeann_refine_header.r_ood << '\n';
      std::cout << "  pipeann_refine_nodes=" << summary.pipeann_refine_nodes << '\n';
      std::cout << "  pipeann_refine_edges=" << summary.pipeann_refine_edges << '\n';
      std::cout << "  pipeann_refine_n_cmps=" << summary.pipeann_refine_header.stats.n_cmps << '\n';
      std::cout << "  pipeann_refine_cpu_us=" << summary.pipeann_refine_header.stats.cpu_us << '\n';
    }
    std::cout << "  has_pipeann_refine_nodes=" << (summary.has_pipeann_refine_nodes ? 1 : 0) << '\n';
    if (summary.has_page) {
      std::cout << "  page_id=" << summary.page.page_id << '\n';
      std::cout << "  page_target_id=" << summary.page.target_id << '\n';
      std::cout << "  page_layout_size=" << summary.page.layout.size() << '\n';
      std::cout << "  page_layout=";
      for (size_t i = 0; i < summary.page.layout.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << summary.page.layout[i];
      }
      std::cout << '\n';
      std::cout << "  page_base_degrees=";
      for (size_t i = 0; i < summary.page.nnbrs.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << summary.page.nnbrs[i];
      }
      std::cout << '\n';
      std::cout << "  page_dense_degrees=";
      for (size_t i = 0; i < summary.page.n_dense_nbrs.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << summary.page.n_dense_nbrs[i];
      }
      std::cout << '\n';
      if (summary.has_pipeann_refine && !summary.page_refine_neighbors.empty()) {
        std::cout << "  page_refine_degree=" << summary.page_refine_degree << '\n';
        std::cout << "  page_refine_max_eh=" << summary.page_refine_max_eh << '\n';
        std::cout << "  page_refine_neighbors=";
        for (size_t i = 0; i < summary.page_refine_neighbors.size(); ++i) {
          std::cout << (i == 0 ? "" : ",") << summary.page_refine_neighbors[i];
        }
        std::cout << '\n';
        std::cout << "  page_refine_ehs=";
        for (size_t i = 0; i < summary.page_refine_ehs.size(); ++i) {
          std::cout << (i == 0 ? "" : ",") << summary.page_refine_ehs[i];
        }
        std::cout << '\n';
      }
    }
    if (summary.has_pq) {
      std::cout << "  pq_dim=" << summary.pq_metadata.dim << '\n';
      std::cout << "  pq_num_points=" << summary.pq_metadata.num_points << '\n';
      std::cout << "  pq_num_subspaces=" << summary.pq_metadata.num_subspaces << '\n';
      std::cout << "  pq_subspace_dim=" << summary.pq_metadata.subspace_dim << '\n';
      std::cout << "  pq_centroids_per_subspace=" << summary.pq_metadata.centroids_per_subspace << '\n';
      std::cout << "  pipeann_pq_pivots_file_size=" << summary.pq_codebook_file_size << '\n';
      std::cout << "  pipeann_pq_compressed_file_size=" << summary.pq_codes_file_size << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Inspect failed: " << e.what() << '\n';
    return 1;
  }
}
