#include <exception>
#include <iostream>
#include <string>

#include "tools/build_pipeline.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_build [options]\n"
      << "  --mode toy|text|fvecs|bvecs|bin\n"
      << "  --input PATH\n"
      << "  --train_query_mode text|fvecs|bvecs|bin\n"
      << "  --train_query_path PATH\n"
      << "  --output_dir DIR\n"
      << "  --dataset_name NAME\n"
      << "  --degree N\n"
      << "  --dense_degree N\n"
      << "  --r_ood N\n"
      << "  --build_l N\n"
      << "  --build_ram_budget_gb N\n"
      << "  --build_threads N\n"
      << "  --l_ood N\n"
      << "  --page_nodes N\n"
      << "  --partition_scale N\n"
      << "  --partition_ldg_times N\n"
      << "  --project_compatible_output\n"
      << "  --gorgeous_native_output\n"
      << "  --entry_id N\n"
      << "  --pq_subspaces N\n"
      << "  --pq_centroids N\n"
      << "  --pq_iterations N\n"
      << "  --toy_points N\n"
      << "  --help\n";
}

uint32_t ParseUint32(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint32_t>(std::stoul(value));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + name);
  }
}

float ParseFloat(const std::string &name, const std::string &value) {
  try {
    return std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid floating-point value for " + name);
  }
}

hybrid::BuildConfig ParseArgs(int argc, char **argv) {
  hybrid::BuildConfig config;
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
    if (arg == "--mode") {
      const std::string value = need_value("--mode");
      if (value == "toy") {
        config.input_mode = hybrid::VectorInputMode::kToy;
      } else if (value == "text") {
        config.input_mode = hybrid::VectorInputMode::kText;
      } else if (value == "fvecs") {
        config.input_mode = hybrid::VectorInputMode::kFvecs;
      } else if (value == "bvecs") {
        config.input_mode = hybrid::VectorInputMode::kBvecs;
      } else if (value == "bin") {
        config.input_mode = hybrid::VectorInputMode::kBin;
      } else {
        throw std::runtime_error("unsupported mode: " + value);
      }
      continue;
    }
    if (arg == "--input") {
      config.input_path = need_value("--input");
      continue;
    }
    if (arg == "--train_query_mode") {
      const std::string value = need_value("--train_query_mode");
      if (value == "text") {
        config.train_query_mode = hybrid::VectorInputMode::kText;
      } else if (value == "fvecs") {
        config.train_query_mode = hybrid::VectorInputMode::kFvecs;
      } else if (value == "bvecs") {
        config.train_query_mode = hybrid::VectorInputMode::kBvecs;
      } else if (value == "bin") {
        config.train_query_mode = hybrid::VectorInputMode::kBin;
      } else {
        throw std::runtime_error("unsupported train_query_mode: " + value);
      }
      continue;
    }
    if (arg == "--train_query_path") {
      config.train_query_path = need_value("--train_query_path");
      continue;
    }
    if (arg == "--output_dir") {
      config.output_dir = need_value("--output_dir");
      continue;
    }
    if (arg == "--dataset_name") {
      config.dataset_name = need_value("--dataset_name");
      continue;
    }
    if (arg == "--degree") {
      config.degree = ParseUint32("--degree", need_value("--degree"));
      continue;
    }
    if (arg == "--dense_degree") {
      config.dense_degree = ParseUint32("--dense_degree", need_value("--dense_degree"));
      continue;
    }
    if (arg == "--r_ood") {
      config.r_ood = ParseUint32("--r_ood", need_value("--r_ood"));
      continue;
    }
    if (arg == "--build_l") {
      config.build_l = ParseUint32("--build_l", need_value("--build_l"));
      continue;
    }
    if (arg == "--build_ram_budget_gb") {
      config.build_ram_budget_gb = ParseUint32("--build_ram_budget_gb", need_value("--build_ram_budget_gb"));
      continue;
    }
    if (arg == "--build_threads") {
      config.build_threads = ParseUint32("--build_threads", need_value("--build_threads"));
      continue;
    }
    if (arg == "--build_candidates") {
      config.build_candidates = ParseUint32("--build_candidates", need_value("--build_candidates"));
      continue;
    }
    if (arg == "--build_alpha") {
      config.build_alpha = ParseFloat("--build_alpha", need_value("--build_alpha"));
      continue;
    }
    if (arg == "--l_ood") {
      config.l_ood = ParseUint32("--l_ood", need_value("--l_ood"));
      continue;
    }
    if (arg == "--page_nodes") {
      config.page_nodes = ParseUint32("--page_nodes", need_value("--page_nodes"));
      continue;
    }
    if (arg == "--partition_scale") {
      config.partition_scale = ParseUint32("--partition_scale", need_value("--partition_scale"));
      continue;
    }
    if (arg == "--partition_ldg_times") {
      config.partition_ldg_times = ParseUint32("--partition_ldg_times", need_value("--partition_ldg_times"));
      continue;
    }
    if (arg == "--project_compatible_output") {
      config.output_mode = hybrid::BuildOutputMode::kProjectCompatible;
      continue;
    }
    if (arg == "--gorgeous_native_output") {
      config.output_mode = hybrid::BuildOutputMode::kGorgeousNative;
      continue;
    }
    if (arg == "--entry_id") {
      config.use_explicit_entry_id = true;
      config.entry_id = ParseUint32("--entry_id", need_value("--entry_id"));
      continue;
    }
    if (arg == "--pq_subspaces") {
      config.pq_subspaces = ParseUint32("--pq_subspaces", need_value("--pq_subspaces"));
      continue;
    }
    if (arg == "--pq_centroids") {
      config.pq_centroids = ParseUint32("--pq_centroids", need_value("--pq_centroids"));
      continue;
    }
    if (arg == "--pq_iterations") {
      config.pq_iterations = ParseUint32("--pq_iterations", need_value("--pq_iterations"));
      continue;
    }
    if (arg == "--toy_points") {
      config.toy_points = ParseUint32("--toy_points", need_value("--toy_points"));
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const hybrid::BuildConfig config = ParseArgs(argc, argv);
    const hybrid::BuildArtifacts artifacts = hybrid::RunBuildPipeline(config);
    std::cout << "Build completed\n";
    std::cout << "  dataset=" << config.dataset_name << '\n';
    std::cout << "  builder=pipeann_original\n";
    std::cout << "  points=" << artifacts.num_points << '\n';
    std::cout << "  dim=" << artifacts.dim << '\n';
    std::cout << "  output_mode="
              << (artifacts.output_mode == hybrid::BuildOutputMode::kProjectCompatible
                      ? "project-compatible"
                      : "gorgeous-native")
              << '\n';
    std::cout << "  r_ood=" << config.r_ood << '\n';
    std::cout << "  build_n_cmps=" << artifacts.pipeann_build_stats.n_cmps << '\n';
    std::cout << "  build_n_prunes=" << artifacts.pipeann_build_stats.n_prunes << '\n';
    std::cout << "  build_n_refine_queries=" << artifacts.pipeann_build_stats.n_refine_queries << '\n';
    std::cout << "  build_n_refine_edges=" << artifacts.pipeann_build_stats.n_refine_edges << '\n';
    std::cout << "  build_n_dense_edges=" << artifacts.pipeann_build_stats.n_dense_edges << '\n';
    std::cout << "  build_cpu_us=" << artifacts.pipeann_build_stats.cpu_us << '\n';
    std::cout << "  build_ram_budget_gb=" << config.build_ram_budget_gb << '\n';
    std::cout << "  build_threads=" << config.build_threads << '\n';
    std::cout << "  partition_scale=" << config.partition_scale << '\n';
    std::cout << "  partition_ldg_times=" << config.partition_ldg_times << '\n';
    std::cout << "  workflow_prefix=" << artifacts.workflow_prefix << '\n';
    std::cout << "  pipeann_base_data=" << artifacts.pipeann_base_data_path << '\n';
    if (!artifacts.pipeann_train_query_path.empty()) {
      std::cout << "  pipeann_train_query=" << artifacts.pipeann_train_query_path << '\n';
    }
    std::cout << "  pipeann_index_prefix=" << artifacts.pipeann_index_prefix << '\n';
    std::cout << "  pipeann_disk_index=" << artifacts.raw_disk_index_path << '\n';
    std::cout << "  gorgeous_partition=" << artifacts.gorgeous_partition_bin_path << '\n';
    std::cout << "  gorgeous_relayout=" << artifacts.gorgeous_relayout_index_path << '\n';
    std::cout << "  index=" << artifacts.index_path << '\n';
    std::cout << "  partition=" << artifacts.partition_path << '\n';
    std::cout << "  reorder=" << artifacts.reorder_path << '\n';
    if (!artifacts.pipeann_refine_sidecar_path.empty()) {
      std::cout << "  pipeann_refine_sidecar=" << artifacts.pipeann_refine_sidecar_path << '\n';
      std::cout << "  pipeann_refine_manifest=" << artifacts.pipeann_refine_manifest_path << '\n';
      std::cout << "  pipeann_refine_nodes=" << artifacts.pipeann_refine_nodes_path << '\n';
    }
    std::cout << "  full_data=" << artifacts.approx_path << '\n';
    std::cout << "  pipeann_pq_pivots=" << artifacts.pq_codebook_path << '\n';
    std::cout << "  pipeann_pq_compressed=" << artifacts.pq_codes_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Build failed: " << e.what() << '\n';
    return 1;
  }
}
