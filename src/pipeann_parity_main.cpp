#include <algorithm>
#include <exception>
#include <iostream>
#include <string>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_bench_pipeann [options]\n"
      << "  --index_prefix PATH\n"
      << "  --queries PATH\n"
      << "  --query_format text|fvecs|bvecs|bin\n"
      << "  --ground_truth PATH\n"
      << "  --generate_ground_truth PATH\n"
      << "  --ground_truth_k N\n"
      << "  --top_k N\n"
      << "  --beam_width N | --beamwidth N\n"
      << "  --l_search N | --L N\n"
      << "  --threads N\n"
      << "  --pipeann_mode 0|1|2 | --mode 0|1|2\n"
      << "  --pipeann_layout equal|gorgeous | --layout equal|gorgeous\n"
      << "  --gp_partition PATH\n"
      << "  --mem_l N | --mem_L N\n"
      << "  --export PATH\n"
      << "  --help\n";
}

uint32_t ParseUint32(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint32_t>(std::stoul(value));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + name);
  }
}

hybrid::QueryInputMode ParseQueryFormat(const std::string &value) {
  if (value == "text") {
    return hybrid::QueryInputMode::kText;
  }
  if (value == "fvecs") {
    return hybrid::QueryInputMode::kFvecs;
  }
  if (value == "bvecs") {
    return hybrid::QueryInputMode::kBvecs;
  }
  if (value == "bin") {
    return hybrid::QueryInputMode::kBin;
  }
  throw std::runtime_error("unsupported query_format: " + value);
}

hybrid::pipeann_parity::PipeannSearchMode ParsePipeannMode(const std::string &value) {
  if (value == "0" || value == "beam") {
    return hybrid::pipeann_parity::PipeannSearchMode::kBeam;
  }
  if (value == "1" || value == "page") {
    return hybrid::pipeann_parity::PipeannSearchMode::kPage;
  }
  if (value == "2" || value == "pipe") {
    return hybrid::pipeann_parity::PipeannSearchMode::kPipe;
  }
  throw std::runtime_error("unsupported pipeann mode: " + value);
}

hybrid::pipeann_parity::PipeannLayout ParsePipeannLayout(const std::string &value) {
  if (value == "equal") {
    return hybrid::pipeann_parity::PipeannLayout::kEqual;
  }
  if (value == "gorgeous") {
    return hybrid::pipeann_parity::PipeannLayout::kGorgeous;
  }
  throw std::runtime_error("unsupported pipeann layout: " + value);
}

const char *PipeannModeName(hybrid::pipeann_parity::PipeannSearchMode mode) {
  switch (mode) {
    case hybrid::pipeann_parity::PipeannSearchMode::kBeam:
      return "beam";
    case hybrid::pipeann_parity::PipeannSearchMode::kPage:
      return "page";
    case hybrid::pipeann_parity::PipeannSearchMode::kPipe:
      return "pipe";
  }
  throw std::runtime_error("unsupported pipeann mode");
}

const char *PipeannLayoutName(hybrid::pipeann_parity::PipeannLayout layout) {
  switch (layout) {
    case hybrid::pipeann_parity::PipeannLayout::kEqual:
      return "equal";
    case hybrid::pipeann_parity::PipeannLayout::kGorgeous:
      return "gorgeous";
  }
  throw std::runtime_error("unsupported pipeann layout");
}

struct ParsedArgs {
  hybrid::BenchToolConfig config;
  std::string export_path;
  std::string generate_ground_truth_path;
};

ParsedArgs ParseArgs(int argc, char **argv) {
  ParsedArgs parsed;
  hybrid::BenchToolConfig &config = parsed.config;
  std::string queries_path;
  hybrid::QueryInputMode query_format = hybrid::QueryInputMode::kText;

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
    if (arg == "--index_prefix") {
      config.index_prefix = need_value("--index_prefix");
      continue;
    }
    if (arg == "--queries" || arg == "--query_file") {
      queries_path = need_value(arg.c_str());
      continue;
    }
    if (arg == "--query_format") {
      query_format = ParseQueryFormat(need_value("--query_format"));
      continue;
    }
    if (arg == "--ground_truth") {
      config.ground_truth_ids = hybrid::LoadGroundTruthIds(need_value("--ground_truth"));
      continue;
    }
    if (arg == "--generate_ground_truth") {
      parsed.generate_ground_truth_path = need_value("--generate_ground_truth");
      continue;
    }
    if (arg == "--ground_truth_k") {
      config.recall_at_k = ParseUint32("--ground_truth_k", need_value("--ground_truth_k"));
      continue;
    }
    if (arg == "--top_k" || arg == "--K") {
      config.top_k = ParseUint32(arg, need_value(arg.c_str()));
      continue;
    }
    if (arg == "--beam_width" || arg == "--beamwidth") {
      config.beam_width = ParseUint32(arg, need_value(arg.c_str()));
      continue;
    }
    if (arg == "--l_search" || arg == "--L") {
      config.l_search = ParseUint32(arg, need_value(arg.c_str()));
      continue;
    }
    if (arg == "--threads") {
      config.num_threads = ParseUint32("--threads", need_value("--threads"));
      continue;
    }
    if (arg == "--pipeann_mode" || arg == "--mode") {
      config.pipeann_mode = ParsePipeannMode(need_value(arg.c_str()));
      continue;
    }
    if (arg == "--pipeann_layout" || arg == "--layout") {
      config.pipeann_layout = ParsePipeannLayout(need_value(arg.c_str()));
      continue;
    }
    if (arg == "--gp_partition") {
      config.gp_partition_path = need_value("--gp_partition");
      continue;
    }
    if (arg == "--mem_l" || arg == "--mem_L") {
      config.mem_l = ParseUint32(arg, need_value(arg.c_str()));
      continue;
    }
    if (arg == "--export") {
      parsed.export_path = need_value("--export");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (config.index_prefix.empty()) {
    throw std::runtime_error("--index_prefix is required");
  }
  if (queries_path.empty()) {
    throw std::runtime_error("--queries is required");
  }
  config.queries = hybrid::LoadQueryVectors(queries_path, query_format);
  return parsed;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    ParsedArgs parsed = ParseArgs(argc, argv);
    if (!parsed.generate_ground_truth_path.empty()) {
      const uint32_t ground_truth_k =
          parsed.config.recall_at_k == 0 ? parsed.config.top_k : std::max(parsed.config.top_k, parsed.config.recall_at_k);
      parsed.config.ground_truth_ids =
          hybrid::GenerateGroundTruthIds(parsed.config.index_prefix,
                                         parsed.config.approx_path,
                                         parsed.config.queries,
                                         ground_truth_k);
      hybrid::WriteGroundTruthIds(parsed.generate_ground_truth_path, parsed.config.ground_truth_ids);
    }

    const hybrid::BenchToolSummary summary = hybrid::RunPipeannParityBenchTool(parsed.config);
    if (!parsed.export_path.empty()) {
      hybrid::ExportBenchSummariesTsv(parsed.export_path, {summary});
    }

    std::cout << "PipeANN parity bench completed\n"
              << "  engine=pipeann_gorgeous_layout\n"
              << "  index_prefix=" << parsed.config.index_prefix << '\n'
              << "  layout=" << PipeannLayoutName(summary.pipeann_layout) << '\n'
              << "  mode=" << PipeannModeName(summary.pipeann_mode) << '\n'
              << "  threads=" << summary.num_threads << '\n'
              << "  beam_width=" << summary.beam_width << '\n'
              << "  L=" << summary.l_search << '\n'
              << "  mem_l=" << summary.mem_l << '\n'
              << "  qps=" << summary.qps << '\n'
              << "  avg_lat_us=" << summary.mean_latency_us << '\n'
              << "  p99_lat_us=" << summary.p99_latency_us << '\n'
              << "  mean_hops=" << summary.mean_hops << '\n'
              << "  mean_ios=" << summary.mean_ios << '\n'
              << "  unique_pg_io=" << summary.unique_pg_io << '\n'
              << "  dup_pg_hit=" << summary.dup_pg_hit << '\n'
              << "  polls=" << summary.polls << '\n';
    if (summary.has_recall) {
      std::cout << "  recall_at_k=" << summary.average_recall << '\n';
    }
    if (!parsed.export_path.empty()) {
      std::cout << "  export_tsv=" << parsed.export_path << '\n';
    }
    if (!parsed.generate_ground_truth_path.empty()) {
      std::cout << "  generated_ground_truth=" << parsed.generate_ground_truth_path << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "PipeANN parity bench failed: " << e.what() << '\n';
    return 1;
  }
}
