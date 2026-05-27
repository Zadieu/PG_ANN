#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_ground_truth [options]\n"
      << "  --index PATH\n"
      << "  --approx PATH (optional full-precision data override)\n"
      << "  --queries PATH\n"
      << "  --query_format text|fvecs|bvecs|bin\n"
      << "  --top_k N\n"
      << "  --output PATH\n"
      << "  --help\n";
}

uint32_t ParseUint32(const std::string &name, const std::string &value) {
  try {
    return static_cast<uint32_t>(std::stoul(value));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + name);
  }
}

hybrid::QueryInputMode ParseQueryInputMode(const std::string &value) {
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

struct ParsedArgs {
  std::string index_path;
  std::string approx_path;
  std::string queries_path;
  std::string output_path;
  hybrid::QueryInputMode query_format = hybrid::QueryInputMode::kText;
  uint32_t top_k = 0;
};

ParsedArgs ParseArgs(int argc, char **argv) {
  ParsedArgs args;
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
      args.index_path = need_value("--index");
      continue;
    }
    if (arg == "--approx") {
      args.approx_path = need_value("--approx");
      continue;
    }
    if (arg == "--queries") {
      args.queries_path = need_value("--queries");
      continue;
    }
    if (arg == "--query_format") {
      args.query_format = ParseQueryInputMode(need_value("--query_format"));
      continue;
    }
    if (arg == "--top_k") {
      args.top_k = ParseUint32("--top_k", need_value("--top_k"));
      continue;
    }
    if (arg == "--output") {
      args.output_path = need_value("--output");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (args.index_path.empty() || args.queries_path.empty() || args.output_path.empty()) {
    throw std::runtime_error("--index, --queries, and --output are required");
  }
  if (args.top_k == 0) {
    throw std::runtime_error("--top_k must be greater than zero");
  }
  return args;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const ParsedArgs args = ParseArgs(argc, argv);
    const std::vector<std::vector<float>> queries =
        hybrid::LoadQueryVectors(args.queries_path, args.query_format);
    const std::vector<std::vector<uint32_t>> truth =
        hybrid::GenerateGroundTruthIds(args.index_path, args.approx_path, queries, args.top_k);
    hybrid::WriteGroundTruthIds(args.output_path, truth);

    std::cout << "Ground truth completed\n";
    std::cout << "  queries=" << truth.size() << '\n';
    std::cout << "  top_k=" << args.top_k << '\n';
    std::cout << "  output=" << args.output_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Ground truth failed: " << e.what() << '\n';
    return 1;
  }
}
