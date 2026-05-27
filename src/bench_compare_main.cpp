#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_bench_compare [options]\n"
      << "  --baseline PATH\n"
      << "  --candidate PATH\n"
      << "  --output PATH\n"
      << "  --format markdown|tsv\n"
      << "  --help\n";
}

struct ParsedArgs {
  std::string baseline_path;
  std::string candidate_path;
  std::string output_path;
  std::string format = "markdown";
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
    if (arg == "--baseline") {
      args.baseline_path = need_value("--baseline");
      continue;
    }
    if (arg == "--candidate") {
      args.candidate_path = need_value("--candidate");
      continue;
    }
    if (arg == "--output") {
      args.output_path = need_value("--output");
      continue;
    }
    if (arg == "--format") {
      args.format = need_value("--format");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (args.baseline_path.empty() || args.candidate_path.empty() || args.output_path.empty()) {
    throw std::runtime_error("--baseline, --candidate, and --output are required");
  }
  if (args.format != "markdown" && args.format != "tsv") {
    throw std::runtime_error("--format must be markdown or tsv");
  }
  return args;
}

std::string LabelFromPath(const std::string &path) {
  const std::filesystem::path fs_path(path);
  if (fs_path.filename().empty()) {
    return path;
  }
  return fs_path.filename().string();
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const ParsedArgs args = ParseArgs(argc, argv);
    const std::vector<hybrid::BenchToolSummary> baseline = hybrid::LoadBenchSummariesTsv(args.baseline_path);
    const std::vector<hybrid::BenchToolSummary> candidate = hybrid::LoadBenchSummariesTsv(args.candidate_path);
    const hybrid::BenchComparisonSummary comparison =
        hybrid::CompareBenchSummaries(LabelFromPath(args.baseline_path), baseline,
                                      LabelFromPath(args.candidate_path), candidate);

    if (args.format == "markdown") {
      hybrid::ExportBenchComparisonMarkdown(args.output_path, comparison);
    } else {
      hybrid::ExportBenchComparisonTsv(args.output_path, comparison);
    }

    std::cout << "Bench compare completed\n";
    std::cout << "  baseline=" << args.baseline_path << '\n';
    std::cout << "  candidate=" << args.candidate_path << '\n';
    std::cout << "  matched_runs=" << comparison.rows.size() << '\n';
    std::cout << "  output=" << args.output_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Bench compare failed: " << e.what() << '\n';
    return 1;
  }
}
