#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tools/tool_cli.h"

namespace {

void PrintUsage() {
  std::cout
      << "Usage: pipeann_gorgeous_bench_merge [options]\n"
      << "  --inputs PATH1,PATH2,...\n"
      << "  --output PATH\n"
      << "  --help\n";
}

std::vector<std::string> ParseInputPaths(const std::string &text) {
  std::vector<std::string> paths;
  std::istringstream in(text);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (!token.empty()) {
      paths.push_back(token);
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("input path list must not be empty");
  }
  return paths;
}

struct ParsedArgs {
  std::vector<std::string> input_paths;
  std::string output_path;
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
    if (arg == "--inputs") {
      args.input_paths = ParseInputPaths(need_value("--inputs"));
      continue;
    }
    if (arg == "--output") {
      args.output_path = need_value("--output");
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (args.input_paths.empty() || args.output_path.empty()) {
    throw std::runtime_error("--inputs and --output are required");
  }
  return args;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const ParsedArgs args = ParseArgs(argc, argv);
    hybrid::MergeBenchSummaryTsvFiles(args.output_path, args.input_paths);
    std::cout << "Bench merge completed\n";
    std::cout << "  inputs=" << args.input_paths.size() << '\n';
    std::cout << "  output=" << args.output_path << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Bench merge failed: " << e.what() << '\n';
    return 1;
  }
}
