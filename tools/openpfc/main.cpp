// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <iostream>

int main(int argc, char **argv) {
  using namespace openpfc_cli;
  try {
    auto args = parse(argc, argv);
    if (args.has("version")) {
      args.allow({"version"});
      if (!args.command.empty()) throw Error("--version takes no command");
      const auto file = prefix() / "share/openpfc/version";
      auto value = fs::is_regular_file(file) ? read(file) : OPENPFC_CLI_VERSION;
      while (!value.empty() && (value.back() == '\n' || value.back() == '\r'))
        value.pop_back();
      std::cout << "OpenPFC " << value << '\n';
      return 0;
    }
    if (args.has("help") || args.command == "help") {
      args.allow({"help"});
      if (args.command == "help") {
        if (args.positional.size() > 1)
          throw Error("help expects at most one command");
        print_help(args.positional.empty() ? "" : args.positional.front());
      } else {
        print_help(args.command);
      }
      return 0;
    }
    if (args.command.empty()) {
      args.allow({});
      print_help();
      return 0;
    }
    if (args.command == "doctor") return doctor(args);
    if (args.command == "init" && args.has("model")) return init_model(args);
    if (args.command == "init" || args.command == "apps")
      return catalog_command(args);
    if (args.command == "export") return export_record(args);
    if (args.command == "bundle") return bundle_model(args);
    if (args.command != "compile" && args.command != "run")
      throw Error("unknown command: " + args.command + "; use --help");
    if (args.positional.empty())
      throw Error(args.command + " expects a case directory or C++ source file");
    const auto extension = fs::path(args.positional.front()).extension();
    const bool source =
        extension == ".cpp" || extension == ".cc" || extension == ".cxx";
    if (!source)
      return args.command == "compile" ? compile_case(args) : run_case(args);
    std::vector<std::string> options{
        "sdk",       "file", "define", "build-type", "toolchain", "source-root",
        "generator", "cxx",  "cc",     "jobs",       "backend",   "profile"};
    if (args.command == "run")
      options.insert(options.end(),
                     {"input-file", "ranks", "launcher", "arg", "record-env"});
    args.allow(options);
    if (args.command == "compile") {
      if (args.positional.size() != 1)
        throw Error("compile source.cpp takes one source pathname");
      (void)compile_source(args);
      return 0;
    }
    return run_source(args);
  } catch (const Error &error) {
    std::cerr << "openpfc: " << error.what() << '\n';
    return error.code;
  } catch (const std::exception &error) {
    std::cerr << "openpfc: " << error.what() << '\n';
    return 2;
  }
}
