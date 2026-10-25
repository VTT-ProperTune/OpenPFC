// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace openpfc_cli {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Command = std::vector<std::string>;
struct Error : std::runtime_error {
  int code;
  explicit Error(std::string message, int status = 2)
      : std::runtime_error(std::move(message)), code(status) {}
};
struct Arguments {
  std::string command;
  std::vector<std::string> positional;
  std::map<std::string, std::vector<std::string>> options;
  [[nodiscard]] bool has(const std::string &key) const;
  [[nodiscard]] std::string get(const std::string &key,
                                std::string fallback = {}) const;
  [[nodiscard]] std::vector<std::string> all(const std::string &key) const;
  [[nodiscard]] int positive(const std::string &key, int fallback = 1) const;
  void allow(const std::vector<std::string> &keys) const;
};
Arguments parse(int argc, char **argv);
void print_help(const std::string &command = {});
std::string env(const std::string &name);
fs::path executable();
fs::path prefix();
fs::path checkout(const Arguments &args);
std::string read(const fs::path &path);
Json read_json(const fs::path &path);
void write(const fs::path &path, const std::string &bytes);
void write_json(const fs::path &path, const Json &value);
void replace_json(const fs::path &path, const Json &value);
std::string sha256(const std::string &bytes);
std::string file_sha256(const fs::path &path);
std::string unique_id();
fs::path absolute_path(const fs::path &path);
fs::path find_program(const std::string &name);
void require_executable(const fs::path &path);
void freeze_file(const fs::path &path);
bool outer_mpi();
void require_single_process();
int run(const Command &command, const fs::path &cwd,
        const fs::path &stdout_path = {}, const fs::path &stderr_path = {});
void checked_run(const Command &command, const fs::path &cwd,
                 const fs::path &stdout_path, const fs::path &stderr_path);
void show_command(const Command &command, const fs::path &cwd);
Command launcher(const Arguments &args, const std::string &recorded_mpiexec = {});
std::string cmake_quote(const std::string &value);
fs::path cache_root();
void safe_directory(const fs::path &path);
class Lock {
  int fd_ = -1;

public:
  explicit Lock(const fs::path &path);
  ~Lock();
  Lock(const Lock &) = delete;
  Lock &operator=(const Lock &) = delete;
};
struct App {
  std::string directory;
  std::map<std::string, std::string> binaries;
  std::vector<std::pair<std::string, std::string>> presets;
  std::string description;
};
const std::map<std::string, App> &catalog();
int catalog_command(const Arguments &args);
int init_model(const Arguments &args);
int compile_case(const Arguments &args);
int run_case(const Arguments &args);
struct Build {
  fs::path root;
  Json manifest;
};
Build compile_source(const Arguments &args);
int doctor(const Arguments &args);
int run_source(const Arguments &args);
int export_record(const Arguments &args);
struct RuntimeLibrary {
  std::string soname;
  fs::path path;
  int glibc_major = 0;
  int glibc_minor = 0;
};
struct RuntimeClosure {
  fs::path executable;
  std::string interpreter;
  int glibc_major = 0;
  int glibc_minor = 0;
  std::vector<RuntimeLibrary> libraries;
};
RuntimeClosure dynamic_closure(const fs::path &executable);
int bundle_model(const Arguments &args);
} // namespace openpfc_cli
