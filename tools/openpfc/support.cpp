// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <openssl/evp.h>
#include <random>
#include <set>
#include <sstream>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace openpfc_cli {
bool Arguments::has(const std::string &key) const { return options.contains(key); }
std::string Arguments::get(const std::string &key, std::string fallback) const {
  const auto found = options.find(key);
  if (found == options.end()) return fallback;
  if (found->second.size() != 1) throw Error("option --" + key + " must occur once");
  return found->second.front();
}
std::vector<std::string> Arguments::all(const std::string &key) const {
  auto found = options.find(key);
  return found == options.end() ? std::vector<std::string>{} : found->second;
}
int Arguments::positive(const std::string &key, int fallback) const {
  if (!has(key)) return fallback;
  const auto value = get(key);
  std::size_t consumed = 0;
  int number = 0;
  try {
    number = std::stoi(value, &consumed);
  } catch (const std::exception &) {
    throw Error("--" + key + " must be a positive integer");
  }
  if (number <= 0 || consumed != value.size())
    throw Error("--" + key + " must be a positive integer");
  return number;
}
void Arguments::allow(const std::vector<std::string> &keys) const {
  for (const auto &[key, values] : options) {
    if (std::ranges::find(keys, key) == keys.end())
      throw Error("unknown option --" + key + " for " + command);
    (void)values;
  }
}
Arguments parse(int argc, char **argv) {
  Arguments args;
  const std::set<std::string> flags{"version", "help", "dry-run", "no-test",
                                    "prepare-only"};
  bool positional_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string value(argv[i]);
    if (value == "--") {
      positional_only = true;
      continue;
    }
    if (!positional_only &&
        (value.starts_with("--") || value == "-o" || value == "-h")) {
      if (value == "-o") value = "--output";
      if (value == "-h") value = "--help";
      auto separator = value.find('=');
      auto key = value.substr(2, separator == std::string::npos ? separator
                                                                : separator - 2);
      std::string setting;
      if (flags.contains(key)) {
        if (separator != std::string::npos)
          throw Error("--" + key + " takes no value");
        setting = "true";
      } else if (separator != std::string::npos) {
        setting = value.substr(separator + 1);
      } else {
        if (i + 1 == argc || std::string(argv[i + 1]).starts_with("--"))
          throw Error("missing value for --" + key);
        setting = argv[++i];
      }
      if (setting.empty()) throw Error("empty value for --" + key);
      args.options[key].push_back(setting);
    } else if (args.command.empty())
      args.command = value;
    else
      args.positional.push_back(value);
  }
  return args;
}
std::string env(const std::string &name) {
  const char *value = std::getenv(name.c_str());
  return value ? value : "";
}
fs::path absolute_path(const fs::path &path) {
  return fs::absolute(path).lexically_normal();
}
fs::path executable() { return fs::read_symlink("/proc/self/exe"); }
fs::path prefix() { return executable().parent_path().parent_path(); }
fs::path checkout(const Arguments &args) {
  auto selected = args.get("source", env("OPENPFC_SOURCE_DIR"));
  fs::path root =
      selected.empty() ? fs::path(OPENPFC_CLI_SOURCE_HINT) : absolute_path(selected);
  if (!fs::is_regular_file(root / "scripts/build.sh"))
    throw Error("case compilation needs an OpenPFC checkout: pass --source or set "
                "OPENPFC_SOURCE_DIR");
  return root;
}
std::string read(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot read " + path.string());
  std::ostringstream result;
  result << stream.rdbuf();
  if (stream.bad()) throw Error("read failed: " + path.string());
  return result.str();
}
Json read_json(const fs::path &path) {
  try {
    return Json::parse(read(path));
  } catch (const Json::exception &error) {
    throw Error("invalid JSON in " + path.string() + ": " + error.what());
  }
}
void write(const fs::path &path, const std::string &bytes) {
  const int fd = ::open(path.c_str(),
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0)
    throw Error("cannot create " + path.string() + ": " + std::strerror(errno));
  std::size_t position = 0;
  while (position < bytes.size()) {
    const auto count = ::write(fd, bytes.data() + position, bytes.size() - position);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      ::close(fd);
      throw Error("write failed: " + path.string());
    }
    position += static_cast<std::size_t>(count);
  }
  if (::fsync(fd) != 0) {
    ::close(fd);
    throw Error("fsync failed: " + path.string());
  }
  if (::close(fd) != 0) throw Error("close failed: " + path.string());
}
void write_json(const fs::path &path, const Json &value) {
  write(path, value.dump(2) + "\n");
}
void replace_json(const fs::path &path, const Json &value) {
  auto temporary = path;
  temporary += ".tmp-" + unique_id();
  write_json(temporary, value);
  fs::rename(temporary, path);
}
std::string sha256(const std::string &bytes) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha256(),
                 nullptr) != 1)
    throw Error("SHA-256 failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned i = 0; i < length; ++i) {
    result += hex[digest[i] >> 4];
    result += hex[digest[i] & 15];
  }
  return result;
}
std::string file_sha256(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw Error("cannot hash " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                  &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw Error("cannot initialize file SHA-256");
  std::array<char, 65536> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (stream.gcount() > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(stream.gcount())) != 1)
      throw Error("file SHA-256 update failed");
  }
  if (stream.bad()) throw Error("read failed while hashing " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1)
    throw Error("file SHA-256 finalization failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned i = 0; i < length; ++i) {
    result += hex[digest[i] >> 4];
    result += hex[digest[i] & 15];
  }
  return result;
}
std::string unique_id() {
  std::random_device random;
  return sha256(std::to_string(::getpid()) + ":" + std::to_string(random()) + ":" +
                std::to_string(random()))
      .substr(0, 24);
}
fs::path find_program(const std::string &name) {
  fs::path program(name);
  if (program.has_parent_path()) {
    require_executable(absolute_path(program));
    return absolute_path(program);
  }
  std::istringstream paths(env("PATH"));
  for (std::string part; std::getline(paths, part, ':');) {
    auto candidate = absolute_path(fs::path(part.empty() ? "." : part) / name);
    if (fs::is_regular_file(candidate) && ::access(candidate.c_str(), X_OK) == 0)
      return candidate;
  }
  throw Error("executable missing: " + name +
              "; load the pre-provisioned compiler/MPI/runtime environment");
}
void require_executable(const fs::path &path) {
  if (!fs::is_regular_file(path) || ::access(path.c_str(), X_OK) != 0)
    throw Error("executable missing or not executable: " + path.string());
}
void freeze_file(const fs::path &path) {
  fs::permissions(path,
                  fs::perms::owner_write | fs::perms::group_write |
                      fs::perms::others_write,
                  fs::perm_options::remove);
}
bool outer_mpi() {
  for (const auto *name :
       {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"})
    if (!env(name).empty()) return true;
  return false;
}
void require_single_process() {
  if (outer_mpi())
    throw Error("compile/source workflow must run once, outside an MPI launch; use "
                "--ranks on openpfc run");
}
void show_command(const Command &command, const fs::path &cwd) {
  std::cout << Json{{"command", command}, {"cwd", cwd.string()}}.dump() << '\n';
}
int run(const Command &command, const fs::path &cwd, const fs::path &out,
        const fs::path &err) {
  if (command.empty()) throw Error("empty process command");
  std::vector<char *> argv;
  for (const auto &argument : command)
    argv.push_back(const_cast<char *>(argument.c_str()));
  argv.push_back(nullptr);
  auto open_log = [](const fs::path &path) {
    if (path.empty()) return -1;
    const int fd = ::open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) throw Error("cannot create process log " + path.string());
    return fd;
  };
  const int stdout_fd = open_log(out);
  int stderr_fd = -1;
  try {
    stderr_fd = open_log(err);
  } catch (...) {
    if (stdout_fd >= 0) ::close(stdout_fd);
    throw;
  }
  std::cout.flush();
  std::cerr.flush();
  const pid_t pid = ::fork();
  if (pid == 0) {
    // Only async-signal-safe operations between fork and exec; no shell parsing.
    if (::chdir(cwd.c_str()) != 0 ||
        (stdout_fd >= 0 && ::dup2(stdout_fd, STDOUT_FILENO) < 0) ||
        (stderr_fd >= 0 && ::dup2(stderr_fd, STDERR_FILENO) < 0))
      ::_exit(126);
    if (stdout_fd >= 0) ::close(stdout_fd);
    if (stderr_fd >= 0) ::close(stderr_fd);
    ::execvp(argv.front(), argv.data());
    constexpr char message[] = "openpfc: process exec failed\n";
    const auto written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)written;
    ::_exit(127);
  }
  if (stdout_fd >= 0) ::close(stdout_fd);
  if (stderr_fd >= 0) ::close(stderr_fd);
  if (pid < 0) throw Error("fork failed: " + std::string(std::strerror(errno)));
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) throw Error("waitpid failed");
  }
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 126;
}
void checked_run(const Command &command, const fs::path &cwd, const fs::path &out,
                 const fs::path &err) {
  const int code = run(command, cwd, out, err);
  if (code)
    throw Error("command failed (" + std::to_string(code) +
                    "). Logs: " + out.string() + " and " + err.string() + "\n" +
                    read(err) + "\n" + read(out),
                code);
}
Command launcher(const Arguments &args, const std::string &recorded_mpiexec) {
  auto selected = args.get("launcher", "auto");
  if (selected != "auto" && selected != "none" && selected != "mpirun" &&
      selected != "srun")
    throw Error("--launcher must be auto, none, mpirun or srun");
  if (outer_mpi()) {
    if (args.has("ranks") || (selected != "auto" && selected != "none"))
      throw Error("already inside MPI: set ranks on the outer mpirun/srun command");
    return {};
  }
  if (selected == "auto") selected = env("SLURM_JOB_ID").empty() ? "mpirun" : "srun";
  if (selected == "none") {
    if (args.positive("ranks") != 1)
      throw Error("--launcher=none cannot launch multiple ranks");
    return {};
  }
  if (selected == "srun") {
    Command result{"srun"};
    if (args.has("ranks")) {
      result.emplace_back("--ntasks");
      result.push_back(std::to_string(args.positive("ranks")));
    }
    return result;
  }
  return {recorded_mpiexec.empty() ? "mpirun" : recorded_mpiexec, "-n",
          std::to_string(args.positive("ranks"))};
}
std::string cmake_quote(const std::string &value) {
  std::string equals;
  while (value.find("]" + equals + "]") != std::string::npos) equals += '=';
  return "[" + equals + "[" + value + "]" + equals + "]";
}
void safe_directory(const fs::path &path) {
  if (fs::is_symlink(fs::symlink_status(path)))
    throw Error("refusing symlink directory: " + path.string());
  fs::create_directories(path);
  if (!fs::is_directory(path)) throw Error("not a directory: " + path.string());
}
fs::path cache_root() {
  auto selected = env("OPENPFC_CACHE");
  if (selected.empty()) {
    auto xdg = env("XDG_CACHE_HOME");
    auto home = env("HOME");
    if (xdg.empty() && home.empty())
      throw Error("set OPENPFC_CACHE, XDG_CACHE_HOME or HOME");
    selected =
        (xdg.empty() ? fs::path(home) / ".cache/openpfc" : fs::path(xdg) / "openpfc")
            .string();
  }
  auto root = absolute_path(selected);
  const bool existed = fs::exists(root);
  safe_directory(root);
  struct stat status{};
  if (::lstat(root.c_str(), &status) != 0 || status.st_uid != ::geteuid())
    throw Error("cache must be owned by the invoking user: " + root.string());
  if (!existed)
    fs::permissions(root, fs::perms::owner_all);
  else if ((status.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    throw Error("cache must not be writable by other users: " + root.string());
  for (const auto *directory : {"builds", "runs", "bundles", "locks", "staging"})
    safe_directory(root / directory);
  return root;
}
Lock::Lock(const fs::path &path) {
  fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd_ < 0) throw Error("cannot open cache lock: " + path.string());
  while (::flock(fd_, LOCK_EX) < 0) {
    if (errno == EINTR) continue;
    ::close(fd_);
    fd_ = -1;
    throw Error("cannot lock cache: " + path.string());
  }
}
Lock::~Lock() {
  if (fd_ >= 0) ::close(fd_);
}
} // namespace openpfc_cli
