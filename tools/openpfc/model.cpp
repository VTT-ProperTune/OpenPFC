// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <set>
#include <unistd.h>

namespace openpfc_cli {
namespace {
using Files = std::map<std::string, std::string>;
void relative_path(const fs::path &path) {
  if (path.empty() || path.is_absolute() || path == "." ||
      (path.string().find(';') != std::string::npos ||
       path.string().find('\0') != std::string::npos) ||
      std::ranges::find(path, "..") != path.end())
    throw Error(
        "expected a safe relative path without '..' or CMake list separators: " +
        path.string());
}
std::string local_file(const fs::path &root, const fs::path &relative) {
  relative_path(relative);
  auto path = root / relative;
  if (fs::canonical(path) != path.lexically_normal())
    throw Error("source/input symlinks are not snapshot dependencies: " +
                path.string());
  if (!fs::is_regular_file(path))
    throw Error("not a regular file: " + path.string());
  return read(path);
}
fs::path source_root(const Arguments &args) {
  const auto source = absolute_path(args.positional.front());
  return fs::canonical(args.get("source-root", source.parent_path().string()));
}
fs::path primary_source(const Arguments &args) {
  const auto source = absolute_path(args.positional.front());
  const auto path = fs::canonical(source.parent_path()) / source.filename();
  auto name = path.lexically_relative(source_root(args));
  relative_path(name);
  return name;
}
Files source_files(const Arguments &args) {
  const auto root = source_root(args);
  Files files;
  std::vector<fs::path> pending{primary_source(args)};
  for (const auto &value : args.all("file")) pending.emplace_back(value);
  const std::regex include(R"(^\s*#\s*include\s*([<"])([^">]+)[">])");
  while (!pending.empty()) {
    auto name = pending.back().lexically_normal();
    pending.pop_back();
    relative_path(name);
    if (files.contains(name.generic_string())) continue;
    auto bytes = local_file(root, name);
    std::istringstream lines(bytes);
    for (std::string line; std::getline(lines, line);) {
      std::smatch match;
      if (!std::regex_search(line, match, include)) continue;
      const fs::path requested(match[2].str());
      if (requested.is_absolute())
        throw Error("absolute source includes are mutable external inputs: " +
                    requested.string());
      if (match[1].str() == "<") relative_path(requested);
      if (match[1].str() == "\"") {
        auto dependency = (name.parent_path() / requested).lexically_normal();
        relative_path(dependency);
        if (fs::is_regular_file(root / dependency))
          pending.push_back(dependency);
        else {
          relative_path(requested);
          if (fs::is_regular_file(root / requested)) pending.push_back(requested);
          // Otherwise let the compiler search the installed SDK/system headers.
          // The original mutable source directory is never an include path.
        }
      }
    }
    files.emplace(name.generic_string(), std::move(bytes));
  }
  return files;
}
Json inventory(const Files &files) {
  Json result = Json::object();
  for (const auto &[path, bytes] : files) result[path] = sha256(bytes);
  return result;
}
void snapshot(const fs::path &root, const Files &files) {
  safe_directory(root);
  for (const auto &[name, bytes] : files) {
    auto path = root / name;
    safe_directory(path.parent_path());
    write(path, bytes);
    freeze_file(path);
  }
}
void verify_files(const fs::path &root, const Json &files) {
  for (auto it = files.begin(); it != files.end(); ++it) {
    relative_path(it.key());
    auto path = root / it.key();
    if (fs::is_symlink(fs::symlink_status(path)) || !fs::is_regular_file(path) ||
        file_sha256(path) != it.value().get<std::string>())
      throw Error("snapshot verification failed: " + path.string() +
                  "; preserve/remove the damaged cache entry explicitly");
  }
}
fs::path sdk_directory(const Arguments &args) {
  auto given = args.get("sdk", env("OpenPFC_DIR"));
  if (given.empty() &&
      fs::is_regular_file(prefix() / "lib/cmake/OpenPFC/OpenPFCConfig.cmake"))
    return prefix() / "lib/cmake/OpenPFC";
  if (given.empty()) return {};
  auto path = absolute_path(given);
  if (fs::is_regular_file(path / "OpenPFCConfig.cmake")) return path;
  if (fs::is_regular_file(path / "lib/cmake/OpenPFC/OpenPFCConfig.cmake"))
    return path / "lib/cmake/OpenPFC";
  throw Error("OpenPFC SDK missing at " + path.string() +
              "; select a compatible installed SDK with --sdk, OpenPFC_DIR or "
              "CMAKE_PREFIX_PATH");
}
Command configure(const Arguments &args, const fs::path &project,
                  const fs::path &build) {
  Command command{find_program("cmake").string(),
                  "-S",
                  project.string(),
                  "-B",
                  build.string(),
                  "-G",
                  args.get("generator", "Ninja"),
                  "-DCMAKE_BUILD_TYPE=" + args.get("build-type", "Release")};
  auto sdk = sdk_directory(args);
  if (!sdk.empty()) command.push_back("-DOpenPFC_DIR=" + sdk.string());
  if (args.has("toolchain"))
    command.push_back("-DCMAKE_TOOLCHAIN_FILE=" +
                      absolute_path(args.get("toolchain")).string());
  for (const auto &[key, variable] : std::map<std::string, std::string>{
           {"cxx", "CMAKE_CXX_COMPILER"}, {"cc", "CMAKE_C_COMPILER"}})
    if (args.has(key))
      command.push_back("-D" + variable + "=" +
                        find_program(args.get(key)).string());
  return command;
}
std::string project_files(const Arguments &args, const Files &files) {
  std::string result = "add_executable(model\n";
  std::set<std::string> translation_units{primary_source(args).generic_string()};
  for (const auto &path : args.all("file"))
    translation_units.insert(fs::path(path).lexically_normal().generic_string());
  for (const auto &[path, bytes] : files) {
    if (!translation_units.contains(path)) continue;
    const auto extension = fs::path(path).extension();
    if (extension == ".cpp" || extension == ".cc" || extension == ".cxx")
      result += "  " + cmake_quote("src/" + path) + "\n";
    (void)bytes;
  }
  result += ")\n";
  if (args.has("define")) {
    result += "target_compile_definitions(model PRIVATE\n";
    for (const auto &value : args.all("define")) {
      if (!std::regex_match(value, std::regex("[A-Za-z_][A-Za-z0-9_]*(=.*)?")) ||
          value.find(';') != std::string::npos)
        throw Error(
            "--define expects NAME or NAME=VALUE without CMake list separators");
      result += "  " + cmake_quote(value) + "\n";
    }
    result += ")\n";
  }
  return result;
}
void validate_build(const Build &build) {
  if (build.manifest.at("build_id").get<std::string>() !=
      sha256(build.manifest.at("identity").dump()))
    throw Error("build manifest fingerprint does not match its contents");
  verify_files(build.root / "src", build.manifest.at("identity").at("sources"));
  if (file_sha256(build.root / "CMakeLists.txt") !=
          build.manifest.at("identity").at("template_sha256").get<std::string>() ||
      file_sha256(build.root / "model-files.cmake") !=
          build.manifest.at("identity").at("model_files_sha256").get<std::string>())
    throw Error("cached CMake project changed; refuse to reuse this build");
  require_executable(build.root / "model");
  if (file_sha256(build.root / "model") !=
      build.manifest.at("binary_sha256").get<std::string>())
    throw Error(
        "cached executable fingerprint mismatch; refuse to reuse this build");
}
void no_login_build() {
  if ((!env("LUMI_SYSTEM_NAME").empty() ||
       env("LMOD_SYSTEM_NAME").starts_with("lumi")) &&
      env("SLURM_JOB_ID").empty())
    throw Error("configure/build the model in a permitted build environment or "
                "allocation, not a LUMI login node");
}
void model_environment(const Arguments &args) {
  require_single_process();
  no_login_build();
  if (args.get("backend", "cpu") != "cpu" ||
      (args.get("profile", "local") != "local" &&
       args.get("profile") != "local-cpu"))
    throw Error("source workflow currently qualifies local CPU only; CUDA/HIP "
                "target profiles require explicit device/MPI qualification");
  const auto type = args.get("build-type", "Release");
  if (type != "Release" && type != "Debug" && type != "RelWithDebInfo" &&
      type != "MinSizeRel")
    throw Error("unsupported --build-type");
  (void)args.positive("jobs", 2);
}
Files input_files(const Arguments &args) {
  const auto input = absolute_path(args.positional.at(1));
  auto reserved = [](const fs::path &name) {
    relative_path(name);
    const auto top = name.begin()->string();
    if (top == "results" || top == "stdout.log" || top == "stderr.log" ||
        top == "status.json")
      throw Error("input path occupies an attempt's reserved output name: " +
                  name.string());
  };
  reserved(input.filename());
  Files files;
  files.emplace(input.filename().generic_string(),
                local_file(fs::canonical(input.parent_path()), input.filename()));
  for (const auto &name : args.all("input-file")) {
    reserved(name);
    files[fs::path(name).generic_string()] =
        local_file(fs::canonical(input.parent_path()), name);
  }
  const auto json = Json::parse(files.at(input.filename().generic_string()));
  auto check = [&](auto &&self, const Json &value) -> void {
    if (value.is_object() || value.is_array()) {
      for (const auto &child : value) self(self, child);
    } else if (value.is_string()) {
      const auto text = value.get<std::string>();
      if (text.empty()) return;
      const fs::path path(text);
      const bool pathlike =
          text.find('/') != std::string::npos ||
          std::regex_search(
              text,
              std::regex("\\.(json|bin|dat|csv|h5|hdf5|vti|vtk|txt|npy|npz)$"));
      if (!pathlike) return;
      relative_path(path);
      if (text == "results" || text.starts_with("results/")) return;
      if (!files.contains(path.generic_string()))
        throw Error(
            "input references an undeclared file: " + text +
            "; use --input-file relative/path; outputs must be under results/");
    }
  };
  check(check, json);
  return files;
}
Json runtime_environment(const Arguments &args) {
  std::set<std::string> names{"PATH",
                              "LD_LIBRARY_PATH",
                              "LD_PRELOAD",
                              "OMP_NUM_THREADS",
                              "OMP_DYNAMIC",
                              "OMP_SCHEDULE",
                              "MKL_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS",
                              "OMP_PROC_BIND",
                              "OMP_PLACES",
                              "CUDA_VISIBLE_DEVICES",
                              "HIP_VISIBLE_DEVICES",
                              "ROCR_VISIBLE_DEVICES",
                              "MPICH_GPU_SUPPORT_ENABLED",
                              "UCX_TLS",
                              "UCX_NET_DEVICES",
                              "FI_PROVIDER",
                              "OMPI_MCA_pml",
                              "OMPI_MCA_btl",
                              "OMPI_MCA_osc"};
  for (const auto &name : args.all("record-env")) {
    auto upper = name;
    std::ranges::transform(upper, upper.begin(), [](unsigned char c) {
      return static_cast<char>(std::toupper(c));
    });
    if (!std::regex_match(name, std::regex("[A-Za-z_][A-Za-z0-9_]*")) ||
        std::regex_search(upper,
                          std::regex("TOKEN|PASSWORD|SECRET|CREDENTIAL|KEY|AUTH")))
      throw Error("refusing credential-like or invalid --record-env name");
    names.insert(name);
  }
  Json values = Json::object();
  for (const auto &name : names) {
    const auto *value = std::getenv(name.c_str());
    values[name] = value ? Json(value) : Json(nullptr);
  }
  return values;
}
void copy_record(const fs::path &from, const fs::path &to) {
  if (!fs::create_directory(to))
    throw Error("export destination already exists: " + to.string());
  for (const auto &item : fs::recursive_directory_iterator(from)) {
    if (item.is_symlink())
      throw Error("refusing symlink in exported record: " + item.path().string());
    auto destination = to / item.path().lexically_relative(from);
    if (item.is_directory())
      safe_directory(destination);
    else if (item.is_regular_file())
      fs::copy_file(item.path(), destination, fs::copy_options::none);
    else
      throw Error("unsupported file in record: " + item.path().string());
  }
}
} // namespace
int doctor(const Arguments &args) {
  args.allow({"sdk", "cc", "cxx", "generator", "toolchain", "build-type", "backend",
              "profile"});
  if (!args.positional.empty()) throw Error("doctor takes no positional arguments");
  model_environment(args);
  std::cout << "OpenPFC doctor: CPU source build environment\n";
  fs::path stage;
  try {
    const auto cache = cache_root();
    stage = cache / "staging" / ("doctor-" + unique_id());
    safe_directory(stage);
    write(stage / "CMakeLists.txt",
          read(prefix() / "share/openpfc/templates/model.cmake"));
    auto command = configure(args, stage, stage / "query");
    command.insert(command.end(),
                   {"-DOPENPFC_DISCOVERY_ONLY=ON", "-DOPENPFC_DOCTOR=ON",
                    "-DOPENPFC_QUERY_REPORT=" + (stage / "consumer.json").string()});
    checked_run(command, stage, stage / "configure.stdout.log",
                stage / "configure.stderr.log");
    const auto report = read_json(stage / "consumer.json");
    auto value = [&](const std::string &name) {
      return report.at(name).at("value").get<std::string>();
    };
    std::cout << "SDK: " << value("OpenPFC_DIR")
              << "\nCMake: " << value("CMAKE_VERSION") << " ("
              << value("CMAKE_GENERATOR")
              << ")\nC++ compiler: " << value("CMAKE_CXX_COMPILER") << " ("
              << value("CMAKE_CXX_COMPILER_ID") << ' '
              << value("CMAKE_CXX_COMPILER_VERSION")
              << ")\nMPI compiler: " << value("MPI_CXX_COMPILER") << '\n';
    const auto mpi = value("MPIEXEC_EXECUTABLE");
    try {
      const auto resolved = find_program(mpi);
      std::cout << "MPI launcher: " << resolved.string() << " (not launched)\n";
    } catch (const Error &) {
      std::cout << "MPI launcher unavailable; use --launcher=none for one rank "
                   "or load matching MPI runtime modules.\n";
    }
    std::cout << "SDK backends:";
    for (const auto *backend : {"cpu", "cuda", "hip"})
      std::cout << ' ' << backend << '='
                << (report.at("sdk").at("capabilities").value(backend, false)
                        ? "yes"
                        : "no");
    std::cout << '\n' << value("sdk_report") << "\nDiagnostics: " << stage << '\n';
    if (value("sdk_status") == "compatible") {
      std::cout << "Ready for CPU source builds. MPI launches and GPU execution "
                   "require separate runtime checks.\n";
      return 0;
    }
  } catch (const std::exception &error) {
    std::cerr << "openpfc doctor: " << error.what() << '\n';
    if (!stage.empty()) std::cerr << "Diagnostics: " << stage << '\n';
  }
  std::cerr << "Load compatible compiler/MPI modules, select an installed SDK "
               "with --sdk or OpenPFC_DIR/CMAKE_PREFIX_PATH, and make the "
               "requested CMake generator available. Then rerun openpfc doctor.\n";
  return 1;
}
Build compile_source(const Arguments &args) {
  if (args.positional.empty()) throw Error("compile expects a C++ translation unit");
  model_environment(args);
  const auto type = args.get("build-type", "Release");
  auto files = source_files(args);
  auto contents = read(prefix() / "share/openpfc/templates/model.cmake");
  const auto declarations = project_files(args, files);
  const auto cache = cache_root();
  const auto stage = cache / "staging" / ("build-" + unique_id());
  safe_directory(stage);
  snapshot(stage / "src", files);
  write(stage / "CMakeLists.txt", contents);
  write(stage / "model-files.cmake", declarations);
  auto command = configure(args, stage, stage / "query");
  command.insert(command.end(),
                 {"-DOPENPFC_DISCOVERY_ONLY=ON",
                  "-DOPENPFC_QUERY_REPORT=" + (stage / "consumer.json").string()});
  checked_run(command, stage, stage / "query.stdout.log",
              stage / "query.stderr.log");
  auto consumer = read_json(stage / "consumer.json");
  // Explicitly declared toolchain inputs participate without dumping the
  // environment.
  Json toolchain_environment = Json::object();
  for (const auto *name :
       {"CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH", "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH", "LIBRARY_PATH", "CMAKE_PREFIX_PATH", "LD_LIBRARY_PATH",
        "LD_PRELOAD", "PATH"})
    if (!env(name).empty()) toolchain_environment[name] = sha256(env(name));
  Json identity{{"schema_version", 1},
                {"domain", "openpfc.model-build.v1"},
                {"sources", inventory(files)},
                {"template_sha256", sha256(contents)},
                {"model_files_sha256", sha256(declarations)},
                {"backend", "cpu"},
                {"build_jobs", args.positive("jobs", 2)},
                {"consumer", consumer},
                {"toolchain_environment", toolchain_environment}};
  const auto id = sha256(identity.dump());
  const auto root = cache / "builds" / ("openpfc-build-" + id);
  Lock lock(cache / "locks" / ("build-" + id + ".lock"));
  if (fs::exists(root)) {
    if (fs::is_symlink(fs::symlink_status(root)))
      throw Error("refusing symlink cache entry");
    Build result{root, read_json(root / "build-manifest.json")};
    if (result.manifest.at("identity") != identity)
      throw Error("cached manifest identity mismatch");
    validate_build(result);
    fs::remove_all(stage);
    std::cout << "Cache hit: " << root << '\n';
    return result;
  }
  command = configure(args, stage, stage / "build");
  command.push_back("-DOPENPFC_QUERY_REPORT=" +
                    (stage / "built-consumer.json").string());
  checked_run(command, stage, stage / "configure.stdout.log",
              stage / "configure.stderr.log");
  if (read_json(stage / "built-consumer.json") != consumer)
    throw Error("SDK/toolchain changed between discovery and configuration; failed "
                "snapshot preserved at " +
                stage.string());
  checked_run({find_program("cmake").string(), "--build", (stage / "build").string(),
               "--config", type, "--parallel",
               std::to_string(args.positive("jobs", 2))},
              stage, stage / "compile.stdout.log", stage / "compile.stderr.log");
  auto verify_command = configure(args, stage, stage / "query");
  verify_command.insert(
      verify_command.end(),
      {"-DOPENPFC_DISCOVERY_ONLY=ON",
       "-DOPENPFC_QUERY_REPORT=" + (stage / "verified-consumer.json").string()});
  checked_run(verify_command, stage, stage / "verify.stdout.log",
              stage / "verify.stderr.log");
  if (read_json(stage / "verified-consumer.json") != consumer)
    throw Error(
        "SDK/toolchain changed during compilation; staged build preserved at " +
        stage.string());
  fs::path binary = stage / "build/model";
  if (!fs::is_regular_file(binary)) binary = stage / "build" / type / "model";
  require_executable(binary);
  fs::copy_file(binary, stage / "model");
  freeze_file(stage / "model");
  Json manifest{{"schema_version", 1},
                {"build_id", id},
                {"identity", identity},
                {"binary_sha256", file_sha256(stage / "model")}};
  write_json(stage / "build-manifest.json", manifest);
  freeze_file(stage / "build-manifest.json");
  freeze_file(stage / "CMakeLists.txt");
  freeze_file(stage / "model-files.cmake");
  verify_files(stage / "src", identity.at("sources"));
  fs::rename(stage, root);
  std::cout << "Built: " << root << '\n';
  return {root, manifest};
}
int compile_model(const Arguments &args) {
  const auto requested = absolute_path(
      args.get("output", fs::path(args.positional.front()).stem().string()));
  // Resolve the parent, but never follow an existing output symlink. A missing
  // parent is an error, as with a compiler's ordinary -o option.
  const auto output = fs::canonical(requested.parent_path()) / requested.filename();
  const auto status = fs::symlink_status(output);
  if (fs::is_symlink(status) || (fs::exists(status) && !fs::is_regular_file(status)))
    throw Error("output must be a regular file, not a symlink or directory: " +
                output.string());
  const auto cache_relative = output.lexically_relative(fs::canonical(cache_root()));
  if (!cache_relative.empty() && *cache_relative.begin() != "..")
    throw Error("output must be outside the OpenPFC cache: " + output.string());
  const auto source = source_root(args);
  for (const auto &[name, bytes] : source_files(args)) {
    (void)bytes;
    const auto input = source / name;
    if (output == input || (fs::exists(output) && fs::equivalent(output, input)))
      throw Error("output would overwrite a model source file: " + output.string());
  }

  const auto build = compile_source(args);
  const auto temporary = output.parent_path() /
                         ("." + output.filename().string() + ".tmp-" + unique_id());
  try {
    // A separate copy lets users move/edit/delete their executable without
    // changing the verified cache. Rename also permits recompilation while the
    // previous executable is running, without exposing a partially copied file.
    fs::copy_file(build.root / "model", temporary);
    fs::permissions(temporary, fs::perms::owner_write, fs::perm_options::add);
    fs::rename(temporary, output);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
  std::cout << "Executable: " << output << '\n';
  return 0;
}

int run_source(const Arguments &args) {
  if (args.positional.size() != 2)
    throw Error("run source.cpp expects one input JSON pathname");
  // Freeze requested inputs before invoking a potentially long compiler process.
  const auto files = input_files(args);
  auto build = compile_source(args);
  auto execution = args;
  if (!execution.has("ranks")) execution.options["ranks"] = {"1"};
  auto command = launcher(execution, build.manifest.at("identity")
                                         .at("consumer")
                                         .at("MPIEXEC_EXECUTABLE")
                                         .at("value")
                                         .get<std::string>());
  if (!command.empty()) command.front() = find_program(command.front()).string();
  Json identity{{"schema_version", 1},
                {"domain", "openpfc.model-run.v1"},
                {"build_id", build.manifest.at("build_id")},
                {"inputs", inventory(files)},
                {"input", fs::path(args.positional[1]).filename().generic_string()},
                {"ranks", args.positive("ranks")},
                {"launcher", command},
                {"arguments", args.all("arg")},
                {"environment", runtime_environment(args)}};
  if (!command.empty()) identity["launcher_sha256"] = file_sha256(command.front());
  const auto id = sha256(identity.dump());
  const auto cache = cache_root();
  const auto root = cache / "runs" / ("openpfc-run-" + id);
  fs::path attempt;
  {
    Lock lock(cache / "locks" / ("run-" + id + ".lock"));
    if (!fs::exists(root)) {
      auto stage = cache / "staging" / ("run-" + unique_id());
      safe_directory(stage);
      snapshot(stage / "inputs", files);
      safe_directory(stage / "model");
      fs::copy_file(build.root / "model", stage / "model/model");
      freeze_file(stage / "model/model");
      fs::copy_file(build.root / "build-manifest.json",
                    stage / "model/build-manifest.json");
      fs::copy_file(build.root / "CMakeLists.txt", stage / "model/CMakeLists.txt");
      fs::copy_file(build.root / "model-files.cmake",
                    stage / "model/model-files.cmake");
      fs::copy(build.root / "src", stage / "model/src", fs::copy_options::recursive);
      write_json(stage / "run-manifest.json",
                 {{"schema_version", 1},
                  {"run_id", id},
                  {"identity", identity},
                  {"binary_sha256", build.manifest.at("binary_sha256")}});
      freeze_file(stage / "run-manifest.json");
      safe_directory(stage / "attempts");
      fs::rename(stage, root);
    }
    if (fs::is_symlink(fs::symlink_status(root)))
      throw Error("refusing symlink run record");
    auto manifest = read_json(root / "run-manifest.json");
    if (manifest.at("identity") != identity ||
        manifest.at("run_id").get<std::string>() != sha256(identity.dump()))
      throw Error("run record fingerprint mismatch");
    verify_files(root / "inputs", identity.at("inputs"));
    if (file_sha256(root / "model/model") !=
        build.manifest.at("binary_sha256").get<std::string>())
      throw Error("run model fingerprint mismatch");
    attempt = root / "attempts" / unique_id();
    safe_directory(attempt);
    snapshot(attempt, files);
    safe_directory(attempt / "results");
    write_json(attempt / "status.json", {{"state", "running"}});
  }
  command.insert(command.end(),
                 {(root / "model/model").string(),
                  (attempt / fs::path(args.positional[1]).filename()).string()});
  const auto extra = args.all("arg");
  command.insert(command.end(), extra.begin(), extra.end());
  std::cout << "Run: " << root << "\nAttempt: " << attempt << '\n';
  int code = 126;
  try {
    code = run(command, attempt, attempt / "stdout.log", attempt / "stderr.log");
    Json outputs = Json::object();
    for (const auto &file : fs::recursive_directory_iterator(attempt / "results")) {
      if (file.is_symlink())
        throw Error("refusing symlink output checksum: " + file.path().string());
      if (file.is_regular_file())
        outputs[file.path().lexically_relative(attempt).generic_string()] =
            file_sha256(file.path());
    }
    verify_files(attempt, identity.at("inputs"));
    replace_json(attempt / "status.json",
                 {{"state", code == 0 ? "succeeded" : "failed"},
                  {"exit_code", code},
                  {"outputs", outputs},
                  {"stdout_sha256", file_sha256(attempt / "stdout.log")},
                  {"stderr_sha256", file_sha256(attempt / "stderr.log")}});
  } catch (const std::exception &error) {
    replace_json(attempt / "status.json", {{"state", "failed"},
                                           {"exit_code", code == 0 ? 126 : code},
                                           {"error", error.what()}});
    throw;
  }
  if (code)
    std::cerr << "Model failed (" << code
              << "); logs and partial output preserved at " << attempt << '\n';
  return code;
}
int export_record(const Arguments &args) {
  args.allow({});
  if (args.positional.size() != 2)
    throw Error("export expects a cached build/run directory and a new destination");
  auto from = fs::canonical(args.positional[0]);
  auto cache = fs::canonical(cache_root());
  if (from.parent_path() != cache / "builds" && from.parent_path() != cache / "runs")
    throw Error("export source must be a cached OpenPFC build or run record");
  const bool is_build = fs::is_regular_file(from / "build-manifest.json");
  const auto record =
      read_json(from / (is_build ? "build-manifest.json" : "run-manifest.json"));
  const auto record_id =
      record.at(is_build ? "build_id" : "run_id").get<std::string>();
  if (!std::regex_match(record_id, std::regex("[0-9a-f]{64}")) ||
      from.filename() !=
          std::string(is_build ? "openpfc-build-" : "openpfc-run-") + record_id)
    throw Error("cached record directory does not match its fingerprint");
  // Stop new attempts being allocated between the active-status check and copy.
  Lock lock(cache / "locks" /
            (std::string(is_build ? "build-" : "run-") + record_id + ".lock"));
  if (is_build)
    validate_build({from, read_json(from / "build-manifest.json")});
  else {
    auto manifest = read_json(from / "run-manifest.json");
    if (manifest.at("run_id").get<std::string>() !=
        sha256(manifest.at("identity").dump()))
      throw Error("run manifest identity mismatch");
    verify_files(from / "inputs", manifest.at("identity").at("inputs"));
    if (file_sha256(from / "model/model") !=
        manifest.at("binary_sha256").get<std::string>())
      throw Error("run executable changed");
    for (const auto &attempt : fs::directory_iterator(from / "attempts")) {
      auto status = read_json(attempt.path() / "status.json");
      if (status.at("state").get<std::string>() == "running")
        throw Error("cannot export a run while an attempt is active");
      verify_files(attempt.path(), status.at("outputs"));
      verify_files(attempt.path(), manifest.at("identity").at("inputs"));
      for (const auto *stream : {"stdout", "stderr"}) {
        if (file_sha256(attempt.path() / (std::string(stream) + ".log")) !=
            status.at(std::string(stream) + "_sha256").get<std::string>())
          throw Error("attempt log fingerprint mismatch");
      }
    }
  }
  auto destination = absolute_path(args.positional[1]);
  if (destination == from || destination.string().starts_with(from.string() + "/"))
    throw Error("export destination cannot be inside its source");
  copy_record(from, destination);
  std::cout << "Exported: " << destination << '\n';
  return 0;
}
} // namespace openpfc_cli
