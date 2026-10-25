// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string_view>

namespace openpfc_cli {
namespace {
const std::regex k_glibc(
    R"(lib(?:c|m|mvec|dl|pthread|rt|util|resolv|anl|nss_\w+)\.so(\.[0-9]+)*)");
const std::regex
    k_openmpi(R"(lib(?:mpi[^/]*|open-pal|pmix|ucp|uct|ucs|ucm)\.so(\.[0-9]+)*)");
const std::regex
    k_cray(R"(lib(?:mpi[^/]*|fabric|cxi|pmi2?|pals|xpmem|dsmml)\.so(\.[0-9]+)*)");
// Interconnect libraries that must stay on the host even when they share a
// directory with ordinary system libraries. Binding that directory would hide
// the base image's C library, so these are selected one file at a time.
const std::regex k_fabric(
    R"(lib(?:fabric|efa|ibverbs|rdmacm|psm2|xpmem|cxi|hwloc)\.so(\.[0-9]+)*)");
const std::regex k_gpu(
    R"(lib(?:cudart|cuda|cublas|cublasLt|cufft|curand|cusparse|cusolver|nvrtc|nvidia-ml|amdhip64|hiprtc|rocfft|rocsparse|rocblas|rocsolver|hsa-runtime64|amd_comgr|rccl)\.so(\.[0-9]+)*)");

[[nodiscard]] bool matches(const std::regex &pattern, const std::string &name) {
  return std::regex_match(name, pattern);
}
[[nodiscard]] bool is_loader(const std::string &name) {
  return name.starts_with("ld-linux") || name.starts_with("ld64.so") ||
         name.starts_with("ld-musl");
}
[[nodiscard]] bool system_library_directory(const fs::path &directory) {
  const auto text = directory.lexically_normal().string();
  static constexpr std::string_view k_directories[] = {
      "/lib",
      "/lib64",
      "/usr/lib",
      "/usr/lib64",
      "/lib/x86_64-linux-gnu",
      "/usr/lib/x86_64-linux-gnu",
      "/lib/aarch64-linux-gnu",
      "/usr/lib/aarch64-linux-gnu",
  };
  return std::ranges::find(k_directories, std::string_view(text)) !=
         std::end(k_directories);
}
[[nodiscard]] std::optional<std::string> provider_leaf(const std::string &soname) {
  if (soname.starts_with("libfabric.so")) return "libfabric";
  if (soname.starts_with("libibverbs.so") || soname.starts_with("librdmacm.so"))
    return "libibverbs";
  if (soname.starts_with("libhwloc.so")) return "hwloc";
  return std::nullopt;
}
[[nodiscard]] bool contained(const fs::path &path, const fs::path &prefix) {
  const auto relative = path.lexically_relative(prefix);
  if (relative.empty()) return false;
  return relative.begin()->string() != "..";
}
void reject_separators(const std::string &text, const std::string &label) {
  if (text.find_first_of(",:\n") != std::string::npos)
    throw Error(label + " must not contain commas, colons, or newlines");
}
[[nodiscard]] std::optional<fs::path> provider_directory(const fs::path &library,
                                                         const std::string &leaf) {
  const auto candidate = library.parent_path() / leaf;
  std::error_code error;
  if (!fs::is_directory(candidate, error)) return std::nullopt;
  auto canonical = fs::weakly_canonical(candidate, error);
  if (error || !fs::is_directory(canonical) || canonical == "/" ||
      system_library_directory(canonical))
    return std::nullopt;
  reject_separators(canonical.string(), "provider directory");
  return canonical;
}
[[nodiscard]] fs::path canonical_directory(const fs::path &path,
                                           const std::string &label) {
  std::error_code error;
  const auto absolute = absolute_path(path);
  if (!fs::is_directory(absolute, error))
    throw Error(label + " must be a directory: " + path.string());
  auto canonical = fs::weakly_canonical(absolute, error);
  if (error) throw Error("cannot resolve " + label + ": " + path.string());
  if (canonical == "/") throw Error(label + " must not be the filesystem root");
  reject_separators(canonical.string(), label);
  reject_separators(absolute.lexically_normal().string(), label);
  return canonical;
}
[[nodiscard]] fs::path canonical_file(const fs::path &path,
                                      const std::string &label) {
  std::error_code error;
  const auto absolute = absolute_path(path);
  if (!fs::is_regular_file(absolute, error))
    throw Error(label + " is not a file: " + path.string());
  auto canonical = fs::weakly_canonical(absolute, error);
  if (error || !fs::is_regular_file(canonical))
    throw Error("cannot resolve " + label + ": " + path.string());
  reject_separators(canonical.string(), label);
  return canonical;
}
[[nodiscard]] std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char character : value) {
    if (character == '\'')
      quoted += "'\\''";
    else
      quoted += character;
  }
  quoted += "'";
  return quoted;
}
void publish(const fs::path &path, const std::string &bytes, bool executable) {
  write(path, bytes);
  auto mode = fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read;
  if (executable)
    mode |= fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec;
  fs::permissions(path, mode);
}
[[nodiscard]] Json library_records(const std::map<std::string, fs::path> &items) {
  Json records = Json::object();
  for (const auto &[name, path] : items)
    records[name] = Json{{"path", path.string()}, {"sha256", file_sha256(path)}};
  return records;
}
[[nodiscard]] std::string glibc_text(int major, int minor) {
  if (major <= 0) return {};
  return std::to_string(major) + "." + std::to_string(minor);
}
void raise_glibc(int &major, int &minor, int candidate_major, int candidate_minor) {
  if (candidate_major > major ||
      (candidate_major == major && candidate_minor > minor)) {
    major = candidate_major;
    minor = candidate_minor;
  }
}
[[nodiscard]] std::uintmax_t tree_bytes(const fs::path &root) {
  std::uintmax_t total = 0;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) total += entry.file_size();
  }
  return total;
}
[[nodiscard]] std::pair<int, int> base_glibc(const fs::path &runtime,
                                             const fs::path &base,
                                             const fs::path &directory) {
  const std::vector<std::string> probes{
      "/lib64/libc.so.6",
      "/usr/lib64/libc.so.6",
      "/lib/x86_64-linux-gnu/libc.so.6",
      "/usr/lib/x86_64-linux-gnu/libc.so.6",
      "/lib/aarch64-linux-gnu/libc.so.6",
  };
  std::vector<Command> commands;
  for (const auto &probe : probes) {
    commands.push_back(
        {runtime.string(), "exec", "--containall", base.string(), probe});
    commands.push_back({runtime.string(), "exec", base.string(), probe});
  }
  commands.push_back({runtime.string(), "exec", "--containall", base.string(),
                      "/bin/sh", "-c", "getconf GNU_LIBC_VERSION"});
  const std::regex version_pattern(R"((?:version|glibc)\s+([0-9]+)\.([0-9]+))");
  std::string last_output;
  for (const auto &command : commands) {
    const auto stamp = unique_id();
    const auto out = directory / ("base-glibc-" + stamp + ".stdout.log");
    const auto err = directory / ("base-glibc-" + stamp + ".stderr.log");
    const auto code = run(command, directory, out, err);
    const auto text = read(out) + "\n" + read(err);
    last_output = text;
    std::smatch match;
    if (code == 0 && std::regex_search(text, match, version_pattern))
      return {std::stoi(match[1].str()), std::stoi(match[2].str())};
  }
  throw Error("cannot read the base image GLIBC version; --base must be a runnable "
              "SIF with glibc and the selected container runtime.\n" +
              last_output);
}
// Current Apptainer and Singularity SIF files start with
// "#!/usr/bin/env run-singularity\n", then a NUL, then the SIF header.
// Older images put SIF_MAGIC at offset 0. Either layout is a SIF.
[[nodiscard]] bool sif_header(const fs::path &image) {
  std::string header(64, '\0');
  std::ifstream stream(image, std::ios::binary);
  stream.read(header.data(), static_cast<std::streamsize>(header.size()));
  header.resize(
      static_cast<std::size_t>(std::max<std::streamsize>(0, stream.gcount())));
  constexpr std::string_view magic = "SIF_MAGIC";
  if (header.starts_with(magic)) return true;
  constexpr std::string_view launch = "#!/usr/bin/env run-singularity\n";
  const auto offset = launch.size() + 1;
  return header.size() >= offset + magic.size() && header.starts_with(launch) &&
         std::string_view(header).substr(offset, magic.size()) == magic;
}
[[nodiscard]] std::string helper_script(const std::string &runtime,
                                        const std::vector<std::string> &binds,
                                        const std::string &library_path,
                                        const std::string &filename) {
  std::ostringstream script;
  // TMPDIR is where Open MPI / PRTE puts the per-launch session. It does not
  // exist until mpirun starts, so the helper binds it at launch rather than
  // baking one directory into the recipe. The bind is writable: the client
  // records its rank there. System library directories are refused.
  script << "#!/bin/sh\n"
            "set -eu\n"
            "bundle_dir=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
            "session_binds=\n"
            "add_session() {\n"
            "  directory=$1\n"
            "  case $directory in\n"
            "    /|/usr|/usr/lib|/usr/lib64|/lib|/lib64) return ;;\n"
            "    /*) ;;\n"
            "    *) return ;;\n"
            "  esac\n"
            "  case $directory in\n"
            "    *[,:]*) return ;;\n"
            "    *' '*) return ;;\n"
            "  esac\n"
            "  if [ -d \"$directory\" ]; then\n"
            "    session_binds=\"$session_binds --bind $directory:$directory\"\n"
            "  fi\n"
            "}\n"
            "if [ -n \"${TMPDIR:-}\" ]; then\n"
            "  add_session \"$TMPDIR\"\n"
            "fi\n"
            "if [ -n \"${PMIX_SERVER_TMPDIR:-}\" ]; then\n"
            "  add_session \"$PMIX_SERVER_TMPDIR\"\n"
            "fi\n"
            "exec "
         << shell_quote(runtime) << " run";
  for (const auto &bind : binds) script << " --bind " << shell_quote(bind);
  script << " $session_binds --env "
         << shell_quote("LD_LIBRARY_PATH=" + library_path) << " \"$bundle_dir\"/"
         << shell_quote(filename) << " \"$@\"\n";
  return script.str();
}
} // namespace

int bundle_model(const Arguments &args) {
  args.allow({"output",   "base",       "prepare-only", "runtime",     "host-prefix",
              "host-lib", "site",       "input",        "sdk",         "file",
              "define",   "build-type", "toolchain",    "source-root", "generator",
              "cxx",      "cc",         "jobs",         "backend",     "profile"});
  if (args.positional.size() != 1)
    throw Error("bundle expects one C++ model source file");
  const auto extension = fs::path(args.positional.front()).extension();
  if (extension != ".cpp" && extension != ".cc" && extension != ".cxx")
    throw Error("bundle packages a C++ model such as model.cpp; catalogued app "
                "cases still use scripts/bake_case.py");
  const auto requested = args.get("profile", "local-cpu");
  if (requested != "local" && requested != "local-cpu")
    throw Error("profile " + requested +
                " is not qualified for native bundles; CPU host-MPI bundles use "
                "--profile=local-cpu. tohtori-cuda and lumi-g remain unverified");
  if (args.get("backend", "cpu") != "cpu")
    throw Error(
        "backend " + args.get("backend") +
        " is not qualified for native bundles; CPU bundles use --backend=cpu");
  const auto site = args.get("site", "openmpi");
  if (site != "openmpi" && site != "cray")
    throw Error("--site must be openmpi or cray");
  const auto runtime_name = args.get("runtime", "apptainer");
  if (runtime_name != "apptainer" && runtime_name != "singularity")
    throw Error("--runtime must be apptainer or singularity");
  if (!args.has("output")) throw Error("bundle requires -o or --output IMAGE.sif");
  const auto output = absolute_path(args.get("output"));
  const auto sidecar = fs::path(output.string() + ".sha256");
  const auto launched = fs::path(output.string() + ".run");
  if (fs::exists(output) || fs::exists(sidecar) || fs::exists(launched))
    throw Error("refusing to overwrite " + output.string() +
                "; remove the image, its .sha256 sidecar and its .run helper first");
  if (!args.has("base"))
    throw Error("CPU bundles need --base BASE.sif, a pinned image that provides "
                "glibc and /bin/sh; scratch images are not qualified yet");
  const auto base = canonical_file(args.get("base"), "--base");
  safe_directory(output.parent_path());
  std::vector<fs::path> prefixes;
  std::set<std::string> bind_prefixes;
  for (const auto &given : args.all("host-prefix")) {
    const auto absolute = absolute_path(given).lexically_normal();
    const auto canonical = canonical_directory(given, "--host-prefix");
    if (system_library_directory(canonical))
      throw Error("--host-prefix is a system library directory (" +
                  canonical.string() +
                  "); pass --host-lib for each host library that lives beside the C "
                  "library");
    prefixes.push_back(canonical);
    bind_prefixes.insert(absolute.string());
    bind_prefixes.insert(canonical.string());
  }
  struct RequestedHostLib {
    std::string filename;
    fs::path canonical;
  };
  std::vector<RequestedHostLib> host_libs;
  for (const auto &given : args.all("host-lib")) {
    const fs::path path(given);
    const auto filename = path.filename().string();
    if (filename.empty() || filename.find('/') != std::string::npos)
      throw Error("--host-lib needs a file name");
    host_libs.push_back(
        RequestedHostLib{filename, canonical_file(path, "--host-lib")});
  }
  std::string input_name;
  std::string input_bytes;
  if (args.has("input")) {
    const auto input = canonical_file(args.get("input"), "--input");
    input_name = input.filename().string();
    if (input_name.empty() || input_name == "." || input_name == ".." ||
        input_name.find_first_of("/\n'\"\\") != std::string::npos)
      throw Error("--input must be a single file name without quotes");
    input_bytes = read(input);
  }
  auto build = compile_source(args);
  const auto model = build.root / "model";
  require_executable(model);
  const auto closure = dynamic_closure(model);
  const auto &mpi_pattern = site == "cray" ? k_cray : k_openmpi;
  std::map<std::string, fs::path> bundled;
  std::map<std::string, fs::path> host_from_prefix;
  std::map<std::string, fs::path> host_from_file;
  std::map<std::string, fs::path> system;
  std::set<fs::path> used_host_libs;
  std::set<std::string> provider_binds;
  int glibc_major = closure.glibc_major;
  int glibc_minor = closure.glibc_minor;
  std::vector<RuntimeLibrary> ordered = closure.libraries;
  std::ranges::sort(ordered,
                    [](const RuntimeLibrary &left, const RuntimeLibrary &right) {
                      return left.soname < right.soname;
                    });
  for (const auto &library : ordered) {
    if (matches(k_gpu, library.soname) || library.soname.starts_with("libpython"))
      throw Error("model links " + library.soname +
                  "; native bundles currently qualify CPU host-MPI only, with no "
                  "Python runtime. tohtori-cuda and lumi-g remain unverified");
    const bool covered = std::ranges::any_of(prefixes, [&](const fs::path &prefix) {
      return contained(library.path, prefix);
    });
    const RequestedHostLib *explicit_host = nullptr;
    for (const auto &item : host_libs) {
      if (item.filename != library.soname && item.canonical != library.path)
        continue;
      if (item.canonical != library.path)
        throw Error("--host-lib does not match the resolved library: " +
                    library.soname + " (" + item.canonical.string() + ")");
      explicit_host = &item;
    }
    if (explicit_host != nullptr) {
      host_from_file.emplace(library.soname, library.path);
      used_host_libs.insert(library.path);
      if (const auto leaf = provider_leaf(library.soname)) {
        if (const auto directory = provider_directory(library.path, *leaf))
          provider_binds.insert(directory->string());
      }
    } else if (covered) {
      host_from_prefix.emplace(library.soname, library.path);
    } else if (matches(mpi_pattern, library.soname) ||
               matches(k_fabric, library.soname)) {
      if (system_library_directory(library.path.parent_path()))
        throw Error("host MPI stack library shares a system directory; pass "
                    "--host-lib FILE: " +
                    library.path.string());
      throw Error("host MPI stack library needs --host-prefix or --host-lib: " +
                  library.path.string());
    } else if (matches(k_glibc, library.soname) || is_loader(library.soname)) {
      system.emplace(library.soname, library.path);
    } else {
      bundled.emplace(library.soname, library.path);
    }
    raise_glibc(glibc_major, glibc_minor, library.glibc_major, library.glibc_minor);
  }
  for (const auto &item : host_libs) {
    if (!used_host_libs.contains(item.canonical))
      throw Error("--host-lib is not a runtime dependency of the model: " +
                  item.filename + " (" + item.canonical.string() + ")");
  }
  const auto floor = glibc_text(glibc_major, glibc_minor);
  const auto &consumer = build.manifest.at("identity").at("consumer");
  Json sdk_manifest = nullptr;
  if (consumer.contains("sdk_manifest_sha256"))
    sdk_manifest = consumer.at("sdk_manifest_sha256").at("value");
  Json host_records = library_records(host_from_prefix);
  for (const auto &[name, path] : host_from_file)
    host_records[name] =
        Json{{"path", path.string()}, {"sha256", file_sha256(path)}};
  Json provenance{
      {"format_version", 1},
      {"kind", "openpfc.model-bundle"},
      {"profile", "local-cpu"},
      {"profile_qualification", "cpu-host-mpi"},
      {"unqualified_profiles", {"tohtori-cuda", "lumi-g"}},
      {"mpi_mode", "host-bind"},
      {"site", site},
      {"runtime", runtime_name},
      {"build_id", build.manifest.at("build_id")},
      {"binary_sha256", build.manifest.at("binary_sha256")},
      {"base_sha256", file_sha256(base)},
      {"input_sha256",
       input_name.empty() ? Json(nullptr) : Json(sha256(input_bytes))},
      {"sdk_manifest_sha256", sdk_manifest},
      {"interpreter", closure.interpreter},
      {"closure", "dt-needed"},
      {"closure_note",
       "Direct and transitive ELF NEEDED entries and their GLIBC "
       "symbol requirements. A library opened only with dlopen is not "
       "copied. When a host fabric library is bound as one file, an "
       "adjacent libfabric, libibverbs or hwloc provider directory is "
       "bind-mounted if it exists and is not itself a system library "
       "directory."},
      {"host_glibc_requirement", floor.empty() ? Json(nullptr) : Json(floor)},
      {"bundled_libraries", library_records(bundled)},
      {"host_libraries", host_records},
      {"base_provided_glibc", library_records(system)},
      {"host_prefixes",
       std::vector<std::string>(bind_prefixes.begin(), bind_prefixes.end())},
      {"host_provider_directories",
       std::vector<std::string>(provider_binds.begin(), provider_binds.end())},
      {"source_identity_note",
       "build_id is the content-addressed model build. The SIF digest "
       "is stored beside the image, not in this file."}};
  Json identity{{"schema_version", 1},
                {"domain", "openpfc.model-bundle.v1"},
                {"provenance", provenance}};
  const auto id = sha256(identity.dump());
  const auto cache = cache_root();
  const auto root = cache / "bundles" / ("openpfc-bundle-" + id);
  Lock lock(cache / "locks" / ("bundle-" + id + ".lock"));
  auto verify = [&] {
    if (fs::is_symlink(fs::symlink_status(root)))
      throw Error("refusing symlink bundle cache entry");
    const auto manifest = read_json(root / "bundle-manifest.json");
    if (manifest.at("identity") != identity ||
        manifest.at("bundle_id").get<std::string>() != id)
      throw Error("cached bundle fingerprint mismatch; remove " + root.string());
    if (file_sha256(root / "payload/bin/model") !=
        build.manifest.at("binary_sha256").get<std::string>())
      throw Error("cached bundle model changed; remove " + root.string());
    for (auto library = provenance.at("bundled_libraries").begin();
         library != provenance.at("bundled_libraries").end(); ++library) {
      const auto file = root / "payload/lib" / library.key();
      if (file_sha256(file) != library.value().at("sha256").get<std::string>())
        throw Error("cached bundled library changed: " + file.string());
    }
    for (const auto *name :
         {"bundle.def", "run-host.sh", "payload/provenance.json"}) {
      if (file_sha256(root / name) != manifest.at(name).get<std::string>())
        throw Error("cached bundle recipe changed: " + (root / name).string());
    }
  };
  if (!fs::exists(root)) {
    const auto stage = cache / "staging" / ("bundle-" + unique_id());
    safe_directory(stage);
    safe_directory(stage / "payload/bin");
    safe_directory(stage / "payload/lib");
    safe_directory(stage / "payload/hostlib");
    publish(stage / "payload/hostlib/.keep", "", false);
    publish(stage / "payload/bin/model", read(model), true);
    for (const auto &[name, path] : bundled)
      publish(stage / "payload/lib" / name, read(path), false);
    if (!input_name.empty()) {
      safe_directory(stage / "payload/inputs");
      publish(stage / "payload/inputs" / input_name, input_bytes, false);
    }
    publish(stage / "payload/build-manifest.json",
            read(build.root / "build-manifest.json"), false);
    publish(stage / "payload/provenance.json", provenance.dump(2) + "\n", false);
    std::error_code error;
    fs::create_symlink(base, stage / "base.sif", error);
    if (error)
      throw Error("cannot record the base image symlink: " + error.message());
    std::string runscript = "    exec /opt/openpfc/bin/model \"$@\"\n";
    if (!input_name.empty()) {
      runscript =
          "    if [ \"$#\" -eq 0 ]; then\n        exec /opt/openpfc/bin/model "
          "/opt/openpfc/inputs/" +
          input_name + "\n    fi\n    exec /opt/openpfc/bin/model \"$@\"\n";
    }
    const std::string definition =
        "Bootstrap: localimage\nFrom: base.sif\n\n"
        "%files\n    payload /opt/openpfc\n\n"
        "%environment\n    export PATH=/opt/openpfc/bin:$PATH\n"
        "    export LD_LIBRARY_PATH=/opt/openpfc/lib:${LD_LIBRARY_PATH:-}\n\n"
        "%runscript\n" +
        runscript;
    publish(stage / "bundle.def", definition, false);
    std::vector<std::string> binds;
    for (const auto &prefix : bind_prefixes)
      binds.push_back(prefix + ":" + prefix + ":ro");
    for (const auto &directory : provider_binds)
      binds.push_back(directory + ":" + directory + ":ro");
    if (site == "cray") binds.emplace_back("/var/spool/slurmd:/var/spool/slurmd");
    for (const auto &[name, path] : host_from_file)
      binds.push_back(path.string() + ":/opt/openpfc/hostlib/" + name + ":ro");
    std::set<std::string> host_directories;
    for (const auto &[name, path] : host_from_prefix) {
      host_directories.insert(path.parent_path().string());
      (void)name;
    }
    std::string library_path = "/opt/openpfc/lib";
    if (!host_from_file.empty()) library_path += ":/opt/openpfc/hostlib";
    for (const auto &directory : host_directories) library_path += ":" + directory;
    publish(stage / "run-host.sh",
            helper_script(runtime_name, binds, library_path, "model.sif"), true);
    const auto payload_bytes = tree_bytes(stage / "payload");
    Json manifest{
        {"schema_version", 1},
        {"bundle_id", id},
        {"identity", identity},
        {"payload_bytes", payload_bytes},
        {"bundle.def", file_sha256(stage / "bundle.def")},
        {"run-host.sh", file_sha256(stage / "run-host.sh")},
        {"payload/provenance.json", file_sha256(stage / "payload/provenance.json")}};
    publish(stage / "bundle-manifest.json", manifest.dump(2) + "\n", false);
    fs::rename(stage, root);
  }
  verify();
  const auto payload_bytes = read_json(root / "bundle-manifest.json")
                                 .at("payload_bytes")
                                 .get<std::uintmax_t>();
  std::cout << "Bundle: " << root.string() << "\nProfile: local-cpu (" << site
            << " host MPI)\nPayload bytes: " << payload_bytes << '\n';
  if (!floor.empty()) std::cout << "GLIBC requirement: " << floor << '\n';
  std::cout << "Unverified profiles: tohtori-cuda lumi-g\n";
  if (args.has("prepare-only")) {
    std::cout << "Prepared bundle recipe; image build skipped.\n";
    return 0;
  }
  const auto runtime = find_program(runtime_name);
  if (!floor.empty()) {
    const auto [major, minor] = base_glibc(runtime, base, root);
    if (major < glibc_major || (major == glibc_major && minor < glibc_minor))
      throw Error("base image GLIBC " + glibc_text(major, minor) +
                  " is older than the host libraries require (" + floor + ")");
    std::cout << "Base GLIBC: " << glibc_text(major, minor) << '\n';
  }
  const auto staged_image = root / "model.sif";
  if (!fs::exists(staged_image)) {
    const auto stamp = unique_id();
    try {
      checked_run(
          {runtime.string(), "build", "--disable-cache", "model.sif", "bundle.def"},
          root, root / ("sif-" + stamp + ".stdout.log"),
          root / ("sif-" + stamp + ".stderr.log"));
    } catch (const Error &) {
      std::error_code error;
      fs::remove(staged_image, error);
      throw;
    }
  }
  if (!sif_header(staged_image))
    throw Error("container runtime did not produce a SIF at " +
                staged_image.string());
  if (fs::exists(fs::symlink_status(output)) || fs::exists(sidecar) ||
      fs::exists(launched))
    throw Error("refusing to overwrite " + output.string());
  const auto temporary = output.parent_path() /
                         (output.filename().string() + ".partial-" + unique_id());
  try {
    fs::copy_file(staged_image, temporary);
    fs::rename(temporary, output);
    const auto digest = file_sha256(output);
    publish(sidecar, digest + "\n", false);
    std::vector<std::string> binds;
    for (const auto &prefix : bind_prefixes)
      binds.push_back(prefix + ":" + prefix + ":ro");
    for (const auto &directory : provider_binds)
      binds.push_back(directory + ":" + directory + ":ro");
    if (site == "cray") binds.emplace_back("/var/spool/slurmd:/var/spool/slurmd");
    for (const auto &[name, path] : host_from_file)
      binds.push_back(path.string() + ":/opt/openpfc/hostlib/" + name + ":ro");
    std::set<std::string> host_directories;
    for (const auto &[name, path] : host_from_prefix) {
      host_directories.insert(path.parent_path().string());
      (void)name;
    }
    std::string library_path = "/opt/openpfc/lib";
    if (!host_from_file.empty()) library_path += ":/opt/openpfc/hostlib";
    for (const auto &directory : host_directories) library_path += ":" + directory;
    publish(
        launched,
        helper_script(runtime_name, binds, library_path, output.filename().string()),
        true);
    std::cout << "Image: " << output.string()
              << "\nImage bytes: " << fs::file_size(output)
              << "\nDigest: " << sidecar.string() << '\n';
  } catch (const std::exception &) {
    std::error_code error;
    fs::remove(temporary, error);
    fs::remove(output, error);
    fs::remove(sidecar, error);
    fs::remove(launched, error);
    throw;
  }
  return 0;
}
} // namespace openpfc_cli
