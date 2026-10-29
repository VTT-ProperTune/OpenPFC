// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "catalog.hpp"
#include "cli.hpp"
#include <iostream>
#include <regex>

namespace openpfc_cli {
const std::map<std::string, App> &catalog() {
  static const auto apps = [] {
    std::map<std::string, App> result;
    const auto data = Json::parse(catalog_json);
    for (auto it = data.begin(); it != data.end(); ++it) {
      App app;
      app.directory = it.value().at("directory").get<std::string>();
      app.binaries =
          it.value().at("binaries").get<std::map<std::string, std::string>>();
      app.description = it.value().at("description").get<std::string>();
      for (const auto &preset : it.value().at("presets"))
        app.presets.emplace_back(
            preset.at(0).get<std::string>(),
            preset.at(1).is_null() ? "" : preset.at(1).get<std::string>());
      result.emplace(it.key(), std::move(app));
    }
    return result;
  }();
  return apps;
}
std::pair<fs::path, Json> case_manifest(const Arguments &args) {
  if (args.positional.size() != 1) throw Error("expected one case directory");
  auto project = absolute_path(args.positional.front());
  const auto manifest = read_json(project / "openpfc.json");
  if (!manifest.is_object() || manifest.value("format_version", 0) != 1 ||
      !manifest.contains("app") || !manifest["app"].is_string() ||
      !catalog().contains(manifest["app"].get<std::string>()) ||
      manifest.value("input", "") != "input.json")
    throw Error(
        "unsupported openpfc.json: expected a version-1 catalogued app case");
  if (manifest.contains("profile")) {
    if (!manifest["profile"].is_string())
      throw Error("unsupported profile in openpfc.json");
    const auto profile = manifest["profile"].get<std::string>();
    if (profile != "local" && profile != "tohtori" && profile != "lumi")
      throw Error("unsupported profile in openpfc.json");
  }
  if (!fs::is_regular_file(project / "input.json"))
    throw Error("case input.json is missing");
  return {project, manifest};
}
namespace {
std::string backend(const std::string &profile) {
  if (profile == "local") return "cpu";
  if (profile == "tohtori") return "cuda";
  if (profile == "lumi") return "hip";
  throw Error("unsupported profile: " + profile);
}
Json smoke() {
  return Json::parse(R"({
    "model":{"name":"tungsten","params":{}},
    "domain":{"size":[16,16,16],"lower":[0,0,0],"upper":[16,16,16]},
    "timestepping":{"t0":0,"t1":0.1,"dt":0.01,"saveat":0.05},
    "initial_conditions":[{"target":"psi","type":"constant","n0":-0.4},
      {"target":"psi","type":"single_seed","amp_eq":0.215936,"rho_seed":-0.047}],
    "fields":[{"name":"psi","data":"results/psi_%04d.vti"}],
    "checkpoint":{"every":10,"directory":"results/checkpoints"}
  })");
}
} // namespace
int catalog_command(const Arguments &args) {
  if (args.command == "apps") {
    args.allow({});
    if (!args.positional.empty()) throw Error("apps takes no positional arguments");
    for (const auto &[name, app] : catalog()) {
      std::cout << name << ": " << app.description << "\n  presets: ";
      bool first = true;
      for (const auto &[preset, path] : app.presets) {
        if (!first) std::cout << ", ";
        first = false;
        std::cout << preset;
        (void)path;
      }
      std::cout << "; backends:";
      for (const auto &[kind, binary] : app.binaries) {
        std::cout << ' ' << kind;
        (void)binary;
      }
      std::cout << '\n';
    }
    std::cout << "Backend support describes build targets, not binaries installed "
                 "on this machine.\n";
    return 0;
  }
  args.allow({"app", "preset"});
  if (args.positional.size() != 1) throw Error("init expects one case directory");
  const auto name = args.get("app", "tungsten");
  if (!catalog().contains(name))
    throw Error("unknown app: " + name + "; use openpfc apps");
  const auto &app = catalog().at(name);
  const auto selected = args.get("preset", app.presets.front().first);
  const auto found = std::ranges::find_if(
      app.presets, [&](const auto &p) { return p.first == selected; });
  if (found == app.presets.end())
    throw Error("unknown preset for " + name + "; use openpfc apps");
  Json input = smoke();
  if (!found->second.empty()) {
    auto path = prefix() / "share/openpfc/cases" / app.directory / found->second;
    if (!fs::is_regular_file(path))
      throw Error("preset data missing; use a complete OpenPFC installation");
    input = read_json(path);
  }
  const auto destination = absolute_path(args.positional.front());
  if (!fs::create_directories(destination))
    throw Error("refusing to overwrite existing case: " + destination.string());
  write_json(destination / "input.json", input);
  write_json(destination / "openpfc.json", {{"format_version", 1},
                                            {"app", name},
                                            {"preset", selected},
                                            {"input", "input.json"}});
  write(destination / "README.md",
        "# " + name + " / " + selected + "\n\n" + app.description +
            "\n\nEdit input.json, then run `openpfc compile . --profile=local` and\n"
            "`openpfc run . --ranks=1` in the matching compiler/MPI environment.\n"
            "Output paths are relative to this case, under results/. Use fresh "
            "output\n"
            "paths for each run. Presets are demonstrations, not calibrated "
            "predictions.\n");
  std::cout << "Created " << name << " / " << selected << " case: " << destination
            << '\n';
  return 0;
}
int compile_case(const Arguments &args) {
  args.allow({"profile", "source", "build-dir", "jobs", "no-test", "dry-run"});
  auto [project, manifest] = case_manifest(args);
  const auto selected = args.get("profile", "local");
  const auto kind = backend(selected);
  const auto &app = catalog().at(manifest.at("app").get<std::string>());
  if (!app.binaries.contains(kind))
    throw Error(manifest.at("app").get<std::string>() + " has no " + kind +
                " target");
  const auto root = checkout(args);
  fs::path build;
  if (args.has("build-dir"))
    build = absolute_path(args.get("build-dir"));
  else if (selected == "lumi") {
    const auto flash = env("LUMI_FLASH_ROOT");
    build =
        fs::path(flash.empty() ? "/flash/project_462001519/juaho/build" : flash) /
        "openpfc-cli-lumi";
  } else
    build = root / "builds" / ("cli-" + selected);
  Command command{"bash", (root / "scripts/build.sh").string()};
  if (selected == "local")
    command.insert(command.end(), {"--cpu", "--no-submit"});
  else if (selected == "tohtori")
    command.insert(command.end(), {"--machine=tohtori", "--with-cuda"});
  else
    command.insert(command.end(), {"--machine=lumi", "--with-rocm", "--wait"});
  command.insert(command.end(),
                 {"--build-dir=" + build.string(), "--without-timestamp",
                  args.has("no-test") ? "--no-test" : "--test"});
  if (args.has("jobs"))
    command.push_back("--jobs=" + std::to_string(args.positive("jobs")));
  show_command(command, root);
  if (args.has("dry-run")) return 0;
  require_single_process();
  const auto code = run(command, root);
  if (code) return code;
  const auto binary = build / "apps" / app.directory / app.binaries.at(kind);
  require_executable(binary);
  manifest["profile"] = selected;
  manifest["source"] = root.string();
  manifest["build_dir"] = build.string();
  manifest["executable"] = binary.string();
  manifest.erase("mpiexec");
  if (fs::is_regular_file(build / "CMakeCache.txt")) {
    const auto cache = read(build / "CMakeCache.txt");
    const std::regex pattern("(?:^|\\n)MPIEXEC_EXECUTABLE:[^=]+=([^\\n]+)");
    std::smatch match;
    if (std::regex_search(cache, match, pattern) &&
        fs::is_regular_file(match[1].str()))
      manifest["mpiexec"] = match[1].str();
  }
  replace_json(project / "openpfc.json", manifest);
  std::cout << "Ready: openpfc run " << project << '\n';
  return 0;
}
int run_case(const Arguments &args) {
  args.allow({"ranks", "launcher", "executable", "dry-run"});
  auto [project, manifest] = case_manifest(args);
  const auto &app = catalog().at(manifest.at("app").get<std::string>());
  const auto selected = args.get("executable", manifest.value("executable", ""));
  fs::path binary;
  if (!selected.empty())
    binary = absolute_path(selected);
  else {
    const auto kind = backend(manifest.value("profile", "local"));
    if (!app.binaries.contains(kind))
      throw Error("recorded profile is unsupported for this app");
    binary = executable().parent_path() / app.binaries.at(kind);
    if (!fs::is_regular_file(binary))
      binary = prefix() / "apps" / app.directory / app.binaries.at(kind);
  }
  if (!fs::is_regular_file(binary))
    throw Error("app executable missing; compile the case or pass --executable");
  require_executable(binary);
  auto command =
      launcher(args, args.has("executable") ? "" : manifest.value("mpiexec", ""));
  command.insert(command.end(),
                 {binary.string(), (project / "input.json").string()});
  show_command(command, project);
  if (args.has("dry-run")) return 0;
  command[0] = find_program(command.front()).string();
  safe_directory(project / "results");
  return run(command, project);
}
} // namespace openpfc_cli
