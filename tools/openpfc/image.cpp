// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <iostream>
#include <set>
#include <unistd.h>

namespace openpfc_cli {
namespace {
void output_path(const Json &value) {
  if (!value.is_string()) throw Error("bundled output path must be a string");
  const fs::path path(value.get<std::string>());
  if (path.empty() || path.is_absolute() || *path.begin() != "results" ||
      std::ranges::find(path, fs::path("..")) != path.end())
    throw Error("bundled output paths must be relative paths under results/");
}
} // namespace

void validate_bundle_input(const Json &settings) {
  if (!settings.is_object()) throw Error("case input must be a JSON object");
  if (settings.contains("restart_from"))
    throw Error(
        "bundling restart inputs is not supported; use a self-contained case");
  for (const auto *key : {"initial_conditions", "boundary_conditions"}) {
    if (!settings.contains(key)) continue;
    const auto &initial = settings.at(key);
    if (!initial.is_array())
      throw Error(std::string("bundled ") + key +
                  " must be an array of self-contained modifiers");
    static const std::set<std::string> types{"constant",
                                             "single_seed",
                                             "seed_grid",
                                             "seed_grid_fcc",
                                             "random_seeds",
                                             "fourier_modes",
                                             "cosine_mode",
                                             "indexed_noise",
                                             "seeded_noise",
                                             "gaussian_pulse",
                                             "kdv_soliton",
                                             "wave_packet",
                                             "gaussian_inclusion",
                                             "circular_inclusion",
                                             "lattice_seed",
                                             "fixed",
                                             "moving"};
    for (const auto &condition : initial) {
      if (!condition.is_object() || !types.contains(condition.value("type", "")))
        throw Error("bundle requires self-contained initial/boundary conditions; "
                    "file inputs "
                    "are not supported");
    }
  }
  for (const auto &field : settings.value("fields", Json::array()))
    output_path(field.at("data"));
  if (settings.contains("checkpoint") &&
      settings.at("checkpoint").contains("directory"))
    output_path(settings.at("checkpoint").at("directory"));
  if (settings.contains("diagnostics") && settings.at("diagnostics").contains("csv"))
    output_path(settings.at("diagnostics").at("csv"));
}

int image_command(const Arguments &args) {
  const auto root = prefix();
  const auto provenance = read_json(root / "provenance.json");
  if (provenance.value("kind", "") != "openpfc.case-bundle")
    throw Error("image commands require a bundled app case");
  if (args.has("version") || args.has("provenance")) {
    args.allow(args.has("version") ? std::vector<std::string>{"version"}
                                   : std::vector<std::string>{"provenance"});
    if (!args.positional.empty()) throw Error("image query takes no arguments");
    if (args.has("provenance"))
      std::cout << read(root / "provenance.json");
    else {
      auto version = read(root / "share/openpfc/version");
      while (!version.empty() && (version.back() == '\n' || version.back() == '\r'))
        version.pop_back();
      std::cout << "OpenPFC " << version << '\n';
    }
    return 0;
  }
  args.allow({});
  if (args.positional.size() != 2 ||
      (args.positional[0] != "init" && args.positional[0] != "run"))
    throw Error("usage: image {--version | --provenance | init CASE | run CASE}");
  const auto project = absolute_path(args.positional[1]);
  if (args.positional[0] == "init") {
    require_single_process();
    safe_directory(project.parent_path());
    if (!fs::create_directory(project))
      throw Error("refusing to overwrite existing case: " + project.string());
    // These are the only case files delivered in the image. Never copy an
    // arbitrary tree, run output or an absolute build-host executable path.
    for (const auto *name : {"input.json", "openpfc.json"})
      write(project / name, read(root / "case" / name));
    std::cout << "Created bundled case: " << project << '\n';
    return 0;
  }
  Arguments selected;
  selected.positional = {project.string()};
  const auto [directory, manifest] = case_manifest(selected);
  if (manifest.at("app") != provenance.at("case_app"))
    throw Error("case app does not match the bundled executable");
  validate_bundle_input(read_json(directory / "input.json"));
  safe_directory(directory / "results");
  // Replace this process, including under outer mpirun/srun: no nested
  // launcher, extra child process, Python or MPI-linked CLI is needed.
  const auto binary = (root / "bin/model").string();
  const auto input = (directory / "input.json").string();
  if (::chdir(directory.c_str()) != 0) throw Error("cannot enter case directory");
  ::execl(binary.c_str(), binary.c_str(), input.c_str(),
          static_cast<char *>(nullptr));
  throw Error("cannot execute bundled model: " + binary);
}
} // namespace openpfc_cli
