// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "cli.hpp"
#include <iostream>

namespace openpfc_cli {
int init_model(const Arguments &args) {
  args.allow({"model"});
  if (args.positional.size() != 1)
    throw Error("init --model=diffusion expects one new directory");
  if (args.get("model") != "diffusion")
    throw Error("unknown starter model; use --model=diffusion");
  const auto source = prefix() / "share/openpfc/templates/diffusion";
  std::map<std::string, std::string> files;
  for (const auto *name :
       {"model.cpp", "input.json", "README.md", "input.json.license"})
    files.emplace(name, read(source / name));
  const auto destination = absolute_path(args.positional.front());
  fs::create_directories(destination.parent_path());
  if (!fs::create_directory(destination))
    throw Error("refusing to overwrite existing model directory: " +
                destination.string());
  for (const auto &[name, contents] : files) write(destination / name, contents);
  std::cout
      << "Created diffusion model: " << destination
      << "\nFrom that directory, with the matching compiler/MPI environment:\n"
         "  openpfc doctor\n"
         "  openpfc compile model.cpp -o model\n"
         "  ./model input.json\n"
         "Add --sdk PREFIX when needed; see README.md and openpfc help compile.\n";
  return 0;
}
} // namespace openpfc_cli
