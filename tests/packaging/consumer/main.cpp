// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Smallest possible use of the installed OpenPFC public API: construct a World
// and query it. If this configures, links, and runs, find_package(OpenPFC) and
// the exported targets/transitive deps are wired correctly.
#include <openpfc/kernel/data/world.hpp>

#include <cstdio>

int main() {
  auto world = pfc::world::create(pfc::GridSize({8, 8, 8}));
  const auto size = pfc::world::get_size(world);
  std::printf("OpenPFC consumer OK: world %dx%dx%d\n", size[0], size[1], size[2]);
  return (size[0] == 8 && size[1] == 8 && size[2] == 8) ? 0 : 1;
}
