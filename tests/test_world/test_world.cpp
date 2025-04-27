// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/world.hpp>

using namespace pfc;

int main() {
  World world({10, 20, 30}, {0.0, 0.0, 0.0}, {0.1, 0.1, 0.1});
  if (world.get_Lx() != 10) {
    return -1;
  }
  return 0;
}
