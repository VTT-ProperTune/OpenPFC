// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/openpfc.hpp>

int main() {
  using namespace pfc;
  auto world = world::create({32, 32, 32});
  std::cout << world << std::endl;
}
