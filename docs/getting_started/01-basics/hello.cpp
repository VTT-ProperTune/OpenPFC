// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/openpfc.hpp>

int main() {
  using namespace pfc;
  Domain domain({32, 32, 32});
  std::cout << domain << std::endl;
}
