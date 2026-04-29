// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/cpu/tungsten.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_app_main<Tungsten>(argc, argv);
}
