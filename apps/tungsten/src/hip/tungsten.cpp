// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "tungsten/hip requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/hip/tungsten.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_app_main<TungstenHIP<double>>(argc, argv);
}
