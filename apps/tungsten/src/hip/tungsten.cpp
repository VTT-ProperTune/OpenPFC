// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_HIP)
#error "tungsten/hip requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <tungsten/hip/tungsten.hpp>

int main(int argc, char *argv[]) {
  std::cout << std::fixed;
  std::cout.precision(3);
  pfc::ui::App<TungstenHIP<double>> app(argc, argv);
  return app.main();
}
