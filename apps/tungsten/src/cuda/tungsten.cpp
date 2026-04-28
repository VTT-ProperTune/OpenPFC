// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "tungsten/cuda requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <cstdlib>
#include <iostream>
#include <tungsten/cuda/tungsten.hpp>

int main(int argc, char *argv[]) {
  try {
    std::cout << std::fixed;
    std::cout.precision(3);
    pfc::ui::App<TungstenCUDA<double>> app(argc, argv);
    return app.main();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
