// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "tungsten/cuda requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <tungsten/cuda/tungsten.hpp>

int main(int argc, char *argv[]) {
  cout << std::fixed;
  cout.precision(3);
  App<TungstenCUDA<double>> app(argc, argv);
  return app.main();
}
