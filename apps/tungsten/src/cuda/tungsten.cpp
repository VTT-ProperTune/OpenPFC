// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "tungsten/cuda requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/cuda/tungsten.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_app_main<TungstenCUDA<double>>(argc, argv);
}
