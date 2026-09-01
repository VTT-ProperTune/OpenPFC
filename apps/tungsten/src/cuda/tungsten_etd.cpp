// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#error "tungsten_etd_cuda requires CUDA spectral support"
#endif

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/tungsten_etd_gpu_session.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_etd_main<tungsten::TungstenETDCUDASession>(
      argc, argv, "tungsten_etd_cuda");
}
