// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_cuda_vtk.cpp
 * @brief Test Tungsten CUDA model with initial/boundary conditions and VTK output
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "This test requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include <nlohmann/json.hpp>
#include <tungsten/common/run_tungsten_gpu_vtk.hpp>
#include <tungsten/cuda/tungsten_model.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_gpu_vtk_main<TungstenCUDA<double>>(
      argc, argv, "tungsten_single_seed_256_cuda.json",
      "Tungsten CUDA Test with VTK Output",
      [](const nlohmann::json &j) {
        return pfc::ui::cuda_spectral_plan_options_from_json(j);
      },
      [](TungstenCUDA<double> &m) -> decltype(auto) { return m.get_cuda_fft(); });
}
