// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_hip_vtk.cpp
 * @brief Test Tungsten HIP model with initial/boundary conditions and VTK output
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "This test requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <nlohmann/json.hpp>
#include <tungsten/common/run_tungsten_gpu_vtk.hpp>
#include <tungsten/hip/tungsten_model.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_gpu_vtk_main<TungstenHIP<double>>(
      argc, argv, "tungsten_single_seed_256_hip.json",
      "Tungsten HIP Test with VTK Output",
      [](const nlohmann::json &j) {
        return pfc::ui::hip_spectral_plan_options_from_json(j);
      },
      [](TungstenHIP<double> &m) -> decltype(auto) { return m.get_hip_fft(); });
}
