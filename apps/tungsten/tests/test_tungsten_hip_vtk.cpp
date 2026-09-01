// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_hip_vtk.cpp
 * @brief Drive TungstenETDHIPSession with VTK output (catalog `vtk`).
 */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "This test requires HIP spectral support"
#endif

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/tungsten_etd_gpu_session.hpp>

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    char default_config[] = "tungsten_single_seed_256_hip.json";
    char *fallback[] = {argv[0], default_config};
    return tungsten::run_tungsten_etd_main<tungsten::TungstenETDHIPSession>(
        2, fallback, "tungsten_hip_vtk");
  }
  return tungsten::run_tungsten_etd_main<tungsten::TungstenETDHIPSession>(
      argc, argv, "tungsten_hip_vtk");
}
