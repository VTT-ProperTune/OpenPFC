// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_cuda_vtk.cpp
 * @brief Drive TungstenCUDASession with VTK output (catalog `vtk`, `.vti` paths).
 */

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <tungsten/tungsten_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<tungsten::TungstenCUDASession>(
      argc, argv, "tungsten_cuda_vtk", tungsten::register_catalog);
}
