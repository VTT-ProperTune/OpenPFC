// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_hip_vtk.cpp
 * @brief Drive TungstenHIPSession with VTK output (catalog `vtk`, `.vti` paths).
 */

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <tungsten/tungsten_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<tungsten::TungstenHIPSession>(
      argc, argv, "tungsten_hip_vtk", tungsten::register_catalog);
}
