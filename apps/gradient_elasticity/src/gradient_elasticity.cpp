// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file gradient_elasticity.cpp  JSON → Helmholtz–Navier gradient elasticity on
 * CPU. */

#include <gradient_elasticity/gradient_elasticity_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<
      gradient_elasticity::GradientElasticityCPUSession>(
      argc, argv, "gradient_elasticity", gradient_elasticity::register_catalog);
}
