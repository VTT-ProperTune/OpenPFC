// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file surface_diffusion.cpp  JSON → Mullins surface diffusion on CPU. */

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <surface_diffusion/surface_diffusion_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<surface_diffusion::SurfaceDiffusionSession>(
      argc, argv, "surface_diffusion", surface_diffusion::register_catalog);
}
