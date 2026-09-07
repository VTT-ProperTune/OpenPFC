// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file surface_diffusion.cpp  JSON → Mullins surface diffusion on HIP. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "surface_diffusion_hip requires HIP spectral support (rocFFT HeFFTe)"
#endif

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <surface_diffusion/surface_diffusion_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<
      surface_diffusion::SurfaceDiffusionHIPSession>(
      argc, argv, "surface_diffusion_hip", surface_diffusion::register_catalog);
}
