// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file cahn_hilliard.cpp  JSON → Fe–Cr Cahn–Hilliard spectral-ETD on HIP. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "cahn_hilliard_hip requires HIP spectral support (rocFFT HeFFTe)"
#endif

#include <cahn_hilliard/cahn_hilliard_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<cahn_hilliard::CahnHilliardHIPSession>(
      argc, argv, "cahn_hilliard_hip", cahn_hilliard::register_catalog);
}
