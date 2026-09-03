// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file aluminum_cuda.cpp  JSON → aluminum spectral-ETD run on the CUDA stack. */

#if !defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#error "aluminum_etd_cuda requires CUDA spectral support"
#endif

#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<aluminum::AluminumCUDASession>(
      argc, argv, "aluminum_etd_cuda", aluminum::register_catalog);
}
