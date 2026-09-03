// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file tungsten.cpp  JSON → tungsten spectral-ETD run on the CUDA stack. */

#if !defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
#error "tungsten_cuda requires CUDA spectral support"
#endif

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <tungsten/tungsten_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<tungsten::TungstenCUDASession>(
      argc, argv, "tungsten_cuda", tungsten::register_catalog);
}
