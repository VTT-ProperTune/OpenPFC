// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file tungsten.cpp  JSON → tungsten spectral-ETD run on the HIP stack. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "tungsten_hip requires HIP spectral support"
#endif

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <tungsten/tungsten_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<tungsten::TungstenHIPSession>(
      argc, argv, "tungsten_hip", tungsten::register_catalog);
}
