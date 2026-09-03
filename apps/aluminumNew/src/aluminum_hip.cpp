// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file aluminum_hip.cpp  JSON → aluminum spectral-ETD run on the HIP stack. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "aluminum_etd_hip requires HIP spectral support"
#endif

#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<aluminum::AluminumHIPSession>(
      argc, argv, "aluminum_etd_hip", aluminum::register_catalog);
}
