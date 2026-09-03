// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file aluminum.cpp  JSON → aluminum spectral-ETD run on the CPU stack. */

#include <aluminum/aluminum_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<aluminum::AluminumSession>(
      argc, argv, "aluminumNew", aluminum::register_catalog);
}
