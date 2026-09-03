// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file tungsten.cpp  JSON → tungsten spectral-ETD run on the CPU stack. */

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <tungsten/tungsten_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<tungsten::TungstenSession>(
      argc, argv, "tungsten", tungsten::register_catalog);
}
