// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_etd.cpp
 * @brief Alias of production `tungsten`: JSON → TungstenETDSession.
 */

#include <tungsten/common/tungsten_app_main.hpp>
#include <tungsten/tungsten_etd_session.hpp>

int main(int argc, char *argv[]) {
  return tungsten::run_tungsten_etd_main<tungsten::TungstenETDSession>(
      argc, argv, "tungsten_etd");
}
