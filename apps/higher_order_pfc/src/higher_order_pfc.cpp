// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file higher_order_pfc.cpp  JSON → higher-order PFC spectral-ETD on CPU. */

#include <higher_order_pfc/higher_order_pfc_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<higher_order_pfc::HigherOrderPFCSession>(
      argc, argv, "higher_order_pfc", higher_order_pfc::register_catalog);
}
