// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file kawahara.cpp  JSON → Kawahara capillary–gravity waves on CPU. */

#include <kawahara/kawahara_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<kawahara::KawaharaSession>(
      argc, argv, "kawahara", kawahara::register_catalog);
}
