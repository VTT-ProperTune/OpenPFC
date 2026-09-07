// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file ehd_film.cpp  JSON → EHD film under a flexible plate on CPU. */

#include <ehd_film/ehd_film_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<ehd_film::EhdFilmSession>(
      argc, argv, "ehd_film", ehd_film::register_catalog);
}
