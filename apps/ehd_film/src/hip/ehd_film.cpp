// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file ehd_film.cpp  JSON → EHD film under a flexible plate on HIP. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "ehd_film_hip requires HIP spectral support (rocFFT HeFFTe)"
#endif

#include <ehd_film/ehd_film_session.hpp>
#include <openpfc/frontend/ui/json_session_main.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<ehd_film::EhdFilmHIPSession>(
      argc, argv, "ehd_film_hip", ehd_film::register_catalog);
}
