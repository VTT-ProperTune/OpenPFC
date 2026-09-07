// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/** @file thin_film.cpp  JSON → lubrication thin-film spectral-ETD on HIP. */

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "thin_film_hip requires HIP spectral support (rocFFT HeFFTe)"
#endif

#include <openpfc/frontend/ui/json_session_main.hpp>
#include <thin_film/thin_film_session.hpp>

int main(int argc, char *argv[]) {
  return pfc::ui::run_json_session_main<thin_film::ThinFilmHIPSession>(
      argc, argv, "thin_film_hip", thin_film::register_catalog);
}
