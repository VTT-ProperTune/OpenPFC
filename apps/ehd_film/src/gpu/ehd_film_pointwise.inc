// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file ehd_film_pointwise.inc
 * @brief Device instantiation of the EHD-film \(\Pi\) remainder.
 */

#include <ehd_film/ehd_film_pointwise.hpp>
#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(ehd_film::EhdFilmPointwise)
