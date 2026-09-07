// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_pointwise.inc
 * @brief Device instantiation of thin-film pointwise Π remainder.
 */

#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>
#include <thin_film/thin_film_pointwise.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(thin_film::ThinFilmPointwise)
