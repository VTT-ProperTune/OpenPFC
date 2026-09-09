// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file thin_film_pointwise.inc
 * @brief Device instantiations for thin-film pointwise/flux kernels.
 *
 * Three functors, one device translation unit each links into:
 *  - `ThinFilmPointwise`        — the ETD remainder used by `thin_film`
 *    (JSON session, constant mobility);
 *  - `PotentialPointwise`       — the full disjoining pressure \f$\Pi(h)\f$,
 *    used by `thin_film_nonlinear[_hip]`'s potential evaluation;
 *  - `pfc::apps::MobilityGradPointwise<CubicMobility>` — the fused
 *    \f$M(h)\cdot\nabla p\f$ kernel behind `pfc::apps::SpectralFlux`.
 */

#include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>
#include <openpfc_apps/spectral_flux.hpp>
#include <thin_film/nonlinear.hpp>
#include <thin_film/thin_film_pointwise.hpp>

OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(thin_film::ThinFilmPointwise)
OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(thin_film::PotentialPointwise)
OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(
    pfc::apps::MobilityGradPointwise<thin_film::CubicMobility>)
