// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file defaults.hpp
 * @brief Material and output defaults matching the historical Julia `kobayashi_v1` script.
 */

namespace kobayashi {

/** Dimensionless parameters from the Julia reference implementation. */
inline constexpr double kTau = 0.0003;
inline constexpr double kEpsilonb = 0.01;
inline constexpr double kKappa = 1.8;
inline constexpr double kDelta = 0.02;
inline constexpr double kAniso = 6.0;
inline constexpr double kAlpha = 0.9;
inline constexpr double kGamma = 10.0;
inline constexpr double kTeq = 1.0;
inline constexpr double kTheta0 = 0.2;
/** Nucleus threshold: global indices satisfy \f$(gi - N_x/2)^2 + (gj - N_y/2)^2 <
 * \texttt{kSeed}\f$ (Julia used integer division for the center). */
inline constexpr double kSeed = 5.0;

inline constexpr int kNprint = 200;
inline constexpr int kNsave = 2000;

} // namespace kobayashi
