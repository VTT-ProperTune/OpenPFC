#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CGM Δ(V) for the Pinomaa Al-Cu FTA model (JCG 2020 Eq. 19)."""

from __future__ import annotations

import math

KE = 0.17
MLE = 5.3  # |m_l^e| K/at%
CLO = 4.5
BETA0 = 0.1  # s/m
EPS_K = 0.12
VD_PF = 2.0  # m/s
G = 5.0e6  # K/m


def k_pf(V: float, vd: float = VD_PF, ke: float = KE) -> float:
    """k^PF(V) = ke exp(√2 (1-k) V/V_D^PF), solved by fixed point."""
    k = ke
    for _ in range(80):
        kn = ke * math.exp(math.sqrt(2.0) * (1.0 - k) * V / vd)
        kn = min(0.999999, max(ke, kn))
        if abs(kn - k) < 1e-14:
            return kn
        k = 0.5 * (k + kn)
    return k


def f_of_k(k: float, drag: float, ke: float = KE) -> float:
    """Acta 2019 Eq. (3). drag=1 full solute drag, 0 zero drag."""
    return (1.0 / (1.0 - ke)) * (
        (k + drag * (1.0 - k)) * math.log(k / ke) + 1.0 - k
    )


def delta_eq19(V: float, drag: float = 0.38, eps_k: float = EPS_K) -> float:
    """Dimensionless undercooling Δ = (T_l-T)/((1-k_e)|m_l^e| c_0)."""
    k = k_pf(V)
    f = f_of_k(k, drag)
    return (1.0 / (1.0 - KE)) * ((1.0 - KE) * (1.0 - eps_k) * BETA0 * V + f / k - 1.0)


def delta_from_xtip(x_tip: float, t: float, x_tl: float, Vp: float, G: float = G) -> float:
    Tl_minus_T = G * (x_tl + Vp * t - x_tip)
    return Tl_minus_T / ((1.0 - KE) * MLE * CLO)
