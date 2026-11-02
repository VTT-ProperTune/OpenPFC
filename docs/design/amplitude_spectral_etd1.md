<!-- SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Complex-amplitude ETD1

`SpectralETDSystem` integrates one real field. This note is a separate
stepper for several complex amplitudes on the same real-to-complex stack.

Each amplitude is a pair of real fields. On the half spectrum the two
combinations `re^ + i im^` and `re^ - i im^` advance with the caller's
symbols `s(q)` and `s(-q)`:

d = exp(s dt) c + phi1(s, dt) N.

`phi1` is `dt (1 + s dt / 2)` when `|s dt| < 1e-8`, and
`(exp(s dt) - 1) / s` otherwise. The caller owns any Nyquist reduction.
The stepper does not replace that choice.

The nonlinearity is a host callback over every amplitude. On CUDA the
transforms and the half-spectrum update use the device FFT. HIP has the
same kernel entry and was not run. This is not an aluminum calibration
and not a grain-boundary law. The one-field spectral system is unchanged.
