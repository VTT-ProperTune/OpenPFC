<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Nonlinear Fourier teaching model

The [web course](https://vtt-propertune.github.io/OpenPFC/spectral/) explains
Fourier normalization, nonlinear aliasing, cubic dealiasing, ETD2, verification
and distributed FFT ownership. This directory owns its runnable model and
independent verifier.

`allen_cahn.cpp` solves `u_t = D u_xx + u - u^3` on `[0, 2π)` with initial
field `mean + a1*cos(x) + a3*cos(3*x)`. It embeds that one-dimensional problem
in an `n × 4 × 4` periodic grid using OpenPFC's HeFFTe/FFTW CPU stack. Only
`(kx,0,0)` with `|kx| <= cutoff` remain live. Requiring `4*cutoff < n` makes
the cubic product alias-free in the retained band. It does not use the
quadratic two-thirds mask. The reaction does not conserve the mean.

Build from the repository root through the supported script:

```sh
./scripts/build.sh --cpu --build-dir=builds/release
```

From this directory, with a compatible installed spectral CPU SDK, the
source-model workflow is:

```sh
openpfc doctor --sdk /path/to/sdk
openpfc run allen_cahn.cpp input.json --sdk /path/to/sdk --launcher=none
openpfc run allen_cahn.cpp input.json --sdk /path/to/sdk --ranks=2 --launcher=mpirun
```

Invoke the CLI once outside an MPI launch. Each printed attempt contains
`results/diagnostics.csv`, `summary.json`, and per-rank field, normalized
mode and layout files. Simulation output belongs under `results/`.

Replace the paths below with the two attempts printed by the CLI:

```sh
python3 verify.py input.json /path/to/serial/results /path/to/two-rank/results
```

The Python standard-library checker verifies complete, nonoverlapping inbox
and outbox ownership, FFT reconstruction and Parseval, diagnostics, rank
agreement, and the solution against direct non-circular cubic convolution
with refined classical RK4. Its default absolute field-error budget `2e-5`
suits `input.json`. It reports the RK4 step-doubling change separately from
the ETD2 time error. This is an independent solution at fixed Fourier cutoff,
not an exact PDE solution. Refine both cutoff and time step for PDE accuracy.

The example's CTest entries execute the program and verifier for nonlinear,
uniform-ODE and zero-linear-symbol cases. The two-rank test follows
`OpenPFC_RUN_MPI_SUITES`. No performance or GPU qualification is implied.
