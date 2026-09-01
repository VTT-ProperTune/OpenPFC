<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Decomposition and FFT

OpenPFC spectral workflows partition a `Domain` across MPI ranks and transform
local data through an implementation of the FFT contract. Read the
[spectral-stack concept](../concepts/spectral_stack.md) before using the exact
API below.

## `pfc::decomposition::Decomposition`

```{doxygenstruct} pfc::decomposition::Decomposition
:project: OpenPFC
:members:
:no-link:
```

## `pfc::fft::IHostFFT`

```{doxygenstruct} pfc::fft::IHostFFT
:project: OpenPFC
:members:
:protected-members:
:no-link:
```

## `pfc::fft::CPUFFT`

```{doxygenclass} pfc::fft::CPUFFT
:project: OpenPFC
:members:
:protected-members:
:no-link:
```
