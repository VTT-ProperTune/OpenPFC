<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# When OpenPFC is (and is not) the right tool

OpenPFC targets **large, parallel, FFT-heavy phase-field crystal (PFC) and related** simulations with a clear C++ / MPI / HeFFTe story. This page is an **honest fit guide**—not a comparison table against every other code.

## Good fit

- You want **distributed 3D grids**, **MPI**, and **HeFFTe-backed FFTs** (CPU or GPU) with a **library + optional JSON `App`** workflow.
- You are building or running **spectral semi-implicit** style models (linear pieces in *k*-space, nonlinear pieces in real space) as in the shipped **tungsten** app and the `examples/` ladder.
- You can invest in **correct MPI + HeFFTe + (optional) CUDA/HIP** alignment ([`INSTALL.md`](../INSTALL.md), [`troubleshooting.md`](troubleshooting.md)).

## Spectral limitations—and what we are building toward

Purely **spectral** discretizations are excellent for many PFC regimes but can be limiting when you need **localized stencil control**, certain **boundary treatments**, or **mixed operator** formulations where FFT-global assumptions are awkward.

**We are actively implementing and extending finite-difference (FD) machinery** alongside the spectral path so those limitations can be addressed in-framework. The design direction is a **unified, abstract treatment of spatial operators** (gradients, Laplacians, and related terms): where the stack supports it, you will be able to choose whether those operators are applied **spectrally** (FFT / HeFFTe) or **via FD stencils**, using the same decomposition and halo infrastructure ([`halo_exchange.md`](halo_exchange.md)).

Today, **shipped spectral `App` pipelines** (for example **tungsten**) remain **FFT-centric** in practice. **FD building blocks** already live in the kernel (e.g. `pfc::field::fd` in [`include/openpfc/kernel/field/finite_difference.hpp`](../include/openpfc/kernel/field/finite_difference.hpp)) and in examples such as finite-difference heat; the **full user-facing switch** for “gradient operator: spectral vs FD” across every app is **still in progress**. For roadmap context, see [`adr/0002-gradient-operators-fd-vs-spectral.md`](adr/0002-gradient-operators-fd-vs-spectral.md) and [`refactoring_roadmap.md`](refactoring_roadmap.md).

## When another tool might be simpler (today)

| Situation | Consider |
|-----------|----------|
| You only need **small 1D/2D toy FFTs** with no MPI | A minimal NumPy / SciPy or FFTW tutorial may be faster to learn. |
| You require a **different community model** (e.g. phase-field **without** this PFC formulation) | Compare domain-specific codes in that community; still read our [`science_tungsten_quicklook.md`](science_tungsten_quicklook.md) for what tungsten actually solves. |
| You cannot use **MPI** or **HeFFTe**-compatible builds | OpenPFC’s value is in the parallel spectral + I/O stack; without that, the install cost may outweigh the benefit. |
| You need **guaranteed production support** or a **turn-key GUI** | OpenPFC is a research-oriented C++ framework; plan engineering effort for your site. |

## See also

- [`spectral_stack.md`](spectral_stack.md) — default spectral data-flow story  
- [`science_numerics_limits.md`](science_numerics_limits.md) — stability and discretization caveats  
- [`architecture.md`](architecture.md) — spectral vs FD coexistence at the layer level  
