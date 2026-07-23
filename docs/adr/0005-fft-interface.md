<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# ADR 0005: FFT interface — split host and device transforms

## Status

Accepted (2026-07-23).

## Context

The audit (§5/§7, AB‑4) found the polymorphic FFT interface dishonest: a factory
returning `std::unique_ptr<IFFT>` for a GPU backend produced an object whose
*every* virtual method throws, because GPU transforms are non-virtual member
templates over `DataBuffer` while `IFFT` only declares host-`std::vector`
transforms. Polymorphic call sites (e.g. `SpectralGradient` holding `IFFT*`)
compile against a GPU backend and fail at runtime. The `Backend` enum also
omitted HIP (fixed in M3).

An interface must be substitutable by all of its implementations.

## Decision

Split the single `IFFT` into two honest interfaces (M5):

- **`IHostFft`** — forward/backward over host containers (`std::vector` /
  host `FieldView`). Implemented by the CPU (FFTW/HeFFTe) backend.
- **`IDeviceFft<MemorySpace>`** — forward/backward over `DataBuffer<MemorySpace,T>`.
  Implemented by the CUDA (cuFFT) and HIP (rocFFT) backends.

`FFT_Impl<BackendTag>` implements whichever interface(s) apply to its backend.
Factories are **honest**: host factories return `IHostFft` for host backends
only; device factories (`create_cuda`/`create_hip` and the string/enum-driven
equivalents) return `IDeviceFft`. Requesting a host FFT for a device backend (or
vice versa) **throws at construction** with a clear message — never returns an
object that throws on use.

Rejected alternative: a single interface templated on a buffer family. Rejected
because the configuration layer already knows the backend when it constructs the
session, so two concrete interfaces are simpler and avoid leaking a buffer-family
template parameter through every call site.

## Consequences

- `SpectralGradient` and other spectral consumers bind to the interface matching
  their memory space; no runtime "throws on every method" objects exist.
- The `Backend` enum and `backend_from_string` cover CPU/CUDA/HIP honestly (M3).
- GPU spectral models no longer need a dead host `CpuFft` (`dummy_fft`);
  the session constructs the device FFT directly (M8/M10).
- Precision policy (ADR 0006) applies: workspace is allocated only for the
  instantiated precision.
