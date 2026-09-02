// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gpu_spectral_stack.hpp
 * @brief Device counterpart of `pfc::sim::stacks::SpectralCPUStack`.
 *
 * @details
 * Lives in runtime because `create_cuda` / `create_hip` and `IDeviceFFT`
 * are runtime (kernel must not include runtime). Bundles
 *
 *     Domain → Decomposition → device FFT → Field<double, MemorySpace>
 *
 * Non-copyable / non-movable: sub-objects would dangle after a move.
 * Optional `heffte::plan_options` overlay matches `SpectralCPUStack`. JSON
 * overlay stays in `spectral_fft_stack_factory.hpp`
 * (`cuda_spectral_plan_options_from_json` / `hip_spectral_plan_options_from_json`).
 */

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#include <openpfc/runtime/gpu/fft_gpu.hpp>

namespace pfc::sim::stacks {

template <class MemorySpace> struct gpu_fft_for;

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
template <> struct gpu_fft_for<CUDASpace> {
  using type = fft::FFT_CUDA;
  static heffte::plan_options default_plan_options() {
    return heffte::default_options<heffte::backend::cufft>();
  }
  static type create(const pfc::decomposition::Decomposition &decomp, int rank,
                     MPI_Comm comm, const heffte::plan_options &options) {
    return pfc::fft::create_cuda(decomp, rank, comm, 0, options);
  }
};
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <> struct gpu_fft_for<HIPSpace> {
  using type = fft::FFT_HIP;
  static heffte::plan_options default_plan_options() {
    return heffte::default_options<heffte::backend::rocfft>();
  }
  static type create(const pfc::decomposition::Decomposition &decomp, int rank,
                     MPI_Comm comm, const heffte::plan_options &options) {
    return pfc::fft::create_hip(decomp, rank, comm, 0, options);
  }
};
#endif

/**
 * @brief Programmatic spectral GPU stack: Domain + Decomposition + device FFT
 *        + `Field<double, MemorySpace>` sized to the FFT inbox.
 */
template <class MemorySpace> class GPUSpectralStack {
public:
  using fft_type = typename gpu_fft_for<MemorySpace>::type;
  using field_type = data::Field<double, MemorySpace>;

  GPUSpectralStack(const GPUSpectralStack &) = delete;
  GPUSpectralStack &operator=(const GPUSpectralStack &) = delete;
  GPUSpectralStack(GPUSpectralStack &&) = delete;
  GPUSpectralStack &operator=(GPUSpectralStack &&) = delete;

  explicit GPUSpectralStack(pfc::Domain domain, int rank, int nproc, MPI_Comm comm,
                            const heffte::plan_options &options)
      : m_domain(std::move(domain)),
        m_decomp(pfc::decomposition::create(m_domain, nproc)),
        m_fft(gpu_fft_for<MemorySpace>::create(m_decomp, rank, comm, options)),
        m_u(m_domain, m_fft.get_inbox_bounds(), 0), m_rank(rank), m_nproc(nproc),
        m_comm(comm) {}

  explicit GPUSpectralStack(pfc::Domain domain, int rank, int nproc,
                            MPI_Comm comm = MPI_COMM_WORLD)
      : GPUSpectralStack(std::move(domain), rank, nproc, comm,
                         gpu_fft_for<MemorySpace>::default_plan_options()) {}

  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] pfc::fft::IDeviceFFT<MemorySpace> &fft() noexcept { return m_fft; }
  [[nodiscard]] const pfc::fft::IDeviceFFT<MemorySpace> &fft() const noexcept {
    return m_fft;
  }

  [[nodiscard]] field_type &u() noexcept { return m_u; }
  [[nodiscard]] const field_type &u() const noexcept { return m_u; }

  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  pfc::Domain m_domain{};
  pfc::decomposition::Decomposition m_decomp;
  fft_type m_fft;
  field_type m_u;
  int m_rank{0};
  int m_nproc{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
using CUDASpectralStack = GPUSpectralStack<CUDASpace>;
#endif
#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
using HIPSpectralStack = GPUSpectralStack<HIPSpace>;
#endif

} // namespace pfc::sim::stacks

#endif // OpenPFC_ENABLE_CUDA_SPECTRAL || OpenPFC_ENABLE_HIP_SPECTRAL
