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

#include <memory>
#include <stdexcept>
#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#include <openpfc/runtime/gpu/bind_local_device.hpp>
#include <openpfc/runtime/gpu/fft_gpu.hpp>
#include <openpfc/runtime/gpu/fft_single_device.hpp>

namespace pfc::sim::stacks {

/**
 * @brief Which device FFT `GPUSpectralStack::fft()` exposes.
 *
 * `heffte` (default) is the distributed heFFTe backend. `single_device`
 * (one rank only) uses `pfc::fft::SingleDeviceFFT`: one vendor 3D plan,
 * the same boxes and scaling, far fewer kernel launches (issue #382).
 */
enum class DeviceFFTChoice { heffte, single_device };

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
      : GPUSpectralStack(std::move(domain), rank, nproc, comm,
                         DeviceFFTChoice::heffte, options) {}

  explicit GPUSpectralStack(pfc::Domain domain, int rank, int nproc,
                            MPI_Comm comm = MPI_COMM_WORLD)
      : GPUSpectralStack(std::move(domain), rank, nproc, comm,
                         gpu_fft_for<MemorySpace>::default_plan_options()) {}

  /**
   * @brief Construct only the selected FFT and size the field from its inbox.
   * @throws std::invalid_argument for `single_device` unless `comm` has one
   *         rank and the supplied rank/size agree with that communicator.
   */
  GPUSpectralStack(pfc::Domain domain, int rank, int nproc, MPI_Comm comm,
                   DeviceFFTChoice choice,
                   const heffte::plan_options &options =
                       gpu_fft_for<MemorySpace>::default_plan_options())
      : m_domain(std::move(domain)),
        m_decomp(make_decomposition(m_domain, rank, nproc, comm, choice)),
        m_fft(make_fft(m_domain, m_decomp, rank, comm, choice, options)),
        m_u(m_domain, m_fft->get_inbox_bounds(), 0), m_choice(choice), m_rank(rank),
        m_nproc(nproc), m_comm(comm) {}

  /// The selected implementation; the other implementation is not constructed.
  [[nodiscard]] DeviceFFTChoice fft_choice() const noexcept { return m_choice; }

  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] pfc::fft::IDeviceFFT<MemorySpace> &fft() noexcept { return *m_fft; }
  [[nodiscard]] const pfc::fft::IDeviceFFT<MemorySpace> &fft() const noexcept {
    return *m_fft;
  }

  [[nodiscard]] field_type &u() noexcept { return m_u; }
  [[nodiscard]] const field_type &u() const noexcept { return m_u; }

  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  static pfc::decomposition::Decomposition
  make_decomposition(const pfc::Domain &domain, int rank, int nproc, MPI_Comm comm,
                     DeviceFFTChoice choice) {
    if (choice == DeviceFFTChoice::single_device) {
      if (comm == MPI_COMM_NULL)
        throw std::invalid_argument(
            "GPUSpectralStack: local FFT needs a valid communicator");
      int comm_size = 0, comm_rank = -1;
      MPI_Comm_size(comm, &comm_size);
      MPI_Comm_rank(comm, &comm_rank);
      if (nproc != 1 || rank != 0 || comm_size != 1 || comm_rank != 0) {
        throw std::invalid_argument(
            "GPUSpectralStack: local FFT needs exactly one rank and matching "
            "communicator rank/size");
      }
    } else if (choice != DeviceFFTChoice::heffte) {
      throw std::invalid_argument("GPUSpectralStack: invalid FFT choice");
    }
    // Validate before device binding or any vendor-plan allocation.
    pfc::runtime::gpu::bind_local_device(comm);
    return pfc::decomposition::create(
        domain, pfc::decomposition::spectral_fft_proc_grid(domain.size, nproc));
  }

  static std::unique_ptr<pfc::fft::IDeviceFFT<MemorySpace>>
  make_fft(const pfc::Domain &domain,
           const pfc::decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
           DeviceFFTChoice choice, const heffte::plan_options &options) {
    if (choice == DeviceFFTChoice::single_device) {
      if (!(pfc::decomposition::local_box(decomp, rank) ==
            pfc::domain::index_box(domain))) {
        throw std::invalid_argument(
            "GPUSpectralStack: local FFT needs the whole domain");
      }
      return std::make_unique<pfc::fft::SingleDeviceFFT<MemorySpace>>(domain.size);
    }
    // Construct the factory prvalue in place: FFT_Impl is not movable.
    return std::unique_ptr<pfc::fft::IDeviceFFT<MemorySpace>>(
        new fft_type(gpu_fft_for<MemorySpace>::create(decomp, rank, comm, options)));
  }

  pfc::Domain m_domain{};
  pfc::decomposition::Decomposition m_decomp;
  std::unique_ptr<pfc::fft::IDeviceFFT<MemorySpace>> m_fft;
  field_type m_u;
  DeviceFFTChoice m_choice;
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
