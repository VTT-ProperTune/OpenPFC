// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_heat_propagator_hip.hpp
 * @brief HIP implicit-Euler heat propagator (2 device FFTs/step).
 *
 * @details Device twin of `SpectralHeatPropagator`. The multiplier table is
 *          filled on the host by `fill_implicit_euler_symbol` and uploaded
 *          once. Each `step` is a device forward FFT, a complex×real multiply,
 *          and a device inverse FFT — the same 2-FFT implicit Euler as the
 *          CPU spectral Heat3D driver, not the 4-FFT point-wise path.
 */

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <vector>

#include <heat3d/spectral_heat_propagator.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace heat3d {

class SpectralHeatPropagatorHIP {
public:
  using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;
  using RealField = pfc::data::Field<double, pfc::HIPSpace>;

  /**
   * @param fft Device FFT plan to reuse (borrowed; must outlive the propagator).
   * @param u   Field whose global grid + spacing define the symbol table.
   * @param D   Diffusion coefficient.
   * @param dt  Time-step size.
   */
  SpectralHeatPropagatorHIP(FFT &fft, const RealField &u, double D, double dt)
      : m_fft(fft), m_psi_F(fft.size_outbox()), m_opL(fft.size_outbox()) {
    std::vector<double> host_opL(fft.size_outbox());
    fill_implicit_euler_symbol(host_opL, fft.get_outbox_bounds(), u.global_size(),
                               u.spacing(), D, dt);
    m_opL.copy_from_host(host_opL);
  }

  /** Advance `u` by one implicit-Euler step (1 fwd FFT + 1 inv FFT). */
  void step(RealField &u) {
    u.sync_to_device();
    m_fft.forward(u.buffer(), m_psi_F);
    pfc::multiply_complex_real_hip_impl(m_psi_F.data(), m_opL.data(), m_psi_F.data(),
                                        m_psi_F.size());
    m_fft.backward(m_psi_F, u.buffer());
    u.note_device_write();
  }

private:
  FFT &m_fft;
  FFT::ComplexBuffer m_psi_F;
  FFT::RealBuffer m_opL;
};

} // namespace heat3d

#endif // OpenPFC_ENABLE_HIP_SPECTRAL
