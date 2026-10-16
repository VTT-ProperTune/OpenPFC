// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file projector.hpp
 * @brief Compatible projection of de Geus et al., Eq. (19).
 *
 * On every non-zero Fourier mode the projected tensor is
 * `(G : B)_ij = (B q)_i q_j / (q · q)`, and the zero mode is set to zero.
 * The zero mode is left for the prescribed macroscopic deformation, which
 * the Newton solver adds in real space.
 *
 * `q` uses `kspace::k_component_odd`. On an odd grid that is `k_component`,
 * so the ratio matches the integer-frequency projector in the published
 * GooseFFT program. On an even grid each Nyquist component is zero. A real
 * Nyquist coefficient has no unique imaginary derivative, and `q_i q_j`
 * built from the one-sided `+k_Nyquist` is not Hermitian. A mode that this
 * rule reduces to `q = 0`, including the pure Nyquist corner, is removed
 * with the zero mode. The device projector still rejects even grids.
 */

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/fft/fft_interface.hpp>
#include <openpfc/kernel/fft/kspace.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/mechanics/tensor.hpp>

namespace pfc::finite_strain {

[[nodiscard]] inline bool odd_grid(const Domain &domain) noexcept {
  const auto size = domain::get_size(domain);
  return size[0] % 2 == 1 && size[1] % 2 == 1 && size[2] % 2 == 1;
}

inline void require_odd_grid(const Domain &domain) {
  if (!odd_grid(domain)) {
    const auto size = domain::get_size(domain);
    throw std::invalid_argument(
        "finite-strain displacement reconstruction is implemented for odd grids "
        "only, got " +
        std::to_string(size[0]) + "x" + std::to_string(size[1]) + "x" +
        std::to_string(size[2]));
  }
}

/**
 * Real-to-complex compatible projection on the local FFT inbox.
 *
 * `in` and `out` are stored in the inbox's x-fastest order and may alias.
 * Fourier workspace is kept on the projector and reused. `apply` is const
 * and is not safe to call concurrently on the same projector.
 */
class CompatibleProjector {
public:
  CompatibleProjector(fft::IHostFFT &fft, Domain domain)
      : m_fft(fft), m_domain(domain) {
    const auto inbox = m_fft.get_inbox_bounds();
    if (inbox.count() != static_cast<long long>(m_fft.size_inbox())) {
      throw std::invalid_argument(
          "CompatibleProjector: inbox bounds do not match size_inbox");
    }
  }

  [[nodiscard]] std::size_t local_size() const noexcept {
    return m_fft.size_inbox();
  }

  [[nodiscard]] const Domain &domain() const noexcept { return m_domain; }

  [[nodiscard]] fft::IHostFFT &fft() const noexcept { return m_fft; }

  void apply(const std::vector<Tensor2> &in, std::vector<Tensor2> &out) const {
    if (in.size() != local_size()) {
      throw std::invalid_argument(
          "CompatibleProjector: field size does not match the FFT inbox");
    }
    const std::size_t n_real = m_fft.size_inbox();
    const std::size_t n_freq = m_fft.size_outbox();
    if (m_real.size() != n_real) {
      m_real.resize(n_real);
    }
    for (auto &component : m_hat) {
      if (component.size() != n_freq) {
        component.resize(n_freq);
      }
    }
    std::vector<double> &real = m_real;
    std::array<std::vector<std::complex<double>>, 9> &hat = m_hat;

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (std::size_t n = 0; n < n_real; ++n) {
          real[n] = in[n](a, b);
        }
        m_fft.forward(real, hat[static_cast<std::size_t>(a * 3 + b)]);
      }
    }

    const auto outbox = m_fft.get_outbox_bounds();
    const auto size = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    const auto scale = fft::kspace::k_frequency_scaling(m_domain);
    fft::kspace::for_each_kpoint(
        outbox, size, spacing,
        [&](std::size_t idx, double, double, double, int i, int j, int k) {
          const double qx = fft::kspace::k_component_odd(i, size[0], scale[0]);
          const double qy = fft::kspace::k_component_odd(j, size[1], scale[1]);
          const double qz = fft::kspace::k_component_odd(k, size[2], scale[2]);
          const double q2 = qx * qx + qy * qy + qz * qz;
          const double q[3] = {qx, qy, qz};
          std::complex<double> projected[3][3];
          if (q2 == 0.0) {
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                projected[a][b] = 0.0;
              }
            }
          } else {
            for (int a = 0; a < 3; ++a) {
              std::complex<double> contracted = 0.0;
              for (int c = 0; c < 3; ++c) {
                contracted += hat[static_cast<std::size_t>(a * 3 + c)][idx] * q[c];
              }
              for (int b = 0; b < 3; ++b) {
                projected[a][b] = contracted * q[b] / q2;
              }
            }
          }
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              hat[static_cast<std::size_t>(a * 3 + b)][idx] = projected[a][b];
            }
          }
        });

    if (out.size() != n_real) {
      out.assign(n_real, Tensor2{});
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        m_fft.backward(hat[static_cast<std::size_t>(a * 3 + b)], real);
        for (std::size_t n = 0; n < n_real; ++n) {
          out[n](a, b) = real[n];
        }
      }
    }
  }

private:
  fft::IHostFFT &m_fft;
  Domain m_domain;
  mutable std::vector<double> m_real;
  mutable std::array<std::vector<std::complex<double>>, 9> m_hat;
};

} // namespace pfc::finite_strain
