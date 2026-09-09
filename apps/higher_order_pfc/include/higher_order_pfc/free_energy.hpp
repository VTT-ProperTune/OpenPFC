// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file free_energy.hpp
 * @brief Free-energy density and reciprocal-space peak sampling for the
 *        two-mode PFC crystal-selection benchmark (`#118`).
 *
 * @details
 * Evaluates the same free energy the kernel comment in
 * `higher_order_pfc_physics.hpp` writes down,
 *
 * \f[
 *   f = \frac1V\int\Bigl[\tfrac12\psi\,\Lambda(\nabla^2)\,\psi
 *                        -\tfrac{g}{3}\psi^3+\tfrac14\psi^4\Bigr]\,d\mathbf r ,
 * \f]
 *
 * as a mean over grid cells (uniform `dx` makes the volume integral and the
 * cell average the same number). The quadratic term is evaluated the way
 * `cahn_hilliard::Diagnostics` evaluates its gradient energy: apply the
 * kernel to the transform, invert, and dot with \f$\psi\f$ in real space,
 * rather than trusting a hand-rolled Parseval normalisation. The same
 * transform feeds `pfc::apps::shell_average` for the reciprocal-space report,
 * so this costs one extra FFT pair per sample, not two.
 *
 * Single-rank use only (see `higher_order_pfc_benchmark.cpp`): the benchmark
 * driver needs the whole grid on one rank anyway for the real-space order
 * metric in `order_parameter.hpp`, so this sampler does not attempt a
 * decomposition-aware gather.
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/fft/kspace_iterator.hpp>
#include <openpfc/kernel/simulation/spectral_etd_ops.hpp>

#include <higher_order_pfc/higher_order_pfc_physics.hpp>
#include <openpfc_apps/structure_factor.hpp>

namespace higher_order_pfc {

/// One diagnostic sample: mass, free-energy density and the reciprocal-space
/// report needed to classify which correlation band(s) carry the pattern.
struct FreeEnergySample {
  double mean_psi{0.0};
  double free_energy_density{0.0};
  pfc::apps::StructureFactor sf{};
  double S_at_1{0.0};   ///< shell power in the bin nearest \f$|k|=1\f$
  double S_at_q1{0.0};  ///< shell power in the bin nearest \f$|k|=q_1\f$
};

/// Nearest-bin lookup: `sf.k`/`sf.S` are shell centres, not a dense grid, so
/// picking a value at a target wavenumber means finding the closest shell.
///
/// A target that sits exactly on a bin boundary (as `k=1` does for the
/// shipped grid's default `n_bins=64`, where the bin width divides 1 evenly)
/// is equidistant from two shells. Break that tie towards the shell that
/// actually carries power, not the lower-index one: for a sharp, nearly
/// single-mode field (a `lattice_seed` run, say) the "wrong" neighbour of an
/// exact-boundary target can be essentially empty, which would silently
/// report zero at a wavenumber that is, in fact, the dominant peak.
[[nodiscard]] inline double power_near(const pfc::apps::StructureFactor &sf,
                                       double k_target) {
  if (sf.k.empty()) return 0.0;
  std::size_t best = 0;
  double best_d = std::abs(sf.k[0] - k_target);
  for (std::size_t i = 1; i < sf.k.size(); ++i) {
    const double d = std::abs(sf.k[i] - k_target);
    if (d < best_d || (d == best_d && sf.S[i] > sf.S[best])) {
      best_d = d;
      best = i;
    }
  }
  return sf.S[best];
}

template <class MemorySpace = pfc::HostSpace> class FreeEnergySampler {
  using Ops = pfc::sim::SpectralETDOps<MemorySpace>;

public:
  FreeEnergySampler(const pfc::Domain &domain, typename Ops::FFT &fft, MPI_Comm comm)
      : m_domain(domain), m_fft(fft), m_comm(comm),
        m_hat(domain, fft.get_outbox_bounds(), 0),
        m_quad(domain, fft.get_inbox_bounds(), 0) {}

  FreeEnergySample sample(typename Ops::RealField &psi, const HigherOrderPFCParams &p,
                          int sf_bins = 64) {
    FreeEnergySample out;
    const auto n = pfc::domain::get_size(m_domain);
    const double count = double(n[0]) * n[1] * n[2];

    double local_mean = 0.0, local_local_terms = 0.0;
    psi.with_host_view([&](double *v, std::size_t m) {
      for (std::size_t i = 0; i < m; ++i) {
        local_mean += v[i];
        const double v2 = v[i] * v[i];
        local_local_terms += -(p.g / 3.0) * v2 * v[i] + 0.25 * v2 * v2;
      }
    });

    // Lambda(u) applied on the transform, inverted, dotted with psi in real
    // space: the quadratic term of F, computed the same way the ETD system
    // computes any other spectral multiplier, not via a separate Parseval sum.
    Ops::forward(m_fft, psi, m_hat);
    std::vector<double> weights(m_fft.size_outbox(), 0.0);
    pfc::fft::kspace::for_each_kpoint(
        m_fft.get_outbox_bounds(), m_domain,
        [&](std::size_t i, double kx, double ky, double kz, int, int, int) {
          weights[i] = p.kernel(-(kx * kx + ky * ky + kz * kz));
        });
    typename Ops::real_coeffs w;
    Ops::upload(w, weights);
    typename Ops::complex_scratch work = Ops::make_complex(m_fft.size_outbox());
    Ops::multiply(m_hat, w, work);
    Ops::backward(m_fft, work, m_quad);

    double local_quad = 0.0;
    psi.with_host_view([&](double *v, std::size_t m) {
      m_quad.with_host_view(
          [&](double *lv, std::size_t) {
            for (std::size_t i = 0; i < m; ++i) local_quad += 0.5 * v[i] * lv[i];
          });
    });

    m_hat.with_host_view([&](std::complex<double> *hat, std::size_t) {
      out.sf = pfc::apps::shell_average(m_fft.get_outbox_bounds(), m_domain, hat,
                                        m_comm, sf_bins);
    });

    double global[2]{};
    double local[2] = {local_mean, local_local_terms + local_quad};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, m_comm);
    out.mean_psi = global[0] / count;
    out.free_energy_density = global[1] / count;
    out.S_at_1 = power_near(out.sf, 1.0);
    out.S_at_q1 = power_near(out.sf, p.q1);
    return out;
  }

private:
  pfc::Domain m_domain;
  typename Ops::FFT &m_fft;
  MPI_Comm m_comm;
  typename Ops::ComplexField m_hat;
  typename Ops::RealField m_quad;
};

} // namespace higher_order_pfc
