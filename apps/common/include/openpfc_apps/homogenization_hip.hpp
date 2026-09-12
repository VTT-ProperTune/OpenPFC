// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#error "homogenization_hip.hpp requires OpenPFC_ENABLE_HIP_SPECTRAL"
#endif

#include <algorithm>
#include <array>

#include <openpfc_apps/homogenization.hpp>
#include <openpfc_apps/microelasticity_hip.hpp>

namespace pfc::apps {

class PeriodicHomogenizerHIP {
public:
  using RealField = pfc::data::Field<double>;
  using FFT = pfc::fft::IDeviceFFT<pfc::HIPSpace>;
  using SymRealFields = EigenstrainMicroelasticity::SymRealFields;

  PeriodicHomogenizerHIP(const pfc::Domain &domain, FFT &fft,
                         MicroelasticityParams params)
      : m_solver(domain, fft, params),
        m_amp(pfc::data::field_from_inbox<double>(domain, fft.get_inbox_bounds())),
        m_n_local(m_amp.size()) {
    std::fill(m_amp.vec().begin(), m_amp.vec().end(), 0.0);
    m_amp.note_host_write();
    const auto box = fft.get_inbox_bounds();
    for (int a = 0; a < kVoigtDim; ++a)
      m_strain[static_cast<std::size_t>(a)] = make_sym(domain, box);
    const auto gs = m_amp.global_size();
    m_n_global = static_cast<double>(gs[0]) * gs[1] * gs[2];
  }

  HomogenizationResult compute(const RealField &h) {
    HomogenizationResult out;
    double local = 0.0;
    for (std::size_t i = 0; i < m_n_local; ++i) local += h.data()[i];
    double glo = 0.0;
    MPI_Allreduce(&local, &glo, 1, MPI_DOUBLE, MPI_SUM, comm());
    out.volume_fraction = glo / m_n_global;
    m_solver.params().warm_start = false;
    for (int a = 0; a < kVoigtDim; ++a) {
      m_solver.params().applied_strain = engineering_unit_strain(a);
      m_solver.reset();
      out.reports[static_cast<std::size_t>(a)] = m_solver.solve(h, m_amp);
      copy_sym(m_solver.strain(), m_strain[static_cast<std::size_t>(a)]);
      out.mean_stress[static_cast<std::size_t>(a)] = mean_stress(h);
      for (int i = 0; i < kVoigtDim; ++i)
        out.stiffness(i, a) = out.mean_stress[static_cast<std::size_t>(a)][i];
    }
    out.stiffness = out.stiffness.symmetrized();
    m_last = out;
    m_has = true;
    return out;
  }

  [[nodiscard]] const HomogenizationResult &last() const { return m_last; }
  [[nodiscard]] const SymRealFields &strain(int a) const {
    return m_strain[static_cast<std::size_t>(a)];
  }

  void objective_sensitivity(const RealField &h, const Voigt6 &Cstar, const Voigt6 &W,
                             RealField &dJdh) const {
    const Stiffness dC =
        Stiffness::blend(m_solver.params().c_solid, 1.0, m_solver.params().c_liquid,
                         -1.0);
    const Voigt6 &C = m_last.stiffness;
    Voigt6 dJdC;
    for (int i = 0; i < kVoigtDim; ++i)
      for (int j = 0; j < kVoigtDim; ++j)
        dJdC(i, j) = W(i, j) * W(i, j) * (C(i, j) - Cstar(i, j));
    double *g = dJdh.data();
    for (std::size_t i = 0; i < m_n_local; ++i) {
      double acc = 0.0;
      for (int a = 0; a < kVoigtDim; ++a) {
        Sym3 ea;
        for (int c = 0; c < kVoigtDim; ++c)
          ea[c] = m_strain[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)]
                      .data()[i];
        const Sym3 dCea = dC.contract(ea);
        for (int b = 0; b < kVoigtDim; ++b) {
          Sym3 eb;
          for (int c = 0; c < kVoigtDim; ++c)
            eb[c] = m_strain[static_cast<std::size_t>(b)][static_cast<std::size_t>(c)]
                        .data()[i];
          acc += dJdC(a, b) * ddot(dCea, eb) / m_n_global;
        }
      }
      g[i] = acc;
    }
    dJdh.note_host_write();
  }

private:
  static SymRealFields make_sym(const pfc::Domain &domain, const pfc::Box3i &box) {
    return SymRealFields{pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box),
                         pfc::data::field_from_inbox<double>(domain, box)};
  }
  void copy_sym(const SymRealFields &src, SymRealFields &dst) {
    for (int c = 0; c < kVoigtDim; ++c) {
      std::copy(src[static_cast<std::size_t>(c)].vec().begin(),
                src[static_cast<std::size_t>(c)].vec().end(),
                dst[static_cast<std::size_t>(c)].vec().begin());
      dst[static_cast<std::size_t>(c)].note_host_write();
    }
  }
  Sym3 mean_stress(const RealField &h) const {
    const auto &eps = m_solver.strain();
    Sym3 s{};
    for (std::size_t i = 0; i < m_n_local; ++i) {
      Sym3 e;
      for (int c = 0; c < kVoigtDim; ++c)
        e[c] = eps[static_cast<std::size_t>(c)].data()[i];
      const Stiffness C = Stiffness::blend(m_solver.params().c_solid, h.data()[i],
                                           m_solver.params().c_liquid,
                                           1.0 - h.data()[i]);
      const Sym3 si = C.contract(e);
      for (int c = 0; c < kVoigtDim; ++c) s[c] += si[c];
    }
    double loc[6], glo[6];
    for (int c = 0; c < 6; ++c) loc[c] = s[c];
    MPI_Allreduce(loc, glo, 6, MPI_DOUBLE, MPI_SUM, comm());
    for (int c = 0; c < 6; ++c) s[c] = glo[c] / m_n_global;
    return s;
  }
  MPI_Comm comm() const noexcept { return m_solver.params().comm; }

  DeviceEigenstrainMicroelasticity m_solver;
  RealField m_amp;
  std::array<SymRealFields, 6> m_strain{};
  std::size_t m_n_local{0};
  double m_n_global{1.0};
  bool m_has{false};
  HomogenizationResult m_last{};
};

} // namespace pfc::apps
