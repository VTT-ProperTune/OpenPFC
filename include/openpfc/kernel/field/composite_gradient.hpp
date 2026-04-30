// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file composite_gradient.hpp
 * @brief Multi-field per-point evaluator built from per-field evaluators.
 *
 * @details
 * `CompositeGradient<Composite, PerField...>` bundles N per-field
 * evaluators (typically `pfc::field::FdGradient<UGrads>`,
 * `pfc::field::SpectralGradient<VGrads>`, ...) into a single
 * `eval(i,j,k) -> Composite` callable suitable for
 * `pfc::sim::for_each_interior`.
 *
 * The `Composite` type is the model-owned aggregate the framework
 * passes to `model.rhs(t, composite)`. Its members are listed in the
 * **same order** as the per-field evaluators in the parameter pack and
 * must be **brace-initializable** from the per-field evaluator return
 * values, e.g.
 *
 * @code
 * struct WaveLocal {
 *   UGrads u;   // populated by the first PerField
 *   VGrads v;   // populated by the second PerField
 * };
 * @endcode
 *
 * `prepare()` fans out to each sub-evaluator (so spectral evaluators
 * still run their FFTs once per step, FD evaluators are a no-op). The
 * interior bounds (`imin/imax/jmin/jmax/kmin/kmax`) are taken from the
 * **first** sub-evaluator: all fields are assumed to share the
 * rank-local layout (true for any setup where every field lives on the
 * same World/Decomposition).
 *
 * @see grad_concepts.hpp for the per-member detection concepts
 * @see fd_gradient.hpp for an FD per-field evaluator
 * @see spectral_gradient.hpp for a spectral per-field evaluator
 * @see openpfc/kernel/simulation/for_each_interior.hpp for the driver loop
 */

#include <cstddef>
#include <tuple>
#include <utility>

namespace pfc::field {

/**
 * @brief Variadic composite per-point evaluator.
 *
 * @tparam Composite Model-owned aggregate listing one member per field, in
 *                   the same order as `PerField...`.
 * @tparam PerField  Per-field evaluators (must each provide
 *                   `prepare()`, `imin/imax/jmin/jmax/kmin/kmax`,
 *                   `idx(i,j,k)`, and `operator()(i,j,k)`).
 */
template <class Composite, class... PerField> class CompositeGradient {
public:
  static_assert(sizeof...(PerField) >= 1,
                "CompositeGradient: at least one per-field evaluator required.");

  explicit CompositeGradient(PerField... evals) : m_evals(std::move(evals)...) {}

  /** Forward `prepare()` to every sub-evaluator. */
  void prepare() {
    std::apply([](auto &...e) { (e.prepare(), ...); }, m_evals);
  }

  /** Build the composite by calling each sub-evaluator at `(ix,iy,iz)`. */
  inline Composite operator()(int ix, int iy, int iz) const noexcept {
    return std::apply([&](const auto &...e) { return Composite{e(ix, iy, iz)...}; },
                      m_evals);
  }

  /**
   * @brief Linear index from the first sub-evaluator (shared layout
   *        assumption).
   */
  inline std::size_t idx(int ix, int iy, int iz) const noexcept {
    return std::get<0>(m_evals).idx(ix, iy, iz);
  }

  int imin() const noexcept { return std::get<0>(m_evals).imin(); }
  int imax() const noexcept { return std::get<0>(m_evals).imax(); }
  int jmin() const noexcept { return std::get<0>(m_evals).jmin(); }
  int jmax() const noexcept { return std::get<0>(m_evals).jmax(); }
  int kmin() const noexcept { return std::get<0>(m_evals).kmin(); }
  int kmax() const noexcept { return std::get<0>(m_evals).kmax(); }

private:
  std::tuple<PerField...> m_evals;
};

/**
 * @brief Free-function factory: deduces `PerField...` from arguments.
 *
 * Caller supplies `Composite` explicitly so the framework knows which
 * model-owned aggregate to construct.
 *
 * @code
 * auto grad = pfc::field::create_composite<WaveLocal>(grad_u, grad_v);
 * @endcode
 */
template <class Composite, class... PerField>
inline CompositeGradient<Composite, PerField...>
create_composite(PerField... evals) {
  return CompositeGradient<Composite, PerField...>(std::move(evals)...);
}

} // namespace pfc::field
