// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file explicit_rk_gpu.hpp
 * @brief Device-resident multi-field explicit RK factory (CUDA/HIP).
 *
 * Stage construction, RHS evaluation, and Butcher accumulation stay on
 * device buffers. Include this from a CUDA or HIP translation unit: the
 * bound `for_each_interior_device` launch is instantiated there.
 *
 * The factory is `pfc::sim::steppers::create(std::tie(u, v[, w[, q]]), eval,
 * model, dt, tableau)` for two, three, or four `Field<double, CUDASpace>` /
 * `HIPSpace` fields. It uses the same `ButcherTableau`, model `rhs(t, g)`,
 * and composite evaluator as the host factory. `eval.prepare()` refreshes
 * Full halos after each stage is published into the bound fields.
 *
 * @see kernel/simulation/steppers/explicit_rk.hpp
 * @see runtime/gpu/elementwise_ops_gpu.hpp
 * @see runtime/gpu/for_each_interior_device_gpu.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <array>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <openpfc/runtime/gpu/fd_gradient_device_gpu.hpp>
#include <openpfc/runtime/gpu/fill_gpu.hpp>
#include <openpfc/runtime/gpu/for_each_interior_device_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim::steppers {
namespace detail {

template <class Space> constexpr bool is_cuda_space() {
#if defined(OpenPFC_ENABLE_CUDA)
  return std::is_same_v<Space, pfc::CUDASpace>;
#else
  return false;
#endif
}

template <class Space> constexpr bool is_hip_space() {
#if defined(OpenPFC_ENABLE_HIP)
  return std::is_same_v<Space, pfc::HIPSpace>;
#else
  return false;
#endif
}

template <class Space> inline void device_fill(double *p, std::size_t n, double v) {
  if constexpr (is_cuda_space<Space>()) {
#if defined(OpenPFC_ENABLE_CUDA)
    pfc::fill_cuda_impl(p, n, v);
#endif
  } else if constexpr (is_hip_space<Space>()) {
#if defined(OpenPFC_ENABLE_HIP)
    pfc::fill_hip_impl(p, n, v);
#endif
  }
}

template <class Space>
inline void device_copy(double *out, const double *x, std::size_t n) {
  if constexpr (is_cuda_space<Space>()) {
#if defined(OpenPFC_ENABLE_CUDA)
    pfc::axpy_fill_cuda_impl(out, x, 1.0, 0.0, n);
#endif
  } else if constexpr (is_hip_space<Space>()) {
#if defined(OpenPFC_ENABLE_HIP)
    pfc::axpy_fill_hip_impl(out, x, 1.0, 0.0, n);
#endif
  }
}

template <class Space>
inline void device_combine(const double *x0, const double *x1, const double *w0,
                           const double *w1, double *out, std::size_t n) {
  if constexpr (is_cuda_space<Space>()) {
#if defined(OpenPFC_ENABLE_CUDA)
    pfc::combine_two_term_cuda_impl(x0, x1, w0, w1, out, n);
#endif
  } else if constexpr (is_hip_space<Space>()) {
#if defined(OpenPFC_ENABLE_HIP)
    pfc::combine_two_term_hip_impl(x0, x1, w0, w1, out, n);
#endif
  }
}

/// `y += alpha * x` via ones/weight buffers and `combine_two_term`.
template <class Space>
inline void device_scale_add(double *y, const double *x, double alpha, double *ones,
                             double *weights, std::size_t n) {
  if (alpha == 0.0 || n == 0) return;
  device_fill<Space>(weights, n, alpha);
  device_combine<Space>(y, x, ones, weights, y, n);
}

template <class Inc>
OPENPFC_HD inline pfc::sim::gpu::DeviceInc2 as_device_inc2(const Inc &inc) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Inc>,
                               pfc::sim::gpu::DeviceInc2>) {
    return inc;
  } else {
    return pfc::sim::gpu::DeviceInc2{inc.du, inc.dv};
  }
}

template <class Inc>
OPENPFC_HD inline pfc::sim::gpu::DeviceInc3 as_device_inc3(const Inc &inc) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Inc>,
                               pfc::sim::gpu::DeviceInc3>) {
    return inc;
  } else {
    return pfc::sim::gpu::DeviceInc3{inc.du, inc.dv, inc.dw};
  }
}

template <class Inc>
OPENPFC_HD inline pfc::sim::gpu::DeviceInc4 as_device_inc4(const Inc &inc) {
  if constexpr (std::is_same_v<std::remove_cvref_t<Inc>,
                               pfc::sim::gpu::DeviceInc4>) {
    return inc;
  } else {
    return pfc::sim::gpu::DeviceInc4{inc.du, inc.dv, inc.dw, inc.dq};
  }
}

template <class Model, class Composite> struct DeviceInc2Adapter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc2 rhs(double t,
                                           const Composite &g) const noexcept {
    return as_device_inc2(model.rhs(t, g));
  }
};

template <class Model, class Composite> struct DeviceInc3Adapter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc3 rhs(double t,
                                           const Composite &g) const noexcept {
    return as_device_inc3(model.rhs(t, g));
  }
};

template <class Model, class Composite> struct DeviceInc4Adapter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc4 rhs(double t,
                                           const Composite &g) const noexcept {
    return as_device_inc4(model.rhs(t, g));
  }
};

template <class FieldT, std::size_t N>
void check_device_rk_fields(const std::array<FieldT *, N> &fields) {
  static_assert(N >= 2 && N <= 4, "device explicit RK supports 2, 3, or 4 fields");
  if (fields[0]->storage_halo() <= 0) {
    throw std::invalid_argument(
        "device explicit RK: padded Fields (storage_halo > 0) are required");
  }
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i + 1; j < N; ++j) {
      if (fields[i] == fields[j] || fields[i]->data() == fields[j]->data()) {
        throw std::invalid_argument("RK stage binding: duplicate field references");
      }
    }
    if (i == 0) continue;
    const auto &a = *fields[0];
    const auto &b = *fields[i];
    if (a.domain() != b.domain() || a.box().low != b.box().low ||
        a.local_size() != b.local_size() || a.storage_halo() != b.storage_halo()) {
      throw std::invalid_argument("RK multi-field factory: layouts differ");
    }
  }
}

/**
 * N-field explicit RK on device-resident padded Fields (N = 2, 3, or 4).
 *
 * Bound fields hold the accepted state. Each stage is formed in scratch,
 * published into the fields for stencil reads, prepared (halos), and
 * evaluated on device. Accepted contents are restored after the RHS so a
 * failed stage does not leave a half-applied stage in the fields.
 */
template <class Space, class Eval, class Model, class Composite, class... Gs>
class ResidentMultiExplicitRK {
public:
  static constexpr std::size_t N = sizeof...(Gs);
  static_assert(N >= 2 && N <= 4, "device explicit RK supports 2, 3, or 4 fields");
  using FieldT = pfc::data::Field<double, Space>;
  using Buffer =
      pfc::core::DataBuffer<pfc::memory_space_to_backend_t<Space>, double>;

  ResidentMultiExplicitRK(std::array<FieldT *, N> fields, Eval &eval,
                          const Model &model, double dt,
                          ButcherTableau<double> tableau)
      : m_fields(fields), m_eval(eval), m_model(model), m_dt(dt),
        m_tableau(std::move(tableau)), m_n(fields[0]->size()),
        m_nx(fields[0]->local_size()[0]), m_ny(fields[0]->local_size()[1]),
        m_nz(fields[0]->local_size()[2]), m_ones(m_n), m_weights(m_n) {
    static_assert(is_cuda_space<Space>() || is_hip_space<Space>(),
                  "device explicit RK requires CUDASpace or HIPSpace");
    const unsigned int s = m_tableau.stage_count();
    for (std::size_t f = 0; f < N; ++f) {
      m_backup[f] = Buffer(m_n);
      m_stage[f] = Buffer(m_n);
      m_du[f] = Buffer(m_n);
      m_k[f].reserve(s);
      for (unsigned int i = 0; i < s; ++i) m_k[f].emplace_back(m_n);
    }
    device_fill<Space>(m_ones.data(), m_n, 1.0);
  }

  template <class... Fs> double step(double t, Fs &...fs) {
    return step_checked(t, [](double) {}, fs...);
  }

  /**
   * Check the published final state on the host; throw to reject it.
   * The callback returns void and must not write the bound fields or change
   * their storage. Caller-owned auxiliaries and collective agreement stay
   * with the caller. Rejection restores every padded device value in place,
   * using the existing backup; no full-state host upload is performed.
   */
  template <class Check, class... Fs>
    requires std::invocable<Check &, double> &&
             std::is_void_v<std::invoke_result_t<Check &, double>>
  double step_checked(double t, Check &&check, Fs &...fs) {
    static_assert(sizeof...(Fs) == N, "device explicit RK: wrong field count");
    const std::array<FieldT *, N> got{&fs...};
    for (std::size_t f = 0; f < N; ++f) {
      if (got[f] != m_fields[f]) {
        throw std::invalid_argument(
            "device explicit RK: step fields are not the bound set");
      }
      if (got[f]->size() != m_n) {
        throw std::invalid_argument(
            "device explicit RK: field storage size changed");
      }
      got[f]->sync_to_device();
    }
    const unsigned int s = m_tableau.stage_count();
    for (std::size_t f = 0; f < N; ++f)
      device_copy<Space>(m_backup[f].data(), m_fields[f]->data(), m_n);
    for (unsigned int i = 0; i < s; ++i) {
      form_stage(i);
      publish_stage();
      try {
        m_eval.prepare();
        evaluate_rhs(t + m_tableau.c(i) * m_dt);
        for (std::size_t f = 0; f < N; ++f)
          device_copy<Space>(m_k[f][i].data(), m_du[f].data(), m_n);
      } catch (...) {
        restore_accepted();
        throw;
      }
      restore_accepted();
    }
    try {
      accumulate();
      check(t + m_dt);
    } catch (...) {
      restore_accepted();
      throw;
    }
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  void evaluate_rhs(double stage_time) {
    for (std::size_t f = 0; f < N; ++f) device_fill<Space>(m_du[f].data(), m_n, 0.0);
    if constexpr (N == 2) {
      DeviceInc2Adapter<Model, Composite> adapter{m_model};
      const pfc::sim::gpu::DevicePtrPack2 pack{m_du[0].data(), m_du[1].data()};
      pfc::sim::gpu::for_each_interior_device<DeviceInc2Adapter<Model, Composite>,
                                              Composite, Gs...>(
          adapter, m_eval, pack, stage_time, m_nx, m_ny, m_nz);
    } else if constexpr (N == 3) {
      DeviceInc3Adapter<Model, Composite> adapter{m_model};
      const pfc::sim::gpu::DevicePtrPack3 pack{m_du[0].data(), m_du[1].data(),
                                               m_du[2].data()};
      pfc::sim::gpu::for_each_interior_device<DeviceInc3Adapter<Model, Composite>,
                                              Composite, Gs...>(
          adapter, m_eval, pack, stage_time, m_nx, m_ny, m_nz);
    } else {
      DeviceInc4Adapter<Model, Composite> adapter{m_model};
      const pfc::sim::gpu::DevicePtrPack4 pack{m_du[0].data(), m_du[1].data(),
                                               m_du[2].data(), m_du[3].data()};
      pfc::sim::gpu::for_each_interior_device<DeviceInc4Adapter<Model, Composite>,
                                              Composite, Gs...>(
          adapter, m_eval, pack, stage_time, m_nx, m_ny, m_nz);
    }
  }

  void form_stage(unsigned int stage_idx) {
    for (std::size_t f = 0; f < N; ++f) {
      device_copy<Space>(m_stage[f].data(), m_backup[f].data(), m_n);
      for (unsigned int j = 0; j < stage_idx; ++j) {
        const double a_ij = m_tableau.a(stage_idx, j);
        device_scale_add<Space>(m_stage[f].data(), m_k[f][j].data(), m_dt * a_ij,
                                m_ones.data(), m_weights.data(), m_n);
      }
    }
  }

  void publish_stage() {
    for (std::size_t f = 0; f < N; ++f) {
      device_copy<Space>(m_fields[f]->data(), m_stage[f].data(), m_n);
      m_fields[f]->note_device_write();
    }
  }

  void restore_accepted() {
    for (std::size_t f = 0; f < N; ++f) {
      device_copy<Space>(m_fields[f]->data(), m_backup[f].data(), m_n);
      m_fields[f]->note_device_write();
    }
  }

  void accumulate() {
    const unsigned int s = m_tableau.stage_count();
    for (std::size_t f = 0; f < N; ++f) {
      for (unsigned int i = 0; i < s; ++i) {
        device_scale_add<Space>(m_fields[f]->data(), m_k[f][i].data(),
                                m_dt * m_tableau.b(i), m_ones.data(),
                                m_weights.data(), m_n);
      }
      m_fields[f]->note_device_write();
    }
  }

  std::array<FieldT *, N> m_fields{};
  Eval &m_eval;
  const Model &m_model;
  double m_dt{0.0};
  ButcherTableau<double> m_tableau;
  std::size_t m_n{0};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::array<Buffer, N> m_backup{};
  std::array<Buffer, N> m_stage{};
  std::array<Buffer, N> m_du{};
  std::array<std::vector<Buffer>, N> m_k{};
  Buffer m_ones;
  Buffer m_weights;
};

} // namespace detail

/**
 * @brief Device-resident two-field explicit RK factory.
 *
 * Same arguments as the host `create(std::tie(u, v), eval, model, dt,
 * tableau)`. `u` and `v` stay on device. After each stage is written into
 * the bound fields, `eval.prepare()` runs (bind Full halo exchange there).
 * The stepper does not copy a full field to the host between stages.
 *
 * Include from a CUDA/HIP TU. `Model::rhs` must be `OPENPFC_HD` and return
 * `DeviceInc2` or a two-field increment with `du`/`dv` members.
 */
template <class Space, class Composite, class G0, class G1, class Model>
[[nodiscard]] auto create(
    std::tuple<pfc::data::Field<double, Space> &, pfc::data::Field<double, Space> &>
        fields,
    pfc::gpu::CompositeGradientDevice<Composite, G0, G1> &eval, const Model &model,
    double dt, const ButcherTableau<double> &tableau) {
  using FieldT = pfc::data::Field<double, Space>;
  FieldT &u = std::get<0>(fields);
  FieldT &v = std::get<1>(fields);
  const std::array<FieldT *, 2> bound{&u, &v};
  detail::check_device_rk_fields(bound);
  using Eval = pfc::gpu::CompositeGradientDevice<Composite, G0, G1>;
  return detail::ResidentMultiExplicitRK<Space, Eval, Model, Composite, G0, G1>(
      bound, eval, model, dt, tableau);
}

/**
 * @brief Device-resident three-field explicit RK factory.
 *
 * Same contract as the two-field factory. `Model::rhs` returns `DeviceInc3`
 * or `{du, dv, dw}`. Stage buffers stay on device.
 */
template <class Space, class Composite, class G0, class G1, class G2, class Model>
[[nodiscard]] auto create(
    std::tuple<pfc::data::Field<double, Space> &, pfc::data::Field<double, Space> &,
               pfc::data::Field<double, Space> &>
        fields,
    pfc::gpu::CompositeGradientDevice<Composite, G0, G1, G2> &eval,
    const Model &model, double dt, const ButcherTableau<double> &tableau) {
  using FieldT = pfc::data::Field<double, Space>;
  const std::array<FieldT *, 3> bound{&std::get<0>(fields), &std::get<1>(fields),
                                      &std::get<2>(fields)};
  detail::check_device_rk_fields(bound);
  using Eval = pfc::gpu::CompositeGradientDevice<Composite, G0, G1, G2>;
  return detail::ResidentMultiExplicitRK<Space, Eval, Model, Composite, G0, G1, G2>(
      bound, eval, model, dt, tableau);
}

/**
 * @brief Device-resident four-field explicit RK factory.
 *
 * Same contract as the two-field factory. `Model::rhs` returns `DeviceInc4`
 * or `{du, dv, dw, dq}`. Stage buffers stay on device.
 */
template <class Space, class Composite, class G0, class G1, class G2, class G3,
          class Model>
[[nodiscard]] auto create(
    std::tuple<pfc::data::Field<double, Space> &, pfc::data::Field<double, Space> &,
               pfc::data::Field<double, Space> &, pfc::data::Field<double, Space> &>
        fields,
    pfc::gpu::CompositeGradientDevice<Composite, G0, G1, G2, G3> &eval,
    const Model &model, double dt, const ButcherTableau<double> &tableau) {
  using FieldT = pfc::data::Field<double, Space>;
  const std::array<FieldT *, 4> bound{&std::get<0>(fields), &std::get<1>(fields),
                                      &std::get<2>(fields), &std::get<3>(fields)};
  detail::check_device_rk_fields(bound);
  using Eval = pfc::gpu::CompositeGradientDevice<Composite, G0, G1, G2, G3>;
  return detail::ResidentMultiExplicitRK<Space, Eval, Model, Composite, G0, G1, G2,
                                         G3>(bound, eval, model, dt, tableau);
}

} // namespace pfc::sim::steppers

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
