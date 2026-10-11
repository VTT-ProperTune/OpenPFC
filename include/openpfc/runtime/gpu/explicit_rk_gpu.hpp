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
 * The factory is `pfc::sim::steppers::create(std::tie(u, v), eval, model,
 * dt, tableau)` for `Field<double, CUDASpace>` / `HIPSpace`. It uses the
 * same `ButcherTableau`, model `rhs(t, g)`, and composite evaluator as
 * the host factory. `eval.prepare()` refreshes Full halos after each
 * stage is published into the bound fields.
 *
 * @see kernel/simulation/steppers/explicit_rk.hpp
 * @see runtime/gpu/elementwise_ops_gpu.hpp
 * @see runtime/gpu/for_each_interior_device_gpu.hpp
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <array>
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

template <class Model, class Composite> struct DeviceInc2Adapter {
  Model model;
  OPENPFC_HD pfc::sim::gpu::DeviceInc2 rhs(double t,
                                           const Composite &g) const noexcept {
    return as_device_inc2(model.rhs(t, g));
  }
};

/**
 * Two-field explicit RK on device-resident padded Fields.
 *
 * Bound fields hold the accepted state. Each stage is formed in scratch,
 * published into the fields for stencil reads, prepared (halos), and
 * evaluated on device. Accepted contents are restored after the RHS so a
 * failed stage does not leave a half-applied stage in the fields.
 */
template <class Space, class Eval, class Model, class Composite, class G0, class G1>
class ResidentMultiExplicitRK {
public:
  using FieldT = pfc::data::Field<double, Space>;
  using Buffer =
      pfc::core::DataBuffer<pfc::memory_space_to_backend_t<Space>, double>;

  ResidentMultiExplicitRK(std::array<FieldT *, 2> fields, Eval &eval,
                          const Model &model, double dt,
                          ButcherTableau<double> tableau)
      : m_fields(fields), m_eval(eval), m_model(model), m_dt(dt),
        m_tableau(std::move(tableau)), m_n(fields[0]->size()),
        m_nx(fields[0]->local_size()[0]), m_ny(fields[0]->local_size()[1]),
        m_nz(fields[0]->local_size()[2]), m_ones(m_n), m_weights(m_n) {
    static_assert(is_cuda_space<Space>() || is_hip_space<Space>(),
                  "device explicit RK requires CUDASpace or HIPSpace");
    const unsigned int s = m_tableau.stage_count();
    for (int f = 0; f < 2; ++f) {
      m_backup[static_cast<std::size_t>(f)] = Buffer(m_n);
      m_stage[static_cast<std::size_t>(f)] = Buffer(m_n);
      m_du[static_cast<std::size_t>(f)] = Buffer(m_n);
      m_k[static_cast<std::size_t>(f)].reserve(s);
      for (unsigned int i = 0; i < s; ++i)
        m_k[static_cast<std::size_t>(f)].emplace_back(m_n);
    }
    device_fill<Space>(m_ones.data(), m_n, 1.0);
  }

  double step(double t, FieldT &u, FieldT &v) {
    if (&u != m_fields[0] || &v != m_fields[1]) {
      throw std::invalid_argument(
          "device explicit RK: step fields are not the bound pair");
    }
    if (u.size() != m_n || v.size() != m_n) {
      throw std::invalid_argument("device explicit RK: field storage size changed");
    }
    u.sync_to_device();
    v.sync_to_device();
    const unsigned int s = m_tableau.stage_count();
    for (int f = 0; f < 2; ++f) {
      const auto fi = static_cast<std::size_t>(f);
      device_copy<Space>(m_backup[fi].data(), m_fields[fi]->data(), m_n);
    }
    DeviceInc2Adapter<Model, Composite> adapter{m_model};
    for (unsigned int i = 0; i < s; ++i) {
      form_stage(i);
      publish_stage();
      try {
        m_eval.prepare();
        device_fill<Space>(m_du[0].data(), m_n, 0.0);
        device_fill<Space>(m_du[1].data(), m_n, 0.0);
        const pfc::sim::gpu::DevicePtrPack2 pack{m_du[0].data(), m_du[1].data()};
        const double stage_time = t + m_tableau.c(i) * m_dt;
        pfc::sim::gpu::for_each_interior_device<DeviceInc2Adapter<Model, Composite>,
                                                Composite, G0, G1>(
            adapter, m_eval, pack, stage_time, m_nx, m_ny, m_nz);
        device_copy<Space>(m_k[0][i].data(), m_du[0].data(), m_n);
        device_copy<Space>(m_k[1][i].data(), m_du[1].data(), m_n);
      } catch (...) {
        restore_accepted();
        throw;
      }
      restore_accepted();
    }
    accumulate();
    return t + m_dt;
  }

  double dt() const noexcept { return m_dt; }

private:
  void form_stage(unsigned int stage_idx) {
    for (int f = 0; f < 2; ++f) {
      const auto fi = static_cast<std::size_t>(f);
      device_copy<Space>(m_stage[fi].data(), m_backup[fi].data(), m_n);
      for (unsigned int j = 0; j < stage_idx; ++j) {
        const double a_ij = m_tableau.a(stage_idx, j);
        device_scale_add<Space>(m_stage[fi].data(), m_k[fi][j].data(), m_dt * a_ij,
                                m_ones.data(), m_weights.data(), m_n);
      }
    }
  }

  void publish_stage() {
    for (int f = 0; f < 2; ++f) {
      const auto fi = static_cast<std::size_t>(f);
      device_copy<Space>(m_fields[fi]->data(), m_stage[fi].data(), m_n);
      m_fields[fi]->note_device_write();
    }
  }

  void restore_accepted() {
    for (int f = 0; f < 2; ++f) {
      const auto fi = static_cast<std::size_t>(f);
      device_copy<Space>(m_fields[fi]->data(), m_backup[fi].data(), m_n);
      m_fields[fi]->note_device_write();
    }
  }

  void accumulate() {
    const unsigned int s = m_tableau.stage_count();
    for (int f = 0; f < 2; ++f) {
      const auto fi = static_cast<std::size_t>(f);
      for (unsigned int i = 0; i < s; ++i) {
        device_scale_add<Space>(m_fields[fi]->data(), m_k[fi][i].data(),
                                m_dt * m_tableau.b(i), m_ones.data(),
                                m_weights.data(), m_n);
      }
      m_fields[fi]->note_device_write();
    }
  }

  std::array<FieldT *, 2> m_fields{};
  Eval &m_eval;
  const Model &m_model;
  double m_dt{0.0};
  ButcherTableau<double> m_tableau;
  std::size_t m_n{0};
  int m_nx{0};
  int m_ny{0};
  int m_nz{0};
  std::array<Buffer, 2> m_backup{};
  std::array<Buffer, 2> m_stage{};
  std::array<Buffer, 2> m_du{};
  std::array<std::vector<Buffer>, 2> m_k{};
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
  if (&u == &v || u.data() == v.data()) {
    throw std::invalid_argument("RK stage binding: duplicate field references");
  }
  if (u.domain() != v.domain() || u.box().low != v.box().low ||
      u.local_size() != v.local_size() || u.storage_halo() != v.storage_halo()) {
    throw std::invalid_argument("RK multi-field factory: layouts differ");
  }
  if (u.storage_halo() <= 0) {
    throw std::invalid_argument(
        "device explicit RK: padded Fields (storage_halo > 0) are required");
  }
  using Eval = pfc::gpu::CompositeGradientDevice<Composite, G0, G1>;
  return detail::ResidentMultiExplicitRK<Space, Eval, Model, Composite, G0, G1>(
      {&u, &v}, eval, model, dt, tableau);
}

} // namespace pfc::sim::steppers

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
