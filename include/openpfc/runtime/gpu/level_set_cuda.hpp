// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file level_set_cuda.hpp
 * @brief One-rank CUDA Hamilton–Jacobi level set.
 *
 * The step is the periodic form of `advance_level_set`. Halo exchange stays
 * on the host. Buffers are allocated once. HIP has the same kernel entry
 * and is not executed here.
 */

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/field/level_set.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>

namespace pfc::field::level_set {

/// Bytes and launches of the last successful or attempted device step.
struct LevelSetProfile {
  int kernel_launches{0};
  std::size_t host_to_device_bytes{0};
  std::size_t device_to_host_bytes{0};
};

/// Device SSP-RK2 on one periodic grid. The constructor allocates every buffer.
class LevelSetCuda {
public:
  explicit LevelSetCuda(LevelSetGrid grid)
      : m_grid(grid), m_n(checked_count(grid)), m_phi(m_n), m_vx(m_n), m_vy(m_n),
        m_vz(m_n), m_vn(m_n), m_phi0(m_n), m_rate(m_n), m_stage(m_n) {}

  LevelSetCuda(const LevelSetCuda &) = delete;
  LevelSetCuda &operator=(const LevelSetCuda &) = delete;

  [[nodiscard]] const LevelSetGrid &grid() const noexcept { return m_grid; }
  [[nodiscard]] const LevelSetProfile &profile() const noexcept { return m_profile; }

  /// One SSP-RK2 step. A throw leaves `phi` unchanged.
  [[nodiscard]] LevelSetStep
  advance(const LevelSetGrid &grid, std::span<double> phi, LevelSetMode mode,
          std::span<const double> velocity_x, std::span<const double> velocity_y,
          std::span<const double> velocity_z, std::span<const double> normal_speed,
          std::span<const double> sign_field, double dt) {
    m_profile = {};
    require_grid(grid);
    if (!(dt > 0.0) || !std::isfinite(dt))
      throw std::invalid_argument("level-set time step must be positive");
    require_values(phi, "field");
    const int mode_id = mode_index(mode);
    upload(m_phi, phi);
    if (mode == LevelSetMode::Advection) {
      require_values(velocity_x, "velocity");
      require_values(velocity_y, "velocity");
      require_values(velocity_z, "velocity");
      upload(m_vx, velocity_x);
      upload(m_vy, velocity_y);
      upload(m_vz, velocity_z);
    } else if (mode == LevelSetMode::NormalSpeed) {
      require_values(normal_speed, "normal speed");
      upload(m_vn, normal_speed);
    } else {
      require_values(sign_field, "sign field");
      upload(m_phi0, sign_field);
    }
    device_rk2(mode_id, dt);
    return download(phi, dt);
  }

  /// Sussman redistance. The sign is the field uploaded once, before the loop.
  [[nodiscard]] LevelSetStep redistance(const LevelSetGrid &grid,
                                        std::span<double> phi,
                                        std::span<const double> sign_field,
                                        int substeps, double cfl) {
    m_profile = {};
    require_grid(grid);
    if (substeps < 1)
      throw std::invalid_argument("level-set redistance needs a step");
    const double dt = level_set_stable_dt(grid, 1.0, cfl);
    require_values(phi, "field");
    require_values(sign_field, "sign field");
    upload(m_phi, phi);
    upload(m_phi0, sign_field);
    for (int s = 0; s < substeps; ++s) device_rk2(2, dt);
    LevelSetStep out = download(phi, dt);
    out.dt = dt;
    return out;
  }

private:
  static std::size_t checked_count(const LevelSetGrid &grid) {
    detail::validate_grid(grid);
    return level_set_count(grid);
  }

  void require_grid(const LevelSetGrid &grid) const {
    if (grid.nx != m_grid.nx || grid.ny != m_grid.ny || grid.nz != m_grid.nz ||
        grid.hx != m_grid.hx || grid.hy != m_grid.hy || grid.hz != m_grid.hz)
      throw std::invalid_argument("level-set grid does not match the device buffer");
  }

  void require_values(std::span<const double> values, const char *what) const {
    if (values.size() != m_n)
      throw std::invalid_argument(std::string("level-set ") + what +
                                  " size does not match the grid");
    for (double v : values) {
      if (!std::isfinite(v))
        throw std::domain_error(std::string("nonfinite level-set ") + what);
    }
  }

  static int mode_index(LevelSetMode mode) {
    if (mode == LevelSetMode::Advection) return 0;
    if (mode == LevelSetMode::NormalSpeed) return 1;
    if (mode == LevelSetMode::Redistance) return 2;
    throw std::invalid_argument("level-set mode is not recognised");
  }

  void upload(pfc::core::DataBuffer<pfc::backend::CUDATag, double> &buffer,
              std::span<const double> values) {
    buffer.copy_from_host(values);
    m_profile.host_to_device_bytes += values.size() * sizeof(double);
  }

  void device_rk2(int mode, double dt) {
    const double *phi = m_phi.data();
    const double *vx = m_vx.data();
    const double *vy = m_vy.data();
    const double *vz = m_vz.data();
    const double *vn = m_vn.data();
    const double *phi0 = m_phi0.data();
    double *rate = m_rate.data();
    double *stage = m_stage.data();
    const int nx = m_grid.nx;
    const int ny = m_grid.ny;
    const int nz = m_grid.nz;
    level_set_rate_cuda_impl(phi, vx, vy, vz, vn, phi0, rate, nx, ny, nz, m_grid.hx,
                             m_grid.hy, m_grid.hz, mode);
    level_set_combine3_cuda_impl(phi, rate, rate, stage, 1.0, dt, 0.0, m_n);
    level_set_rate_cuda_impl(stage, vx, vy, vz, vn, phi0, rate, nx, ny, nz,
                             m_grid.hx, m_grid.hy, m_grid.hz, mode);
    level_set_combine3_cuda_impl(phi, stage, rate, m_phi.data(), 0.5, 0.5, 0.5 * dt,
                                 m_n);
    m_profile.kernel_launches += 4;
  }

  [[nodiscard]] LevelSetStep download(std::span<double> phi, double dt) {
    std::vector<double> host(m_n);
    m_phi.copy_to_host(host.data(), host.size());
    m_profile.device_to_host_bytes += m_n * sizeof(double);
    for (double v : host) {
      if (!std::isfinite(v)) throw std::domain_error("nonfinite level-set update");
    }
    std::copy(host.begin(), host.end(), phi.begin());
    LevelSetStep out;
    out.dt = dt;
    out.negative_volume = level_set_negative_centroid(m_grid, phi).volume;
    return out;
  }

  LevelSetGrid m_grid;
  std::size_t m_n;
  LevelSetProfile m_profile;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_phi;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_vx;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_vy;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_vz;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_vn;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_phi0;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_rate;
  pfc::core::DataBuffer<pfc::backend::CUDATag, double> m_stage;
};

} // namespace pfc::field::level_set
