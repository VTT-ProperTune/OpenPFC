// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file crystal_observables.hpp
 * @brief Host observables for a planar crystal front and its orientation.
 *
 * The front amplitude at a station is the root-mean-square of the owned
 * samples in the two transverse directions. Ranks reduce those sums with
 * MPI_Allreduce. A clean front is one interface of the requested polarity
 * and none of the other. Its position is the linear crossing of `level`
 * between the two stations that bracket it, so a translated front can move
 * by less than one grid spacing. Velocity is only the time-series quotient
 * `(x(t1) - x(t0)) / (t1 - t0)` of two valid samples. It does not unwrap
 * the periodic box.
 *
 * Orientation is the wavevector of one real density wave
 * `A cos(k · x + φ)`. Order-2 central differences recover
 * `|2 sin(k_a h_a / 2) / h_a|` from `-u ∂_{aa} u / u²`, and the mixed
 * differences recover the relative signs. A real wave cannot tell `k`
 * from `-k`: the cosine gauge flips the vector so its first nonzero
 * component is positive. Further identification is the caller's crystal
 * symmetry, never a material threshold:
 *
 * - `none` keeps that gauge. The angle is `atan2(k_y, k_x)` in (-π, π].
 * - `four_fold` rotates about +z by a multiple of π/2 into [0, π/2).
 * - `six_fold` rotates about +z by a multiple of π/3 into [0, π/3).
 * - `cubic` searches the 24 proper cubic images of both `k` and `-k` and
 *   keeps the image with the greatest z, then y, then x.
 *
 * The symmetry axis of the cyclic groups is z. Grain labels are votes for
 * the nearest supplied candidate under that same representative. Cells
 * whose stencil crosses a grain boundary are the caller's to exclude.
 * This is a small-field host reference, not a generic multi-peak grain fit
 * or scalable global replicated-field diagnostic. Incoherent moments will
 * be rejected rather than identified as an invented crystal direction.
 * Nothing here reads device memory.
 */

#include <openpfc/kernel/data/box3i.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc::observables {

enum class FrontValidity { valid, no_crossing, not_single_front, nonfinite };

enum class CrystalSymmetry { none, four_fold, six_fold, cubic };

struct OwnedDensity {
  /// x-fastest owned samples, no halo. The count is `box.count()`.
  std::span<const double> values;
  std::array<int, 3> global_size{};
  std::array<double, 3> spacing{};
  std::array<double, 3> origin{};
  Box3i box{};
};

struct PlanarFrontOptions {
  int axis{0};
  double level{0.5};
  bool rising{true};
};

struct FrontSample {
  double position{std::numeric_limits<double>::quiet_NaN()};
  FrontValidity validity{FrontValidity::no_crossing};
  /// Crossings of the requested polarity. Opposite interfaces are not counted.
  int crossings{0};
};

struct FrontVelocity {
  double velocity{std::numeric_limits<double>::quiet_NaN()};
  bool valid{false};
};

struct IndexWindow {
  std::array<int, 3> low{};
  std::array<int, 3> high{};
};

struct OrientationSample {
  std::array<double, 3> wavevector{};
  double angle_xy{std::numeric_limits<double>::quiet_NaN()};
  bool valid{false};
};

struct GrainStatistics {
  std::vector<long long> counts;
  std::vector<double> fraction;
  long long voting_cells{0};
  bool valid{false};
};

/// `(x1 - x0) / (t1 - t0)` when both samples are valid and `t1 > t0`.
[[nodiscard]] inline FrontVelocity front_velocity(const FrontSample &earlier,
                                                  double t0,
                                                  const FrontSample &later,
                                                  double t1) noexcept {
  FrontVelocity out;
  if (earlier.validity != FrontValidity::valid ||
      later.validity != FrontValidity::valid)
    return out;
  if (!(t1 > t0) || !std::isfinite(t0) || !std::isfinite(t1)) return out;
  if (!std::isfinite(earlier.position) || !std::isfinite(later.position)) return out;
  out.velocity = (later.position - earlier.position) / (t1 - t0);
  out.valid = std::isfinite(out.velocity);
  if (!out.valid) out.velocity = std::numeric_limits<double>::quiet_NaN();
  return out;
}

[[nodiscard]] inline std::array<double, 3>
representative_wavevector(std::array<double, 3> wave, CrystalSymmetry symmetry);

[[nodiscard]] inline FrontSample
planar_front_position(const OwnedDensity &density, MPI_Comm comm,
                      const PlanarFrontOptions &options);

[[nodiscard]] inline OrientationSample
region_orientation(const OwnedDensity &density, MPI_Comm comm, IndexWindow window,
                   CrystalSymmetry symmetry);

[[nodiscard]] inline GrainStatistics
label_grains(const OwnedDensity &density, MPI_Comm comm, IndexWindow window,
             CrystalSymmetry symmetry,
             std::span<const std::array<double, 3>> candidates);

namespace crystal_detail {

inline constexpr double kTie = 1e-12;

[[noreturn]] inline void fail(const char *what) {
  throw std::invalid_argument(what);
}

inline void require_comm(MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) fail("crystal observable: communicator is null");
}

inline int wrap_index(int i, int n) {
  const int m = i % n;
  return m < 0 ? m + n : m;
}

inline std::size_t grid_index(int i, int j, int k, const std::array<int, 3> &n) {
  i = wrap_index(i, n[0]);
  j = wrap_index(j, n[1]);
  k = wrap_index(k, n[2]);
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(n[0]) *
             (static_cast<std::size_t>(j) +
              static_cast<std::size_t>(n[1]) * static_cast<std::size_t>(k));
}

inline std::size_t grid_count(const std::array<int, 3> &n) {
  if (n[0] < 1 || n[1] < 1 || n[2] < 1) {
    fail("crystal observable: grid size must be positive");
  }
  const auto nx = static_cast<std::size_t>(n[0]);
  const auto ny = static_cast<std::size_t>(n[1]);
  const auto nz = static_cast<std::size_t>(n[2]);
  if (nx > std::numeric_limits<std::size_t>::max() / ny) {
    fail("crystal observable: grid is too large");
  }
  const auto nxy = nx * ny;
  if (nxy > std::numeric_limits<std::size_t>::max() / nz) {
    fail("crystal observable: grid is too large");
  }
  return nxy * nz;
}

inline void require_geometry(const OwnedDensity &density) {
  if (!density.box.is_consistent()) {
    fail("crystal observable: owned box is inconsistent");
  }
  if (density.values.size() != static_cast<std::size_t>(density.box.count())) {
    fail("crystal observable: sample count does not match the owned box");
  }
  if (density.box.count() > std::numeric_limits<int>::max()) {
    fail("crystal observable: owned box does not fit an MPI count");
  }
  grid_count(density.global_size);
  for (int d = 0; d < 3; ++d) {
    if (density.box.low[d] < 0 || density.box.high[d] >= density.global_size[d]) {
      fail("crystal observable: owned box leaves the global grid");
    }
    if (!(density.spacing[d] > 0.0) || !std::isfinite(density.spacing[d]) ||
        !std::isfinite(density.origin[d])) {
      fail("crystal observable: spacing and origin must be finite");
    }
  }
}

inline void require_window(IndexWindow window, const std::array<int, 3> &n) {
  for (int d = 0; d < 3; ++d) {
    if (window.low[d] < 0 || window.high[d] >= n[d] ||
        window.low[d] > window.high[d]) {
      fail("crystal observable: window is outside the grid");
    }
  }
}

// Coordinate rank-local input errors before the first collective. A malformed
// local field must not strand other MPI ranks inside Allreduce/Allgatherv.
template <class Check> inline void collective_validate(MPI_Comm comm, Check check) {
  int valid = 1;
  try {
    check();
  } catch (const std::exception &) {
    valid = 0;
  }
  int all_valid = 0;
  const int mpi_result =
      MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm);
  if (mpi_result != MPI_SUCCESS) {
    fail("crystal observable: collective validation failed");
  }
  if (all_valid == 0) {
    fail("crystal observable: invalid input on at least one MPI rank");
  }
}

inline std::vector<double> replicate(const OwnedDensity &density, MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  const std::size_t total_cells = grid_count(density.global_size);
  // This is a bounded reference implementation that replicates the entire
  // field on every rank; larger fields need a local-halo/reduction design.
  if (total_cells > (std::size_t{1} << 20)) {
    fail("crystal observable: reference replication exceeds 2^20 cells");
  }
  const int local_n = static_cast<int>(density.box.count());
  std::vector<int> counts(static_cast<std::size_t>(size));
  std::vector<int> displs(static_cast<std::size_t>(size));
  if (MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) !=
      MPI_SUCCESS) {
    fail("crystal observable: MPI_Allgather of owned counts failed");
  }
  int total = 0;
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    if (counts[static_cast<std::size_t>(r)] < 0 ||
        total >
            std::numeric_limits<int>::max() - counts[static_cast<std::size_t>(r)]) {
      fail("crystal observable: replicated field does not fit an MPI count");
    }
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<double> packed(static_cast<std::size_t>(total));
  if (MPI_Allgatherv(density.values.data(), local_n, MPI_DOUBLE, packed.data(),
                     counts.data(), displs.data(), MPI_DOUBLE,
                     comm) != MPI_SUCCESS) {
    fail("crystal observable: MPI_Allgatherv of the density failed");
  }
  const int meta[6] = {density.box.low[0],  density.box.low[1],
                       density.box.low[2],  density.box.size[0],
                       density.box.size[1], density.box.size[2]};
  std::vector<int> metas(static_cast<std::size_t>(6 * size));
  if (MPI_Allgather(meta, 6, MPI_INT, metas.data(), 6, MPI_INT, comm) !=
      MPI_SUCCESS) {
    fail("crystal observable: MPI_Allgather of owned boxes failed");
  }
  const std::size_t ncell = grid_count(density.global_size);
  std::vector<double> full(ncell, std::numeric_limits<double>::quiet_NaN());
  std::vector<char> seen(ncell, 0);
  for (int r = 0; r < size; ++r) {
    const int *m = metas.data() + static_cast<std::size_t>(6 * r);
    const int sx = m[3];
    const int sy = m[4];
    const int sz = m[5];
    if (sx < 1 || sy < 1 || sz < 1) fail("crystal observable: empty owned box");
    const double *src = packed.data() + displs[static_cast<std::size_t>(r)];
    std::size_t p = 0;
    for (int k = 0; k < sz; ++k) {
      for (int j = 0; j < sy; ++j) {
        for (int i = 0; i < sx; ++i, ++p) {
          const int gx = m[0] + i;
          const int gy = m[1] + j;
          const int gz = m[2] + k;
          if (gx < 0 || gy < 0 || gz < 0 || gx >= density.global_size[0] ||
              gy >= density.global_size[1] || gz >= density.global_size[2]) {
            fail("crystal observable: gathered box leaves the global grid");
          }
          const std::size_t id = grid_index(gx, gy, gz, density.global_size);
          if (seen[id] != 0) fail("crystal observable: owned boxes overlap");
          seen[id] = 1;
          full[id] = src[p];
        }
      }
    }
  }
  for (char bit : seen) {
    if (bit == 0) fail("crystal observable: owned boxes do not cover the grid");
  }
  return full;
}

struct Derivatives {
  double u;
  double xx, yy, zz, xy, xz, yz;
};

inline Derivatives differentiate(const std::vector<double> &full,
                                 const std::array<int, 3> &n,
                                 const std::array<double, 3> &h, int i, int j,
                                 int k) {
  const auto at = [&](int a, int b, int c) { return full[grid_index(a, b, c, n)]; };
  const double u = at(i, j, k);
  const double hx = h[0];
  const double hy = h[1];
  const double hz = h[2];
  Derivatives d;
  d.u = u;
  d.xx = (at(i + 1, j, k) - 2.0 * u + at(i - 1, j, k)) / (hx * hx);
  d.yy = (at(i, j + 1, k) - 2.0 * u + at(i, j - 1, k)) / (hy * hy);
  d.zz = (at(i, j, k + 1) - 2.0 * u + at(i, j, k - 1)) / (hz * hz);
  d.xy = (at(i + 1, j + 1, k) - at(i + 1, j - 1, k) - at(i - 1, j + 1, k) +
          at(i - 1, j - 1, k)) /
         (4.0 * hx * hy);
  d.xz = (at(i + 1, j, k + 1) - at(i + 1, j, k - 1) - at(i - 1, j, k + 1) +
          at(i - 1, j, k - 1)) /
         (4.0 * hx * hz);
  d.yz = (at(i, j + 1, k + 1) - at(i, j + 1, k - 1) - at(i, j - 1, k + 1) +
          at(i, j - 1, k - 1)) /
         (4.0 * hy * hz);
  return d;
}

inline double invert_symbol(double q_over_s, double h) {
  if (!(q_over_s > 0.0) || !std::isfinite(q_over_s)) return 0.0;
  double arg = std::sqrt(q_over_s) * h / 2.0;
  if (arg > 1.0) arg = 1.0;
  return (2.0 / h) * std::asin(arg);
}

inline std::array<double, 3> wave_from_moments(double S, double Qxx, double Qyy,
                                               double Qzz, double Qxy, double Qxz,
                                               double Qyz,
                                               const std::array<double, 3> &h) {
  const double kx = invert_symbol(Qxx / S, h[0]);
  const double ky = invert_symbol(Qyy / S, h[1]);
  const double kz = invert_symbol(Qzz / S, h[2]);
  int pivot = 0;
  double peak = kx;
  if (ky > peak) {
    pivot = 1;
    peak = ky;
  }
  if (kz > peak) {
    pivot = 2;
    peak = kz;
  }
  const double cross[3] = {0.0, Qxy / S, Qxz / S};
  const double yz = Qyz / S;
  auto signed_component = [](double magnitude, double product) {
    if (!(magnitude > 0.0)) return 0.0;
    return product < 0.0 ? -magnitude : magnitude;
  };
  std::array<double, 3> wave{kx, ky, kz};
  if (pivot == 0) {
    wave[1] = signed_component(ky, cross[1]);
    wave[2] = signed_component(kz, cross[2]);
  } else if (pivot == 1) {
    wave[0] = signed_component(kx, Qxy / S);
    wave[2] = signed_component(kz, yz);
    wave[1] = ky;
  } else {
    wave[0] = signed_component(kx, Qxz / S);
    wave[1] = signed_component(ky, yz);
    wave[2] = kz;
  }
  return wave;
}

// For one Fourier cosine, the mixed central differences must match the
// product of the one-dimensional first-derivative stencil symbols.
// A sum of independent lattice waves generally violates these identities.
inline bool coherent_single_mode(double S, double Qxx, double Qyy, double Qzz,
                                 double Qxy, double Qxz, double Qyz,
                                 const std::array<double, 3> &h,
                                 const std::array<double, 3> &wave) {
  if (!(S > 1.0e-12) || !std::isfinite(S)) return false;
  const double diag[3] = {Qxx / S, Qyy / S, Qzz / S};
  for (double value : diag) {
    if (!std::isfinite(value) || value < -1.0e-10) return false;
  }
  const double scale =
      std::max({std::abs(diag[0]), std::abs(diag[1]), std::abs(diag[2])});
  if (!(scale > 1.0e-12)) return false;
  const double symbol[3] = {std::sin(wave[0] * h[0]) / h[0],
                            std::sin(wave[1] * h[1]) / h[1],
                            std::sin(wave[2] * h[2]) / h[2]};
  const double observed[3] = {Qxy / S, Qxz / S, Qyz / S};
  const double predicted_xy = symbol[0] * symbol[1];
  const double predicted_xz = symbol[0] * symbol[2];
  const double predicted_yz = symbol[1] * symbol[2];
  const double predicted[3] = {predicted_xy, predicted_xz, predicted_yz};
  for (int pair = 0; pair < 3; ++pair) {
    if (!std::isfinite(observed[pair]) ||
        std::abs(observed[pair] - predicted[pair]) > 0.10 * scale + 1.0e-10)
      return false;
  }
  return true;
}

inline bool finite_wave(const std::array<double, 3> &wave) {
  const double n2 = wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2];
  return std::isfinite(n2) && n2 > 1e-16;
}

inline double det_rows(const double R[3][3]) {
  return R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
         R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
         R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
}

inline std::array<double, 3> apply_row(const double R[3][3],
                                       const std::array<double, 3> &v) {
  return {R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
          R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
          R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]};
}

inline bool lex_greater(const std::array<double, 3> &a,
                        const std::array<double, 3> &b) {
  for (int axis : {2, 1, 0}) {
    if (a[static_cast<std::size_t>(axis)] > b[static_cast<std::size_t>(axis)] + kTie)
      return true;
    if (b[static_cast<std::size_t>(axis)] > a[static_cast<std::size_t>(axis)] + kTie)
      return false;
  }
  return false;
}

inline std::array<double, 3> cosine_gauge(std::array<double, 3> wave) {
  const double scale =
      1.0 + std::abs(wave[0]) + std::abs(wave[1]) + std::abs(wave[2]);
  for (int i = 0; i < 3; ++i) {
    if (wave[static_cast<std::size_t>(i)] > kTie * scale) return wave;
    if (wave[static_cast<std::size_t>(i)] < -kTie * scale) {
      return {-wave[0], -wave[1], -wave[2]};
    }
  }
  return wave;
}

inline std::array<double, 3> fold_about_z(std::array<double, 3> wave,
                                          double period) {
  const double kxy = std::hypot(wave[0], wave[1]);
  if (!(kxy > 0.0)) return wave;
  const double angle = std::atan2(wave[1], wave[0]);
  double wrapped = angle - period * std::floor(angle / period);
  if (wrapped < 0.0) wrapped = 0.0;
  // A representative on the closed end of the sector is the next copy of angle 0.
  if (wrapped >= period || period - wrapped <= 1e-12) wrapped = 0.0;
  wave[0] = kxy * std::cos(wrapped);
  wave[1] = kxy * std::sin(wrapped);
  return wave;
}

inline std::array<double, 3> fold_cubic(std::array<double, 3> wave) {
  static constexpr int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                      {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  std::array<double, 3> best = wave;
  bool have = false;
  for (const std::array<double, 3> &source :
       {wave, std::array<double, 3>{-wave[0], -wave[1], -wave[2]}}) {
    for (const auto &perm : perms) {
      for (int mask = 0; mask < 8; ++mask) {
        double R[3][3] = {};
        const double sign[3] = {(mask & 1) != 0 ? -1.0 : 1.0,
                                (mask & 2) != 0 ? -1.0 : 1.0,
                                (mask & 4) != 0 ? -1.0 : 1.0};
        for (int row = 0; row < 3; ++row) R[row][perm[row]] = sign[row];
        if (det_rows(R) <= 0.5) continue;
        const std::array<double, 3> image = apply_row(R, source);
        if (!have || lex_greater(image, best)) {
          best = image;
          have = true;
        }
      }
    }
  }
  return best;
}

inline double angle_of(const std::array<double, 3> &wave) {
  if (!(std::hypot(wave[0], wave[1]) > 0.0)) return 0.0;
  return std::atan2(wave[1], wave[0]);
}

inline std::array<double, 3> unit_wave(const std::array<double, 3> &wave) {
  const double n =
      std::sqrt(wave[0] * wave[0] + wave[1] * wave[1] + wave[2] * wave[2]);
  if (!(n > 0.0) || !std::isfinite(n)) return {0.0, 0.0, 0.0};
  return {wave[0] / n, wave[1] / n, wave[2] / n};
}

} // namespace crystal_detail

inline std::array<double, 3> representative_wavevector(std::array<double, 3> wave,
                                                       CrystalSymmetry symmetry) {
  wave = crystal_detail::cosine_gauge(wave);
  if (symmetry == CrystalSymmetry::four_fold) {
    wave = crystal_detail::fold_about_z(wave, std::numbers::pi / 2.0);
  } else if (symmetry == CrystalSymmetry::six_fold) {
    wave = crystal_detail::fold_about_z(wave, std::numbers::pi / 3.0);
  } else if (symmetry == CrystalSymmetry::cubic) {
    wave = crystal_detail::fold_cubic(wave);
  }
  return wave;
}

inline FrontSample planar_front_position(const OwnedDensity &density, MPI_Comm comm,
                                         const PlanarFrontOptions &options) {
  crystal_detail::require_comm(comm);
  crystal_detail::collective_validate(comm, [&] {
    crystal_detail::require_geometry(density);
    if (options.axis < 0 || options.axis > 2) {
      crystal_detail::fail("crystal observable: front axis must be 0, 1 or 2");
    }
    if (!std::isfinite(options.level)) {
      crystal_detail::fail("crystal observable: front level must be finite");
    }
    if (density.global_size[static_cast<std::size_t>(options.axis)] < 2) {
      crystal_detail::fail("crystal observable: front axis needs two stations");
    }
  });
  const int axis = options.axis;
  const int stations = density.global_size[static_cast<std::size_t>(axis)];
  std::vector<double> reduced(static_cast<std::size_t>(2 * stations), 0.0);
  const int nx = density.box.size[0];
  const int ny = density.box.size[1];
  const int nz = density.box.size[2];
  bool local_bad = false;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t local =
            static_cast<std::size_t>(i) +
            static_cast<std::size_t>(nx) *
                (static_cast<std::size_t>(j) +
                 static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const double u = density.values[local];
        if (!std::isfinite(u)) local_bad = true;
        const int global[3] = {density.box.low[0] + i, density.box.low[1] + j,
                               density.box.low[2] + k};
        const int station = global[axis];
        reduced[static_cast<std::size_t>(station)] += u * u;
        reduced[static_cast<std::size_t>(stations + station)] += 1.0;
      }
    }
  }
  if (MPI_Allreduce(MPI_IN_PLACE, reduced.data(), 2 * stations, MPI_DOUBLE, MPI_SUM,
                    comm) != MPI_SUCCESS) {
    crystal_detail::fail("crystal observable: MPI_Allreduce of the front failed");
  }
  FrontSample sample;
  if (local_bad) {
    sample.validity = FrontValidity::nonfinite;
    return sample;
  }
  std::vector<double> amplitude(static_cast<std::size_t>(stations));
  for (int s = 0; s < stations; ++s) {
    const double sum2 = reduced[static_cast<std::size_t>(s)];
    const double count = reduced[static_cast<std::size_t>(stations + s)];
    if (!std::isfinite(sum2) || !(count > 0.0)) {
      sample.validity = FrontValidity::nonfinite;
      return sample;
    }
    amplitude[static_cast<std::size_t>(s)] = std::sqrt(sum2 / count);
  }
  int rising = 0;
  int falling = 0;
  double crossing = 0.0;
  const double h = density.spacing[static_cast<std::size_t>(axis)];
  const double x0 = density.origin[static_cast<std::size_t>(axis)];
  for (int s = 0; s + 1 < stations; ++s) {
    const double a0 = amplitude[static_cast<std::size_t>(s)] - options.level;
    const double a1 = amplitude[static_cast<std::size_t>(s + 1)] - options.level;
    if (!std::isfinite(a0) || !std::isfinite(a1)) {
      sample.validity = FrontValidity::nonfinite;
      return sample;
    }
    if (a0 < 0.0 && a1 >= 0.0) {
      ++rising;
      crossing = x0 + (static_cast<double>(s) + a0 / (a0 - a1)) * h;
    } else if (a0 > 0.0 && a1 <= 0.0) {
      ++falling;
      crossing = x0 + (static_cast<double>(s) + a0 / (a0 - a1)) * h;
    }
  }
  const int requested = options.rising ? rising : falling;
  const int opposite = options.rising ? falling : rising;
  sample.crossings = requested;
  if (requested == 1 && opposite == 0) {
    sample.validity = FrontValidity::valid;
    sample.position = crossing;
    return sample;
  }
  sample.validity = (requested == 0 && opposite == 0)
                        ? FrontValidity::no_crossing
                        : FrontValidity::not_single_front;
  return sample;
}

inline OrientationSample region_orientation(const OwnedDensity &density,
                                            MPI_Comm comm, IndexWindow window,
                                            CrystalSymmetry symmetry) {
  crystal_detail::require_comm(comm);
  crystal_detail::collective_validate(comm, [&] {
    crystal_detail::require_geometry(density);
    crystal_detail::require_window(window, density.global_size);
  });
  const std::vector<double> full = crystal_detail::replicate(density, comm);
  double S = 0.0;
  double Qxx = 0.0, Qyy = 0.0, Qzz = 0.0, Qxy = 0.0, Qxz = 0.0, Qyz = 0.0;
  bool bad = false;
  for (int k = window.low[2]; k <= window.high[2]; ++k) {
    for (int j = window.low[1]; j <= window.high[1]; ++j) {
      for (int i = window.low[0]; i <= window.high[0]; ++i) {
        const auto d = crystal_detail::differentiate(full, density.global_size,
                                                     density.spacing, i, j, k);
        if (!std::isfinite(d.u) || !std::isfinite(d.xx) || !std::isfinite(d.yy) ||
            !std::isfinite(d.zz) || !std::isfinite(d.xy) || !std::isfinite(d.xz) ||
            !std::isfinite(d.yz)) {
          bad = true;
          continue;
        }
        S += d.u * d.u;
        Qxx += -d.u * d.xx;
        Qyy += -d.u * d.yy;
        Qzz += -d.u * d.zz;
        Qxy += -d.u * d.xy;
        Qxz += -d.u * d.xz;
        Qyz += -d.u * d.yz;
      }
    }
  }
  OrientationSample sample;
  if (bad || !(S > 0.0) || !std::isfinite(S)) return sample;
  const std::array<double, 3> estimated = crystal_detail::wave_from_moments(
      S, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, density.spacing);
  if (!crystal_detail::finite_wave(estimated)) return sample;
  const bool coherent = crystal_detail::coherent_single_mode(
      S, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, density.spacing, estimated);
  if (!coherent) return sample;
  sample.wavevector = representative_wavevector(estimated, symmetry);
  sample.angle_xy = crystal_detail::angle_of(sample.wavevector);
  sample.valid = crystal_detail::finite_wave(sample.wavevector) &&
                 std::isfinite(sample.angle_xy);
  if (!sample.valid) {
    sample.angle_xy = std::numeric_limits<double>::quiet_NaN();
  }
  return sample;
}

inline GrainStatistics
label_grains(const OwnedDensity &density, MPI_Comm comm, IndexWindow window,
             CrystalSymmetry symmetry,
             std::span<const std::array<double, 3>> candidates) {
  crystal_detail::require_comm(comm);
  crystal_detail::collective_validate(comm, [&] {
    crystal_detail::require_geometry(density);
    crystal_detail::require_window(window, density.global_size);
    if (candidates.empty())
      crystal_detail::fail("crystal observable: grain labeling needs a candidate");
  });
  const std::vector<double> full = crystal_detail::replicate(density, comm);
  GrainStatistics stats;
  stats.counts.assign(candidates.size(), 0);
  stats.fraction.assign(candidates.size(), 0.0);
  std::vector<std::array<double, 3>> refs;
  refs.reserve(candidates.size());
  for (const std::array<double, 3> &candidate : candidates) {
    const std::array<double, 3> rep = representative_wavevector(candidate, symmetry);
    refs.push_back(crystal_detail::unit_wave(rep));
    if (!crystal_detail::finite_wave(refs.back())) {
      crystal_detail::fail("crystal observable: a grain candidate has no direction");
    }
  }
  long long eligible_cells = 0;
  for (int k = window.low[2]; k <= window.high[2]; ++k) {
    for (int j = window.low[1]; j <= window.high[1]; ++j) {
      for (int i = window.low[0]; i <= window.high[0]; ++i) {
        const auto d = crystal_detail::differentiate(full, density.global_size,
                                                     density.spacing, i, j, k);
        if (!std::isfinite(d.u) || !(d.u * d.u > 1e-12)) continue;
        ++eligible_cells;
        const double S = d.u * d.u;
        const double Qxx = -d.u * d.xx, Qyy = -d.u * d.yy;
        const double Qzz = -d.u * d.zz, Qxy = -d.u * d.xy;
        const double Qxz = -d.u * d.xz, Qyz = -d.u * d.yz;
        const std::array<double, 3> estimated = crystal_detail::wave_from_moments(
            S, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, density.spacing);
        if (!crystal_detail::finite_wave(estimated)) continue;
        const bool coherent = crystal_detail::coherent_single_mode(
            S, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz, density.spacing, estimated);
        if (!coherent) continue;
        const std::array<double, 3> got = crystal_detail::unit_wave(
            representative_wavevector(estimated, symmetry));
        std::size_t best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (std::size_t g = 0; g < refs.size(); ++g) {
          const double dx = got[0] - refs[g][0];
          const double dy = got[1] - refs[g][1];
          const double dz = got[2] - refs[g][2];
          const double dist = dx * dx + dy * dy + dz * dz;
          if (dist < best_d) {
            best_d = dist;
            best = g;
          }
        }
        ++stats.counts[best];
        ++stats.voting_cells;
      }
    }
  }
  // Do not report confident grain fractions when the great majority of
  // points cannot be explained by any one local Fourier wave.
  const bool enough_votes = stats.voting_cells * 4 >= eligible_cells * 3;
  if (stats.voting_cells > 0 && enough_votes) {
    stats.valid = true;
    for (std::size_t g = 0; g < stats.counts.size(); ++g) {
      stats.fraction[g] = static_cast<double>(stats.counts[g]) /
                          static_cast<double>(stats.voting_cells);
    }
  }
  return stats;
}

} // namespace pfc::observables
