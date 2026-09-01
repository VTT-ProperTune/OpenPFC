// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file tungsten_field_modifiers.hpp
 * @brief Host-buffer ICs and fixed BC for tungsten ETD sessions.
 *
 * Formulas match Gen-1 `Constant`, `SingleSeed`, and `FixedBC`. Callers pass
 * a host-accessible buffer (host `Field::data()` or a device field's host
 * mirror). Layout is x-fastest, halo 0, local owned box.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/initial_conditions/seed.hpp>

namespace tungsten {

inline std::size_t local_idx(const pfc::Box3i &box, int i, int j, int k) {
  return static_cast<std::size_t>(i) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(box.size[0]) +
         static_cast<std::size_t>(k) * static_cast<std::size_t>(box.size[0]) *
             static_cast<std::size_t>(box.size[1]);
}

inline pfc::Real3 local_coords(const pfc::Domain &domain, const pfc::Box3i &box,
                               int i, int j, int k) {
  const auto &o = pfc::domain::get_origin(domain);
  const auto &s = pfc::domain::get_spacing(domain);
  return {o[0] + static_cast<double>(box.low[0] + i) * s[0],
          o[1] + static_cast<double>(box.low[1] + j) * s[1],
          o[2] + static_cast<double>(box.low[2] + k) * s[2]};
}

inline void fill_constant(double *data, std::size_t n, double n0) {
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = n0;
  }
}

inline void fill_single_seed(const pfc::Domain &domain, const pfc::Box3i &box,
                             double *data, double amp, double rho) {
  const double s = 1.0 / std::sqrt(2.0);
  const double q[6][3] = {{s, s, 0},  {s, 0, s},  {0, s, s},
                          {s, 0, -s}, {s, -s, 0}, {0, s, -s}};
  const double r2 = 64.0 * 64.0;
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto c = local_coords(domain, box, i, j, k);
        const double x = c[0];
        const double y = c[1];
        const double z = c[2];
        if (x * x + y * y + z * z >= r2) {
          data[local_idx(box, i, j, k)] = 0.0;
          continue;
        }
        double u = rho;
        for (int qi = 0; qi < 6; ++qi) {
          u += 2.0 * amp * std::cos(q[qi][0] * x + q[qi][1] * y + q[qi][2] * z);
        }
        data[local_idx(box, i, j, k)] = u;
      }
    }
  }
}

/** Host-buffer SeedGrid; formulas and RNG match Gen-1 `pfc::SeedGrid`. */
inline void fill_seed_grid(const pfc::Domain &domain, const pfc::Box3i &box,
                           double *data, int Ny, int Nz, double X0, double radius,
                           double amplitude, double rho) {
  const auto size = pfc::domain::get_size(domain);
  const auto spacing = pfc::domain::get_spacing(domain);
  const double Dy =
      spacing[1] * static_cast<double>(size[1]) / static_cast<double>(Ny);
  const double Dz =
      spacing[2] * static_cast<double>(size[2]) / static_cast<double>(Nz);
  const double Y0 = Dy / 2.0;
  const double Z0 = Dz / 2.0;
  std::mt19937_64 re(42);
  std::uniform_real_distribution<double> rt(-0.2 * radius, 0.2 * radius);
  std::uniform_real_distribution<double> rr(0.0, 2.0 * std::numbers::pi);
  std::vector<pfc::Seed> seeds;
  seeds.reserve(static_cast<std::size_t>(Ny * Nz));
  for (int j = 0; j < Ny; ++j) {
    for (int k = 0; k < Nz; ++k) {
      const std::array<double, 3> location = {
          X0 + rt(re), Y0 + (Dy * static_cast<double>(j)) + rt(re),
          Z0 + (Dz * static_cast<double>(k)) + rt(re)};
      const std::array<double, 3> orientation = {rr(re), rr(re), rr(re)};
      seeds.emplace_back(location, orientation, radius, rho, amplitude);
    }
  }
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const auto c = local_coords(domain, box, i, j, k);
        double value = 0.0;
        for (const auto &seed : seeds) {
          if (seed.is_inside(c)) {
            value = seed.get_value(c);
            break;
          }
        }
        data[local_idx(box, i, j, k)] = value;
      }
    }
  }
}

inline void apply_ics_from_json(const nlohmann::json &settings,
                                const pfc::Domain &domain, const pfc::Box3i &box,
                                double *data, std::size_t n) {
  if (!settings.contains("initial_conditions") ||
      !settings["initial_conditions"].is_array()) {
    return;
  }
  for (const auto &ic : settings["initial_conditions"]) {
    const std::string type = ic.value("type", std::string{});
    if (type == "constant") {
      fill_constant(data, n, ic.at("n0").get<double>());
    } else if (type == "single_seed") {
      fill_single_seed(domain, box, data, ic.at("amp_eq").get<double>(),
                       ic.at("rho_seed").get<double>());
    } else if (type == "seed_grid") {
      fill_seed_grid(domain, box, data, ic.at("Ny").get<int>(),
                     ic.at("Nz").get<int>(), ic.at("X0").get<double>(),
                     ic.at("radius").get<double>(), ic.at("amplitude").get<double>(),
                     ic.at("rho").get<double>());
    } else {
      throw std::invalid_argument("tungsten: unsupported initial condition type '" +
                                  type + "'");
    }
  }
}

struct FixedBc {
  double rho_low{};
  double rho_high{};
};

inline std::optional<FixedBc> parse_fixed_bc(const nlohmann::json &settings) {
  if (!settings.contains("boundary_conditions") ||
      !settings["boundary_conditions"].is_array()) {
    return std::nullopt;
  }
  std::optional<FixedBc> out;
  for (const auto &bc : settings["boundary_conditions"]) {
    if (bc.value("type", std::string{}) == "fixed") {
      out = FixedBc{bc.at("rho_low").get<double>(), bc.at("rho_high").get<double>()};
    }
  }
  return out;
}

/** Directional-solidification moving BC; algorithm matches Gen-1 `MovingBC`. */
struct MovingBc {
  double rho_low{};
  double rho_high{};
  double width{15.0};
  double alpha{1.0};
  double disp{40.0};
  double xpos{0.0};
  double threshold{0.1};
  int idx{0};
  bool first{true};
  MPI_Comm comm{MPI_COMM_WORLD};
};

inline std::optional<MovingBc> parse_moving_bc(const nlohmann::json &settings,
                                               MPI_Comm comm) {
  if (!settings.contains("boundary_conditions") ||
      !settings["boundary_conditions"].is_array()) {
    return std::nullopt;
  }
  std::optional<MovingBc> out;
  for (const auto &bc : settings["boundary_conditions"]) {
    const std::string type = bc.value("type", std::string{});
    if (type == "fixed") {
      continue;
    }
    if (type != "moving") {
      throw std::invalid_argument("tungsten: unsupported boundary condition type '" +
                                  type + "'");
    }
    MovingBc m;
    m.rho_low = bc.at("rho_low").get<double>();
    m.rho_high = bc.at("rho_high").get<double>();
    m.width = bc.at("width").get<double>();
    m.alpha = bc.at("alpha").get<double>();
    m.disp = bc.at("disp").get<double>();
    m.xpos = bc.at("xpos").get<double>();
    m.comm = comm;
    out = m;
  }
  return out;
}

inline void apply_fixed_bc(const pfc::Domain &domain, const pfc::Box3i &box,
                           double *data, const FixedBc &bc) {
  const auto n = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);
  const double xwidth = 20.0;
  const double alpha = 1.0;
  const double xpos = static_cast<double>(n[0]) * dx[0] - xwidth;
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double x = local_coords(domain, box, i, j, k)[0];
        if (std::abs(x - xpos) < xwidth) {
          const double S = 1.0 / (1.0 + std::exp(-alpha * (x - xpos)));
          data[local_idx(box, i, j, k)] =
              (bc.rho_low * S) + (bc.rho_high * (1.0 - S));
        }
      }
    }
  }
}

inline void apply_moving_bc(const pfc::Domain &domain, const pfc::Box3i &box,
                            double *data, MovingBc &bc) {
  const auto n = pfc::domain::get_size(domain);
  const auto dx = pfc::domain::get_spacing(domain);
  const auto origin = pfc::domain::get_origin(domain);
  const int Lx = n[0];
  std::vector<double> xline(static_cast<std::size_t>(Lx),
                            std::numeric_limits<double>::lowest());
  std::vector<double> global_xline(static_cast<std::size_t>(Lx),
                                   std::numeric_limits<double>::lowest());
  const int nx = box.size[0];
  const int ny = box.size[1];
  const int nz = box.size[2];
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int gi = box.low[0] + i;
        if (gi >= 0 && gi < Lx) {
          xline[static_cast<std::size_t>(gi)] = std::max(
              xline[static_cast<std::size_t>(gi)], data[local_idx(box, i, j, k)]);
        }
      }
    }
  }
  pfc::mpi::throw_on_mpi_error(MPI_Reduce(xline.data(), global_xline.data(), Lx,
                                          MPI_DOUBLE, MPI_MAX, 0, bc.comm),
                               "MovingBc MPI_Reduce");
  int rank = 0;
  MPI_Comm_rank(bc.comm, &rank);
  if (rank == 0) {
    if (bc.first) {
      for (int i = Lx - 1; i >= 0; --i) {
        if (global_xline[static_cast<std::size_t>(i)] > bc.threshold) {
          bc.idx = i;
          break;
        }
      }
    } else {
      int scanned = 0;
      while (global_xline[static_cast<std::size_t>(bc.idx % Lx)] > bc.threshold &&
             scanned < Lx) {
        bc.idx += 1;
        scanned += 1;
      }
    }
  }
  const double new_xpos =
      origin[0] + (static_cast<double>(bc.idx) * dx[0]) + bc.disp;
  bc.xpos = std::max(new_xpos, bc.xpos);
  pfc::mpi::throw_on_mpi_error(MPI_Bcast(&bc.xpos, 1, MPI_DOUBLE, 0, bc.comm),
                               "MovingBc MPI_Bcast");
  bc.first = false;

  const double l = static_cast<double>(Lx) * dx[0];
  const double xpos = std::fmod(bc.xpos, l);
  const double xwidth = bc.width;
  const double alpha = bc.alpha;
  const double rho_low = bc.rho_low;
  const double rho_high = bc.rho_high;
  auto blend = [&](double d) {
    const double S = 1.0 / (1.0 + std::exp(-alpha * d));
    return (rho_low * S) + (rho_high * (1.0 - S));
  };
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const double x = local_coords(domain, box, i, j, k)[0];
        const double dist = x - xpos;
        double *cell = &data[local_idx(box, i, j, k)];
        if (std::abs(dist) < xwidth) {
          *cell = blend(dist);
        } else if (xpos < xwidth && std::abs(dist - l) < xwidth) {
          *cell = blend(dist - l);
        } else if (xpos > l - xwidth && std::abs(dist + l) < xwidth) {
          *cell = blend(dist + l);
        }
      }
    }
  }
}

} // namespace tungsten
