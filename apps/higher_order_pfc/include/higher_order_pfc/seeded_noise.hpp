// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file seeded_noise.hpp
 * @brief Broadband PFC seed whose mean is independent of the MPI decomposition.
 *
 * @details
 * Hashes global cell indices rather than running a rank-local RNG, and removes
 * the mean through an integer reduction, so the same seed gives the same field
 * on any number of ranks. Conserved dynamics never restores a mean that the
 * initial condition got wrong, so this matters more here than it looks:
 * \f$\bar\psi\f$ is a thermodynamic control parameter, not a detail.
 *
 * `amplitude` bounds \f$|\psi-\psi_0|\f$; it is not an RMS value. Collective on
 * the session communicator.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace higher_order_pfc {

class SeededNoise : public pfc::FieldModifier {
public:
  double psi0{0.0}, amplitude{0.01};
  std::uint64_t seed{42};

  const std::string &get_modifier_name() const override {
    static const std::string name{"SeededNoise"};
    return name;
  }

  void apply(pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double time) override {
    // No implicit MPI_COMM_WORLD: standalone use must cover the entire grid.
    const auto n = pfc::domain::get_size(domain);
    if (box.low != pfc::Int3{0, 0, 0} || box.size != n) {
      throw std::invalid_argument(
          "seeded_noise: distributed use needs SimulationContext");
    }
    apply(pfc::SimulationContext(MPI_COMM_SELF), field, domain, box, time);
  }

  void apply(const pfc::SimulationContext &ctx,
             pfc::field::FieldOutput<double> field, const pfc::Domain &domain,
             const pfc::Box3i &box, double) override {
    if (!std::isfinite(psi0) || !std::isfinite(amplitude) || amplitude < 0.0) {
      throw std::invalid_argument(
          "seeded_noise: require finite psi0 and amplitude >= 0");
    }
    const auto n = pfc::domain::get_size(domain);
    const double count = double(n[0]) * n[1] * n[2];
    if (count > double(std::numeric_limits<std::uint64_t>::max() / 65535)) {
      throw std::invalid_argument(
          "seeded_noise: grid too large for exact centering");
    }
    std::uint64_t sum = 0;
    std::size_t offset = 0;
    for (int k = box.low[2]; k <= box.high[2]; ++k)
      for (int j = box.low[1]; j <= box.high[1]; ++j)
        for (int i = box.low[0]; i <= box.high[0]; ++i) {
          std::uint64_t x =
              seed + std::uint64_t(i) +
              std::uint64_t(n[0]) *
                  (std::uint64_t(j) + std::uint64_t(n[1]) * std::uint64_t(k));
          // SplitMix64 finalizer; explicitly unsigned wraparound arithmetic.
          x += 0x9e3779b97f4a7c15ULL;
          x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
          x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
          const auto value = (x ^ (x >> 31)) >> 48;
          field.data()[offset++] = double(value);
          sum += value;
        }
    std::uint64_t global = 0;
    MPI_Allreduce(&sum, &global, 1, MPI_UINT64_T, MPI_SUM, ctx.mpi_comm());
    const double mean = double(global) / count;
    for (std::size_t i = 0; i < offset; ++i)
      field.data()[i] = psi0 + amplitude * ((field.data()[i] - mean) / 65535.0);
  }
};

inline void from_json(const nlohmann::json &j, SeededNoise &ic) {
  if (j.value("type", "") != "seeded_noise")
    throw std::invalid_argument("seeded_noise: incorrect type");
  ic.psi0 = j.at("psi0").get<double>();
  ic.amplitude = j.at("amplitude").get<double>();
  const auto &seed = j.at("seed");
  if (!seed.is_number_integer() ||
      (!seed.is_number_unsigned() && seed.get<std::int64_t>() < 0))
    throw std::invalid_argument("seeded_noise: seed must be a nonnegative integer");
  ic.seed = seed.get<std::uint64_t>();
}

} // namespace higher_order_pfc
