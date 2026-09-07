// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <openpfc/kernel/simulation/field_modifier.hpp>

namespace cahn_hilliard {

/** @brief Broadband grid noise with exactly reproducible global mean removal.
 * Hashes global cell indices, not rank-local RNG streams. Integer reduction
 * gives the same centering offset for every MPI decomposition. Amplitude is
 * an upper bound on |c-c0|, not an RMS value. Collective on the session comm.
 */
class SeededNoise : public pfc::FieldModifier {
public:
  double c0{0.32}, amplitude{0.02};
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
    if (!std::isfinite(c0) || !std::isfinite(amplitude) || c0 <= 0 || c0 >= 1 ||
        amplitude < 0 || amplitude >= std::min(c0, 1 - c0)) {
      throw std::invalid_argument(
          "seeded_noise: require 0<c0<1 and 0<=amplitude<min(c0,1-c0)");
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
      field.data()[i] = c0 + amplitude * ((field.data()[i] - mean) / 65535.0);
  }
};

inline void from_json(const nlohmann::json &j, SeededNoise &ic) {
  if (j.value("type", "") != "seeded_noise")
    throw std::invalid_argument("seeded_noise: incorrect type");
  ic.c0 = j.at("c0").get<double>();
  ic.amplitude = j.at("amplitude").get<double>();
  const auto &seed = j.at("seed");
  if (!seed.is_number_integer() ||
      (!seed.is_number_unsigned() && seed.get<std::int64_t>() < 0))
    throw std::invalid_argument("seeded_noise: seed must be a nonnegative integer");
  ic.seed = seed.get<std::uint64_t>();
}
} // namespace cahn_hilliard
