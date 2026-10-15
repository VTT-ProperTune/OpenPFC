// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file indexed_accumulate.hpp
 * @brief Indexed scatter-add and gather-add over a precomputed bin map.
 *
 * @details
 * A `BinMap` pairs a list of source indices with a list of bin indices:
 * entry `i` says that source cell `source[i]` contributes to bin `bin[i]`.
 * One source cell may feed several bins and one bin may receive many cells.
 * With per-entry weights `w`:
 *
 * - `scatter_add(out, in, map, w)`:      `out[bin[i]] += w[i] * in[source[i]]`
 * - `scatter_add_abs2(out, in, map, w)`: `out[bin[i]] += w[i] * |in[source[i]]|^2`
 *   for complex `in` (binning a power spectrum into radial or detector bins)
 * - `gather_add(out, g, map, w)`:        `out[source[i]] += w[i] * g[bin[i]]`
 *   (the adjoint of `scatter_add` with respect to `in`)
 *
 * The functions accumulate into `out`; they never clear it. Bounds are
 * checked once, when the map is built on the host: the map records one past
 * its largest source and bin index, and each call compares those against the
 * buffer sizes in O(1) and throws before any write. The call never reads the
 * index lists back from a device.
 *
 * This header defines the map and the host (`CPUTag`) path. Include
 * `openpfc/runtime/gpu/indexed_accumulate_gpu.hpp` for the CUDA and HIP
 * overloads, which have the same semantics. The host path accumulates in
 * entry order; the device paths use atomic adds, so duplicate bins sum in an
 * unspecified order and floating-point results may differ in the last bits.
 *
 * @code
 * using Tag = pfc::backend::CPUTag; // or CUDATag / HIPTag
 * pfc::core::BinMap<Tag> map(source_indices, bin_indices); // host spans
 * pfc::core::DataBuffer<Tag, double> power(n_bins);          // zero it first
 * pfc::core::scatter_add_abs2(power, spectrum, map, weights);
 * @endcode
 */

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>

namespace pfc::core {

/// Element types the indexed accumulation supports on every backend.
template <typename T>
concept AccumulateScalar = std::same_as<T, float> || std::same_as<T, double>;

/**
 * @brief Source-to-bin index lists, validated on the host and stored on the
 *        backend that runs the accumulation.
 *
 * @tparam BackendTag `CPUTag`, or `CUDATag` / `HIPTag` after including
 *         `runtime/gpu/databuffer_gpu.hpp`.
 */
template <typename BackendTag> class BinMap {
public:
  BinMap() = default;

  /**
   * @brief Build the map from host index lists of equal length.
   * @throws std::invalid_argument when the lists differ in length.
   */
  BinMap(std::span<const std::size_t> source, std::span<const std::size_t> bin)
      : m_source(source.size()), m_bin(bin.size()) {
    if (source.size() != bin.size()) {
      throw std::invalid_argument(
          "BinMap: source has " + std::to_string(source.size()) +
          " entries but bin has " + std::to_string(bin.size()));
    }
    if (!source.empty()) {
      m_source_end = *std::max_element(source.begin(), source.end()) + 1;
      m_bin_end = *std::max_element(bin.begin(), bin.end()) + 1;
      m_source.copy_from_host(source);
      m_bin.copy_from_host(bin);
    }
  }

  BinMap(const std::vector<std::size_t> &source, const std::vector<std::size_t> &bin)
      : BinMap(std::span<const std::size_t>(source),
               std::span<const std::size_t>(bin)) {}

  /// Number of (source, bin) entries.
  [[nodiscard]] std::size_t size() const noexcept { return m_source.size(); }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  /// One past the largest source index (0 for an empty map).
  [[nodiscard]] std::size_t source_end() const noexcept { return m_source_end; }
  /// One past the largest bin index (0 for an empty map).
  [[nodiscard]] std::size_t bin_end() const noexcept { return m_bin_end; }

  [[nodiscard]] const DataBuffer<BackendTag, std::size_t> &source() const noexcept {
    return m_source;
  }
  [[nodiscard]] const DataBuffer<BackendTag, std::size_t> &bin() const noexcept {
    return m_bin;
  }

private:
  DataBuffer<BackendTag, std::size_t> m_source;
  DataBuffer<BackendTag, std::size_t> m_bin;
  std::size_t m_source_end = 0;
  std::size_t m_bin_end = 0;
};

namespace detail {

/// Shared O(1) checks: the same messages on every backend.
inline void check_indexed_extents(const char *op, std::size_t entries,
                                  std::size_t weights, std::size_t source_end,
                                  std::size_t source_size, std::size_t bin_end,
                                  std::size_t bin_size) {
  if (weights != entries) {
    throw std::invalid_argument(std::string(op) + ": map has " +
                                std::to_string(entries) + " entries but w has " +
                                std::to_string(weights));
  }
  if (source_end > source_size) {
    throw std::out_of_range(std::string(op) + ": source index " +
                            std::to_string(source_end - 1) +
                            " out of range for size " + std::to_string(source_size));
  }
  if (bin_end > bin_size) {
    throw std::out_of_range(std::string(op) + ": bin index " +
                            std::to_string(bin_end - 1) + " out of range for size " +
                            std::to_string(bin_size));
  }
}

} // namespace detail

/// Host `out[bin[i]] += w[i] * in[source[i]]`.
template <AccumulateScalar T>
void scatter_add(DataBuffer<backend::CPUTag, T> &out,
                 const DataBuffer<backend::CPUTag, T> &in,
                 const BinMap<backend::CPUTag> &map,
                 const DataBuffer<backend::CPUTag, T> &w) {
  detail::check_indexed_extents("scatter_add", map.size(), w.size(),
                                map.source_end(), in.size(), map.bin_end(),
                                out.size());
  const std::size_t *src = map.source().data();
  const std::size_t *bin = map.bin().data();
  for (std::size_t i = 0; i < map.size(); ++i) {
    out[bin[i]] += w[i] * in[src[i]];
  }
}

/// Host `out[bin[i]] += w[i] * |in[source[i]]|^2` for complex @p in.
template <AccumulateScalar T>
void scatter_add_abs2(DataBuffer<backend::CPUTag, T> &out,
                      const DataBuffer<backend::CPUTag, std::complex<T>> &in,
                      const BinMap<backend::CPUTag> &map,
                      const DataBuffer<backend::CPUTag, T> &w) {
  detail::check_indexed_extents("scatter_add_abs2", map.size(), w.size(),
                                map.source_end(), in.size(), map.bin_end(),
                                out.size());
  const std::size_t *src = map.source().data();
  const std::size_t *bin = map.bin().data();
  for (std::size_t i = 0; i < map.size(); ++i) {
    const std::complex<T> z = in[src[i]];
    out[bin[i]] += w[i] * (z.real() * z.real() + z.imag() * z.imag());
  }
}

/// Host `out[source[i]] += w[i] * g[bin[i]]` (adjoint of `scatter_add`).
template <AccumulateScalar T>
void gather_add(DataBuffer<backend::CPUTag, T> &out,
                const DataBuffer<backend::CPUTag, T> &g,
                const BinMap<backend::CPUTag> &map,
                const DataBuffer<backend::CPUTag, T> &w) {
  detail::check_indexed_extents("gather_add", map.size(), w.size(), map.source_end(),
                                out.size(), map.bin_end(), g.size());
  const std::size_t *src = map.source().data();
  const std::size_t *bin = map.bin().data();
  for (std::size_t i = 0; i < map.size(); ++i) {
    out[src[i]] += w[i] * g[bin[i]];
  }
}

} // namespace pfc::core
