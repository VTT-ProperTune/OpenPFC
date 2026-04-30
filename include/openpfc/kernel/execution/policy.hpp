// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file policy.hpp
 * @brief Kokkos-compatible execution policies
 *
 * @details
 * Policies define the iteration space for parallel_for. Names and constructor
 * style match Kokkos.
 *
 * - RangePolicy: 1D range [begin, end) or [0, count)
 * - MDRangePolicy: N-D iteration space [start_i, end_i) per dimension
 *
 * @see parallel.hpp for parallel_for
 * @see execution_space.hpp for execution space tags
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <array>
#include <cstddef>
#include <openpfc/kernel/execution/execution_space.hpp>

namespace pfc {

/**
 * @brief Rank tag for MDRangePolicy (Kokkos-compatible)
 * @tparam N Number of dimensions
 */
template <std::size_t N> struct Rank {
  static constexpr std::size_t value = N;
};

/**
 * @brief 1D range policy (Kokkos-compatible)
 *
 * Iteration space [m_begin, m_end). Can be constructed as
 * RangePolicy<ES>(begin, end) or RangePolicy<ES>(count) for [0, count).
 *
 * @tparam ExecutionSpace Serial, OpenMP, Cuda, or HIP
 * @tparam IndexType Integer type for indices (default std::size_t)
 */
template <typename ExecutionSpace, typename IndexType = std::size_t>
struct RangePolicy {
  using execution_space = ExecutionSpace;
  using index_type = IndexType;

  index_type m_begin = 0;
  index_type m_end = 0;

  RangePolicy() = default;

  /** @brief Range [begin, end) */
  RangePolicy(index_type begin, index_type end) : m_begin(begin), m_end(end) {}

  /** @brief Range [0, count) */
  explicit RangePolicy(index_type count) : m_begin(0), m_end(count) {}

  index_type begin() const { return m_begin; }
  index_type end() const { return m_end; }
  index_type size() const { return m_end > m_begin ? m_end - m_begin : 0; }
};

/**
 * @brief Multi-dimensional range policy (Kokkos-compatible)
 *
 * Iteration space [start_i, end_i) for each dimension i.
 *
 * @tparam ExecutionSpace Serial, OpenMP, Cuda, or HIP
 * @tparam RankTag Rank<N> for N dimensions
 * @tparam IndexType Integer type for indices
 */
template <typename ExecutionSpace, typename RankTag,
          typename IndexType = std::size_t>
struct MDRangePolicy {
  static constexpr std::size_t rank = RankTag::value;
  using execution_space = ExecutionSpace;
  using index_type = IndexType;

  std::array<index_type, rank> m_start{};
  std::array<index_type, rank> m_end{};

  MDRangePolicy() = default;

  /** @brief Set bounds for each dimension [start_i, end_i) */
  MDRangePolicy(const std::array<index_type, rank> &start,
                const std::array<index_type, rank> &end)
      : m_start(start), m_end(end) {}

  /** @brief 2D: [start0,end0) x [start1,end1) */
  template <std::size_t R = rank>
    requires(R == 2)
  MDRangePolicy(index_type start0, index_type end0, index_type start1,
                index_type end1)
      : m_start{{start0, start1}}, m_end{{end0, end1}} {}

  /** @brief 3D: [start0,end0) x [start1,end1) x [start2,end2) */
  template <std::size_t R = rank>
    requires(R == 3)
  MDRangePolicy(index_type start0, index_type end0, index_type start1,
                index_type end1, index_type start2, index_type end2)
      : m_start{{start0, start1, start2}}, m_end{{end0, end1, end2}} {}

  const std::array<index_type, rank> &start() const { return m_start; }
  const std::array<index_type, rank> &end() const { return m_end; }

  index_type start(std::size_t dim) const { return m_start[dim]; }
  index_type end(std::size_t dim) const { return m_end[dim]; }
};

} // namespace pfc
