// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file workspace.hpp
 * @brief Integrator-owned workspace for stage storage and scratch buffers
 *
 * @details
 * `pfc::integrator::Workspace<T>` is the one workspace type. Integrators own
 * stage buffers (RK increments) plus one scratch buffer. Models never see it.
 *
 * `pfc::sim::steppers::StageWorkspace<T>` is an alias of this type
 * (`stage_workspace.hpp`).
 *
 * Two constructors:
 * - `(num_stages, local_size)` — stepper local-buffer form
 * - `(extents, num_stages)` — grid-extents form (`local_size = nx*ny*nz`)
 *
 * `stage(i)` is bounds-checked and returns the stage `std::vector<T>`.
 * `clear()` / `reset()` zero every stage and the scratch buffer.
 * The type is move-only.
 *
 * @see kernel/integrator/stage_context.hpp for MPI coordination context
 */

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/types.hpp>

namespace pfc::integrator {

/**
 * @brief Integrator-owned workspace for stage storage and scratch buffers
 *
 * @tparam T Field value type (e.g., double, std::complex<double>)
 */
template <typename T>
class Workspace {
public:
  /**
   * @brief Construct `num_stages` buffers plus one scratch, each `local_size`.
   */
  explicit Workspace(std::size_t num_stages, std::size_t local_size)
      : m_num_stages(num_stages), m_local_size(local_size),
        m_stages(num_stages, std::vector<T>(local_size, T{})),
        m_scratch(local_size, T{}) {}

  /**
   * @brief Construct from grid extents (`local_size = nx * ny * nz`).
   */
  explicit Workspace(const pfc::types::Int3 &extents, std::size_t num_stages)
      : Workspace(num_stages, static_cast<std::size_t>(extents[0]) *
                                  static_cast<std::size_t>(extents[1]) *
                                  static_cast<std::size_t>(extents[2])) {}

  Workspace(Workspace &&other) noexcept
      : m_num_stages(other.m_num_stages), m_local_size(other.m_local_size),
        m_stages(std::move(other.m_stages)),
        m_scratch(std::move(other.m_scratch)) {
    other.m_num_stages = 0;
    other.m_local_size = 0;
  }

  Workspace &operator=(Workspace &&other) noexcept {
    if (this != &other) {
      m_num_stages = other.m_num_stages;
      m_local_size = other.m_local_size;
      m_stages = std::move(other.m_stages);
      m_scratch = std::move(other.m_scratch);
      other.m_num_stages = 0;
      other.m_local_size = 0;
    }
    return *this;
  }

  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  ~Workspace() = default;

  /**
   * @brief Mutable stage buffer (`stage_index` in `[0, num_stages)`).
   */
  std::vector<T> &stage(std::size_t stage_index) {
    if (stage_index >= m_num_stages) {
      throw std::out_of_range("Workspace::stage: stage_index out of range");
    }
    return m_stages[stage_index];
  }

  /**
   * @brief Const stage buffer (`stage_index` in `[0, num_stages)`).
   */
  const std::vector<T> &stage(std::size_t stage_index) const {
    if (stage_index >= m_num_stages) {
      throw std::out_of_range("Workspace::stage: stage_index out of range");
    }
    return m_stages[stage_index];
  }

  /** Scratch buffer, same length as each stage. */
  std::vector<T> &scratch() noexcept { return m_scratch; }

  /** Const scratch buffer. */
  const std::vector<T> &scratch() const noexcept { return m_scratch; }

  /** Zero every stage and the scratch buffer. */
  void clear() noexcept {
    for (auto &stage_buffer : m_stages) {
      std::fill(stage_buffer.begin(), stage_buffer.end(), T{});
    }
    std::fill(m_scratch.begin(), m_scratch.end(), T{});
  }

  /** Alias of `clear()` (historical `StageWorkspace` name). */
  void reset() { clear(); }

  [[nodiscard]] std::size_t stage_count() const noexcept { return m_num_stages; }

  /** Alias of `stage_count()` (historical `StageWorkspace` name). */
  [[nodiscard]] std::size_t num_stages() const noexcept { return m_num_stages; }

  [[nodiscard]] std::size_t stage_size() const noexcept { return m_local_size; }

  /** Alias of `stage_size()` (historical `StageWorkspace` name). */
  [[nodiscard]] std::size_t local_size() const noexcept { return m_local_size; }

private:
  std::size_t m_num_stages;
  std::size_t m_local_size;
  std::vector<std::vector<T>> m_stages;
  std::vector<T> m_scratch;
};

} // namespace pfc::integrator
