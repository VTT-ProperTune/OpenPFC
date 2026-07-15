// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_workspace.hpp
 * @brief Multi-stage scratch workspace for Runge-Kutta time integrators.
 *
 * @details
 * `StageWorkspace<T>` manages scratch memory buffers for multi-stage
 * Runge-Kutta time integration methods (RK2, RK4, IMEX, etc.). Each stage
 * buffer (k₁, k₂, k₃, k₄, ...) stores intermediate derivative or increment
 * vectors computed during a single time step, following the ownership
 * semantics established in ADR 0003: integrators own their stage state,
 * not Model fields.
 *
 * The workspace is constructed with a fixed number of stages and a fixed
 * local buffer size (typically `local_size` from a `LocalField`). All
 * buffers are value-initialized to zero at construction and can be reset
 * to zero via `reset()`. Stage buffers are accessed by zero-based index
 * with bounds checking that throws `std::out_of_range` for invalid indices.
 *
 * Move semantics are supported to enable efficient stepper composition,
 * while copy operations are deleted to establish clear ownership semantics.
 * The template parameter `T` is constrained to floating-point types
 * (`float`, `double`) via `static_assert`.
 *
 * Typical usage in an RK4 stepper:
 *
 *     StageWorkspace<double> ws(4, local_size);  // k₁, k₂, k₃, k₄ buffers
 *     // ... compute k₁ into ws.stage(0) ...
 *     // ... compute k₂ into ws.stage(1) ...
 *     // ... compute k₃ into ws.stage(2) ...
 *     // ... compute k₄ into ws.stage(3) ...
 *     ws.reset();  // prepare for next time step
 *
 * This class is CPU-only and uses `std::vector<std::vector<T>>` storage.
 * Future extensions may add MPI-aware distribution or GPU device memory
 * variants, but the core API (stage access, reset, queries) will remain
 * compatible.
 *
 * @see openpfc/kernel/simulation/steppers/butcher_tableau.hpp for RK
 *      coefficient definitions
 * @see openpfc/kernel/simulation/steppers/euler.hpp for the single-stage
 *      EulerStepper pattern that owns its `m_du` buffer
 */

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace pfc::sim::steppers {

/**
 * @brief Multi-stage scratch workspace for Runge-Kutta time integrators.
 *
 * @tparam T Floating-point type (`float` or `double`) for stage buffer elements.
 */
template<typename T>
class StageWorkspace {
    static_assert(std::is_floating_point_v<T>,
                  "StageWorkspace<T> requires T to be a floating-point type");

public:
    /**
     * @brief Construct a workspace with specified stage count and buffer size.
     *
     * Allocates `num_stages` buffers, each of size `local_size`, with all
     * elements value-initialized to zero.
     *
     * @param num_stages Number of stage buffers to allocate (e.g., 4 for RK4).
     * @param local_size Size of each stage buffer in elements.
     */
    explicit StageWorkspace(std::size_t num_stages, std::size_t local_size)
        : m_num_stages(num_stages),
          m_local_size(local_size),
          m_stages(num_stages, std::vector<T>(local_size, T(0))) {}

    /**
     * @brief Move constructor: transfers ownership from another workspace.
     *
     * The moved-from workspace is left in a valid but unspecified state
     * with `num_stages() == 0` and `local_size() == 0`.
     *
     * @param other Workspace to move from.
     */
    StageWorkspace(StageWorkspace&& other) noexcept
        : m_num_stages(other.m_num_stages),
          m_local_size(other.m_local_size),
          m_stages(std::move(other.m_stages)) {
        other.m_num_stages = 0;
        other.m_local_size = 0;
    }

    /**
     * @brief Move assignment: transfers ownership from another workspace.
     *
     * The moved-from workspace is left in a valid but unspecified state
     * with `num_stages() == 0` and `local_size() == 0`.
     *
     * @param other Workspace to move from.
     * @return Reference to this workspace.
     */
    StageWorkspace& operator=(StageWorkspace&& other) noexcept {
        if (this != &other) {
            m_num_stages = other.m_num_stages;
            m_local_size = other.m_local_size;
            m_stages = std::move(other.m_stages);
            other.m_num_stages = 0;
            other.m_local_size = 0;
        }
        return *this;
    }

    /**
     * @brief Copy constructor is deleted to establish clear ownership.
     */
    StageWorkspace(const StageWorkspace&) = delete;

    /**
     * @brief Copy assignment is deleted to establish clear ownership.
     */
    StageWorkspace& operator=(const StageWorkspace&) = delete;

    /**
     * @brief Destructor (default).
     */
    ~StageWorkspace() = default;

    /**
     * @brief Access a mutable reference to the specified stage buffer.
     *
     * @param stage_index Zero-based stage index (0 to num_stages()-1).
     * @return Mutable reference to the stage buffer.
     * @throws std::out_of_range if `stage_index >= num_stages()`.
     */
    std::vector<T>& stage(std::size_t stage_index) {
        if (stage_index >= m_num_stages) {
            throw std::out_of_range("StageWorkspace::stage: stage_index out of range");
        }
        return m_stages[stage_index];
    }

    /**
     * @brief Access a const reference to the specified stage buffer.
     *
     * @param stage_index Zero-based stage index (0 to num_stages()-1).
     * @return Const reference to the stage buffer.
     * @throws std::out_of_range if `stage_index >= num_stages()`.
     */
    const std::vector<T>& stage(std::size_t stage_index) const {
        if (stage_index >= m_num_stages) {
            throw std::out_of_range("StageWorkspace::stage: stage_index out of range");
        }
        return m_stages[stage_index];
    }

    /**
     * @brief Reset all stage buffers to zero.
     *
     * Fills every element in every stage buffer with `T(0)`.
     */
    void reset() {
        for (auto& stage_buffer : m_stages) {
            std::fill(stage_buffer.begin(), stage_buffer.end(), T(0));
        }
    }

    /**
     * @brief Query the number of stage buffers.
     *
     * @return Number of stages (e.g., 4 for RK4).
     */
    std::size_t num_stages() const noexcept {
        return m_num_stages;
    }

    /**
     * @brief Query the size of each stage buffer.
     *
     * @return Number of elements in each stage buffer.
     */
    std::size_t local_size() const noexcept {
        return m_local_size;
    }

private:
    std::size_t m_num_stages;
    std::size_t m_local_size;
    std::vector<std::vector<T>> m_stages;
};

} // namespace pfc::sim::steppers
