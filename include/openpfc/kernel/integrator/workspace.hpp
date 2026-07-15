// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file workspace.hpp
 * @brief Integrator-owned workspace for stage storage and scratch buffers
 *
 * @details
 * This header provides Workspace<T>, a value-semantic container for integrator-owned
 * storage including:
 *
 * - Stage storage: Intermediate RK stages or method-specific state
 * - Scratch buffers: Temporary workspace for operator evaluations
 *
 * Workspace<T> is owned by integrators and is not exposed to physics models or drivers.
 * Lifetime can be per-method-object, per-step, or pooled depending on integrator needs.
 *
 * Design rationale:
 * - Integrator-owned allocation: Integrators allocate, own, and manage all storage
 * - No exposure to models: Workspace is internal to integrators
 * - Reuse across steps: Avoid allocation overhead in time loops
 * - Clear lifetime: Storage lifetime is explicit and managed
 *
 * @see kernel/integrator/stage_context.hpp for MPI coordination context
 */

#include <vector>

#include <openpfc/kernel/data/world_types.hpp>

namespace pfc::integrator {

/**
 * @brief Integrator-owned workspace for stage storage and scratch buffers
 *
 * @details
 * Workspace<T> provides storage for:
 *
 * - Stage storage: Intermediate Runge-Kutta stages or method-specific state
 * - Scratch buffers: Temporary workspace for operator evaluations
 *
 * The workspace is owned by integrators and is not exposed to physics models or
 * drivers. Storage can be reused across time steps to avoid allocation overhead.
 *
 * Lifetime semantics:
 * - Per-method-object: Workspace lives as long as the integrator object
 * - Per-step: Workspace can be cleared/reclaimed between steps
 * - Pooled: Multiple workspaces can be managed by an execution layer
 *
 * @tparam T Field value type (e.g., double, std::complex<double>)
 */
template<typename T>
class Workspace {
public:
    /**
     * @brief Construct workspace with given extents and number of stages
     *
     * Allocates storage for:
     * - num_stages stage buffers, each sized to extents[0] * extents[1] * extents[2]
     * - One scratch buffer of the same size
     *
     * All storage is value-initialized (zero for arithmetic types).
     *
     * @param extents Grid dimensions (nx, ny, nz)
     * @param num_stages Number of stage buffers to allocate
     */
    explicit Workspace(const pfc::types::Int3& extents, std::size_t num_stages)
        : m_stage_size(static_cast<std::size_t>(extents[0]) *
                       static_cast<std::size_t>(extents[1]) *
                       static_cast<std::size_t>(extents[2]))
        , m_stages(num_stages)
        , m_scratch(m_stage_size)
    {
        // Value-initialize all stage buffers
        for (auto& stage : m_stages) {
            stage.resize(m_stage_size);
        }
    }

    /**
     * @brief Access stage storage by index
     *
     * @param stage_index Stage index (must be < num_stages)
     * @return T* Pointer to stage storage
     */
    T* stage(std::size_t stage_index) noexcept {
        return m_stages[stage_index].data();
    }

    /**
     * @brief Access stage storage by index (const overload)
     *
     * @param stage_index Stage index (must be < num_stages)
     * @return const T* Pointer to stage storage
     */
    const T* stage(std::size_t stage_index) const noexcept {
        return m_stages[stage_index].data();
    }

    /**
     * @brief Access scratch buffer
     *
     * @return T* Pointer to scratch buffer
     */
    T* scratch() noexcept {
        return m_scratch.data();
    }

    /**
     * @brief Access scratch buffer (const overload)
     *
     * @return const T* Pointer to scratch buffer
     */
    const T* scratch() const noexcept {
        return m_scratch.data();
    }

    /**
     * @brief Clear all buffers (set to zero)
     *
     * Resets all stage buffers and scratch buffer to zero.
     * Useful for reclaiming storage between time steps.
     */
    void clear() noexcept {
        for (auto& stage : m_stages) {
            std::fill(stage.begin(), stage.end(), T{});
        }
        std::fill(m_scratch.begin(), m_scratch.end(), T{});
    }

    /**
     * @brief Get number of stage buffers
     *
     * @return std::size_t Number of stage buffers
     */
    std::size_t stage_count() const noexcept {
        return m_stages.size();
    }

    /**
     * @brief Get size of each stage buffer
     *
     * @return std::size_t Number of elements in each stage buffer
     */
    std::size_t stage_size() const noexcept {
        return m_stage_size;
    }

private:
    std::size_t m_stage_size;
    std::vector<std::vector<T>> m_stages;
    std::vector<T> m_scratch;
};

} // namespace pfc::integrator
