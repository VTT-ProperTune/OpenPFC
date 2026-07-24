// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file state_access.hpp
 * @brief Value-semantic field access primitives for state representation
 *
 * @details
 * This header provides minimal value-semantic types for field state access:
 *
 * - `FieldView<T>`: Read-only view of field data for operator inputs
 * - `FieldOutput<T>`: Mutable output storage for operator results
 * - `FieldBundle<Ts...>`: Multi-field bundle for coupled systems
 *
 * These types provide read/write contracts without virtual dispatch and are
 * backend-agnostic (work with CPU and GPU storage). FieldView<T> holds only
 * a const pointer to data and geometry metadata, enabling single-header usage
 * across backends. Backend-specific validation is provided in separate headers
 * (e.g., validation_cuda.hpp) when GPU implementation is pursued.
 *
 * Design rationale:
 * - Value semantics: No virtual dispatch, no heap allocation, copyable
 * - Backend-agnostic: FieldView<T> works with any contiguous storage
 * - Minimal overhead: Thin wrappers around pointers and metadata
 * - Clear contracts: Const views for input, mutable outputs for results
 *
 * @see kernel/field/validation.hpp for validation utilities
 * @see kernel/integrator/workspace.hpp for integrator-owned storage
 */

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/world_types.hpp>

namespace pfc::field {

/**
 * @brief Read-only view of field data for operator inputs
 *
 * @details
 * FieldView<T> provides const access to field data and geometry metadata.
 * It is backend-agnostic and works with any contiguous storage (CPU std::vector,
 * GPU GPUVector, etc.). The view does not own the underlying data.
 *
 * Geometry metadata includes:
 * - Extents: Grid dimensions (nx, ny, nz)
 * - Spacing: Physical spacing per axis (dx, dy, dz)
 * - Origin: Physical origin of the grid (x0, y0, z0)
 *
 * @tparam T Field value type (e.g., double, std::complex<double>)
 */
template<typename T>
class FieldView {
public:
    /**
     * @brief Construct a field view from data pointer and geometry
     *
     * @param data Pointer to field data (must remain valid for view lifetime)
     * @param size Number of elements in the field data
     * @param extents Grid dimensions (nx, ny, nz)
     * @param spacing Physical spacing per axis (dx, dy, dz)
     * @param origin Physical origin of the grid (x0, y0, z0)
     */
    FieldView(const T* data, std::size_t size,
              const pfc::types::Int3& extents,
              const pfc::types::Real3& spacing,
              const pfc::types::Real3& origin) noexcept
        : m_data(data)
        , m_size(size)
        , m_extents(extents)
        , m_spacing(spacing)
        , m_origin(origin)
    {}

    /**
     * @brief Get const pointer to field data
     *
     * @return const T* Pointer to field data
     */
    const T* data() const noexcept { return m_data; }

    /**
     * @brief Get number of elements in the field data
     *
     * @return std::size_t Number of elements
     */
    std::size_t size() const noexcept { return m_size; }

    /**
     * @brief Get grid extents
     *
     * @return pfc::types::Int3 Grid dimensions (nx, ny, nz)
     */
    pfc::types::Int3 extents() const noexcept { return m_extents; }

    /**
     * @brief Get physical spacing per axis
     *
     * @return pfc::types::Real3 Spacing (dx, dy, dz)
     */
    pfc::types::Real3 spacing() const noexcept { return m_spacing; }

    /**
     * @brief Get physical origin of the grid
     *
     * @return pfc::types::Real3 Origin (x0, y0, z0)
     */
    pfc::types::Real3 origin() const noexcept { return m_origin; }

    /**
     * @brief Get the local index box of the view
     *
     * @return pfc::Box3i Index box [0, extents-1] in integer coordinates
     */
    [[nodiscard]] pfc::Box3i box() const noexcept {
        return pfc::Box3i::from_bounds({0, 0, 0},
                                       {m_extents[0] - 1, m_extents[1] - 1, m_extents[2] - 1});
    }

    /**
     * @brief Check if this field is compatible with another field
     *
     * Two fields are compatible if they have matching extents, spacing, and origin.
     * This is a structural check that ensures fields can be used together in
     * spatial operators without layout mismatches.
     *
     * @param other Other field view to compare with
     * @return true if fields are compatible, false otherwise
     */
    bool is_compatible_with(const FieldView& other) const noexcept {
        return m_extents == other.m_extents &&
               m_spacing == other.m_spacing &&
               m_origin == other.m_origin;
    }

private:
    const T* m_data;
    std::size_t m_size;
    pfc::types::Int3 m_extents;
    pfc::types::Real3 m_spacing;
    pfc::types::Real3 m_origin;
};

/**
 * @brief Mutable output storage for operator results
 *
 * @details
 * FieldOutput<T> provides mutable access to caller-provided storage for
 * operator results. It includes aliasing validation to ensure output storage
 * does not alias input fields, preventing subtle bugs from in-place mutation.
 *
 * The output does not own the underlying data; it is a view into caller-provided
 * storage. This allows operators to write into existing buffers without allocation.
 *
 * @tparam T Field value type (e.g., double, std::complex<double>)
 */
template<typename T>
class FieldOutput {
public:
    /**
     * @brief Construct field output from mutable data pointer
     *
     * @param data Pointer to output storage (must remain valid for output lifetime)
     * @param size Number of elements in the output storage
     */
    FieldOutput(T* data, std::size_t size) noexcept
        : m_data(data)
        , m_size(size)
    {}

    /**
     * @brief Get mutable pointer to output storage
     *
     * @return T* Pointer to output storage
     */
    T* data() noexcept { return m_data; }

    /**
     * @brief Get number of elements in the output storage
     *
     * @return std::size_t Number of elements
     */
    std::size_t size() const noexcept { return m_size; }

    /**
     * @brief Validate that output storage does not alias input view
     *
     * Performs pointer comparison to ensure output and input refer to distinct
     * memory regions. Throws std::invalid_argument if aliasing is detected.
     *
     * This check prevents subtle bugs where in-place mutation would produce
     * incorrect results. Documented in-place patterns (e.g., u += dt*du via
     * ScaledField) are explicitly supported and should not use this validation.
     *
     * @tparam InputView Type of input field view
     * @param input Input field view to check for aliasing
     * @throws std::invalid_argument if output storage aliases input storage
     */
    template<typename InputView>
    void validate_no_alias(const InputView& input) const {
        // Perform pointer comparison to detect aliasing
        // Cast to void* to avoid strict aliasing warnings
        const void* output_ptr = static_cast<const void*>(m_data);
        const void* input_ptr = static_cast<const void*>(input.data());

        // Check if pointers overlap by comparing ranges
        const std::size_t output_size = m_size * sizeof(T);
        const std::size_t input_size = input.size() * sizeof(T);

        const unsigned char* output_start = static_cast<const unsigned char*>(output_ptr);
        const unsigned char* output_end = output_start + output_size;
        const unsigned char* input_start = static_cast<const unsigned char*>(input_ptr);
        const unsigned char* input_end = input_start + input_size;

        // Check for overlap
        if (!(output_end <= input_start || input_end <= output_start)) {
            throw std::invalid_argument(
                "FieldOutput::validate_no_alias: output storage aliases input storage");
        }
    }

private:
    T* m_data;
    std::size_t m_size;
};

/**
 * @brief Multi-field bundle for coupled systems
 *
 * @details
 * FieldBundle<Ts...> groups multiple fields (e.g., u/v for wave equation)
 * and provides coordinated access and validation. This is useful for coupled
 * PDE systems where multiple fields must evolve together with consistent geometry.
 *
 * The bundle stores fields by value and provides indexed access via get<I>().
 * All fields in the bundle must have compatible shapes for the system to be
 * well-posed.
 *
 * @tparam Fields Types of fields in the bundle (e.g., FieldView<double>, FieldView<double>)
 */
template<typename... Fields>
class FieldBundle {
public:
    /**
     * @brief Construct a field bundle from multiple fields
     *
     * @param fields Fields to include in the bundle
     */
    explicit FieldBundle(Fields... fields) noexcept
        : m_fields(std::move(fields)...)
    {}

    /**
     * @brief Access field by index
     *
     * @tparam I Index of the field to access
     * @return Reference to the field at index I
     */
    template<std::size_t I>
    auto& get() noexcept {
        return std::get<I>(m_fields);
    }

    /**
     * @brief Access field by index (const overload)
     *
     * @tparam I Index of the field to access
     * @return Const reference to the field at index I
     */
    template<std::size_t I>
    const auto& get() const noexcept {
        return std::get<I>(m_fields);
    }

    /**
     * @brief Validate that all fields in the bundle have compatible shapes
     *
     * Checks that all fields have matching extents, spacing, and origin.
     * Returns false if the bundle is empty or if any field is incompatible.
     *
     * @return true if all fields are compatible, false otherwise
     */
    bool validate_shapes() const noexcept {
        if constexpr (sizeof...(Fields) == 0) {
            return true;  // Empty bundle is trivially valid
        } else if constexpr (sizeof...(Fields) == 1) {
            return true;  // Single field is trivially valid
        } else {
            // Check that all fields are compatible with the first field
            return validate_shapes_impl<1>();
        }
    }

private:
    std::tuple<Fields...> m_fields;

    // Recursive implementation for shape validation
    template<std::size_t I>
    bool validate_shapes_impl() const noexcept {
        if constexpr (I < sizeof...(Fields)) {
            const auto& first = std::get<0>(m_fields);
            const auto& current = std::get<I>(m_fields);
            if (!first.is_compatible_with(current)) {
                return false;
            }
            return validate_shapes_impl<I + 1>();
        } else {
            return true;
        }
    }
};

} // namespace pfc::field
