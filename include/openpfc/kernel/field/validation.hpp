// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file validation.hpp
 * @brief Validation utilities for field state access
 *
 * @details
 * This header provides validation functions for field state access:
 *
 * - `validate_shape_compatibility()`: Check that two fields have compatible shapes
 * - `validate_no_alias()`: Check that output storage does not alias input storage
 *
 * These functions perform runtime validation at binding time to ensure field
 * operations are well-posed. Validation is performed once when fields are bound
 * to operators, not per-evaluation, to avoid runtime overhead.
 *
 * Backend memory-space compatibility validation is deferred to separate
 * backend-specific headers (e.g., validation_cuda.hpp, validation_hip.hpp) when
 * GPU implementation is pursued. FieldView<T> is intentionally backend-agnostic
 * to allow single header usage across CPU and GPU backends.
 *
 * @see kernel/field/state_access.hpp for field access primitives
 */

#include <stdexcept>
#include <string>

#include <openpfc/kernel/field/state_access.hpp>

namespace pfc::field {

/**
 * @brief Validate that two fields have compatible shapes
 *
 * Checks that extents, spacing, and origin match between two fields.
 * Throws std::invalid_argument if incompatible.
 *
 * Shape compatibility ensures that fields can be used together in spatial
 * operators without layout mismatches. This check is performed at binding
 * time, not during evaluation, to avoid per-call overhead.
 *
 * @tparam T Field value type
 * @param field1 First field view
 * @param field2 Second field view
 * @throws std::invalid_argument if shapes are incompatible
 */
template<typename T>
void validate_shape_compatibility(const FieldView<T>& field1,
                                   const FieldView<T>& field2) {
    if (!field1.is_compatible_with(field2)) {
        throw std::invalid_argument(
            "validate_shape_compatibility: fields have incompatible shapes\n"
            "  Field1 extents: [" + 
            std::to_string(field1.extents()[0]) + ", " +
            std::to_string(field1.extents()[1]) + ", " +
            std::to_string(field1.extents()[2]) + "]\n"
            "  Field2 extents: [" + 
            std::to_string(field2.extents()[0]) + ", " +
            std::to_string(field2.extents()[1]) + ", " +
            std::to_string(field2.extents()[2]) + "]\n"
            "  Field1 spacing: [" + 
            std::to_string(field1.spacing()[0]) + ", " +
            std::to_string(field1.spacing()[1]) + ", " +
            std::to_string(field1.spacing()[2]) + "]\n"
            "  Field2 spacing: [" + 
            std::to_string(field2.spacing()[0]) + ", " +
            std::to_string(field2.spacing()[1]) + ", " +
            std::to_string(field2.spacing()[2]) + "]\n"
            "  Field1 origin: [" + 
            std::to_string(field1.origin()[0]) + ", " +
            std::to_string(field1.origin()[1]) + ", " +
            std::to_string(field1.origin()[2]) + "]\n"
            "  Field2 origin: [" + 
            std::to_string(field2.origin()[0]) + ", " +
            std::to_string(field2.origin()[1]) + ", " +
            std::to_string(field2.origin()[2]) + "]");
    }
}

/**
 * @brief Validate that output storage does not alias input storage
 *
 * Performs pointer comparison to ensure output and input refer to
 * distinct memory regions. Throws std::invalid_argument if aliasing detected.
 *
 * Aliasing validation prevents subtle bugs where in-place mutation would
 * produce incorrect results. Documented in-place patterns (e.g., u += dt*du
 * via ScaledField) are explicitly supported and should not use this validation.
 *
 * @tparam T Field value type
 * @tparam InputView Type of input field view (must have data() and size() methods)
 * @param output Output field storage
 * @param input Input field view
 * @throws std::invalid_argument if aliasing detected
 */
template<typename T, typename InputView>
void validate_no_alias(const FieldOutput<T>& output,
                        const InputView& input) {
    output.validate_no_alias(input);
}

// Note: Backend memory-space compatibility validation is deferred to
// separate backend-specific headers (e.g., validation_cuda.hpp, validation_hip.hpp)
// when GPU implementation is pursued. FieldView<T> is intentionally backend-agnostic
// to allow single header usage across CPU and GPU backends without requiring
// BackendTag template parameters that cannot be obtained from FieldView arguments.
//
// Backend-specific headers will wrap the generic validation utilities with
// backend-aware checks when needed. For example:
//
// // include/openpfc/kernel/field/validation_cuda.hpp (future)
// namespace pfc::field {
// template<typename T>
// void validate_backend_compatibility(const FieldView<T>& field1,
//                                      const FieldView<T>& field2) {
//     // Backend-specific checks for CUDA storage
//     // Throws if fields use different backends (CPU vs GPU)
// }
// }

} // namespace pfc::field
