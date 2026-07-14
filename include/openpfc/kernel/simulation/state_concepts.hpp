// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file state_concepts.hpp
 * @brief C++20 concepts for field state access and storage contracts.
 *
 * @details
 * These concepts define the minimum semantic requirements for scalar and
 * coupled real/complex field state and integrator workspace access. They are
 * evidence-driven concepts based on existing field implementations
 * (LocalField<T>, PaddedBrick<T>) and enable compile-time validation of
 * read/write access, shape compatibility, and aliasing safety.
 *
 * The concepts enforce:
 * - Read access for operator inputs (FieldReadable)
 * - Write access for outputs (FieldWritable)
 * - Shape and value_type compatibility (ShapeCompatible)
 * - Aliasing validation (AliasingSafe)
 *
 * FieldReadable requires const access via size(), data() returning const void*,
 * and operator()(int,int,int) const returning const reference to value type.
 *
 * FieldWritable requires mutable access via size(), data() returning void*,
 * operator()(int,int,int) returning lvalue_reference, and explicitly requires
 * the operator() return type be non-const via !std::is_const_v removal.
 *
 * Element type is captured compositionally through operator() return types.
 * ShapeCompatible additionally requires value_type members for explicit
 * type checking.
 *
 * Usage example:
 * @code
 * template <FieldWritable F, ConstField G>
 * void solve(F& out, const G& in) {
 *     // Compile-time guarantees: out supports write, in supports read
 *     static_assert(ShapeCompatible<F, G>);
 *     if constexpr (AliasingSafe<F, G>) {
 *         // Can safely perform in-place operations
 *     }
 * }
 * @endcode
 */

#include <concepts>
#include <type_traits>
#include <cstddef>

namespace pfc::field {

/**
 * @brief Concept requiring read access to field data.
 *
 * Requires:
 * - size() const returning std::size_t
 * - data() const returning type convertible to const void*
 * - operator()(int,int,int) const returning const reference to value type
 *
 * @tparam F Field type to check for readable access
 */
template <class F>
concept FieldReadable = requires(const F& f) {
    { f.size() } -> std::convertible_to<std::size_t>;
    { f.data() } -> std::convertible_to<const void*>;
    { f.operator()(int{}, int{}, int{}) } -> std::same_as<const std::remove_reference_t<decltype(f.operator()(int{}, int{}, int{}))>&>;
};

/**
 * @brief Concept requiring write access to field data.
 *
 * Requires:
 * - size() const returning std::size_t
 * - data() returning type convertible to void*
 * - operator()(int,int,int) returning non-const lvalue_reference
 * - Explicit requirement that operator() return type is non-const via
 *   !std::is_const_v<std::remove_reference_t<...>>
 *
 * @tparam F Field type to check for writable access
 */
template <class F>
concept FieldWritable = requires(F& f) {
    { f.size() } -> std::convertible_to<std::size_t>;
    { f.data() } -> std::convertible_to<void*>;
    requires !std::is_const_v<std::remove_reference_t<decltype(f.operator()(int{}, int{}, int{}))>>;
    requires std::is_lvalue_reference_v<decltype(f.operator()(int{}, int{}, int{}))>;
};

/**
 * @brief Concept requiring both read and write access to field data.
 *
 * Composes FieldReadable and FieldWritable requirements.
 *
 * @tparam F Field type to check for full field access
 */
template <class F>
concept Field = FieldReadable<F> && FieldWritable<F>;

/**
 * @brief Concept alias for read-only field access.
 *
 * Equivalent to FieldReadable - provides semantic clarity for const field
 * usage in operator evaluation contexts.
 *
 * @tparam F Field type to check for read-only access
 */
template <class F>
concept ConstField = FieldReadable<F>;

/**
 * @brief Concept requiring shape and value_type compatibility between two fields.
 *
 * Requires:
 * - Both types have size() const returning std::size_t
 * - a.size() == b.size() at compile time
 * - Both types have value_type member
 * - Both value_type members are the same type
 *
 * Shape verification ensures fields used with same spatial operators have
 * compatible layout for safe element access.
 *
 * @tparam A First field type
 * @tparam B Second field type
 */
template <class A, class B>
concept ShapeCompatible = requires(const A& a, const B& b) {
    { a.size() } -> std::convertible_to<std::size_t>;
    { b.size() } -> std::convertible_to<std::size_t>;
    a.size() == b.size();
    typename A::value_type;
    typename B::value_type;
} && std::is_same_v<typename A::value_type, typename B::value_type>;

/**
 * @brief Concept requiring aliasing safety between input and output.
 *
 * Ensures that input and output types are not the same type after removing
 * cv-qualifiers and reference. This prevents unintended in-place mutations
 * that could corrupt state during operator evaluation.
 *
 * Note: This concept rejects ALL in-place operations (same input and output
 * object), which may exclude valid optimization patterns in future integrator
 * implementations.
 *
 * @tparam Input Input type
 * @tparam Output Output type
 */
template <class Input, class Output>
concept AliasingSafe = requires {
    requires !std::is_same_v<std::remove_cvref_t<Input>, std::remove_cvref_t<Output>>;
};

} // namespace pfc::field
