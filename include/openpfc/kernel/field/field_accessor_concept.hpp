// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file field_accessor_concept.hpp
 * @brief C++20 concept validating a field type's basic access pattern.
 *
 * @details
 * `FieldAccessor<T>` captures the minimal read/write access every field
 * type used by stepper integrators must provide: a `size()` const query
 * and both a mutable and a const `data()` accessor. Stepper integrators
 * currently assume this access pattern without any compile-time guarantee
 * -- as the codebase grows multi-architecture support (CPU/CUDA/HIP),
 * an incompatible field type would otherwise only fail at runtime or
 * during integration on a specific backend. This is a companion to
 * `state_concepts.hpp`'s FieldReadable/FieldWritable (which additionally
 * require `operator()(int,int,int)` element access); FieldAccessor is
 * the narrower, storage-only contract that ADR-0003 contract 7
 * (docs/adr/0003-time-integrator-interface.md) extends via
 * concept-constrained templates.
 *
 * @see local_field.hpp for LocalField<T>, the field type this concept is
 *      verified against.
 */

#include <concepts>
#include <cstddef>

namespace pfc::field {

/**
 * @brief Satisfied by any field type providing size() and both const and
 *        non-const data() accessors.
 *
 * @tparam F Field type to check for basic storage access.
 */
template <class F>
concept FieldAccessor = requires(const F &cf, F &f) {
  { cf.size() } -> std::convertible_to<std::size_t>;
  { cf.data() } -> std::convertible_to<const void *>;
  { f.size() } -> std::convertible_to<std::size_t>;
  { f.data() } -> std::convertible_to<void *>;
};

} // namespace pfc::field
