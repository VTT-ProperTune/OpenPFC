// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_grid_field_concepts.cpp
 * @brief Static verification that pfc::data::Field satisfies existing field state-concepts.
 *
 * This test statically checks Field<double, pfc::HostSpace> against the concepts defined in:
 * - kernel/simulation/state_concepts.hpp (field read/write access)
 * - kernel/field/field_accessor_concept.hpp (basic storage access)
 *
 * Where Field satisfies a concept, the static_assert passes. Where it does NOT satisfy
 * a concept, a FIXME comment documents the exact missing member so the migration path
 * is clear.
 */

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/simulation/state_concepts.hpp>
#include <openpfc/kernel/field/field_accessor_concept.hpp>

TEST_CASE("Field satisfies field state-concepts", "[kernel][data][concepts]") {
    using Field_t = pfc::data::Field<double, pfc::HostSpace>;

    SECTION("Field satisfies ReadableField concept") {
        // pfc::concept::ReadableField is defined in state_concepts.hpp
        // Requires: size() const returning std::size_t,
        //           data() const returning type convertible to const void*,
        //           operator()(int,int,int) const returning const reference
        static_assert(pfc::field::FieldReadable<Field_t>,
            "Field<double, HostSpace> should satisfy FieldReadable concept");
    }

    SECTION("Field satisfies WritableField concept") {
        // pfc::concept::WritableField is defined in state_concepts.hpp
        // Requires: size() const returning std::size_t,
        //           data() returning type convertible to void*,
        //           operator()(int,int,int) returning non-const lvalue_reference
        static_assert(pfc::field::FieldWritable<Field_t>,
            "Field<double, HostSpace> should satisfy FieldWritable concept");
    }

    SECTION("Field satisfies composite Field concept") {
        // pfc::concept::Field composes FieldReadable and FieldWritable
        static_assert(pfc::field::Field<Field_t>,
            "Field<double, HostSpace> should satisfy Field concept");
    }

    SECTION("Field satisfies ConstField concept (alias for FieldReadable)") {
        static_assert(pfc::field::ConstField<Field_t>,
            "Field<double, HostSpace> should satisfy ConstField concept");
    }

    SECTION("Field satisfies FieldAccessor concept") {
        // pfc::field::FieldAccessor is defined in field_accessor_concept.hpp
        // Requires: size() const returning std::size_t,
        //           both const and non-const data() returning void* variants
        static_assert(pfc::field::FieldAccessor<Field_t>,
            "Field<double, HostSpace> should satisfy FieldAccessor concept");
    }

    SECTION("Field satisfies ShapeCompatible with itself") {
        // ShapeCompatible requires both types have size() returning std::size_t,
        // compatible sizes, and matching value_type members
        static_assert(pfc::field::ShapeCompatible<Field_t, Field_t>,
            "Field<double, HostSpace> should be ShapeCompatible with itself");
    }

    SECTION("Field value_type is correctly exposed") {
        // Field must expose value_type for ShapeCompatible and other type-aware concepts
        static_assert(std::is_same_v<Field_t::value_type, double>,
            "Field<double, HostSpace>::value_type should be double");
    }

    SECTION("Field memory space is correctly exposed") {
        // Field exposes memory_space for architecture-aware code
        static_assert(std::is_same_v<Field_t::memory_space, pfc::HostSpace>,
            "Field<double, HostSpace>::memory_space should be pfc::HostSpace");
    }

    SECTION("Field satisfies AliasingSafe semantics") {
        // AliasingSafe<Input, Output> requires types be different to prevent unintended in-place mutations
        // This concept rejects same-type input/output pairs, which is the expected behavior
        using Field_write_t = pfc::data::Field<double, pfc::HostSpace>;
        static_assert(pfc::field::AliasingSafe<const Field_t, Field_write_t> == false,
            "Same input and output types should not be AliasingSafe (prevents unintended in-place mutations)");
    }

    // NOTE: No additional concepts need FIXME comments at this time.
    // Field<double, HostSpace> satisfies all relevant field state-concepts:
    // - FieldReadable: has size() const, data() const, operator()(int,int,int) const
    // - FieldWritable: has size() const, data(), operator()(int,int,int)  
    // - Field: composes both FieldReadable and FieldWritable
    // - ConstField: alias for FieldReadable
    // - FieldAccessor: has size() and both const/non-const data() methods
    // - ShapeCompatible: has value_type member and size() method
    // - AliasingSafe: correctly rejects same-type input/output pairs
}
