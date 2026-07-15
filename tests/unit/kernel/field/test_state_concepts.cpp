// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <type_traits>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;

// Compile-time test: FieldView<T> is copyable
static_assert(std::is_copy_constructible_v<FieldView<double>>);
static_assert(std::is_copy_assignable_v<FieldView<double>>);
static_assert(std::is_move_constructible_v<FieldView<double>>);
static_assert(std::is_move_assignable_v<FieldView<double>>);

// Compile-time test: FieldView<T> with complex types
static_assert(std::is_copy_constructible_v<FieldView<std::complex<double>>>);
static_assert(std::is_copy_assignable_v<FieldView<std::complex<double>>>);

// Compile-time test: FieldOutput<T> is copyable
static_assert(std::is_copy_constructible_v<FieldOutput<double>>);
static_assert(std::is_copy_assignable_v<FieldOutput<double>>);
static_assert(std::is_move_constructible_v<FieldOutput<double>>);
static_assert(std::is_move_assignable_v<FieldOutput<double>>);

// Compile-time test: FieldBundle<Ts...> is copyable
static_assert(std::is_copy_constructible_v<FieldBundle<FieldView<double>>>);
static_assert(std::is_copy_assignable_v<FieldBundle<FieldView<double>>>);
static_assert(std::is_move_constructible_v<FieldBundle<FieldView<double>>>);
static_assert(std::is_move_assignable_v<FieldBundle<FieldView<double>>>);

// Compile-time test: Multi-field bundle
static_assert(std::is_copy_constructible_v<FieldBundle<FieldView<double>, FieldView<double>>>);
static_assert(std::is_copy_assignable_v<FieldBundle<FieldView<double>, FieldView<double>>>);

TEST_CASE("FieldView concept requirements", "[field][state_concepts]") {
    using ViewType = FieldView<double>;

    SECTION("FieldView is copyable") {
        static_assert(std::is_copy_constructible_v<ViewType>);
        static_assert(std::is_copy_assignable_v<ViewType>);
        static_assert(std::is_move_constructible_v<ViewType>);
        static_assert(std::is_move_assignable_v<ViewType>);
    }

    SECTION("FieldView has const member functions") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        const ViewType view(data.data(), data.size(), extents, spacing, origin);

        // Verify const member functions exist and are callable
        static_assert(std::is_same_v<decltype(view.data()), const double*>);
        static_assert(std::is_same_v<decltype(view.size()), std::size_t>);
        static_assert(std::is_same_v<decltype(view.extents()), pfc::types::Int3>);
        static_assert(std::is_same_v<decltype(view.spacing()), pfc::types::Real3>);
        static_assert(std::is_same_v<decltype(view.origin()), pfc::types::Real3>);
        static_assert(std::is_same_v<decltype(view.is_compatible_with(view)), bool>);
    }

    SECTION("FieldView data() returns const T*") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        ViewType view(data.data(), data.size(), extents, spacing, origin);

        const double* const_ptr = view.data();
        REQUIRE(const_ptr != nullptr);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(const_ptr)>>);
    }

    SECTION("FieldView works with complex types") {
        using ComplexViewType = FieldView<std::complex<double>>;

        static_assert(std::is_copy_constructible_v<ComplexViewType>);
        static_assert(std::is_copy_assignable_v<ComplexViewType>);

        std::vector<std::complex<double>> data(64, std::complex<double>(1.0, 2.0));
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        const ComplexViewType view(data.data(), data.size(), extents, spacing, origin);

        const std::complex<double>* const_ptr = view.data();
        REQUIRE(const_ptr != nullptr);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(const_ptr)>>);
    }
}

TEST_CASE("FieldOutput concept requirements", "[field][state_concepts]") {
    using OutputType = FieldOutput<double>;

    SECTION("FieldOutput is copyable") {
        static_assert(std::is_copy_constructible_v<OutputType>);
        static_assert(std::is_copy_assignable_v<OutputType>);
        static_assert(std::is_move_constructible_v<OutputType>);
        static_assert(std::is_move_assignable_v<OutputType>);
    }

    SECTION("FieldOutput data() returns T* (mutable)") {
        std::vector<double> data(64, 0.0);
        OutputType output(data.data(), data.size());

        double* mutable_ptr = output.data();
        REQUIRE(mutable_ptr != nullptr);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutable_ptr)>>);
    }

    SECTION("FieldOutput validate_no_alias is callable") {
        std::vector<double> input_data(64, 1.0);
        std::vector<double> output_data(64, 0.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> input(input_data.data(), input_data.size(), extents, spacing, origin);
        FieldOutput<double> output(output_data.data(), output_data.size());

        // Verify validate_no_alias is callable
        REQUIRE_NOTHROW(output.validate_no_alias(input));
    }

    SECTION("FieldOutput works with complex types") {
        using ComplexOutputType = FieldOutput<std::complex<double>>;

        static_assert(std::is_copy_constructible_v<ComplexOutputType>);
        static_assert(std::is_copy_assignable_v<ComplexOutputType>);

        std::vector<std::complex<double>> data(64, std::complex<double>(0.0, 0.0));
        ComplexOutputType output(data.data(), data.size());

        std::complex<double>* mutable_ptr = output.data();
        REQUIRE(mutable_ptr != nullptr);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutable_ptr)>>);
    }
}

TEST_CASE("FieldBundle concept requirements", "[field][state_concepts]") {
    using BundleType = FieldBundle<FieldView<double>, FieldView<double>>;

    SECTION("FieldBundle is copyable") {
        static_assert(std::is_copy_constructible_v<BundleType>);
        static_assert(std::is_copy_assignable_v<BundleType>);
        static_assert(std::is_move_constructible_v<BundleType>);
        static_assert(std::is_move_assignable_v<BundleType>);
    }

    SECTION("FieldBundle get<I>() returns reference") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

        BundleType bundle(u_view, v_view);

        // Verify get<I>() returns reference
        static_assert(std::is_same_v<decltype(bundle.get<0>()), FieldView<double>&>);
        static_assert(std::is_same_v<decltype(bundle.get<1>()), FieldView<double>&>);
    }

    SECTION("FieldBundle get<I>() const returns const reference") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

        const BundleType bundle(u_view, v_view);

        // Verify const get<I>() returns const reference
        static_assert(std::is_same_v<decltype(bundle.get<0>()), const FieldView<double>&>);
        static_assert(std::is_same_v<decltype(bundle.get<1>()), const FieldView<double>&>);
    }

    SECTION("FieldBundle validate_shapes() is callable") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

        BundleType bundle(u_view, v_view);

        // Verify validate_shapes() is callable
        static_assert(std::is_same_v<decltype(bundle.validate_shapes()), bool>);
        REQUIRE(bundle.validate_shapes());
    }

    SECTION("FieldBundle works with heterogeneous field types") {
        std::vector<double> real_data(64, 1.0);
        std::vector<std::complex<double>> complex_data(64, std::complex<double>(1.0, 2.0));

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> real_view(real_data.data(), real_data.size(), extents, spacing, origin);
        FieldView<std::complex<double>> complex_view(complex_data.data(), complex_data.size(), extents, spacing, origin);

        FieldBundle<FieldView<double>, FieldView<std::complex<double>>> bundle(real_view, complex_view);

        static_assert(std::is_copy_constructible_v<decltype(bundle)>);
        static_assert(std::is_same_v<decltype(bundle.get<0>()), FieldView<double>&>);
        static_assert(std::is_same_v<decltype(bundle.get<1>()), FieldView<std::complex<double>>&>);
    }
}

TEST_CASE("Compatible fields concept", "[field][state_concepts]") {
    SECTION("fields with matching extents/spacing/origin are compatible") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin);

        // Verify is_compatible_with() returns true
        REQUIRE(field1.is_compatible_with(field2));
        REQUIRE(field2.is_compatible_with(field1));
    }

    SECTION("fields with different extents are not compatible") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(128, 2.0);

        pfc::types::Int3 extents1{4, 4, 4};
        pfc::types::Int3 extents2{8, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents1, spacing, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents2, spacing, origin);

        // Verify is_compatible_with() returns false
        REQUIRE_FALSE(field1.is_compatible_with(field2));
        REQUIRE_FALSE(field2.is_compatible_with(field1));
    }

    SECTION("fields with different spacing are not compatible") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing1{1.0, 1.0, 1.0};
        pfc::types::Real3 spacing2{2.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents, spacing1, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents, spacing2, origin);

        // Verify is_compatible_with() returns false
        REQUIRE_FALSE(field1.is_compatible_with(field2));
        REQUIRE_FALSE(field2.is_compatible_with(field1));
    }

    SECTION("fields with different origin are not compatible") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin1{0.0, 0.0, 0.0};
        pfc::types::Real3 origin2{1.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin1);
        FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin2);

        // Verify is_compatible_with() returns false
        REQUIRE_FALSE(field1.is_compatible_with(field2));
        REQUIRE_FALSE(field2.is_compatible_with(field1));
    }

    SECTION("is_compatible_with() is const-qualified") {
        std::vector<double> data(64, 1.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        const FieldView<double> field(data.data(), data.size(), extents, spacing, origin);

        // Verify is_compatible_with() is callable on const object
        static_assert(std::is_same_v<decltype(field.is_compatible_with(field)), bool>);
        REQUIRE(field.is_compatible_with(field));
    }
}
