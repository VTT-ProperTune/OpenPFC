// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <catch2/catch_approx.hpp>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;
using Catch::Approx;

TEST_CASE("FieldView const access", "[field][state_access]") {
    std::vector<double> data(64, 1.0);
    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> view(data.data(), data.size(), extents, spacing, origin);

    SECTION("construction and data access") {
        REQUIRE(view.data() == data.data());
        REQUIRE(view.size() == 64);
        REQUIRE(view.extents() == extents);
        REQUIRE(view.spacing() == spacing);
        REQUIRE(view.origin() == origin);
    }

    SECTION("read-only semantics") {
        // Verify that data() returns const pointer
        const double* const_ptr = view.data();
        REQUIRE(const_ptr != nullptr);

        // Verify size is correct
        REQUIRE(view.size() == 64);

        // Verify geometry queries work
        REQUIRE(view.extents()[0] == 4);
        REQUIRE(view.extents()[1] == 4);
        REQUIRE(view.extents()[2] == 4);
        REQUIRE(view.spacing()[0] == 1.0);
        REQUIRE(view.spacing()[1] == 1.0);
        REQUIRE(view.spacing()[2] == 1.0);
        REQUIRE(view.origin()[0] == 0.0);
        REQUIRE(view.origin()[1] == 0.0);
        REQUIRE(view.origin()[2] == 0.0);
    }

    SECTION("is_compatible_with returns correct bool") {
        FieldView<double> compatible(data.data(), data.size(), extents, spacing, origin);
        REQUIRE(view.is_compatible_with(compatible));

        pfc::types::Int3 different_extents{8, 4, 4};
        FieldView<double> incompatible_extent(data.data(), data.size(), different_extents, spacing, origin);
        REQUIRE_FALSE(view.is_compatible_with(incompatible_extent));

        pfc::types::Real3 different_spacing{2.0, 1.0, 1.0};
        FieldView<double> incompatible_spacing(data.data(), data.size(), extents, different_spacing, origin);
        REQUIRE_FALSE(view.is_compatible_with(incompatible_spacing));

        pfc::types::Real3 different_origin{1.0, 0.0, 0.0};
        FieldView<double> incompatible_origin(data.data(), data.size(), extents, spacing, different_origin);
        REQUIRE_FALSE(view.is_compatible_with(incompatible_origin));
    }
}

TEST_CASE("FieldOutput mutable access", "[field][state_access]") {
    std::vector<double> data(64, 0.0);

    FieldOutput<double> output(data.data(), data.size());

    SECTION("construction and data access") {
        REQUIRE(output.data() == data.data());
        REQUIRE(output.size() == 64);
    }

    SECTION("write semantics") {
        // Verify that data() returns mutable pointer
        double* mutable_ptr = output.data();
        REQUIRE(mutable_ptr != nullptr);

        // Write through output
        for (std::size_t i = 0; i < output.size(); ++i) {
            mutable_ptr[i] = static_cast<double>(i);
        }

        // Verify data was written
        for (std::size_t i = 0; i < data.size(); ++i) {
            REQUIRE(data[i] == static_cast<double>(i));
        }
    }
}

TEST_CASE("Field aliasing detection", "[field][state_access]") {
    std::vector<double> input_data(64, 1.0);
    std::vector<double> output_data(64, 0.0);

    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> input(input_data.data(), input_data.size(), extents, spacing, origin);
    FieldOutput<double> output(output_data.data(), output_data.size());

    SECTION("no aliasing when storage is distinct") {
        // Should not throw when storage is distinct
        REQUIRE_NOTHROW(output.validate_no_alias(input));
    }

    SECTION("aliasing detected when storage overlaps") {
        // Create overlapping storage
        std::vector<double> shared_data(64, 1.0);
        FieldView<double> shared_view(shared_data.data(), shared_data.size(), extents, spacing, origin);
        FieldOutput<double> shared_output(shared_data.data(), shared_data.size());

        // Should throw when storage aliases
        REQUIRE_THROWS_AS(shared_output.validate_no_alias(shared_view), std::invalid_argument);
    }

    SECTION("aliasing validation through validate_no_alias function") {
        // Test the free function form
        REQUIRE_NOTHROW(validate_no_alias(output, input));

        std::vector<double> shared_data(64, 1.0);
        FieldView<double> shared_view(shared_data.data(), shared_data.size(), extents, spacing, origin);
        FieldOutput<double> shared_output(shared_data.data(), shared_data.size());

        REQUIRE_THROWS_AS(validate_no_alias(shared_output, shared_view), std::invalid_argument);
    }
}

TEST_CASE("Shape compatibility validation", "[field][state_access]") {
    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    std::vector<double> data1(64, 1.0);
    std::vector<double> data2(64, 1.0);

    FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin);
    FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin);

    SECTION("compatible fields pass validation") {
        REQUIRE_NOTHROW(validate_shape_compatibility(field1, field2));
    }

    SECTION("incompatible extents throw") {
        pfc::types::Int3 different_extents{8, 4, 4};
        std::vector<double> data3(128, 1.0);
        FieldView<double> field3(data3.data(), data3.size(), different_extents, spacing, origin);

        REQUIRE_THROWS_AS(validate_shape_compatibility(field1, field3), std::invalid_argument);
    }

    SECTION("incompatible spacing throws") {
        pfc::types::Real3 different_spacing{2.0, 1.0, 1.0};
        std::vector<double> data3(64, 1.0);
        FieldView<double> field3(data3.data(), data3.size(), extents, different_spacing, origin);

        REQUIRE_THROWS_AS(validate_shape_compatibility(field1, field3), std::invalid_argument);
    }

    SECTION("incompatible origin throws") {
        pfc::types::Real3 different_origin{1.0, 0.0, 0.0};
        std::vector<double> data3(64, 1.0);
        FieldView<double> field3(data3.data(), data3.size(), extents, spacing, different_origin);

        REQUIRE_THROWS_AS(validate_shape_compatibility(field1, field3), std::invalid_argument);
    }

    SECTION("is_compatible_with returns correct bool") {
        REQUIRE(field1.is_compatible_with(field2));

        pfc::types::Int3 different_extents{8, 4, 4};
        std::vector<double> data3(128, 1.0);
        FieldView<double> field3(data3.data(), data3.size(), different_extents, spacing, origin);
        REQUIRE_FALSE(field1.is_compatible_with(field3));
    }
}

TEST_CASE("FieldBundle multi-field", "[field][state_access]") {
    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    std::vector<double> u_data(64, 1.0);
    std::vector<double> v_data(64, 2.0);
    std::vector<double> lap_data(64, 0.0);

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldView<double> lap_view(lap_data.data(), lap_data.size(), extents, spacing, origin);

    SECTION("construction with multiple fields") {
        FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> bundle(u_view, v_view, lap_view);

        REQUIRE(bundle.get<0>().data() == u_data.data());
        REQUIRE(bundle.get<1>().data() == v_data.data());
        REQUIRE(bundle.get<2>().data() == lap_data.data());
    }

    SECTION("get<I>() access to individual fields") {
        FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> bundle(u_view, v_view, lap_view);

        const auto& u = bundle.get<0>();
        const auto& v = bundle.get<1>();
        const auto& lap = bundle.get<2>();

        REQUIRE(u.data() == u_data.data());
        REQUIRE(v.data() == v_data.data());
        REQUIRE(lap.data() == lap_data.data());
    }

    SECTION("validate_shapes() across all fields") {
        FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> bundle(u_view, v_view, lap_view);
        REQUIRE(bundle.validate_shapes());
    }

    SECTION("validate_shapes() detects incompatible fields") {
        pfc::types::Int3 different_extents{8, 4, 4};
        std::vector<double> w_data(128, 3.0);
        FieldView<double> w_view(w_data.data(), w_data.size(), different_extents, spacing, origin);

        FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> bundle(u_view, v_view, w_view);
        REQUIRE_FALSE(bundle.validate_shapes());
    }

    SECTION("empty bundle is trivially valid") {
        FieldBundle<> empty_bundle;
        REQUIRE(empty_bundle.validate_shapes());
    }

    SECTION("single field bundle is trivially valid") {
        FieldBundle<FieldView<double>> single_bundle(u_view);
        REQUIRE(single_bundle.validate_shapes());
    }
}

TEST_CASE("FieldView with complex types", "[field][state_access]") {
    std::vector<std::complex<double>> data(64, std::complex<double>(1.0, 2.0));
    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<std::complex<double>> view(data.data(), data.size(), extents, spacing, origin);

    SECTION("construction and data access") {
        REQUIRE(view.data() == data.data());
        REQUIRE(view.size() == 64);
        REQUIRE(view.extents() == extents);
        REQUIRE(view.spacing() == spacing);
        REQUIRE(view.origin() == origin);
    }

    SECTION("read-only semantics for complex types") {
        const std::complex<double>* const_ptr = view.data();
        REQUIRE(const_ptr != nullptr);
        REQUIRE(const_ptr[0] == std::complex<double>(1.0, 2.0));
    }
}

TEST_CASE("FieldOutput with complex types", "[field][state_access]") {
    std::vector<std::complex<double>> data(64, std::complex<double>(0.0, 0.0));

    FieldOutput<std::complex<double>> output(data.data(), data.size());

    SECTION("write semantics for complex types") {
        std::complex<double>* mutable_ptr = output.data();
        REQUIRE(mutable_ptr != nullptr);

        for (std::size_t i = 0; i < output.size(); ++i) {
            mutable_ptr[i] = std::complex<double>(static_cast<double>(i), static_cast<double>(i) * 2.0);
        }

        for (std::size_t i = 0; i < data.size(); ++i) {
            REQUIRE(data[i].real() == static_cast<double>(i));
            REQUIRE(data[i].imag() == static_cast<double>(i) * 2.0);
        }
    }
}
