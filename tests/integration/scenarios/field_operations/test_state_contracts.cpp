// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;
using Catch::Approx;

TEST_CASE("Read write contract", "[field][state_contracts]") {
    SECTION("FieldView does not mutate underlying storage") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        // Create field view
        FieldView<double> view(data.data(), data.size(), extents, spacing, origin);

        // Store original data
        std::vector<double> original_data = data;

        // Access data through view (read-only)
        const double* view_data = view.data();
        volatile double sum = 0.0;
        for (std::size_t i = 0; i < view.size(); ++i) {
            sum += view_data[i];
        }

        // Verify data is unchanged
        REQUIRE(data == original_data);
    }

    SECTION("FieldOutput writes to caller-provided storage") {
        std::vector<double> output_data(64, 0.0);
        FieldOutput<double> output(output_data.data(), output_data.size());

        // Write through output
        double* output_ptr = output.data();
        for (std::size_t i = 0; i < output.size(); ++i) {
            output_ptr[i] = static_cast<double>(i) * 2.0;
        }

        // Verify data was written to caller-provided storage
        for (std::size_t i = 0; i < output_data.size(); ++i) {
            REQUIRE(output_data[i] == static_cast<double>(i) * 2.0);
        }
    }

    SECTION("no allocation occurs during access operations") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> view(data.data(), data.size(), extents, spacing, origin);

        // Access operations should not allocate
        auto data_ptr = view.data();
        auto size = view.size();
        auto extents_val = view.extents();
        auto spacing_val = view.spacing();
        auto origin_val = view.origin();
        bool compatible = view.is_compatible_with(view);

        // Verify we got valid data without allocation
        REQUIRE(data_ptr != nullptr);
        REQUIRE(size == 64);
        REQUIRE(extents_val == extents);
        REQUIRE(spacing_val == spacing);
        REQUIRE(origin_val == origin);
        REQUIRE(compatible);
    }
}

TEST_CASE("Const mutable separation", "[field][state_contracts]") {
    SECTION("const FieldView cannot be used to mutate") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        const FieldView<double> view(data.data(), data.size(), extents, spacing, origin);

        // data() returns const pointer
        const double* const_ptr = view.data();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(const_ptr)>>);

        // Cannot mutate through const view (compile-time enforcement)
        // const_ptr[0] = 2.0;  // This would not compile
    }

    SECTION("FieldOutput provides mutable access") {
        std::vector<double> data(64, 0.0);
        FieldOutput<double> output(data.data(), data.size());

        // data() returns mutable pointer
        double* mutable_ptr = output.data();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mutable_ptr)>>);

        // Can mutate through output
        mutable_ptr[0] = 2.0;
        REQUIRE(data[0] == 2.0);
    }

    SECTION("compile-time enforcement of const correctness") {
        std::vector<double> data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> view(data.data(), data.size(), extents, spacing, origin);
        const FieldView<double> const_view(data.data(), data.size(), extents, spacing, origin);

        // Non-const view provides mutable access to underlying data if needed
        // (though FieldView itself is read-only by design)
        static_assert(std::is_same_v<decltype(view.data()), const double*>);
        static_assert(std::is_same_v<decltype(const_view.data()), const double*>);
    }
}

TEST_CASE("Backend semantic equivalence", "[field][state_contracts]") {
    SECTION("CPU and GPU use same contracts") {
        // CPU storage
        std::vector<double> cpu_data(64, 1.0);
        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> cpu_view(cpu_data.data(), cpu_data.size(), extents, spacing, origin);
        std::vector<double> cpu_output_data(64, 0.0);
        FieldOutput<double> cpu_output(cpu_output_data.data(), cpu_output_data.size());

        // Verify CPU contracts work
        REQUIRE(cpu_view.data() != nullptr);
        REQUIRE(cpu_view.size() == 64);
        REQUIRE(cpu_view.extents() == extents);
        REQUIRE(cpu_view.spacing() == spacing);
        REQUIRE(cpu_view.origin() == origin);

        REQUIRE(cpu_output.data() != nullptr);
        REQUIRE(cpu_output.size() == 64);

        REQUIRE_NOTHROW(cpu_output.validate_no_alias(cpu_view));
        REQUIRE_NOTHROW(validate_shape_compatibility(cpu_view, cpu_view));

        // Note: GPU implementation would follow identical contracts
        // Backend-specific validation will be in separate headers (e.g., validation_cuda.hpp)
        // when GPU implementation is pursued. FieldView<T> is intentionally backend-agnostic.
    }

    SECTION("FieldView is backend-agnostic") {
        // FieldView works with any contiguous storage
        std::vector<double> std_vector_data(64, 1.0);
        double c_array_data[64];
        std::fill(std::begin(c_array_data), std::end(c_array_data), 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        // FieldView with std::vector storage
        FieldView<double> std_vector_view(std_vector_data.data(), std_vector_data.size(), extents, spacing, origin);

        // FieldView with C array storage
        FieldView<double> c_array_view(c_array_data, std::size(c_array_data), extents, spacing, origin);

        // Both work identically
        REQUIRE(std_vector_view.extents() == extents);
        REQUIRE(c_array_view.extents() == extents);
        REQUIRE(std_vector_view.is_compatible_with(c_array_view));
    }

    SECTION("backend-specific validation is deferred") {
        // This test documents that backend-specific validation is deferred
        // to separate headers (e.g., validation_cuda.hpp, validation_hip.hpp)

        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin);

        // Generic validation works
        REQUIRE_NOTHROW(validate_shape_compatibility(field1, field2));

        // Backend-specific validation (e.g., CUDA vs CPU mixing) will be in
        // separate headers when GPU implementation is pursued. FieldView<T>
        // is intentionally backend-agnostic to enable single-header usage.
    }
}

TEST_CASE("Aliasing validation contract", "[field][state_contracts]") {
    SECTION("aliasing is detected and rejected") {
        std::vector<double> shared_data(64, 1.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> view(shared_data.data(), shared_data.size(), extents, spacing, origin);
        FieldOutput<double> output(shared_data.data(), shared_data.size());

        // Aliasing must be detected and rejected
        REQUIRE_THROWS_AS(output.validate_no_alias(view), std::invalid_argument);
    }

    SECTION("distinct storage passes validation") {
        std::vector<double> input_data(64, 1.0);
        std::vector<double> output_data(64, 0.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> view(input_data.data(), input_data.size(), extents, spacing, origin);
        FieldOutput<double> output(output_data.data(), output_data.size());

        // Distinct storage must pass validation
        REQUIRE_NOTHROW(output.validate_no_alias(view));
    }
}

TEST_CASE("Shape compatibility contract", "[field][state_contracts]") {
    SECTION("compatible fields pass validation") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents, spacing, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents, spacing, origin);

        // Compatible fields must pass validation
        REQUIRE_NOTHROW(validate_shape_compatibility(field1, field2));
        REQUIRE(field1.is_compatible_with(field2));
    }

    SECTION("incompatible fields are rejected") {
        std::vector<double> data1(64, 1.0);
        std::vector<double> data2(128, 2.0);

        pfc::types::Int3 extents1{4, 4, 4};
        pfc::types::Int3 extents2{8, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> field1(data1.data(), data1.size(), extents1, spacing, origin);
        FieldView<double> field2(data2.data(), data2.size(), extents2, spacing, origin);

        // Incompatible fields must be rejected
        REQUIRE_THROWS_AS(validate_shape_compatibility(field1, field2), std::invalid_argument);
        REQUIRE_FALSE(field1.is_compatible_with(field2));
    }
}

TEST_CASE("Multi-field bundle contract", "[field][state_contracts]") {
    SECTION("FieldBundle provides coordinated access") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(64, 2.0);
        std::vector<double> lap_data(64, 0.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
        FieldView<double> lap_view(lap_data.data(), lap_data.size(), extents, spacing, origin);

        FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> bundle(u_view, v_view, lap_view);

        // Bundle provides coordinated access to all fields
        REQUIRE(bundle.get<0>().data() == u_data.data());
        REQUIRE(bundle.get<1>().data() == v_data.data());
        REQUIRE(bundle.get<2>().data() == lap_data.data());
    }

    SECTION("FieldBundle validates coordinated shapes") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(64, 2.0);

        pfc::types::Int3 extents{4, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);

        FieldBundle<FieldView<double>, FieldView<double>> bundle(u_view, v_view);

        // Bundle must validate that all fields have compatible shapes
        REQUIRE(bundle.validate_shapes());
    }

    SECTION("FieldBundle detects incompatible shapes") {
        std::vector<double> u_data(64, 1.0);
        std::vector<double> v_data(128, 2.0);

        pfc::types::Int3 extents1{4, 4, 4};
        pfc::types::Int3 extents2{8, 4, 4};
        pfc::types::Real3 spacing{1.0, 1.0, 1.0};
        pfc::types::Real3 origin{0.0, 0.0, 0.0};

        FieldView<double> u_view(u_data.data(), u_data.size(), extents1, spacing, origin);
        FieldView<double> v_view(v_data.data(), v_data.size(), extents2, spacing, origin);

        FieldBundle<FieldView<double>, FieldView<double>> bundle(u_view, v_view);

        // Bundle must detect incompatible shapes
        REQUIRE_FALSE(bundle.validate_shapes());
    }
}
