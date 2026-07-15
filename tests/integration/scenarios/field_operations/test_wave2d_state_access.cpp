// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;
using Catch::Approx;

/**
 * @brief Test Wave2D multi-field state access pattern
 *
 * This test verifies that the new state access primitives can represent
 * the Wave2D multi-field pattern from apps/wave2d/src/cpu/wave2d_fd.cpp.
 *
 * Wave2D pattern:
 * - Multiple coupled fields (u, v, lap)
 * - Wave equation: d_t u = v, d_t v = c^2 * Laplacian(u)
 * - Tuple-based increments (WaveIncrements{du, dv})
 * - Per-point Laplacian aggregate (WaveLaplacian{lxx, lyy})
 */
TEST_CASE("Wave2D multi-field state access", "[field][wave2d_evidence]") {
    // Create test data matching Wave2D pattern
    // Wave2D uses a 2D grid (nz = 1) with halo padding
    const int nx = 8;
    const int ny = 8;
    const int nz = 1;
    const int halo_width = 1;
    
    const int nx_padded = nx + 2 * halo_width;
    const int ny_padded = ny + 2 * halo_width;
    const int nz_padded = nz + 2 * halo_width;
    
    const std::size_t total_size = static_cast<std::size_t>(nx_padded) * 
                                   static_cast<std::size_t>(ny_padded) * 
                                   static_cast<std::size_t>(nz_padded);

    std::vector<double> u_data(total_size, 0.0);
    std::vector<double> v_data(total_size, 0.0);
    std::vector<double> lap_data(total_size, 0.0);

    // Initialize with a Gaussian-like pattern (simplified)
    const double xc = static_cast<double>(nx) / 2.0;
    const double yc = static_cast<double>(ny) / 2.0;
    const double sigma = 2.0;

    for (int k = halo_width; k < nz + halo_width; ++k) {
        for (int j = halo_width; j < ny + halo_width; ++j) {
            for (int i = halo_width; i < nx + halo_width; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) + 
                                        static_cast<std::size_t>(j) * nx_padded +
                                        static_cast<std::size_t>(k) * nx_padded * ny_padded;
                
                const double dx = static_cast<double>(i - halo_width) - xc;
                const double dy = static_cast<double>(j - halo_width) - yc;
                
                u_data[idx] = std::exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma));
                v_data[idx] = 0.0;  // Initially at rest
            }
        }
    }

    // Construct FieldBundle with multiple FieldView<double>
    // Note: For this test, we'll use owned regions (excluding halo)
    const std::size_t owned_size = static_cast<std::size_t>(nx) * 
                                   static_cast<std::size_t>(ny) * 
                                   static_cast<std::size_t>(nz);
    
    std::vector<double> u_owned(owned_size);
    std::vector<double> v_owned(owned_size);
    std::vector<double> lap_owned(owned_size);

    // Extract owned region from padded storage
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t owned_idx = static_cast<std::size_t>(i) + 
                                              static_cast<std::size_t>(j) * nx +
                                              static_cast<std::size_t>(k) * nx * ny;
                
                const std::size_t padded_idx = static_cast<std::size_t>(i + halo_width) + 
                                               static_cast<std::size_t>(j + halo_width) * nx_padded +
                                               static_cast<std::size_t>(k + halo_width) * nx_padded * ny_padded;
                
                u_owned[owned_idx] = u_data[padded_idx];
                v_owned[owned_idx] = v_data[padded_idx];
                lap_owned[owned_idx] = lap_data[padded_idx];
            }
        }
    }

    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_owned.data(), u_owned.size(), extents, spacing, origin);
    FieldView<double> v_view(v_owned.data(), v_owned.size(), extents, spacing, origin);
    FieldView<double> lap_view(lap_owned.data(), lap_owned.size(), extents, spacing, origin);

    FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> fields(u_view, v_view, lap_view);

    SECTION("FieldBundle construction with multiple fields") {
        // Verify bundle construction
        REQUIRE(fields.get<0>().data() == u_owned.data());
        REQUIRE(fields.get<1>().data() == v_owned.data());
        REQUIRE(fields.get<2>().data() == lap_owned.data());
    }

    SECTION("coordinated shape validation across fields") {
        // All fields should have compatible shapes
        REQUIRE(fields.validate_shapes());
    }

    SECTION("multi-field aliasing detection") {
        // Create output storage for increments
        std::vector<double> du_data(owned_size, 0.0);
        std::vector<double> dv_data(owned_size, 0.0);

        FieldOutput<double> du_output(du_data.data(), du_data.size());
        FieldOutput<double> dv_output(dv_data.data(), dv_data.size());

        // Distinct storage should pass validation
        REQUIRE_NOTHROW(du_output.validate_no_alias(u_view));
        REQUIRE_NOTHROW(dv_output.validate_no_alias(v_view));
        REQUIRE_NOTHROW(du_output.validate_no_alias(v_view));
        REQUIRE_NOTHROW(dv_output.validate_no_alias(u_view));

        // Aliased storage should fail
        FieldOutput<double> aliased_output(u_owned.data(), u_owned.size());
        REQUIRE_THROWS_AS(aliased_output.validate_no_alias(u_view), std::invalid_argument);
    }

    SECTION("numerical results match expected pattern") {
        // Verify initial condition for u
        double max_u = 0.0;
        for (std::size_t i = 0; i < u_view.size(); ++i) {
            max_u = std::max(max_u, u_view.data()[i]);
        }
        REQUIRE(max_u == Approx(1.0).epsilon(1e-10));

        // Verify v is initially zero
        double max_v = 0.0;
        for (std::size_t i = 0; i < v_view.size(); ++i) {
            max_v = std::max(max_v, std::abs(v_view.data()[i]));
        }
        REQUIRE(max_v == 0.0);

        // Verify laplacian is initially zero
        double max_lap = 0.0;
        for (std::size_t i = 0; i < lap_view.size(); ++i) {
            max_lap = std::max(max_lap, std::abs(lap_view.data()[i]));
        }
        REQUIRE(max_lap == 0.0);
    }

    SECTION("FieldBundle provides indexed access to individual fields") {
        // Verify indexed access
        const auto& u = fields.get<0>();
        const auto& v = fields.get<1>();
        const auto& lap = fields.get<2>();

        REQUIRE(u.data() == u_owned.data());
        REQUIRE(v.data() == v_owned.data());
        REQUIRE(lap.data() == lap_owned.data());

        // Verify geometry is consistent across fields
        REQUIRE(u.extents() == extents);
        REQUIRE(v.extents() == extents);
        REQUIRE(lap.extents() == extents);

        REQUIRE(u.spacing() == spacing);
        REQUIRE(v.spacing() == spacing);
        REQUIRE(lap.spacing() == spacing);

        REQUIRE(u.origin() == origin);
        REQUIRE(v.origin() == origin);
        REQUIRE(lap.origin() == origin);
    }

    SECTION("FieldBundle detects incompatible field shapes") {
        // Create field with different extents
        pfc::types::Int3 different_extents{16, 8, 1};
        std::vector<double> w_data(128, 0.0);
        FieldView<double> w_view(w_data.data(), w_data.size(), different_extents, spacing, origin);

        // Bundle with incompatible field should fail validation
        FieldBundle<FieldView<double>, FieldView<double>> incompatible_bundle(u_view, w_view);
        REQUIRE_FALSE(incompatible_bundle.validate_shapes());
    }
}

/**
 * @brief Document migration path from WaveIncrements to FieldBundle
 *
 * This section documents how to migrate from WaveIncrements to FieldBundle.
 */
TEST_CASE("Wave2D migration path from WaveIncrements to FieldBundle", "[field][wave2d_evidence]") {
    // Old pattern (WaveIncrements):
    // struct WaveIncrements {
    //     double du = 0.0;
    //     double dv = 0.0;
    //     auto as_tuple() { return std::tie(du, dv); }
    // };
    // 
    // WaveIncrements increments = model.rhs(t, v, lap);
    // auto [du, dv] = increments.as_tuple();
    
    // New pattern (FieldBundle):
    // FieldBundle<FieldOutput<double>, FieldOutput<double>> outputs(du_output, dv_output);
    // outputs.get<0>().data()[i] = du_value;
    // outputs.get<1>().data()[i] = dv_value;
    
    // For this test, we'll demonstrate the pattern with simple data
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) * 
                            static_cast<std::size_t>(ny) * 
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 1.0);
    std::vector<double> v_data(size, 0.0);
    std::vector<double> du_data(size, 0.0);
    std::vector<double> dv_data(size, 0.0);

    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldOutput<double> du_output(du_data.data(), du_data.size());
    FieldOutput<double> dv_output(dv_data.data(), dv_data.size());

    FieldBundle<FieldView<double>, FieldView<double>> inputs(u_view, v_view);
    FieldBundle<FieldOutput<double>, FieldOutput<double>> outputs(du_output, dv_output);

    // Verify the migration preserves all required information
    REQUIRE(inputs.get<0>().data() == u_data.data());
    REQUIRE(inputs.get<1>().data() == v_data.data());
    REQUIRE(outputs.get<0>().data() == du_data.data());
    REQUIRE(outputs.get<1>().data() == dv_data.data());

    // Simulate RHS computation
    const double c = 1.0;  // Wave speed
    for (std::size_t i = 0; i < size; ++i) {
        // du = v
        outputs.get<0>().data()[i] = inputs.get<1>().data()[i];
        // dv = c^2 * laplacian(u) (simplified: just use u as proxy)
        outputs.get<1>().data()[i] = c * c * 0.1 * inputs.get<0>().data()[i];
    }

    // Verify results
    for (std::size_t i = 0; i < size; ++i) {
        REQUIRE(outputs.get<0>().data()[i] == Approx(0.0).epsilon(1e-10));
        REQUIRE(outputs.get<1>().data()[i] == Approx(c * c * 0.1 * 1.0).epsilon(1e-10));
    }
}

/**
 * @brief Test Wave2D RHS computation pattern
 *
 * This test verifies that the new accessors support the Wave2D RHS computation
 * pattern: compute Laplacian of u, then compute du = v and dv = c^2 * Laplacian(u).
 */
TEST_CASE("Wave2D RHS computation pattern", "[field][wave2d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) * 
                            static_cast<std::size_t>(ny) * 
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 1.0);
    std::vector<double> v_data(size, 0.5);
    std::vector<double> lap_data(size, 0.0);
    std::vector<double> du_data(size, 0.0);
    std::vector<double> dv_data(size, 0.0);

    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldView<double> lap_view(lap_data.data(), lap_data.size(), extents, spacing, origin);
    FieldOutput<double> du_output(du_data.data(), du_data.size());
    FieldOutput<double> dv_output(dv_data.data(), dv_data.size());

    SECTION("output storage is distinct from input") {
        // Verify no aliasing
        REQUIRE_NOTHROW(du_output.validate_no_alias(u_view));
        REQUIRE_NOTHROW(du_output.validate_no_alias(v_view));
        REQUIRE_NOTHROW(dv_output.validate_no_alias(u_view));
        REQUIRE_NOTHROW(dv_output.validate_no_alias(v_view));
    }

    SECTION("shape compatibility across coupled fields") {
        // All fields must have compatible shapes
        REQUIRE(u_view.is_compatible_with(v_view));
        REQUIRE(u_view.is_compatible_with(lap_view));
        REQUIRE(v_view.is_compatible_with(lap_view));
    }

    SECTION("RHS computation uses output storage") {
        // Simulate Laplacian computation (simplified)
        for (std::size_t i = 0; i < size; ++i) {
            lap_data[i] = 0.1 * u_view.data()[i];  // Simplified
        }

        // Simulate RHS computation: du = v, dv = c^2 * laplacian(u)
        const double c = 1.0;  // Wave speed
        for (std::size_t i = 0; i < size; ++i) {
            du_output.data()[i] = v_view.data()[i];
            dv_output.data()[i] = c * c * lap_view.data()[i];
        }

        // Verify results
        for (std::size_t i = 0; i < size; ++i) {
            REQUIRE(du_output.data()[i] == Approx(0.5).epsilon(1e-10));
            REQUIRE(dv_output.data()[i] == Approx(c * c * 0.1 * 1.0).epsilon(1e-10));
        }
    }

    SECTION("FieldBundle supports multi-field operations") {
        // Create bundles for inputs and outputs
        FieldBundle<FieldView<double>, FieldView<double>> inputs(u_view, v_view);
        FieldBundle<FieldOutput<double>, FieldOutput<double>> outputs(du_output, dv_output);

        // Verify bundle validation
        REQUIRE(inputs.validate_shapes());

        // Simulate RHS computation using bundles
        const double c = 1.0;
        for (std::size_t i = 0; i < size; ++i) {
            outputs.get<0>().data()[i] = inputs.get<1>().data()[i];
            outputs.get<1>().data()[i] = c * c * 0.1 * inputs.get<0>().data()[i];
        }

        // Verify results
        for (std::size_t i = 0; i < size; ++i) {
            REQUIRE(outputs.get<0>().data()[i] == Approx(0.5).epsilon(1e-10));
            REQUIRE(outputs.get<1>().data()[i] == Approx(c * c * 0.1 * 1.0).epsilon(1e-10));
        }
    }
}

/**
 * @brief Test Wave2D per-point Laplacian aggregate pattern
 *
 * This test verifies that the new accessors support the Wave2D per-point
 * Laplacian aggregate pattern (WaveLaplacian{lxx, lyy}).
 */
TEST_CASE("Wave2D per-point Laplacian aggregate pattern", "[field][wave2d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) * 
                            static_cast<std::size_t>(ny) * 
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 1.0);
    std::vector<double> lxx_data(size, 0.1);
    std::vector<double> lyy_data(size, 0.2);

    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> lxx_view(lxx_data.data(), lxx_data.size(), extents, spacing, origin);
    FieldView<double> lyy_view(lyy_data.data(), lyy_data.size(), extents, spacing, origin);

    SECTION("per-point Laplacian components are accessible") {
        // Verify individual Laplacian components are accessible
        REQUIRE(lxx_view.data() != nullptr);
        REQUIRE(lyy_view.data() != nullptr);

        // Verify geometry compatibility
        REQUIRE(u_view.is_compatible_with(lxx_view));
        REQUIRE(u_view.is_compatible_with(lyy_view));
        REQUIRE(lxx_view.is_compatible_with(lyy_view));
    }

    SECTION("Laplacian aggregate can be computed") {
        // Simulate per-point Laplacian aggregate computation
        std::vector<double> lap_data(size, 0.0);
        FieldOutput<double> lap_output(lap_data.data(), lap_data.size());

        const double inv_dx2 = 1.0;
        const double inv_dy2 = 1.0;

        for (std::size_t i = 0; i < size; ++i) {
            // laplacian = inv_dx2 * lxx + inv_dy2 * lyy
            lap_output.data()[i] = inv_dx2 * lxx_view.data()[i] + 
                                   inv_dy2 * lyy_view.data()[i];
        }

        // Verify results
        for (std::size_t i = 0; i < size; ++i) {
            const double expected = inv_dx2 * 0.1 + inv_dy2 * 0.2;
            REQUIRE(lap_output.data()[i] == Approx(expected).epsilon(1e-10));
        }
    }

    SECTION("FieldBundle can hold Laplacian components") {
        // Bundle can hold per-point Laplacian components
        FieldBundle<FieldView<double>, FieldView<double>> laplacian_components(lxx_view, lyy_view);

        // Verify bundle validation
        REQUIRE(laplacian_components.validate_shapes());

        // Verify indexed access
        REQUIRE(laplacian_components.get<0>().data() == lxx_data.data());
        REQUIRE(laplacian_components.get<1>().data() == lyy_data.data());
    }
}
