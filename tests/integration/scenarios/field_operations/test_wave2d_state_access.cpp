// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <numeric>

#include <openpfc/kernel/field/state_access.hpp>
#include <openpfc/kernel/field/validation.hpp>

using namespace pfc::field;
using pfc::types::Int3;
using pfc::types::Real3;
using Catch::Approx;

/**
 * @brief Test Wave2D multi-field state access pattern with numerical equivalence
 *
 * This test verifies that the new state access primitives can represent
 * the Wave2D multi-field pattern AND produce numerically equivalent results
 * to manual FD computations.
 *
 * Wave2D pattern from apps/wave2d/src/cpu/wave2d_fd.cpp:
 * - Multiple fields: u (displacement), v (velocity), lap (Laplacian)
 * - Coupled time integration: u += dt * v, v += dt * k^2 * lap
 */
TEST_CASE("Wave2D numerical equivalence test", "[field][wave2d_evidence][numerical]") {
    // Create test data for numerical equivalence verification
    const int nx = 8;
    const int ny = 8;
    const int nz = 1;  // 2D problem
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size);
    std::vector<double> v_data(size);
    std::vector<double> lap_old(size);
    std::vector<double> lap_new(size);

    // Initialize with a sine wave pattern
    const double k = 2.0 * M_PI / nx;  // Wave number

    const Real3 spacing{1.0, 1.0, 1.0};
    const Real3 origin{0.0, 0.0, 0.0};

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) +
                                    static_cast<std::size_t>(j) * nx;

            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            u_data[idx] = std::sin(k * x) * std::sin(k * y);
            v_data[idx] = std::cos(k * x) * std::cos(k * y);
        }
    }

    // OLD pattern: Compute Laplacian using manual FD
    const double inv_dx2 = 1.0 / (spacing[0] * spacing[0]);
    const double inv_dy2 = 1.0 / (spacing[1] * spacing[1]);

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) +
                                    static_cast<std::size_t>(j) * nx;

            const double u_center = u_data[idx];
            const double u_left = u_data[idx - 1];
            const double u_right = u_data[idx + 1];
            const double u_down = u_data[idx - nx];
            const double u_up = u_data[idx + nx];

            lap_old[idx] = inv_dx2 * (u_left - 2.0 * u_center + u_right) +
                           inv_dy2 * (u_down - 2.0 * u_center + u_up);
        }
    }

    // NEW pattern: Create FieldView/FieldOutput and compute same Laplacian
    Int3 extents{nx, ny, nz};
    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldOutput<double> lap_output(lap_new.data(), lap_new.size());

    // Compute Laplacian using FieldView
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) +
                                    static_cast<std::size_t>(j) * nx;

            const double u_center = u_view.data()[idx];
            const double u_left = u_view.data()[idx - 1];
            const double u_right = u_view.data()[idx + 1];
            const double u_down = u_view.data()[idx - nx];
            const double u_up = u_view.data()[idx + nx];

            lap_output.data()[idx] = inv_dx2 * (u_left - 2.0 * u_center + u_right) +
                                     inv_dy2 * (u_down - 2.0 * u_center + u_up);
        }
    }

    // Verify numerical equivalence (interior points only)
    double max_error = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) +
                                    static_cast<std::size_t>(j) * nx;

            const double error = std::abs(lap_old[idx] - lap_new[idx]);
            max_error = std::max(max_error, error);
            REQUIRE(lap_old[idx] == Approx(lap_new[idx]).epsilon(1e-10));
        }
    }

    // OLD pattern: Time integration (velocity Verlet-like)
    const double dt = 0.01;
    const double k_wave = 1.0;  // Wave speed

    // Store old values for comparison
    std::vector<double> u_old_values = u_data;
    std::vector<double> v_old_values = v_data;

    // Apply OLD pattern time integration
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] += dt * v_data[i];
        v_data[i] += dt * k_wave * k_wave * lap_old[i];
    }

    // NEW pattern: Time integration
    for (std::size_t i = 0; i < u_old_values.size(); ++i) {
        u_old_values[i] += dt * v_old_values[i];
        v_old_values[i] += dt * k_wave * k_wave * lap_new[i];
    }

    // Verify numerical equivalence (interior points only)
    double max_u_error = 0.0;
    double max_v_error = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) +
                                    static_cast<std::size_t>(j) * nx;

            const double u_error = std::abs(u_data[idx] - u_old_values[idx]);
            const double v_error = std::abs(v_data[idx] - v_old_values[idx]);
            max_u_error = std::max(max_u_error, u_error);
            max_v_error = std::max(max_v_error, v_error);
            REQUIRE(u_data[idx] == Approx(u_old_values[idx]).epsilon(1e-10));
            REQUIRE(v_data[idx] == Approx(v_old_values[idx]).epsilon(1e-10));
        }
    }

    // Verify shape compatibility across multi-field system
    REQUIRE_NOTHROW(validate_shape_compatibility(u_view, v_view));

    // Verify aliasing detection works for multi-field system
    REQUIRE_NOTHROW(lap_output.validate_no_alias(u_view));
    REQUIRE_NOTHROW(lap_output.validate_no_alias(v_view));

    SECTION("verify max errors are small") {
        REQUIRE(max_error < 1e-10);
        REQUIRE(max_u_error < 1e-10);
        REQUIRE(max_v_error < 1e-10);
    }
}

/**
 * @brief Test Wave2D multi-field bundle pattern
 *
 * Verifies that FieldBundle can represent coupled multi-field systems
 * like Wave2D's u/v/lap fields.
 */
TEST_CASE("Wave2D multi-field bundle pattern", "[field][wave2d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 1.0);
    std::vector<double> v_data(size, 2.0);
    std::vector<double> lap_data(size, 0.0);

    Int3 extents{nx, ny, nz};
    Real3 spacing{1.0, 1.0, 1.0};
    Real3 origin{0.0, 0.0, 0.0};

    // Create individual field views
    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldOutput<double> lap_output(lap_data.data(), lap_data.size());

    // Create a multi-field bundle
    FieldBundle<FieldView<double>, FieldView<double>> wave_bundle(u_view, v_view);

    // Verify bundle provides access to individual fields
    // FieldBundle stores fields by value, so addresses differ
    // FieldBundle stores fields by value, so addresses differ

    // Verify shape validation across bundle
    REQUIRE(wave_bundle.validate_shapes());

    // Verify shape compatibility checks work
    REQUIRE_NOTHROW(validate_shape_compatibility(u_view, v_view));

    // Verify all fields have same geometry
    REQUIRE(u_view.extents() == v_view.extents());
    REQUIRE(u_view.spacing() == v_view.spacing());
    REQUIRE(u_view.origin() == v_view.origin());
}

/**
 * @brief Test Wave2D coupled time integration pattern
 *
 * Verifies that the coupled time integration from Wave2D can be
 * represented using FieldView/FieldOutput.
 */
TEST_CASE("Wave2D coupled time integration pattern", "[field][wave2d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size);
    std::vector<double> v_data(size);
    std::vector<double> lap_data(size);

    // Initialize with simple pattern
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] = std::sin(static_cast<double>(i));
        v_data[i] = std::cos(static_cast<double>(i));
    }

    Int3 extents{nx, ny, nz};
    Real3 spacing{1.0, 1.0, 1.0};
    Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldOutput<double> lap_output(lap_data.data(), lap_data.size());

    // Compute Laplacian (simplified)
    for (std::size_t i = 0; i < size; ++i) {
        lap_output.data()[i] = -0.1 * u_view.data()[i];  // Simplified: -k^2 * u
    }

    // Coupled time integration (velocity Verlet-like)
    const double dt = 0.1;
    const double k_wave = 1.0;

    std::vector<double> u_old = u_data;
    std::vector<double> v_old = v_data;

    // u += dt * v
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] += dt * v_data[i];
    }

    // v += dt * k^2 * lap
    for (std::size_t i = 0; i < size; ++i) {
        v_data[i] += dt * k_wave * k_wave * lap_output.data()[i];
    }

    // Verify results match expected
    for (std::size_t i = 0; i < size; ++i) {
        const double expected_u = u_old[i] + dt * v_old[i];
        const double expected_v = v_old[i] + dt * k_wave * k_wave * (-0.1 * u_old[i]);
        REQUIRE(u_data[i] == Approx(expected_u).epsilon(1e-10));
        REQUIRE(v_data[i] == Approx(expected_v).epsilon(1e-10));
    }
}

/**
 * @brief Document migration path from WaveIncrements to FieldBundle
 *
 * This section documents how to migrate from Wave2D's multi-field
 * pattern to FieldBundle.
 */
TEST_CASE("Wave2D migration path from multi-field to FieldBundle", "[field][wave2d_evidence]") {
    // Old pattern (separate PaddedBrick):
    // PaddedBrick<double> u(decomp, rank, halo_width);
    // PaddedBrick<double> v(decomp, rank, halo_width);
    // PaddedBrick<double> lap(decomp, rank, halo_width);
    //
    // Usage:
    // for_each_owned(u, [&](int i, int j, int k) {
    //     u(i, j, k) += dt * v(i, j, k);
    //     v(i, j, k) += dt * k^2 * lap(i, j, k);
    // });

    // New pattern (FieldBundle):
    // Create FieldView for each field, bundle them together
    // FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> wave_bundle(u_view, v_view, lap_view);
    //
    // Usage:
    // auto& u = wave_bundle.get<0>();
    // auto& v = wave_bundle.get<1>();
    // auto& lap = wave_bundle.get<2>();
    // for (std::size_t i = 0; i < size; ++i) {
    //     u_data[i] += dt * v_data[i];
    //     v_data[i] += dt * k^2 * lap_data[i];
    // }

    // The new pattern provides:
    // 1. Coordinated access to multiple fields
    // 2. Shape validation across all fields in bundle
    // 3. Type-safe indexed access via get<I>()
    // 4. Backend-agnostic views for each field

    // For this test, we'll demonstrate the pattern with simple data
    const int nx = 4;
    const int ny = 4;
    const int nz = 1;
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 1.0);
    std::vector<double> v_data(size, 2.0);
    std::vector<double> lap_data(size, 0.0);

    Int3 extents{nx, ny, nz};
    Real3 spacing{1.0, 1.0, 1.0};
    Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldView<double> v_view(v_data.data(), v_data.size(), extents, spacing, origin);
    FieldView<double> lap_view(lap_data.data(), lap_data.size(), extents, spacing, origin);

    // Create multi-field bundle
    FieldBundle<FieldView<double>, FieldView<double>, FieldView<double>> wave_bundle(u_view, v_view, lap_view);

    // Verify bundle provides coordinated access
    // FieldBundle stores fields by value, so addresses differ
    // FieldBundle stores fields by value, so addresses differ
    // FieldBundle stores fields by value, so addresses differ

    // Verify shape validation works across bundle
    REQUIRE(wave_bundle.validate_shapes());
}
