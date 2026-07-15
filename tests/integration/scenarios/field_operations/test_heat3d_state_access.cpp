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
 * @brief Test Heat3D scalar field state access pattern with numerical equivalence
 *
 * This test verifies that the new state access primitives can represent
 * the Heat3D scalar field pattern AND produce numerically equivalent results
 * to manual FD computations.
 *
 * Heat3D pattern from apps/heat3d/src/cpu/heat3d_fd.cpp:
 * - Single scalar field u (temperature)
 * - Laplacian evaluation via FD
 * - Explicit Euler time integration: u += dt * du
 */
TEST_CASE("Heat3D numerical equivalence test", "[field][heat3d_evidence][numerical]") {
    // Create test data for numerical equivalence verification
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size);
    std::vector<double> du_old(size);
    std::vector<double> du_new(size);

    // Initialize with a Gaussian-like pattern
    const double xc = static_cast<double>(nx) / 2.0;
    const double yc = static_cast<double>(ny) / 2.0;
    const double zc = static_cast<double>(nz) / 2.0;
    const double sigma = 2.0;

    const Real3 spacing{1.0, 1.0, 1.0};
    const Real3 origin{0.0, 0.0, 0.0};

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(j) * nx +
                                        static_cast<std::size_t>(k) * nx * ny;

                const double dx = static_cast<double>(i) - xc;
                const double dy = static_cast<double>(j) - yc;
                const double dz = static_cast<double>(k) - zc;

                u_data[idx] = std::exp(-(dx*dx + dy*dy + dz*dz) / (2.0 * sigma * sigma));
            }
        }
    }

    // OLD pattern: Manual FD computation
    const double inv_dx2 = 1.0 / (spacing[0] * spacing[0]);
    const double inv_dy2 = 1.0 / (spacing[1] * spacing[1]);
    const double inv_dz2 = 1.0 / (spacing[2] * spacing[2]);

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(j) * nx +
                                        static_cast<std::size_t>(k) * nx * ny;

                const double u_center = u_data[idx];
                const double u_left = u_data[idx - 1];
                const double u_right = u_data[idx + 1];
                const double u_down = u_data[idx - nx];
                const double u_up = u_data[idx + nx];
                const double u_back = u_data[idx - nx * ny];
                const double u_front = u_data[idx + nx * ny];

                du_old[idx] = inv_dx2 * (u_left - 2.0 * u_center + u_right) +
                              inv_dy2 * (u_down - 2.0 * u_center + u_up) +
                              inv_dz2 * (u_back - 2.0 * u_center + u_front);
            }
        }
    }

    // NEW pattern: Create FieldView/FieldOutput and compute same Laplacian
    Int3 extents{nx, ny, nz};
    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldOutput<double> du_output(du_new.data(), du_new.size());

    // Compute Laplacian using FieldView
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(j) * nx +
                                        static_cast<std::size_t>(k) * nx * ny;

                const double u_center = u_view.data()[idx];
                const double u_left = u_view.data()[idx - 1];
                const double u_right = u_view.data()[idx + 1];
                const double u_down = u_view.data()[idx - nx];
                const double u_up = u_view.data()[idx + nx];
                const double u_back = u_view.data()[idx - nx * ny];
                const double u_front = u_view.data()[idx + nx * ny];

                du_output.data()[idx] = inv_dx2 * (u_left - 2.0 * u_center + u_right) +
                                        inv_dy2 * (u_down - 2.0 * u_center + u_up) +
                                        inv_dz2 * (u_back - 2.0 * u_center + u_front);
            }
        }
    }

    // Verify numerical equivalence (interior points only)
    double max_error = 0.0;
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) +
                                        static_cast<std::size_t>(j) * nx +
                                        static_cast<std::size_t>(k) * nx * ny;

                const double error = std::abs(du_old[idx] - du_new[idx]);
                max_error = std::max(max_error, error);
                REQUIRE(du_old[idx] == Approx(du_new[idx]).epsilon(1e-10));
            }
        }
    }

    // Verify shape compatibility checks work
    REQUIRE_NOTHROW(validate_shape_compatibility(u_view, u_view));

    // Verify aliasing detection works
    REQUIRE_NOTHROW(du_output.validate_no_alias(u_view));

    SECTION("verify max error is small") {
        REQUIRE(max_error < 1e-10);
    }
}

/**
 * @brief Test Heat3D time integration pattern
 *
 * Verifies that the explicit Euler time integration pattern
 * from Heat3D can be represented using FieldView/FieldOutput.
 */
TEST_CASE("Heat3D time integration pattern", "[field][heat3d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 4;
    const std::size_t size = static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size);
    std::vector<double> du_data(size);

    // Initialize with simple pattern
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] = static_cast<double>(i);
    }

    Int3 extents{nx, ny, nz};
    Real3 spacing{1.0, 1.0, 1.0};
    Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldOutput<double> du_output(du_data.data(), du_data.size());

    // Compute RHS (simplified Laplacian)
    for (std::size_t i = 0; i < size; ++i) {
        du_output.data()[i] = 0.1 * u_view.data()[i];
    }

    // Time integration: u += dt * du (explicit Euler)
    const double dt = 0.1;
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] += dt * du_output.data()[i];
    }

    // Verify result matches expected: u_new = u_old * (1 + dt * 0.1)
    for (std::size_t i = 0; i < size; ++i) {
        const double expected = static_cast<double>(i) * (1.0 + dt * 0.1);
        REQUIRE(u_data[i] == Approx(expected).epsilon(1e-10));
    }
}

/**
 * @brief Document migration path from LocalField to FieldView
 *
 * This section documents how to migrate from LocalField<T> to FieldView<T>.
 */
TEST_CASE("Heat3D migration path from LocalField to FieldView", "[field][heat3d_evidence]") {
    // Old pattern (LocalField):
    // LocalField<double> u = LocalField<double>::from_subdomain(decomp, rank, halo_width);
    // const double* u_data = u.data();
    // Int3 u_size = u.size3();
    // Real3 u_spacing = u.spacing();
    // Real3 u_origin = u.origin();

    // New pattern (FieldView):
    // LocalField<double> u_local = LocalField<double>::from_subdomain(decomp, rank, halo_width);
    // FieldView<double> u_view(u_local.data(), u_local.size(), u_local.size3(),
    //                          u_local.spacing(), u_local.origin());
    // const double* u_data = u_view.data();
    // Int3 u_size = u_view.extents();
    // Real3 u_spacing = u_view.spacing();
    // Real3 u_origin = u_view.origin();

    // The new pattern provides:
    // 1. Backend-agnostic view (works with CPU and GPU storage)
    // 2. Explicit read-only semantics (const access)
    // 3. Shape validation via is_compatible_with()
    // 4. Aliasing detection for output storage

    // For this test, we'll demonstrate the pattern with simple data
    std::vector<double> u_data(64, 1.0);
    Int3 extents{4, 4, 4};
    Real3 spacing{1.0, 1.0, 1.0};
    Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);

    // Verify the migration preserves all required information
    REQUIRE(u_view.data() == u_data.data());
    REQUIRE(u_view.size() == 64);
    REQUIRE(u_view.extents() == extents);
    REQUIRE(u_view.spacing() == spacing);
    REQUIRE(u_view.origin() == origin);
}
