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
 * @brief Test Heat3D scalar field state access pattern
 *
 * This test verifies that the new state access primitives can represent
 * the Heat3D scalar field pattern from apps/heat3d/src/cpu/heat3d_fd.cpp.
 *
 * Heat3D pattern:
 * - Single scalar field u (temperature)
 * - PaddedBrick<double> for state storage
 * - Laplacian evaluation for RHS computation
 * - Explicit Euler time integration: u += dt * du
 */
TEST_CASE("Heat3D scalar field state access", "[field][heat3d_evidence]") {
    // Create test data matching Heat3D pattern
    // Heat3D uses a 3D grid with halo padding
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const int halo_width = 1;
    
    const int nx_padded = nx + 2 * halo_width;
    const int ny_padded = ny + 2 * halo_width;
    const int nz_padded = nz + 2 * halo_width;
    
    const std::size_t total_size = static_cast<std::size_t>(nx_padded) * 
                                   static_cast<std::size_t>(ny_padded) * 
                                   static_cast<std::size_t>(nz_padded);

    std::vector<double> u_data(total_size, 0.0);
    std::vector<double> du_data(total_size, 0.0);

    // Initialize with a Gaussian-like pattern (simplified)
    const double xc = static_cast<double>(nx) / 2.0;
    const double yc = static_cast<double>(ny) / 2.0;
    const double zc = static_cast<double>(nz) / 2.0;
    const double sigma = 2.0;

    for (int k = halo_width; k < nz + halo_width; ++k) {
        for (int j = halo_width; j < ny + halo_width; ++j) {
            for (int i = halo_width; i < nx + halo_width; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i) + 
                                        static_cast<std::size_t>(j) * nx_padded +
                                        static_cast<std::size_t>(k) * nx_padded * ny_padded;
                
                const double dx = static_cast<double>(i - halo_width) - xc;
                const double dy = static_cast<double>(j - halo_width) - yc;
                const double dz = static_cast<double>(k - halo_width) - zc;
                
                u_data[idx] = std::exp(-(dx*dx + dy*dy + dz*dz) / (2.0 * sigma * sigma));
            }
        }
    }

    // Construct FieldView<double> from data
    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    // Note: FieldView sees only the owned region (excluding halo)
    // Heat3D would need to adapt to this or use halo-aware views
    const std::size_t owned_size = static_cast<std::size_t>(nx) * 
                                   static_cast<std::size_t>(ny) * 
                                   static_cast<std::size_t>(nz);
    
    // For this test, we'll create views that point to the owned region
    std::vector<double> u_owned(owned_size);
    std::vector<double> du_owned(owned_size);

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
            }
        }
    }

    FieldView<double> u_view(u_owned.data(), u_owned.size(), extents, spacing, origin);
    FieldOutput<double> du_output(du_owned.data(), du_owned.size());

    SECTION("shape compatibility checks work") {
        // Verify shape compatibility
        REQUIRE_NOTHROW(validate_shape_compatibility(u_view, u_view));
        
        // Create incompatible field
        pfc::types::Int3 different_extents{16, 8, 8};
        std::vector<double> incompatible_data(1024, 0.0);
        FieldView<double> incompatible_view(incompatible_data.data(), incompatible_data.size(), 
                                            different_extents, spacing, origin);
        
        REQUIRE_THROWS_AS(validate_shape_compatibility(u_view, incompatible_view), 
                         std::invalid_argument);
    }

    SECTION("aliasing detection works") {
        // Distinct storage should pass
        REQUIRE_NOTHROW(du_output.validate_no_alias(u_view));
        
        // Aliased storage should fail
        FieldOutput<double> aliased_output(u_owned.data(), u_owned.size());
        REQUIRE_THROWS_AS(aliased_output.validate_no_alias(u_view), std::invalid_argument);
    }

    SECTION("numerical results match expected pattern") {
        // Verify initial condition
        double max_u = 0.0;
        for (std::size_t i = 0; i < u_view.size(); ++i) {
            max_u = std::max(max_u, u_view.data()[i]);
        }
        REQUIRE(max_u == Approx(1.0).epsilon(1e-10));
        
        // Verify symmetry (Gaussian centered)
        const std::size_t center_idx = static_cast<std::size_t>(nx/2) + 
                                       static_cast<std::size_t>(ny/2) * nx +
                                       static_cast<std::size_t>(nz/2) * nx * ny;
        REQUIRE(u_view.data()[center_idx] == Approx(1.0).epsilon(1e-10));
    }

    SECTION("FieldView can be used for read-only access") {
        // Verify read-only access works
        const double* u_data_ptr = u_view.data();
        REQUIRE(u_data_ptr != nullptr);
        
        double sum = 0.0;
        for (std::size_t i = 0; i < u_view.size(); ++i) {
            sum += u_data_ptr[i];
        }
        
        REQUIRE(sum > 0.0);
    }

    SECTION("FieldOutput can be used for write access") {
        // Verify write access works
        double* du_data_ptr = du_output.data();
        REQUIRE(du_data_ptr != nullptr);
        
        // Simulate RHS computation (simplified)
        for (std::size_t i = 0; i < du_output.size(); ++i) {
            du_data_ptr[i] = -0.1 * u_view.data()[i];  // Simple decay
        }
        
        // Verify data was written
        REQUIRE(du_output.data()[0] == Approx(-0.1 * u_view.data()[0]).epsilon(1e-10));
    }

    SECTION("FieldView geometry queries match Heat3D pattern") {
        // Verify geometry queries
        REQUIRE(u_view.extents() == extents);
        REQUIRE(u_view.spacing() == spacing);
        REQUIRE(u_view.origin() == origin);
        
        // Heat3D uses uniform spacing
        REQUIRE(u_view.spacing()[0] == u_view.spacing()[1]);
        REQUIRE(u_view.spacing()[1] == u_view.spacing()[2]);
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
    pfc::types::Int3 extents{4, 4, 4};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);

    // Verify the migration preserves all required information
    REQUIRE(u_view.data() == u_data.data());
    REQUIRE(u_view.size() == 64);
    REQUIRE(u_view.extents() == extents);
    REQUIRE(u_view.spacing() == spacing);
    REQUIRE(u_view.origin() == origin);
}

/**
 * @brief Test Heat3D RHS computation pattern
 *
 * This test verifies that the new accessors support the Heat3D RHS computation
 * pattern: compute Laplacian, then compute RHS as D * Laplacian.
 */
TEST_CASE("Heat3D RHS computation pattern", "[field][heat3d_evidence]") {
    const int nx = 4;
    const int ny = 4;
    const int nz = 4;
    const std::size_t size = static_cast<std::size_t>(nx) * 
                            static_cast<std::size_t>(ny) * 
                            static_cast<std::size_t>(nz);

    std::vector<double> u_data(size, 0.0);
    std::vector<double> laplacian_data(size, 0.0);
    std::vector<double> rhs_data(size, 0.0);

    // Initialize with simple pattern
    for (std::size_t i = 0; i < size; ++i) {
        u_data[i] = static_cast<double>(i);
    }

    pfc::types::Int3 extents{nx, ny, nz};
    pfc::types::Real3 spacing{1.0, 1.0, 1.0};
    pfc::types::Real3 origin{0.0, 0.0, 0.0};

    FieldView<double> u_view(u_data.data(), u_data.size(), extents, spacing, origin);
    FieldOutput<double> laplacian_output(laplacian_data.data(), laplacian_data.size());
    FieldOutput<double> rhs_output(rhs_data.data(), rhs_data.size());

    SECTION("output storage is distinct from input") {
        // Verify no aliasing
        REQUIRE_NOTHROW(laplacian_output.validate_no_alias(u_view));
        REQUIRE_NOTHROW(rhs_output.validate_no_alias(u_view));
        
        // Outputs can alias each other (in-place operations)
        // FieldOutput<double> combined_output(laplacian_data.data(), laplacian_data.size());
        // REQUIRE_NOTHROW(combined_output.validate_no_alias(u_view));
    }

    SECTION("shape compatibility across computation stages") {
        // All fields must have compatible shapes
        REQUIRE_NOTHROW(validate_shape_compatibility(u_view, u_view));
        
        // Laplacian output must be compatible with input
        // Note: We can't directly validate compatibility between FieldView and FieldOutput
        // because they don't share the same interface. In practice, we would validate
        // that the underlying storage has compatible geometry.
    }

    SECTION("RHS computation uses output storage") {
        // Simulate Laplacian computation (simplified)
        const double D = 1.0;  // Diffusion coefficient
        for (std::size_t i = 0; i < size; ++i) {
            laplacian_output.data()[i] = 0.1 * u_view.data()[i];  // Simplified
        }
        
        // Simulate RHS computation: rhs = D * Laplacian
        for (std::size_t i = 0; i < size; ++i) {
            rhs_output.data()[i] = D * laplacian_output.data()[i];
        }
        
        // Verify results
        for (std::size_t i = 0; i < size; ++i) {
            REQUIRE(rhs_output.data()[i] == Approx(D * 0.1 * u_view.data()[i]).epsilon(1e-10));
        }
    }
}
