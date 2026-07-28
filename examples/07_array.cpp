// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <openpfc/frontend/utils/array_to_string.hpp>
#include <openpfc/frontend/utils/typename.hpp>
#include <openpfc/frontend/utils/utils.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

/**
 * \example 07_array.cpp
 *
 * OpenPFC implements a modern `pfc::data::Field<T>` that replaces legacy
 * `Array<T,D>` and `DiscreteField<T,D>` types. This example demonstrates Field
 * construction using factory functions, direct indexing, element-wise operations,
 * and integration with finite-difference kernels.
 *
 * The field API makes it possible to work with decomposed domains by storing
 * geometry information (origin, spacing) alongside the data. Fields are aware
 * of which part of domain decomposition they represent and can map between
 * grid indices and physical coordinates.
 *
 * This example shows three recommended ways to construct Field objects:
 * - `field_from_subdomain()` creates padded fields with halo regions (for FD stencils)
 * - `field_from_subdomain_unpadded()` creates unpadded fields (for spectral methods)
 * - `field_from_inbox()` creates fields from FFT inbox boxes
 *
 * We also demonstrate integration with OpenPFC's finite-difference kernels:
 * - `FDGradient` for evaluating spatial derivatives
 * - `PaddedHaloExchanger` for MPI halo exchange
 */

using namespace pfc;
using namespace pfc::data;

template <typename T> struct SecondOrderTensor {
  std::array<T, 9> data;

  friend std::ostream &operator<<(std::ostream &os,
                                  const SecondOrderTensor<T> &tensor) {
    os << std::string("SecondOrderTensor<") + TypeName<T>::get().data() + ">"
       << utils::array_to_string(tensor.data);
    return os;
  }
};

// Simple gradient structure for FDGradient demonstration
struct SimpleGrads {
  double xx{};
  double yy{};
  double zz{};
};

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (rank == 0) {
    std::cout << "=== Field API Demonstration ===" << std::endl;
    std::cout << "Using " << nproc << " MPI process(es)" << std::endl;
  }

  // Create a global domain
  int Lx = 16;
  int Ly = 8;
  int Lz = 1; // 2D simulation extended to 3D
  auto domain = domain::create({Lx, Ly, Lz});
  
  // Create domain decomposition
  auto decomp = decomposition::create(domain, nproc);

  // === Factory function demonstrations ===
  
  // 1. field_from_subdomain: creates padded fields with halo regions
  // This is the recommended approach for finite-difference methods
  const int halo_width = 1;
  auto field_padded = field_from_subdomain<double>(decomp, rank, halo_width);
  
  if (rank == 0) {
    std::cout << "\n--- field_from_subdomain (padded with halos) ---" << std::endl;
    std::cout << "Local size (with halos): [" 
              << field_padded.local_size()[0] << ","
              << field_padded.local_size()[1] << ","
              << field_padded.local_size()[2] << "]" << std::endl;
    std::cout << "Owned box: [" << field_padded.box().low[0] << ","
              << field_padded.box().low[1] << "," << field_padded.box().low[2]
              << "] to [" << field_padded.box().high[0] << ","
              << field_padded.box().high[1] << "," << field_padded.box().high[2]
              << "]" << std::endl;
  }

  // 2. field_from_subdomain_unpadded: creates unpadded fields
  // This is useful for spectral methods and algorithms that don't need halos
  auto field_unpadded = field_from_subdomain_unpadded<double>(decomp, rank, 0);
  
  if (rank == 0) {
    std::cout << "\n--- field_from_subdomain_unpadded (no padding) ---" << std::endl;
    std::cout << "Local size (unpadded): ["
              << field_unpadded.local_size()[0] << ","
              << field_unpadded.local_size()[1] << ","
              << field_unpadded.local_size()[2] << "]" << std::endl;
  }

  // 3. field_from_inbox: creates fields from FFT inbox boxes
  // This demonstrates compatibility with spectral methods
  auto inbox = decomposition::local_box(decomp, rank);
  auto field_inbox = field_from_inbox<double>(domain, inbox);
  
  if (rank == 0) {
    std::cout << "\n--- field_from_inbox (from inbox box) ---" << std::endl;
    std::cout << "Inbox size: ["
              << field_inbox.local_size()[0] << ","
              << field_inbox.local_size()[1] << ","
              << field_inbox.local_size()[2] << "]" << std::endl;
  }

  // === Direct indexing demonstration ===
  
  // Demonstrate operator() for direct element access
  if (rank == 0) {
    std::cout << "\n--- Direct indexing with operator() ---" << std::endl;
  }
  
  // Set some values using direct indexing
  field_padded(halo_width, halo_width, halo_width) = 42.0;
  field_unpadded(0, 0, 0) = 24.0;
  
  if (rank == 0) {
    std::cout << "field_padded(halo_width, halo_width, halo_width) = "
              << field_padded(halo_width, halo_width, halo_width) << std::endl;
    std::cout << "field_unpadded(0, 0, 0) = "
              << field_unpadded(0, 0, 0) << std::endl;
  }

  // === Direct data access ===
  
  if (rank == 0) {
    std::cout << "\n--- Direct data access via data() ---" << std::endl;
    std::cout << "First element via data(): " 
              << field_padded.data()[0] << std::endl;
  }

  // === Element-wise operations with apply() ===
  
  if (rank == 0) {
    std::cout << "\n--- Element-wise operations with apply() ---" << std::endl;
  }
  
  // Apply a function based on physical coordinates
  field_unpadded.apply([](double x, double y, double /*z*/) {
    return 1.0 + x + y * y;
  });
  
  if (rank == 0) {
    // Check the result at first owned cell
    std::cout << "After apply: field_unpadded(0, 0, 0) = "
              << field_unpadded(0, 0, 0) << std::endl;
    
    // Also check coordinates mapping
    auto coords = field_unpadded.coords(0, 0, 0);
    std::cout << "Physical coordinates at (0,0,0): ["
              << coords[0] << "," << coords[1] << "," << coords[2] << "]" << std::endl;
  }

  // === Iteration with for_each_owned ===
  
  if (rank == 0) {
    std::cout << "\n--- Iteration with for_each_owned ---" << std::endl;
    int count = 0;
    field_padded.for_each_owned([&](int i, int j, int k) {
      (void)i; (void)j; (void)k;
      count++;
    });
    std::cout << "Number of owned cells: " << count << std::endl;
  }

  // === Integration with FDGradient ===
  
  if (rank == 0) {
    std::cout << "\n--- Integration with FDGradient ---" << std::endl;
  }
  
  // Create a separate field for gradient demonstration with sufficient size
  int gx = 8, gy = 8, gz = 4;  // Larger field for gradient testing
  auto grad_domain = domain::create({gx, gy, gz});
  auto grad_decomp = decomposition::create(grad_domain, nproc);
  auto grad_field = field_from_subdomain<double>(grad_decomp, rank, halo_width);
  
  // Create a gradient evaluator for the gradient field
  const int fd_order = 2;
  pfc::gradient::FDGradient<SimpleGrads> grad(grad_field, fd_order);
  
  // Initialize field with a simple quadratic function for gradient testing
  // f(x,y,z) = x^2 + y^2 + z^2, Laplacian should be 6
  grad_field.for_each_owned([&](int i, int j, int k) {
    auto coords = grad_field.coords(i, j, k);
    grad_field(i, j, k) = coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
  });
  
  // Exchange halos before gradient evaluation
  pfc::communication::PaddedHaloExchanger<double> grad_halo(grad_field, grad_decomp, rank,
                                                             MPI_COMM_WORLD);
  pfc::communication::exchange(grad_halo);
  
  // Evaluate gradients at interior points
  if (rank == 0) {
    auto local_size = grad_field.local_size();
    for (int k = halo_width; k < local_size[2] - halo_width; ++k) {
      for (int j = halo_width; j < local_size[1] - halo_width; ++j) {
        for (int i = halo_width; i < local_size[0] - halo_width; ++i) {
          auto g = pfc::gradient::evaluate(grad, pfc::Int3{i, j, k});
          double laplacian = g.xx + g.yy + g.zz;
          std::cout << "Gradient at interior point: xx=" << g.xx << ", yy=" << g.yy 
                    << ", zz=" << g.zz << ", laplacian=" << laplacian
                    << " (expected ~6.0 for quadratic)" << std::endl;
          goto gradient_done;
        }
      }
    }
    gradient_done:;
  }

  // === Field of complex objects ===
  
  if (rank == 0) {
    std::cout << "\n--- Field of complex objects (tensors) ---" << std::endl;
  }
  
  auto tensor_domain = domain::create({2, 2, 1});
  auto tensor_decomp = decomposition::create(tensor_domain, nproc);
  auto tensor_field = field_from_subdomain<SecondOrderTensor<double>>(
      tensor_decomp, rank, 0);
  
  // Set a tensor value
  tensor_field(0, 0, 0) = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  
  if (rank == 0) {
    std::cout << "Tensor at (0,0,0): " << tensor_field(0, 0, 0) << std::endl;
    
    // Iterate over owned cells
    tensor_field.for_each_owned([&](int i, int j, int k) {
      std::cout << "Local index [" << i << "," << j << "," << k << "] => "
                << tensor_field(i, j, k) << std::endl;
    });
  }

  // === Summary ===
  
  if (rank == 0) {
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "This example demonstrated:" << std::endl;
    std::cout << "- field_from_subdomain: padded fields with halos for FD methods" << std::endl;
    std::cout << "- field_from_subdomain_unpadded: unpadded fields for spectral methods" << std::endl;
    std::cout << "- field_from_inbox: fields from FFT inbox boxes" << std::endl;
    std::cout << "- Direct indexing with operator()" << std::endl;
    std::cout << "- Direct data access via data()" << std::endl;
    std::cout << "- Element-wise operations with apply()" << std::endl;
    std::cout << "- Iteration with for_each_owned()" << std::endl;
    std::cout << "- Integration with FDGradient for derivative evaluation" << std::endl;
    std::cout << "- Integration with PaddedHaloExchanger for MPI halo exchange" << std::endl;
    std::cout << "- Fields of complex objects (tensors)" << std::endl;
  }

  MPI_Finalize();
  return 0;
}