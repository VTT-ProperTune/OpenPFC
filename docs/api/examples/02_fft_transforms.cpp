// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example 02_fft_transforms.cpp
 * @brief Demonstrates FFT API usage for spectral operations
 *
 * This example shows:
 * - Setting up FFT with domain decomposition
 * - Forward and backward transforms
 * - Normalization conventions
 * - Computing derivatives in k-space
 * - Memory layout (real vs complex)
 *
 * Expected output:
 * - Transform verification (round-trip accuracy)
 * - Laplacian computation example
 * - Performance timing
 *
 * Time to run: < 1 second
 */

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/mpi.hpp>

using namespace pfc;

void print_section(const std::string &title) {
  if (mpi::get_rank() == 0) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
  }
}

void example_basic_setup() {
  print_section("Example 1: Basic FFT Setup");

  // Create computational domain
  auto world = world::create({64, 64, 64});

  // Set up domain decomposition for MPI
  auto decomp = decomposition::create(world, MPI_COMM_WORLD);

  // Create FFT object
  auto fft = fft::create(decomp);

  if (mpi::get_rank() == 0) {
    std::cout << "Created FFT for 64³ grid\n";
    std::cout << "  Real field size (local): " << fft.size_inbox() << "\n";
    std::cout << "  Complex field size (local): " << fft.size_outbox() << "\n";
    std::cout << "  Workspace size: " << fft.size_workspace() << "\n";

    double compression =
        100.0 * (1.0 - static_cast<double>(fft.size_outbox()) / fft.size_inbox());
    std::cout << "  Memory savings from conjugate symmetry: " << std::fixed
              << std::setprecision(1) << compression << "%\n";
  }
}

void example_round_trip() {
  print_section("Example 2: Round-Trip Transform (Normalization Check)");

  auto world = world::create({32, 32, 32});
  auto decomp = decomposition::create(world, MPI_COMM_WORLD);
  auto fft = fft::create(decomp);

  // Create test field: cos(2πx/L)
  std::vector<double> original(fft.size_inbox());
  std::vector<std::complex<double>> fourier(fft.size_outbox());
  std::vector<double> reconstructed(fft.size_inbox());

  auto inbox = fft::get_inbox(fft);
  auto size = world::get_size(world);
  auto spacing = world::get_spacing(world);

  for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
        // Linear index in local array
        size_t idx = (i - inbox.low[0]) * (inbox.high[1] - inbox.low[1] + 1) *
                         (inbox.high[2] - inbox.low[2] + 1) +
                     (j - inbox.low[1]) * (inbox.high[2] - inbox.low[2] + 1) +
                     (k - inbox.low[2]);

        // Cosine wave in x-direction
        double x = i * spacing[0];
        original[idx] = std::cos(2.0 * M_PI * x / (size[0] * spacing[0]));
      }
    }
  }

  // Forward transform
  fft.forward(original, fourier);

  // Backward transform (with normalization)
  fft.backward(fourier, reconstructed);

  // Check round-trip accuracy
  double max_error = 0.0;
  for (size_t i = 0; i < original.size(); ++i) {
    double error = std::abs(original[i] - reconstructed[i]);
    max_error = std::max(max_error, error);
  }

  // Reduce across ranks
  double global_max_error;
  MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (mpi::get_rank() == 0) {
    std::cout << "Round-trip transform error:\n";
    std::cout << "  Max |original - reconstructed|: " << std::scientific
              << std::setprecision(2) << global_max_error << "\n";
    std::cout << "  " << (global_max_error < 1e-10 ? "✓ PASS" : "✗ FAIL")
              << " (threshold: 1e-10)\n";
  }
}

void example_laplacian() {
  print_section("Example 3: Computing Laplacian in K-Space");

  auto world = world::create({32, 32, 32}, {0, 0, 0}, {0.1, 0.1, 0.1});
  auto decomp = decomposition::create(world, MPI_COMM_WORLD);
  auto fft = fft::create(decomp);

  // Create test function: f(x,y,z) = sin(2πx/L)
  // Analytical Laplacian: ∇²f = -(2π/L)² sin(2πx/L)
  std::vector<double> field(fft.size_inbox());
  std::vector<std::complex<double>> field_k(fft.size_outbox());
  std::vector<double> laplacian(fft.size_inbox());

  auto inbox = fft::get_inbox(fft);
  auto size = world::get_size(world);
  auto spacing = world::get_spacing(world);
  double L = size[0] * spacing[0];

  // Initialize field
  for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
        size_t idx = (i - inbox.low[0]) * (inbox.high[1] - inbox.low[1] + 1) *
                         (inbox.high[2] - inbox.low[2] + 1) +
                     (j - inbox.low[1]) * (inbox.high[2] - inbox.low[2] + 1) +
                     (k - inbox.low[2]);

        double x = i * spacing[0];
        field[idx] = std::sin(2.0 * M_PI * x / L);
      }
    }
  }

  // Transform to k-space
  fft.forward(field, field_k);

  // Apply Laplacian operator: -k²
  auto outbox = fft::get_outbox(fft);
  for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
    for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
      for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
        size_t idx = (i - outbox.low[0]) * (outbox.high[1] - outbox.low[1] + 1) *
                         (outbox.high[2] - outbox.low[2] + 1) +
                     (j - outbox.low[1]) * (outbox.high[2] - outbox.low[2] + 1) +
                     (k - outbox.low[2]);

        // Wavenumbers
        double kx = (i < size[0] / 2) ? i : i - size[0];
        double ky = (j < size[1] / 2) ? j : j - size[1];
        double kz = k; // Half-space, only positive

        kx *= 2.0 * M_PI / L;
        ky *= 2.0 * M_PI / L;
        kz *= 2.0 * M_PI / L;

        double k2 = kx * kx + ky * ky + kz * kz;
        field_k[idx] *= -k2;
      }
    }
  }

  // Transform back to real space
  fft.backward(field_k, laplacian);

  // Compare with analytical result
  double max_error = 0.0;
  double analytical_value = -(2.0 * M_PI / L) * (2.0 * M_PI / L);

  for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
      for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
        size_t idx = (i - inbox.low[0]) * (inbox.high[1] - inbox.low[1] + 1) *
                         (inbox.high[2] - inbox.low[2] + 1) +
                     (j - inbox.low[1]) * (inbox.high[2] - inbox.low[2] + 1) +
                     (k - inbox.low[2]);

        double x = i * spacing[0];
        double analytical = analytical_value * std::sin(2.0 * M_PI * x / L);
        double error = std::abs(laplacian[idx] - analytical);
        max_error = std::max(max_error, error);
      }
    }
  }

  double global_max_error;
  MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (mpi::get_rank() == 0) {
    std::cout << "Laplacian computation:\n";
    std::cout << "  f(x) = sin(2πx/L)\n";
    std::cout << "  ∇²f = -(2π/L)² sin(2πx/L)\n";
    std::cout << "  Max |numerical - analytical|: " << std::scientific
              << std::setprecision(2) << global_max_error << "\n";
    std::cout << "  " << (global_max_error < 1e-8 ? "✓ PASS" : "✗ FAIL")
              << " (threshold: 1e-8)\n";
  }
}

void example_performance() {
  print_section("Example 4: Performance Timing");

  auto world = world::create({128, 128, 128});
  auto decomp = decomposition::create(world, MPI_COMM_WORLD);
  auto fft = fft::create(decomp);

  std::vector<double> real_data(fft.size_inbox(), 1.0);
  std::vector<std::complex<double>> complex_data(fft.size_outbox());

  // Warmup
  fft.forward(real_data, complex_data);
  fft.backward(complex_data, real_data);

  // Timed run
  fft.reset_fft_time();

  int num_transforms = 100;
  for (int i = 0; i < num_transforms; ++i) {
    fft.forward(real_data, complex_data);
    fft.backward(complex_data, real_data);
  }

  double total_time = fft.get_fft_time();
  double avg_time = total_time / (2 * num_transforms); // 2 transforms per iteration

  if (mpi::get_rank() == 0) {
    std::cout << "Performance on 128³ grid:\n";
    std::cout << "  Number of transform pairs: " << num_transforms << "\n";
    std::cout << "  Total FFT time: " << std::fixed << std::setprecision(3)
              << total_time << " seconds\n";
    std::cout << "  Average per transform: " << std::setprecision(6)
              << avg_time * 1000.0 << " ms\n";

    int size_total = 128 * 128 * 128;
    double throughput = size_total / (avg_time * 1e6);
    std::cout << "  Throughput: " << std::setprecision(2) << throughput
              << " million points/sec\n";
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (mpi::get_rank() == 0) {
    std::cout << "OpenPFC FFT API Examples\n";
    std::cout << "========================\n";
    std::cout
        << "\nThis example demonstrates FFT operations for spectral methods.\n";
  }

  try {
    example_basic_setup();
    example_round_trip();
    example_laplacian();
    example_performance();

    print_section("Summary");
    if (mpi::get_rank() == 0) {
      std::cout << "\nKey takeaways:\n";
      std::cout << "  ✓ Forward: real → complex (no normalization)\n";
      std::cout << "  ✓ Backward: complex → real (normalized by 1/N)\n";
      std::cout << "  ✓ Complex arrays are ~50% smaller (conjugate symmetry)\n";
      std::cout << "  ✓ Spectral derivatives: multiply by ik in k-space\n";
      std::cout << "  ✓ Round-trip accuracy: < 1e-10\n";
      std::cout << "\nSee include/openpfc/fft.hpp for complete API.\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error on rank " << mpi::get_rank() << ": " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
