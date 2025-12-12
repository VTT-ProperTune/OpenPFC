#include "heffte.h"
#include <cmath>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <vector>

/*!
 * \brief CUDA FFT example demonstrating Laplace operator in Fourier domain
 *
 * This example:
 * 1. Creates a 3D array on CPU (test function: sin(x)*sin(y)*sin(z))
 * 2. Transfers it to GPU
 * 3. Performs forward FFT
 * 4. Applies Laplace operator in Fourier domain (multiply by -k²) - ALL ON GPU
 * 5. Performs inverse FFT to get Laplacian in real space
 * 6. Transfers result back to CPU
 * 7. Verifies against analytical solution: ∇²(sin(x)sin(y)sin(z)) =
 * -3*sin(x)sin(y)sin(z)
 */

// CUDA kernel for scaling
// Note: std::complex is not available in device code, so we use cuDoubleComplex
__global__ void scale_kernel(cuDoubleComplex *data, int size, double scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = cuCmul(data[idx], make_cuDoubleComplex(scale, 0.0));
  }
}

// CUDA kernel to apply Laplacian operator in Fourier domain
// Multiplies each element by -k² where k² = kx² + ky² + kz²
// Note: std::complex is not available in device code, so we use cuDoubleComplex
__global__ void apply_laplacian_kernel(cuDoubleComplex *data, int nx, int ny, int nz,
                                       int i_low, int i_high, int j_low, int j_high,
                                       int k_low, int k_high, int local_nx,
                                       int local_ny, int local_nz) {
  // Calculate global thread indices
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = local_nx * local_ny * local_nz;

  if (idx >= total_elements) return;

  // Convert linear index to 3D local indices
  int local_k = idx / (local_nx * local_ny);
  int local_j = (idx % (local_nx * local_ny)) / local_nx;
  int local_i = idx % local_nx;

  // Convert to global indices
  int i = i_low + local_i;
  int j = j_low + local_j;
  int k = k_low + local_k;

  // Calculate wavenumbers with Nyquist folding
  double fx = 1.0; // Domain is [0, 2π]
  double fy = 1.0;
  double fz = 1.0;

  double kx = (i <= nx / 2) ? i * fx : (i - nx) * fx;
  double ky = (j <= ny / 2) ? j * fy : (j - ny) * fy;
  double kz = (k <= nz / 2) ? k * fz : (k - nz) * fz;

  // Laplacian in Fourier space: -k²
  double k_squared = kx * kx + ky * ky + kz * kz;
  double laplacian_factor = -k_squared;

  // Multiply the Fourier coefficient by the Laplacian factor
  cuDoubleComplex factor = make_cuDoubleComplex(laplacian_factor, 0.0);
  data[idx] = cuCmul(data[idx], factor);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // Use CUDA backend
  using backend_tag = heffte::backend::cufft;

  int const my_rank = heffte::mpi::comm_rank(MPI_COMM_WORLD);
  int const num_ranks = heffte::mpi::comm_size(MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "=== HeFFTe CUDA FFT Example ===" << std::endl;
    std::cout << "Number of MPI ranks: " << num_ranks << std::endl;
    std::cout << "CUDA device count: " << heffte::gpu::device_count() << std::endl;
  }

  // Set GPU device for this MPI rank (for multi-GPU setups)
  if (heffte::gpu::device_count() > 0) {
    int device_id = my_rank % heffte::gpu::device_count();
    heffte::gpu::device_set(device_id);

    // Verify we're on the correct device and get device properties
    int current_device;
    cudaGetDevice(&current_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    // Format UUID
    char uuid_str[64];
    snprintf(uuid_str, sizeof(uuid_str),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             prop.uuid.bytes[0], prop.uuid.bytes[1], prop.uuid.bytes[2],
             prop.uuid.bytes[3], prop.uuid.bytes[4], prop.uuid.bytes[5],
             prop.uuid.bytes[6], prop.uuid.bytes[7], prop.uuid.bytes[8],
             prop.uuid.bytes[9], prop.uuid.bytes[10], prop.uuid.bytes[11],
             prop.uuid.bytes[12], prop.uuid.bytes[13], prop.uuid.bytes[14],
             prop.uuid.bytes[15]);

    std::cout << "Rank " << my_rank << " using GPU device " << device_id
              << " (verified: " << current_device << ")"
              << " - " << prop.name << " [UUID: " << uuid_str << "]" << std::endl;
  } else {
    if (my_rank == 0) {
      std::cerr << "ERROR: No CUDA devices found!" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Define a 3D domain that works well with multiple ranks
  // Use a larger domain that can be properly decomposed
  int const nx = 64, ny = 64, nz = 64;

  // Create boxes for domain decomposition
  // Simple 1D decomposition along z-axis
  int const nz_per_rank = nz / num_ranks;
  int const z_start = my_rank * nz_per_rank;
  int const z_end =
      (my_rank == num_ranks - 1) ? nz - 1 : (my_rank + 1) * nz_per_rank - 1;

  heffte::box3d<> const inbox = {{0, 0, z_start}, {nx - 1, ny - 1, z_end}};
  heffte::box3d<> const outbox = inbox; // Same geometry for forward and backward

  if (my_rank == 0) {
    std::cout << "Domain size: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Total number of ranks: " << num_ranks << std::endl;
  }

  // Print box information for each rank
  std::cout << "Rank " << my_rank << " box: [" << inbox.low[0] << "," << inbox.low[1]
            << "," << inbox.low[2] << "] to [" << inbox.high[0] << ","
            << inbox.high[1] << "," << inbox.high[2] << "], size: " << inbox.size[0]
            << " x " << inbox.size[1] << " x " << inbox.size[2] << std::endl;

  // Create FFT plan
  heffte::fft3d<backend_tag> fft(inbox, outbox, MPI_COMM_WORLD);

  // Create input data on CPU
  std::vector<std::complex<double>> input(fft.size_inbox());

  // Initialize with a simple pattern: sin(x) * sin(y) * sin(z)
  // HeFFTe uses order (0, 1, 2) where dimension 0 is contiguous
  int local_plane = inbox.size[0] * inbox.size[1];
  int local_stride = inbox.size[0];

  for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
        // Calculate linear index following HeFFTe convention
        int idx = (k - inbox.low[2]) * local_plane +
                  (j - inbox.low[1]) * local_stride + (i - inbox.low[0]);

        double x = 2.0 * M_PI * i / nx;
        double y = 2.0 * M_PI * j / ny;
        double z = 2.0 * M_PI * k / nz;
        input[idx] =
            std::complex<double>(std::sin(x) * std::sin(y) * std::sin(z), 0.0);
      }
    }
  }

  // Transfer input to GPU
  // Verify GPU memory allocation on correct device
  int device_before;
  cudaGetDevice(&device_before);
  size_t free_mem_before, total_mem_before;
  cudaMemGetInfo(&free_mem_before, &total_mem_before);

  heffte::gpu::vector<std::complex<double>> gpu_input =
      heffte::gpu::transfer().load(input);

  // Verify memory was allocated on correct device
  int device_after;
  cudaGetDevice(&device_after);
  size_t free_mem_after, total_mem_after;
  cudaMemGetInfo(&free_mem_after, &total_mem_after);
  size_t mem_used = free_mem_before - free_mem_after;

  std::cout << "Rank " << my_rank << " GPU " << device_after << " memory: used "
            << mem_used / (1024 * 1024) << " MB"
            << " (free: " << free_mem_after / (1024 * 1024) << " MB / "
            << total_mem_after / (1024 * 1024) << " MB total)" << std::endl;

  // Allocate GPU memory for FFT output
  heffte::gpu::vector<std::complex<double>> gpu_output(fft.size_outbox());

  // Allocate workspace
  heffte::fft3d<backend_tag>::buffer_container<std::complex<double>> workspace(
      fft.size_workspace());

  if (my_rank == 0) {
    std::cout << "Performing forward FFT..." << std::endl;
  }

  // Perform forward FFT with scale::none (we'll handle scaling manually)
  fft.forward(gpu_input.data(), gpu_output.data(), workspace.data(),
              heffte::scale::none);

  if (my_rank == 0) {
    std::cout << "Applying Laplace operator in Fourier domain..." << std::endl;
  }

  // Apply Laplace operator in Fourier domain: multiply by -k²
  // The Laplacian in k-space is: -k² = -(kx² + ky² + kz²)
  // Perform this operation directly on GPU using CUDA kernel
  // Cast to cuDoubleComplex* since HeFFTe uses std::complex but they have the same
  // layout
  cuDoubleComplex *gpu_data = reinterpret_cast<cuDoubleComplex *>(gpu_output.data());
  int total_elements = gpu_output.size();

  // Configure CUDA kernel launch parameters
  int threads_per_block = 256;
  int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

  // Launch CUDA kernel to apply Laplacian operator
  apply_laplacian_kernel<<<blocks_per_grid, threads_per_block>>>(
      gpu_data, nx, ny, nz, outbox.low[0], outbox.high[0], outbox.low[1],
      outbox.high[1], outbox.low[2], outbox.high[2], outbox.size[0], outbox.size[1],
      outbox.size[2]);

  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (my_rank == 0) {
      std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err)
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Synchronize to ensure kernel completes
  cudaDeviceSynchronize();

  if (my_rank == 0) {
    std::cout << "Performing inverse FFT..." << std::endl;
  }

  // Perform inverse FFT (using the vector API for convenience)
  // We used scale::none in forward, so we need to scale by 1/N manually
  heffte::gpu::vector<std::complex<double>> gpu_result = fft.backward(gpu_output);

  // Scale by 1/N directly on GPU using CUDA kernel
  cuDoubleComplex *result_data =
      reinterpret_cast<cuDoubleComplex *>(gpu_result.data());
  int result_size = gpu_result.size();
  double scale_factor = 1.0 / (nx * ny * nz);

  // Simple scaling kernel
  int scale_blocks = (result_size + threads_per_block - 1) / threads_per_block;
  scale_kernel<<<scale_blocks, threads_per_block>>>(result_data, result_size,
                                                    scale_factor);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (my_rank == 0) {
      std::cerr << "CUDA scaling kernel error: " << cudaGetErrorString(err)
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  cudaDeviceSynchronize();

  // Transfer result back to CPU only once at the end
  std::vector<std::complex<double>> laplacian_result =
      heffte::gpu::transfer::unload(gpu_result);

  // Verify results against analytical solution
  // For f(x,y,z) = sin(x)*sin(y)*sin(z)
  // Analytical Laplacian: ∇²f = -3*sin(x)*sin(y)*sin(z)
  // (since ∂²/∂x²(sin(x)) = -sin(x), and same for y and z)
  double max_error = 0.0;
  double max_relative_error = 0.0;
  for (int k = inbox.low[2]; k <= inbox.high[2]; k++) {
    for (int j = inbox.low[1]; j <= inbox.high[1]; j++) {
      for (int i = inbox.low[0]; i <= inbox.high[0]; i++) {
        // Calculate linear index
        int idx = (k - inbox.low[2]) * local_plane +
                  (j - inbox.low[1]) * local_stride + (i - inbox.low[0]);

        // Analytical solution: -3 * sin(x) * sin(y) * sin(z)
        double x = 2.0 * M_PI * i / nx;
        double y = 2.0 * M_PI * j / ny;
        double z = 2.0 * M_PI * k / nz;
        std::complex<double> analytical = std::complex<double>(
            -3.0 * std::sin(x) * std::sin(y) * std::sin(z), 0.0);

        // Compare computed Laplacian (real part) with analytical solution
        double computed_real = laplacian_result[idx].real();
        double analytical_real = analytical.real();
        double error = std::abs(computed_real - analytical_real);
        double relative_error = (std::abs(analytical_real) > 1e-10)
                                    ? error / std::abs(analytical_real)
                                    : error;
        max_error = std::max(max_error, error);
        max_relative_error = std::max(max_relative_error, relative_error);
      }
    }
  }

  // Get global maximum error across all ranks
  double global_max_error = 0.0;
  double global_max_relative_error = 0.0;
  MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&max_relative_error, &global_max_relative_error, 1, MPI_DOUBLE,
                MPI_MAX, MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\n=== Verification Results ===" << std::endl;
    std::cout
        << "Computing Laplacian: ∇²(sin(x)sin(y)sin(z)) = -3*sin(x)sin(y)sin(z)"
        << std::endl;
    std::cout << "Maximum absolute error: " << global_max_error << std::endl;
    std::cout << "Maximum relative error: " << global_max_relative_error
              << std::endl;

    // Check if results are acceptable (numerical precision)
    const double tolerance = 1e-8; // Slightly relaxed for Laplacian computation
    if (global_max_error < tolerance) {
      std::cout << "✓ SUCCESS: Computed Laplacian matches analytical solution "
                   "within tolerance ("
                << tolerance << ")" << std::endl;
    } else {
      std::cout << "✗ WARNING: Error exceeds tolerance. This may indicate a problem."
                << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
