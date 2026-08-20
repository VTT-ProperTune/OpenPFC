// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_ops_kernels.cu
 * @brief CUDA kernel implementations for Tungsten-specific operations
 *
 * This file contains CUDA kernel code for Tungsten-specific operations
 * (nonlinear term and stabilization). Complex×real multiply and ETD
 * two-term combine are `runtime/gpu/elementwise_ops_gpu`.
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <tungsten/common/tungsten_ops.hpp>

namespace tungsten {
namespace ops {
namespace detail {

// CUDA kernel: Compute nonlinear term (template-based for precision)
template <typename RealType>
__global__ void compute_nonlinear_kernel(const RealType *u, const RealType *v,
                                         RealType p3, RealType p4, RealType q3,
                                         RealType q4, RealType *out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    RealType u_val = u[idx];
    RealType v_val = v[idx];
    RealType u2 = u_val * u_val;
    RealType u3 = u2 * u_val;
    RealType v2 = v_val * v_val;
    RealType v3 = v2 * v_val;
    out[idx] = p3 * u2 + p4 * u3 + q3 * v2 + q4 * v3;
  }
}

// Explicit instantiations for float and double
template __global__ void compute_nonlinear_kernel<double>(const double *,
                                                          const double *, double,
                                                          double, double, double,
                                                          double *, size_t);
template __global__ void compute_nonlinear_kernel<float>(const float *,
                                                         const float *, float, float,
                                                         float, float, float *,
                                                         size_t);

// CUDA kernel: Apply stabilization (template-based for precision)
template <typename RealType>
__global__ void apply_stabilization_kernel(const RealType *in, const RealType *field,
                                           RealType stabP, RealType *out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] - stabP * field[idx];
  }
}

// Explicit instantiations for float and double
template __global__ void apply_stabilization_kernel<double>(const double *,
                                                            const double *, double,
                                                            double *, size_t);
template __global__ void apply_stabilization_kernel<float>(const float *,
                                                           const float *, float,
                                                           float *, size_t);

// Helper function to launch kernels with appropriate grid/block sizes
// Optimized for modern GPUs (H100): use larger block size for better occupancy
inline void launch_kernel(size_t n, int &blocks, int &threads_per_block) {
  // Use 512 threads per block for better GPU utilization on H100
  // This improves occupancy and memory bandwidth utilization
  threads_per_block = 512;
  blocks = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
}

} // namespace detail
} // namespace ops
} // namespace tungsten

// CUDA specialization implementations (structs are declared in tungsten_ops.hpp)
namespace tungsten {
namespace ops {
namespace detail {

// CUDA specialization for double precision - implement methods
void TungstenOps<pfc::backend::CUDATag, double>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<double>> &a,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &b,
    pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<double>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  pfc::multiply_complex_real_cuda_impl(a.data(), b.data(), out.data(), N);
}

void TungstenOps<pfc::backend::CUDATag, double>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &u,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &v, double p3,
    double p4, double q3, double q4,
    pfc::core::DataBuffer<pfc::backend::CUDATag, double> &out) {
  const size_t N = u.size();
  if (v.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in compute_nonlinear");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::compute_nonlinear_kernel<double><<<blocks, threads_per_block>>>(
      u.data(), v.data(), p3, p4, q3, q4, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch failed (compute_nonlinear<double>): ") +
        cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel sync failed (compute_nonlinear<double>): ") +
        cudaGetErrorString(err));
  }
}

void TungstenOps<pfc::backend::CUDATag, double>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &in,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &field, double stabP,
    pfc::core::DataBuffer<pfc::backend::CUDATag, double> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::apply_stabilization_kernel<double>
      <<<blocks, threads_per_block>>>(in.data(), field.data(), stabP, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch failed (apply_stabilization<double>): ") +
        cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel sync failed (apply_stabilization<double>): ") +
        cudaGetErrorString(err));
  }
}

void TungstenOps<pfc::backend::CUDATag, double>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<double>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<double>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &opL,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, double> &opN,
    pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<double>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  pfc::combine_two_term_cuda_impl(psi_F.data(), psiN_F.data(), opL.data(),
                                  opN.data(), out.data(), N);
}

// CUDA specialization for float precision - implement methods
void TungstenOps<pfc::backend::CUDATag, float>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<float>> &a,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &b,
    pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<float>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  pfc::multiply_complex_real_cuda_impl(a.data(), b.data(), out.data(), N);
}

void TungstenOps<pfc::backend::CUDATag, float>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &u,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &v, float p3, float p4,
    float q3, float q4, pfc::core::DataBuffer<pfc::backend::CUDATag, float> &out) {
  const size_t N = u.size();
  if (v.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in compute_nonlinear");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::compute_nonlinear_kernel<float><<<blocks, threads_per_block>>>(
      u.data(), v.data(), p3, p4, q3, q4, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch failed (compute_nonlinear<float>): ") +
        cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel sync failed (compute_nonlinear<float>): ") +
        cudaGetErrorString(err));
  }
}

void TungstenOps<pfc::backend::CUDATag, float>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &in,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &field, float stabP,
    pfc::core::DataBuffer<pfc::backend::CUDATag, float> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::apply_stabilization_kernel<float>
      <<<blocks, threads_per_block>>>(in.data(), field.data(), stabP, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel launch failed (apply_stabilization<float>): ") +
        cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA kernel sync failed (apply_stabilization<float>): ") +
        cudaGetErrorString(err));
  }
}

void TungstenOps<pfc::backend::CUDATag, float>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<float>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<float>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &opL,
    const pfc::core::DataBuffer<pfc::backend::CUDATag, float> &opN,
    pfc::core::DataBuffer<pfc::backend::CUDATag, std::complex<float>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  pfc::combine_two_term_cuda_impl(psi_F.data(), psiN_F.data(), opL.data(),
                                  opN.data(), out.data(), N);
}

} // namespace detail
} // namespace ops
} // namespace tungsten

#endif // OpenPFC_ENABLE_CUDA
