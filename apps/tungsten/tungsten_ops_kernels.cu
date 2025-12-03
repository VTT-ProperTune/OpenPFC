// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_ops_kernels.cu
 * @brief CUDA kernel implementations for Tungsten-specific operations
 *
 * This file contains CUDA kernel code for Tungsten operations. It is only compiled when CUDA is enabled.
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include "tungsten_ops.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <type_traits>

namespace tungsten {
namespace ops {
namespace detail {

// CUDA kernel: Multiply complex by real (template-based for precision)
template <typename RealType>
__global__ void multiply_complex_real_kernel_impl(
    const typename std::conditional<std::is_same<RealType, double>::value,
                                     cuDoubleComplex, cuFloatComplex>::type *a,
    const RealType *b,
    typename std::conditional<std::is_same<RealType, double>::value,
                               cuDoubleComplex, cuFloatComplex>::type *out,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if constexpr (std::is_same_v<RealType, double>) {
      cuDoubleComplex a_val = a[idx];
      double b_val = b[idx];
      out[idx] = cuCmul(cuDoubleComplex{b_val, 0.0}, a_val);
    } else {
      cuFloatComplex a_val = a[idx];
      float b_val = b[idx];
      out[idx] = cuCmulf(cuFloatComplex{b_val, 0.0f}, a_val);
    }
  }
}

// Explicit instantiations for float and double (required for CUDA template kernels)
template __global__ void multiply_complex_real_kernel_impl<double>(
    const cuDoubleComplex *, const double *, cuDoubleComplex *, size_t);
template __global__ void multiply_complex_real_kernel_impl<float>(
    const cuFloatComplex *, const float *, cuFloatComplex *, size_t);

// CUDA kernel: Compute nonlinear term (template-based for precision)
template <typename RealType>
__global__ void compute_nonlinear_kernel(
    const RealType *u, const RealType *v, RealType p3, RealType p4, RealType q3,
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
template __global__ void compute_nonlinear_kernel<double>(
    const double *, const double *, double, double, double, double, double *,
    size_t);
template __global__ void compute_nonlinear_kernel<float>(
    const float *, const float *, float, float, float, float, float *, size_t);

// CUDA kernel: Apply stabilization (template-based for precision)
template <typename RealType>
__global__ void apply_stabilization_kernel(
    const RealType *in, const RealType *field, RealType stabP, RealType *out,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] - stabP * field[idx];
  }
}

// Explicit instantiations for float and double
template __global__ void apply_stabilization_kernel<double>(
    const double *, const double *, double, double *, size_t);
template __global__ void apply_stabilization_kernel<float>(
    const float *, const float *, float, float *, size_t);

// CUDA kernel: Apply time integration (template-based for precision)
template <typename RealType>
__global__ void apply_time_integration_kernel_impl(
    const typename std::conditional<std::is_same<RealType, double>::value,
                                     cuDoubleComplex, cuFloatComplex>::type *psi_F,
    const typename std::conditional<std::is_same<RealType, double>::value,
                                     cuDoubleComplex, cuFloatComplex>::type *psiN_F,
    const RealType *opL, const RealType *opN,
    typename std::conditional<std::is_same<RealType, double>::value,
                               cuDoubleComplex, cuFloatComplex>::type *out,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if constexpr (std::is_same_v<RealType, double>) {
      cuDoubleComplex psi_F_val = psi_F[idx];
      cuDoubleComplex psiN_F_val = psiN_F[idx];
      double opL_val = opL[idx];
      double opN_val = opN[idx];
      
      // out = opL * psi_F + opN * psiN_F
      cuDoubleComplex term1 = cuCmul(cuDoubleComplex{opL_val, 0.0}, psi_F_val);
      cuDoubleComplex term2 = cuCmul(cuDoubleComplex{opN_val, 0.0}, psiN_F_val);
      out[idx] = cuCadd(term1, term2);
    } else {
      cuFloatComplex psi_F_val = psi_F[idx];
      cuFloatComplex psiN_F_val = psiN_F[idx];
      float opL_val = opL[idx];
      float opN_val = opN[idx];
      
      // out = opL * psi_F + opN * psiN_F
      cuFloatComplex term1 = cuCmulf(cuFloatComplex{opL_val, 0.0f}, psi_F_val);
      cuFloatComplex term2 = cuCmulf(cuFloatComplex{opN_val, 0.0f}, psiN_F_val);
      out[idx] = cuCaddf(term1, term2);
    }
  }
}

// Explicit instantiations for float and double (required for CUDA template kernels)
template __global__ void apply_time_integration_kernel_impl<double>(
    const cuDoubleComplex *, const cuDoubleComplex *, const double *, const double *,
    cuDoubleComplex *, size_t);
template __global__ void apply_time_integration_kernel_impl<float>(
    const cuFloatComplex *, const cuFloatComplex *, const float *, const float *,
    cuFloatComplex *, size_t);

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
void TungstenOps<pfc::backend::CudaTag, double>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &a,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &b,
    pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const cuDoubleComplex *a_ptr = reinterpret_cast<const cuDoubleComplex *>(a.data());
  const double *b_ptr = b.data();
  cuDoubleComplex *out_ptr = reinterpret_cast<cuDoubleComplex *>(out.data());

  detail::multiply_complex_real_kernel_impl<double>
      <<<blocks, threads_per_block>>>(a_ptr, b_ptr, out_ptr, N);

  // Only check for launch errors, don't synchronize
  // Synchronization will happen implicitly when needed (e.g., before FFT or MPI)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (multiply_complex_real): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, double>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &u,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &v,
    double p3, double p4, double q3, double q4,
    pfc::core::DataBuffer<pfc::backend::CudaTag, double> &out) {
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
    throw std::runtime_error("CUDA kernel launch failed (compute_nonlinear): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, double>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &in,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &field,
    double stabP,
    pfc::core::DataBuffer<pfc::backend::CudaTag, double> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::apply_stabilization_kernel<double><<<blocks, threads_per_block>>>(
      in.data(), field.data(), stabP, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (apply_stabilization): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, double>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &opL,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, double> &opN,
    pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const cuDoubleComplex *psi_F_ptr =
      reinterpret_cast<const cuDoubleComplex *>(psi_F.data());
  const cuDoubleComplex *psiN_F_ptr =
      reinterpret_cast<const cuDoubleComplex *>(psiN_F.data());
  const double *opL_ptr = opL.data();
  const double *opN_ptr = opN.data();
  cuDoubleComplex *out_ptr = reinterpret_cast<cuDoubleComplex *>(out.data());

  apply_time_integration_kernel_impl<double>
      <<<blocks, threads_per_block>>>(psi_F_ptr, psiN_F_ptr, opL_ptr, opN_ptr,
                                        out_ptr, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (apply_time_integration): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

// CUDA specialization for float precision - implement methods
void TungstenOps<pfc::backend::CudaTag, float>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &a,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &b,
    pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const cuFloatComplex *a_ptr = reinterpret_cast<const cuFloatComplex *>(a.data());
  const float *b_ptr = b.data();
  cuFloatComplex *out_ptr = reinterpret_cast<cuFloatComplex *>(out.data());

  detail::multiply_complex_real_kernel_impl<float>
      <<<blocks, threads_per_block>>>(a_ptr, b_ptr, out_ptr, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (multiply_complex_real): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, float>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &u,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &v,
    float p3, float p4, float q3, float q4,
    pfc::core::DataBuffer<pfc::backend::CudaTag, float> &out) {
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
    throw std::runtime_error("CUDA kernel launch failed (compute_nonlinear): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, float>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &in,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &field,
    float stabP,
    pfc::core::DataBuffer<pfc::backend::CudaTag, float> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  detail::apply_stabilization_kernel<float><<<blocks, threads_per_block>>>(
      in.data(), field.data(), stabP, out.data(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (apply_stabilization): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

void TungstenOps<pfc::backend::CudaTag, float>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &opL,
    const pfc::core::DataBuffer<pfc::backend::CudaTag, float> &opN,
    pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<float>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const cuFloatComplex *psi_F_ptr =
      reinterpret_cast<const cuFloatComplex *>(psi_F.data());
  const cuFloatComplex *psiN_F_ptr =
      reinterpret_cast<const cuFloatComplex *>(psiN_F.data());
  const float *opL_ptr = opL.data();
  const float *opN_ptr = opN.data();
  cuFloatComplex *out_ptr = reinterpret_cast<cuFloatComplex *>(out.data());

  detail::apply_time_integration_kernel_impl<float>
      <<<blocks, threads_per_block>>>(psi_F_ptr, psiN_F_ptr, opL_ptr, opN_ptr,
                                      out_ptr, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed (apply_time_integration): " +
                             std::string(cudaGetErrorString(err)));
  }
  // Removed cudaDeviceSynchronize() - allows kernel overlap and better GPU utilization
}

} // namespace detail
} // namespace ops
} // namespace tungsten

#endif // OpenPFC_ENABLE_CUDA
