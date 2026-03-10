// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_ops_kernels.hip
 * @brief HIP kernel implementations for Tungsten-specific operations
 *
 * This file contains HIP kernel code for Tungsten operations. It is only
 * compiled when HIP (ROCm) is enabled.
 */

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <tungsten/common/tungsten_ops.hpp>
#include <type_traits>

namespace tungsten {
namespace ops {
namespace detail {

template <typename RealType>
__global__ void multiply_complex_real_kernel_impl(
    const typename std::conditional<std::is_same<RealType, double>::value,
                                    hipDoubleComplex, hipFloatComplex>::type *a,
    const RealType *b,
    typename std::conditional<std::is_same<RealType, double>::value,
                              hipDoubleComplex, hipFloatComplex>::type *out,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if constexpr (std::is_same_v<RealType, double>) {
      hipDoubleComplex a_val = a[idx];
      double b_val = b[idx];
      out[idx] = hipCmul(make_hipDoubleComplex(b_val, 0.0), a_val);
    } else {
      hipFloatComplex a_val = a[idx];
      float b_val = b[idx];
      out[idx] = hipCmulf(make_hipFloatComplex(b_val, 0.0f), a_val);
    }
  }
}

template __global__ void
multiply_complex_real_kernel_impl<double>(const hipDoubleComplex *, const double *,
                                          hipDoubleComplex *, size_t);
template __global__ void
multiply_complex_real_kernel_impl<float>(const hipFloatComplex *, const float *,
                                         hipFloatComplex *, size_t);

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

template __global__ void compute_nonlinear_kernel<double>(const double *,
                                                          const double *, double,
                                                          double, double, double,
                                                          double *, size_t);
template __global__ void compute_nonlinear_kernel<float>(const float *, const float *,
                                                         float, float, float, float,
                                                         float *, size_t);

template <typename RealType>
__global__ void apply_stabilization_kernel(const RealType *in, const RealType *field,
                                           RealType stabP, RealType *out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] - stabP * field[idx];
  }
}

template __global__ void apply_stabilization_kernel<double>(const double *,
                                                            const double *, double,
                                                            double *, size_t);
template __global__ void apply_stabilization_kernel<float>(const float *,
                                                           const float *, float,
                                                           float *, size_t);

template <typename RealType>
__global__ void apply_time_integration_kernel_impl(
    const typename std::conditional<std::is_same<RealType, double>::value,
                                    hipDoubleComplex, hipFloatComplex>::type *psi_F,
    const typename std::conditional<std::is_same<RealType, double>::value,
                                    hipDoubleComplex, hipFloatComplex>::type *psiN_F,
    const RealType *opL, const RealType *opN,
    typename std::conditional<std::is_same<RealType, double>::value,
                              hipDoubleComplex, hipFloatComplex>::type *out,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if constexpr (std::is_same_v<RealType, double>) {
      hipDoubleComplex psi_F_val = psi_F[idx];
      hipDoubleComplex psiN_F_val = psiN_F[idx];
      double opL_val = opL[idx];
      double opN_val = opN[idx];
      hipDoubleComplex term1 = hipCmul(make_hipDoubleComplex(opL_val, 0.0), psi_F_val);
      hipDoubleComplex term2 = hipCmul(make_hipDoubleComplex(opN_val, 0.0), psiN_F_val);
      out[idx] = hipCadd(term1, term2);
    } else {
      hipFloatComplex psi_F_val = psi_F[idx];
      hipFloatComplex psiN_F_val = psiN_F[idx];
      float opL_val = opL[idx];
      float opN_val = opN[idx];
      hipFloatComplex term1 = hipCmulf(make_hipFloatComplex(opL_val, 0.0f), psi_F_val);
      hipFloatComplex term2 = hipCmulf(make_hipFloatComplex(opN_val, 0.0f), psiN_F_val);
      out[idx] = hipCaddf(term1, term2);
    }
  }
}

template __global__ void apply_time_integration_kernel_impl<double>(
    const hipDoubleComplex *, const hipDoubleComplex *, const double *, const double *,
    hipDoubleComplex *, size_t);
template __global__ void apply_time_integration_kernel_impl<float>(
    const hipFloatComplex *, const hipFloatComplex *, const float *, const float *,
    hipFloatComplex *, size_t);

inline void launch_kernel(size_t n, int &blocks, int &threads_per_block) {
  threads_per_block = 512;
  blocks = (static_cast<int>(n) + threads_per_block - 1) / threads_per_block;
}

} // namespace detail
} // namespace ops
} // namespace tungsten

namespace tungsten {
namespace ops {
namespace detail {

void TungstenOps<pfc::backend::HipTag, double>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> &a,
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &b,
    pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const hipDoubleComplex *a_ptr =
      reinterpret_cast<const hipDoubleComplex *>(a.data());
  const double *b_ptr = b.data();
  hipDoubleComplex *out_ptr = reinterpret_cast<hipDoubleComplex *>(out.data());

  hipLaunchKernelGGL(multiply_complex_real_kernel_impl<double>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, a_ptr, b_ptr, out_ptr, N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (multiply_complex_real): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, double>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &u,
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &v, double p3,
    double p4, double q3, double q4,
    pfc::core::DataBuffer<pfc::backend::HipTag, double> &out) {
  const size_t N = u.size();
  if (v.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in compute_nonlinear");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  hipLaunchKernelGGL(compute_nonlinear_kernel<double>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, u.data(), v.data(), p3, p4, q3,
                    q4, out.data(), N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (compute_nonlinear): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, double>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &in,
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &field, double stabP,
    pfc::core::DataBuffer<pfc::backend::HipTag, double> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  hipLaunchKernelGGL(apply_stabilization_kernel<double>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, in.data(), field.data(), stabP,
                    out.data(), N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (apply_stabilization): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, double>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &opL,
    const pfc::core::DataBuffer<pfc::backend::HipTag, double> &opN,
    pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const hipDoubleComplex *psi_F_ptr =
      reinterpret_cast<const hipDoubleComplex *>(psi_F.data());
  const hipDoubleComplex *psiN_F_ptr =
      reinterpret_cast<const hipDoubleComplex *>(psiN_F.data());
  const double *opL_ptr = opL.data();
  const double *opN_ptr = opN.data();
  hipDoubleComplex *out_ptr = reinterpret_cast<hipDoubleComplex *>(out.data());

  hipLaunchKernelGGL(apply_time_integration_kernel_impl<double>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, psi_F_ptr, psiN_F_ptr, opL_ptr,
                    opN_ptr, out_ptr, N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (apply_time_integration): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, float>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &a,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &b,
    pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const hipFloatComplex *a_ptr =
      reinterpret_cast<const hipFloatComplex *>(a.data());
  const float *b_ptr = b.data();
  hipFloatComplex *out_ptr = reinterpret_cast<hipFloatComplex *>(out.data());

  hipLaunchKernelGGL(multiply_complex_real_kernel_impl<float>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, a_ptr, b_ptr, out_ptr, N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (multiply_complex_real): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, float>::compute_nonlinear_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &u,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &v, float p3, float p4,
    float q3, float q4,
    pfc::core::DataBuffer<pfc::backend::HipTag, float> &out) {
  const size_t N = u.size();
  if (v.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in compute_nonlinear");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  hipLaunchKernelGGL(compute_nonlinear_kernel<float>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, u.data(), v.data(), p3, p4, q3,
                    q4, out.data(), N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (compute_nonlinear): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, float>::apply_stabilization_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &in,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &field, float stabP,
    pfc::core::DataBuffer<pfc::backend::HipTag, float> &out) {
  const size_t N = in.size();
  if (field.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_stabilization");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  hipLaunchKernelGGL(apply_stabilization_kernel<float>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, in.data(), field.data(), stabP,
                    out.data(), N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (apply_stabilization): " +
                             std::string(hipGetErrorString(err)));
  }
}

void TungstenOps<pfc::backend::HipTag, float>::apply_time_integration_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &psi_F,
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &psiN_F,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &opL,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &opN,
    pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &out) {
  const size_t N = psi_F.size();
  if (psiN_F.size() != N || opL.size() != N || opN.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in apply_time_integration");
  }
  if (N == 0) {
    return;
  }

  int blocks, threads_per_block;
  launch_kernel(N, blocks, threads_per_block);

  const hipFloatComplex *psi_F_ptr =
      reinterpret_cast<const hipFloatComplex *>(psi_F.data());
  const hipFloatComplex *psiN_F_ptr =
      reinterpret_cast<const hipFloatComplex *>(psiN_F.data());
  const float *opL_ptr = opL.data();
  const float *opN_ptr = opN.data();
  hipFloatComplex *out_ptr = reinterpret_cast<hipFloatComplex *>(out.data());

  hipLaunchKernelGGL(apply_time_integration_kernel_impl<float>, dim3(blocks),
                    dim3(threads_per_block), 0, 0, psi_F_ptr, psiN_F_ptr, opL_ptr,
                    opN_ptr, out_ptr, N);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed (apply_time_integration): " +
                             std::string(hipGetErrorString(err)));
  }
}

} // namespace detail
} // namespace ops
} // namespace tungsten

#endif // OpenPFC_ENABLE_HIP
