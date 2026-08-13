// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file tungsten_ops_kernels.hip
 * @brief HIP kernel implementations for Tungsten-specific operations
 *
 * This file contains HIP kernel code for Tungsten-specific operations
 * (nonlinear term and stabilization). Complex×real multiply and ETD
 * two-term combine are `runtime/gpu/elementwise_ops_gpu`.
 */

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/runtime/gpu/elementwise_ops_gpu.hpp>
#include <tungsten/common/tungsten_ops.hpp>

namespace tungsten {
namespace ops {
namespace detail {

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
  pfc::multiply_complex_real_hip_impl(a.data(), b.data(), out.data(), N);
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
    throw std::runtime_error(
        std::string("HIP kernel launch failed (compute_nonlinear<double>): ") +
        hipGetErrorString(err));
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP kernel sync failed (compute_nonlinear<double>): ") +
        hipGetErrorString(err));
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
    throw std::runtime_error(
        std::string("HIP kernel launch failed (apply_stabilization<double>): ") +
        hipGetErrorString(err));
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP kernel sync failed (apply_stabilization<double>): ") +
        hipGetErrorString(err));
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
  pfc::combine_two_term_hip_impl(psi_F.data(), psiN_F.data(), opL.data(),
                                 opN.data(), out.data(), N);
}

void TungstenOps<pfc::backend::HipTag, float>::multiply_complex_real_impl(
    const pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &a,
    const pfc::core::DataBuffer<pfc::backend::HipTag, float> &b,
    pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<float>> &out) {
  const size_t N = a.size();
  if (b.size() != N || out.size() != N) {
    throw std::runtime_error("Size mismatch in multiply_complex_real");
  }
  pfc::multiply_complex_real_hip_impl(a.data(), b.data(), out.data(), N);
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
    throw std::runtime_error(
        std::string("HIP kernel launch failed (compute_nonlinear<float>): ") +
        hipGetErrorString(err));
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP kernel sync failed (compute_nonlinear<float>): ") +
        hipGetErrorString(err));
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
    throw std::runtime_error(
        std::string("HIP kernel launch failed (apply_stabilization<float>): ") +
        hipGetErrorString(err));
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP kernel sync failed (apply_stabilization<float>): ") +
        hipGetErrorString(err));
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
  pfc::combine_two_term_hip_impl(psi_F.data(), psiN_F.data(), opL.data(),
                                 opN.data(), out.data(), N);
}

} // namespace detail
} // namespace ops
} // namespace tungsten

#endif // OpenPFC_ENABLE_HIP
