// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file elementwise_ops_gpu.inc
 * @brief Single-source generic GPU elementwise kernels (from vendor .cu/.hip).
 *
 * Do not include this from a host TU. Complex values are interleaved
 * real/imag (`std::complex` layout). Error strings keep the vendor kind.
 *
 * @see runtime/gpu/elementwise_ops_gpu.hpp
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc {

#if defined(OPENPFC_GPU_API_CUDA)
constexpr const char *kElemGpuKind = "CUDA";
#elif defined(OPENPFC_GPU_API_HIP)
constexpr const char *kElemGpuKind = "HIP";
#endif

template <typename T>
__global__ void multiply_complex_real_kernel(const T *a, const T *b, T *out,
                                             std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T br = b[i];
    out[2 * i] = a[2 * i] * br;
    out[2 * i + 1] = a[2 * i + 1] * br;
  }
}

template <typename T>
__global__ void combine_two_term_kernel(const T *x0, const T *x1, const T *w0,
                                        const T *w1, T *out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T a = w0[i];
    const T b = w1[i];
    out[2 * i] = a * x0[2 * i] + b * x1[2 * i];
    out[2 * i + 1] = a * x0[2 * i + 1] + b * x1[2 * i + 1];
  }
}

template <typename T>
__global__ void axpy_fill_kernel(T *out, const T *x, T alpha, T beta,
                                 std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * x[i] + beta;
  }
}

void check_elem_launch(const char *op) {
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string(kElemGpuKind) + " " + op +
                             " kernel failed: " + ::pfc::gpuGetErrorString(err));
  }
  err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(
        std::string(kElemGpuKind) + " " + op +
        " kernel sync failed: " + ::pfc::gpuGetErrorString(err));
  }
}

int elem_blocks(std::size_t n) {
  constexpr int threads = 256;
  return static_cast<int>((n + static_cast<std::size_t>(threads) - 1) /
                          static_cast<std::size_t>(threads));
}

template <typename T>
void launch_multiply_complex_real(const T *a, const T *b, T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  multiply_complex_real_kernel<<<elem_blocks(n), threads>>>(a, b, out, n);
  check_elem_launch("multiply_complex_real");
}

template <typename T>
void launch_combine_two_term(const T *x0, const T *x1, const T *w0, const T *w1,
                             T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  combine_two_term_kernel<<<elem_blocks(n), threads>>>(x0, x1, w0, w1, out, n);
  check_elem_launch("combine_two_term");
}

template <typename T>
__global__ void combine_two_term_real_kernel(const T *x0, const T *x1, const T *w0,
                                             const T *w1, T *out, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = w0[i] * x0[i] + w1[i] * x1[i];
  }
}

template <typename T>
void launch_combine_two_term_real(const T *x0, const T *x1, const T *w0, const T *w1,
                                  T *out, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  combine_two_term_real_kernel<<<elem_blocks(n), threads>>>(x0, x1, w0, w1, out, n);
  check_elem_launch("combine_two_term_real");
}

template <typename T>
void launch_axpy_fill(T *out, const T *x, T alpha, T beta, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  axpy_fill_kernel<<<elem_blocks(n), threads>>>(out, x, alpha, beta, n);
  check_elem_launch("axpy_fill");
}

/** n = c1*psi + c2*psi^2 + c3*psi^3 + d1*mf + d2*mf^2 + d3*mf^3 */
template <typename T>
__global__ void polynomial_nl_kernel(const T *psi, const T *mf, T *out, T c_psi,
                                     T c_psi2, T c_psi3, T c_mf, T c_mf2, T c_mf3,
                                     std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const T p = psi[i];
    const T m = mf[i];
    const T p2 = p * p;
    const T m2 = m * m;
    out[i] = c_psi * p + c_psi2 * p2 + c_psi3 * p2 * p + c_mf * m + c_mf2 * m2 +
             c_mf3 * m2 * m;
  }
}

template <typename T>
void launch_polynomial_nl(const T *psi, const T *mf, T *out, T c_psi, T c_psi2,
                          T c_psi3, T c_mf, T c_mf2, T c_mf3, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr int threads = 256;
  polynomial_nl_kernel<<<elem_blocks(n), threads>>>(psi, mf, out, c_psi, c_psi2,
                                                    c_psi3, c_mf, c_mf2, c_mf3, n);
  check_elem_launch("polynomial_nl");
}

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void multiply_complex_real_cuda_impl(const std::complex<double> *a, const double *b,
                                     std::complex<double> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const double *>(a), b,
                               reinterpret_cast<double *>(out), n);
}
void multiply_complex_real_cuda_impl(const std::complex<float> *a, const float *b,
                                     std::complex<float> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const float *>(a), b,
                               reinterpret_cast<float *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<double> *x0,
                                const std::complex<double> *x1, const double *w0,
                                const double *w1, std::complex<double> *out,
                                std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const double *>(x0),
                          reinterpret_cast<const double *>(x1), w0, w1,
                          reinterpret_cast<double *>(out), n);
}
void combine_two_term_cuda_impl(const std::complex<float> *x0,
                                const std::complex<float> *x1, const float *w0,
                                const float *w1, std::complex<float> *out,
                                std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const float *>(x0),
                          reinterpret_cast<const float *>(x1), w0, w1,
                          reinterpret_cast<float *>(out), n);
}
void combine_two_term_cuda_impl(const double *x0, const double *x1, const double *w0,
                                const double *w1, double *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void combine_two_term_cuda_impl(const float *x0, const float *x1, const float *w0,
                                const float *w1, float *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void axpy_fill_cuda_impl(double *out, const double *x, double alpha, double beta,
                         std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void axpy_fill_cuda_impl(float *out, const float *x, float alpha, float beta,
                         std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void polynomial_nl_cuda_impl(const double *psi, const double *mf, double *out,
                             double c_psi, double c_psi2, double c_psi3, double c_mf,
                             double c_mf2, double c_mf3, std::size_t n) {
  launch_polynomial_nl(psi, mf, out, c_psi, c_psi2, c_psi3, c_mf, c_mf2, c_mf3, n);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void multiply_complex_real_hip_impl(const std::complex<double> *a, const double *b,
                                    std::complex<double> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const double *>(a), b,
                               reinterpret_cast<double *>(out), n);
}
void multiply_complex_real_hip_impl(const std::complex<float> *a, const float *b,
                                    std::complex<float> *out, std::size_t n) {
  launch_multiply_complex_real(reinterpret_cast<const float *>(a), b,
                               reinterpret_cast<float *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<double> *x0,
                               const std::complex<double> *x1, const double *w0,
                               const double *w1, std::complex<double> *out,
                               std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const double *>(x0),
                          reinterpret_cast<const double *>(x1), w0, w1,
                          reinterpret_cast<double *>(out), n);
}
void combine_two_term_hip_impl(const std::complex<float> *x0,
                               const std::complex<float> *x1, const float *w0,
                               const float *w1, std::complex<float> *out,
                               std::size_t n) {
  launch_combine_two_term(reinterpret_cast<const float *>(x0),
                          reinterpret_cast<const float *>(x1), w0, w1,
                          reinterpret_cast<float *>(out), n);
}
void combine_two_term_hip_impl(const double *x0, const double *x1, const double *w0,
                               const double *w1, double *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void combine_two_term_hip_impl(const float *x0, const float *x1, const float *w0,
                               const float *w1, float *out, std::size_t n) {
  launch_combine_two_term_real(x0, x1, w0, w1, out, n);
}
void axpy_fill_hip_impl(double *out, const double *x, double alpha, double beta,
                        std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void axpy_fill_hip_impl(float *out, const float *x, float alpha, float beta,
                        std::size_t n) {
  launch_axpy_fill(out, x, alpha, beta, n);
}
void polynomial_nl_hip_impl(const double *psi, const double *mf, double *out,
                            double c_psi, double c_psi2, double c_psi3, double c_mf,
                            double c_mf2, double c_mf3, std::size_t n) {
  launch_polynomial_nl(psi, mf, out, c_psi, c_psi2, c_psi3, c_mf, c_mf2, c_mf3, n);
}
#endif

} // namespace pfc
