// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/runtime/gpu/complex_line_fft_cuda.hpp>

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace pfc::fft {
namespace {
void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CudaComplexLineFFT: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cufft(cufftResult status, const char *what) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("CudaComplexLineFFT: ") + what +
                             " status " + std::to_string(static_cast<int>(status)));
  }
}

void require_size(std::size_t actual, int expected, const char *buffer) {
  if (actual != static_cast<std::size_t>(expected)) {
    throw std::invalid_argument(std::string("CudaComplexLineFFT: ") + buffer +
                                " size " + std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}

struct Buffer {
  cufftDoubleComplex *data = nullptr;
  ~Buffer() {
    if (data != nullptr) cudaFree(data);
  }
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

struct Plan {
  cufftHandle handle = 0;
  ~Plan() {
    if (handle != 0) cufftDestroy(handle);
  }
  Plan() = default;
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
};
} // namespace

struct CudaComplexLineFFT::Impl {
  int length;
  Buffer in;
  Buffer out;
  Plan plan;

  explicit Impl(int n) : length(n) {
    if (n < 2) {
      throw std::invalid_argument("CudaComplexLineFFT: length must be at least 2");
    }
    const auto count = static_cast<std::size_t>(n);
    if (count >
        std::numeric_limits<std::size_t>::max() / sizeof(cufftDoubleComplex)) {
      throw std::invalid_argument(
          "CudaComplexLineFFT: length exceeds addressable storage");
    }
    const auto bytes = sizeof(cufftDoubleComplex) * count;
    void *input = nullptr;
    void *output = nullptr;
    check_cuda(cudaMalloc(&input, bytes), "allocate input");
    check_cuda(cudaMalloc(&output, bytes), "allocate output");
    in.data = static_cast<cufftDoubleComplex *>(input);
    out.data = static_cast<cufftDoubleComplex *>(output);
    check_cufft(cufftPlan1d(&plan.handle, n, CUFFT_Z2Z, 1), "plan");
  }

  void copy_in(std::span<const std::complex<double>> samples) {
    check_cuda(
        cudaMemcpy(in.data, samples.data(),
                   sizeof(std::complex<double>) * static_cast<std::size_t>(length),
                   cudaMemcpyHostToDevice),
        "copy input");
  }

  void copy_out(std::span<std::complex<double>> samples) const {
    check_cuda(
        cudaMemcpy(samples.data(), out.data,
                   sizeof(std::complex<double>) * static_cast<std::size_t>(length),
                   cudaMemcpyDeviceToHost),
        "copy output");
  }

  void execute(int direction) {
    check_cufft(cufftExecZ2Z(plan.handle, in.data, out.data, direction), "execute");
    check_cuda(cudaDeviceSynchronize(), "synchronize");
  }
};

static_assert(sizeof(std::complex<double>) == sizeof(cufftDoubleComplex));

CudaComplexLineFFT::CudaComplexLineFFT(int length)
    : m_impl(std::make_unique<Impl>(length)) {}

CudaComplexLineFFT::~CudaComplexLineFFT() = default;

void CudaComplexLineFFT::forward(std::span<const std::complex<double>> in,
                                 std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "forward input");
  require_size(out.size(), m_impl->length, "forward output");
  m_impl->copy_in(in);
  m_impl->execute(CUFFT_FORWARD);
  m_impl->copy_out(out);
}

void CudaComplexLineFFT::backward(std::span<const std::complex<double>> in,
                                  std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "backward input");
  require_size(out.size(), m_impl->length, "backward output");
  m_impl->copy_in(in);
  m_impl->execute(CUFFT_INVERSE);
  m_impl->copy_out(out);
}

int CudaComplexLineFFT::length() const { return m_impl->length; }

} // namespace pfc::fft

#endif
