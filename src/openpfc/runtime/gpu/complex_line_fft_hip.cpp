// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/runtime/gpu/complex_line_fft_hip.hpp>

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime_api.h>
#include <rocfft/rocfft.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace pfc::fft {
namespace {
void check_hip(hipError_t status, const char *what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("HipComplexLineFFT: ") + what + ": " +
                             hipGetErrorString(status));
  }
}

void check_roc(rocfft_status status, const char *what) {
  if (status != rocfft_status_success) {
    throw std::runtime_error(std::string("HipComplexLineFFT: ") + what + " status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void require_size(std::size_t actual, int expected, const char *buffer) {
  if (actual != static_cast<std::size_t>(expected)) {
    throw std::invalid_argument(std::string("HipComplexLineFFT: ") + buffer +
                                " size " + std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}

struct DeviceBuffer {
  void *data = nullptr;
  ~DeviceBuffer() {
    if (data != nullptr) (void)hipFree(data);
  }
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct RocPlan {
  rocfft_plan handle = nullptr;
  ~RocPlan() {
    if (handle != nullptr) rocfft_plan_destroy(handle);
  }
  RocPlan() = default;
  RocPlan(const RocPlan &) = delete;
  RocPlan &operator=(const RocPlan &) = delete;
};

struct ExecInfo {
  rocfft_execution_info info = nullptr;
  void *work = nullptr;
  ~ExecInfo() {
    if (info != nullptr) rocfft_execution_info_destroy(info);
    if (work != nullptr) (void)hipFree(work);
  }
  ExecInfo() = default;
  ExecInfo(const ExecInfo &) = delete;
  ExecInfo &operator=(const ExecInfo &) = delete;
};
} // namespace

struct HipComplexLineFFT::Impl {
  int length;
  DeviceBuffer in;
  DeviceBuffer out;
  RocPlan forward_plan;
  RocPlan inverse_plan;
  ExecInfo exec;

  explicit Impl(int n) : length(n) {
    if (n < 2) {
      throw std::invalid_argument("HipComplexLineFFT: length must be at least 2");
    }
    const auto count = static_cast<std::size_t>(n);
    if (count >
        std::numeric_limits<std::size_t>::max() / sizeof(std::complex<double>)) {
      throw std::invalid_argument(
          "HipComplexLineFFT: length exceeds addressable storage");
    }
    check_roc(rocfft_setup(), "rocfft_setup");
    const auto bytes = sizeof(std::complex<double>) * count;
    check_hip(hipMalloc(&in.data, bytes), "allocate input");
    check_hip(hipMalloc(&out.data, bytes), "allocate output");
    const std::size_t len = count;
    check_roc(rocfft_plan_create(&forward_plan.handle, rocfft_placement_notinplace,
                                 rocfft_transform_type_complex_forward,
                                 rocfft_precision_double, 1, &len, 1, nullptr),
              "plan forward");
    check_roc(rocfft_plan_create(&inverse_plan.handle, rocfft_placement_notinplace,
                                 rocfft_transform_type_complex_inverse,
                                 rocfft_precision_double, 1, &len, 1, nullptr),
              "plan inverse");
    std::size_t forward_work = 0;
    std::size_t inverse_work = 0;
    check_roc(rocfft_plan_get_work_buffer_size(forward_plan.handle, &forward_work),
              "work forward");
    check_roc(rocfft_plan_get_work_buffer_size(inverse_plan.handle, &inverse_work),
              "work inverse");
    check_roc(rocfft_execution_info_create(&exec.info), "execution info");
    const std::size_t work =
        forward_work > inverse_work ? forward_work : inverse_work;
    if (work > 0) {
      check_hip(hipMalloc(&exec.work, work), "allocate work");
      check_roc(rocfft_execution_info_set_work_buffer(exec.info, exec.work, work),
                "set work");
    }
  }

  void copy_in(std::span<const std::complex<double>> samples) {
    check_hip(
        hipMemcpy(in.data, samples.data(),
                  sizeof(std::complex<double>) * static_cast<std::size_t>(length),
                  hipMemcpyHostToDevice),
        "copy input");
  }

  void copy_out(std::span<std::complex<double>> samples) const {
    check_hip(
        hipMemcpy(samples.data(), out.data,
                  sizeof(std::complex<double>) * static_cast<std::size_t>(length),
                  hipMemcpyDeviceToHost),
        "copy output");
  }

  void execute(rocfft_plan plan) {
    void *input[1] = {in.data};
    void *output[1] = {out.data};
    check_roc(rocfft_execute(plan, input, output, exec.info), "execute");
    check_hip(hipDeviceSynchronize(), "synchronize");
  }
};

static_assert(sizeof(std::complex<double>) == 16);

HipComplexLineFFT::HipComplexLineFFT(int length)
    : m_impl(std::make_unique<Impl>(length)) {}

HipComplexLineFFT::~HipComplexLineFFT() = default;

void HipComplexLineFFT::forward(std::span<const std::complex<double>> in,
                                std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "forward input");
  require_size(out.size(), m_impl->length, "forward output");
  m_impl->copy_in(in);
  m_impl->execute(m_impl->forward_plan.handle);
  m_impl->copy_out(out);
}

void HipComplexLineFFT::backward(std::span<const std::complex<double>> in,
                                 std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "backward input");
  require_size(out.size(), m_impl->length, "backward output");
  m_impl->copy_in(in);
  m_impl->execute(m_impl->inverse_plan.handle);
  m_impl->copy_out(out);
}

int HipComplexLineFFT::length() const { return m_impl->length; }

} // namespace pfc::fft

#endif
