// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/complex_line_fft.hpp>

#include <fftw3.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace pfc::fft {
namespace {
struct Free {
  void operator()(void *p) const noexcept { fftw_free(p); }
};
struct Plan {
  fftw_plan handle = nullptr;
  ~Plan() {
    if (handle) fftw_destroy_plan(handle);
  }
  Plan() = default;
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
};
void require_size(std::size_t actual, int expected, const char *buffer) {
  if (actual != static_cast<std::size_t>(expected)) {
    throw std::invalid_argument(std::string("ComplexLineFFT: ") + buffer + " size " +
                                std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}
} // namespace

struct ComplexLineFFT::Impl {
  int length;
  std::unique_ptr<fftw_complex, Free> in;
  std::unique_ptr<fftw_complex, Free> out;
  Plan forward;
  Plan backward;

  explicit Impl(int n) : length(n) {
    if (n < 2) {
      throw std::invalid_argument("ComplexLineFFT: length must be at least 2");
    }
    const auto count = static_cast<std::size_t>(n);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(fftw_complex)) {
      throw std::invalid_argument(
          "ComplexLineFFT: length exceeds addressable storage");
    }
    const auto bytes = sizeof(fftw_complex) * count;
    in.reset(static_cast<fftw_complex *>(fftw_malloc(bytes)));
    out.reset(static_cast<fftw_complex *>(fftw_malloc(bytes)));
    if (!in || !out) throw std::bad_alloc();
    forward.handle =
        fftw_plan_dft_1d(n, in.get(), out.get(), FFTW_FORWARD, FFTW_ESTIMATE);
    backward.handle =
        fftw_plan_dft_1d(n, in.get(), out.get(), FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!forward.handle || !backward.handle) {
      throw std::runtime_error("ComplexLineFFT: FFTW planning failed");
    }
  }

  void copy_in(std::span<const std::complex<double>> samples) {
    for (int i = 0; i < length; ++i) {
      in.get()[i][0] = samples[static_cast<std::size_t>(i)].real();
      in.get()[i][1] = samples[static_cast<std::size_t>(i)].imag();
    }
  }

  void copy_out(std::span<std::complex<double>> samples) const {
    for (int i = 0; i < length; ++i) {
      samples[static_cast<std::size_t>(i)] = {out.get()[i][0], out.get()[i][1]};
    }
  }
};

ComplexLineFFT::ComplexLineFFT(int length)
    : m_impl(std::make_unique<Impl>(length)) {}
ComplexLineFFT::~ComplexLineFFT() = default;

void ComplexLineFFT::forward(std::span<const std::complex<double>> in,
                             std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "forward input");
  require_size(out.size(), m_impl->length, "forward output");
  m_impl->copy_in(in);
  fftw_execute(m_impl->forward.handle);
  m_impl->copy_out(out);
}

void ComplexLineFFT::backward(std::span<const std::complex<double>> in,
                              std::span<std::complex<double>> out) {
  require_size(in.size(), m_impl->length, "backward input");
  require_size(out.size(), m_impl->length, "backward output");
  m_impl->copy_in(in);
  fftw_execute(m_impl->backward.handle);
  m_impl->copy_out(out);
}

int ComplexLineFFT::length() const { return m_impl->length; }

} // namespace pfc::fft
