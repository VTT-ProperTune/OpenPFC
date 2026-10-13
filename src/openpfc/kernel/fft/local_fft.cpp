// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/local_fft.hpp>

#include <algorithm>
#include <chrono>
#include <fftw3.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace pfc::fft {
namespace {
std::size_t checked_count(const std::array<int, 3> &n, std::size_t bytes) {
  std::size_t count = 1;
  for (int v : n) {
    if (v < 1) throw std::invalid_argument("LocalFFT: grid sizes must be positive");
    if (count > std::numeric_limits<std::size_t>::max() / bytes /
                    static_cast<std::size_t>(v))
      throw std::invalid_argument("LocalFFT: grid size exceeds addressable storage");
    count *= static_cast<std::size_t>(v);
  }
  return count;
}
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
void require_size(std::size_t actual, std::size_t expected, const char *buffer) {
  if (actual != expected)
    throw std::invalid_argument(std::string("LocalFFT: ") + buffer + " size " +
                                std::to_string(actual) + ", expected " +
                                std::to_string(expected));
}
} // namespace

struct LocalFFT::Impl {
  std::array<int, 3> n;
  std::size_t nr, nc;
  std::unique_ptr<double, Free> real;
  std::unique_ptr<fftw_complex, Free> complex;
  Plan forward, backward;
  double time = 0.0;

  explicit Impl(const std::array<int, 3> &size)
      : n(size), nr(checked_count(n, sizeof(double))),
        nc(checked_count({n[0] / 2 + 1, n[1], n[2]}, sizeof(fftw_complex))),
        real(fftw_alloc_real(nr)), complex(fftw_alloc_complex(nc)) {
    if (!real || !complex) throw std::bad_alloc();
    forward.handle = fftw_plan_dft_r2c_3d(n[2], n[1], n[0], real.get(),
                                          complex.get(), FFTW_ESTIMATE);
    backward.handle = fftw_plan_dft_c2r_3d(n[2], n[1], n[0], complex.get(),
                                           real.get(), FFTW_ESTIMATE);
    if (!forward.handle || !backward.handle)
      throw std::runtime_error("LocalFFT: FFTW planning failed");
  }
};

LocalFFT::LocalFFT(const std::array<int, 3> &size)
    : m_impl(std::make_unique<Impl>(size)) {}
LocalFFT::~LocalFFT() = default;

void LocalFFT::forward(const RealVector &in, ComplexVector &out) {
  require_size(in.size(), size_inbox(), "forward real input");
  require_size(out.size(), size_outbox(), "forward complex output");
  const auto start = std::chrono::steady_clock::now();
  std::copy(in.begin(), in.end(), m_impl->real.get());
  fftw_execute(m_impl->forward.handle);
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = {m_impl->complex.get()[i][0], m_impl->complex.get()[i][1]};
  m_impl->time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
}
void LocalFFT::backward(const ComplexVector &in, RealVector &out) {
  require_size(in.size(), size_outbox(), "backward complex input");
  require_size(out.size(), size_inbox(), "backward real output");
  const auto start = std::chrono::steady_clock::now();
  const double scale = 1.0 / static_cast<double>(size_inbox());
  for (std::size_t i = 0; i < in.size(); ++i) {
    m_impl->complex.get()[i][0] = in[i].real() * scale;
    m_impl->complex.get()[i][1] = in[i].imag() * scale;
  }
  fftw_execute(m_impl->backward.handle);
  std::copy_n(m_impl->real.get(), out.size(), out.begin());
  m_impl->time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
}
void LocalFFT::reset_fft_time() { m_impl->time = 0.0; }
double LocalFFT::get_fft_time() const { return m_impl->time; }
std::size_t LocalFFT::size_inbox() const { return m_impl->nr; }
std::size_t LocalFFT::size_outbox() const { return m_impl->nc; }
std::size_t LocalFFT::size_workspace() const {
  return (m_impl->nr + 1) / 2 + m_impl->nc;
}
std::size_t LocalFFT::get_allocated_memory_bytes() const {
  return m_impl->nr * sizeof(double) + m_impl->nc * sizeof(fftw_complex);
}
Box3i LocalFFT::get_inbox_bounds() const {
  const auto &n = m_impl->n;
  return Box3i::from_bounds({0, 0, 0}, {n[0] - 1, n[1] - 1, n[2] - 1});
}
Box3i LocalFFT::get_outbox_bounds() const {
  const auto &n = m_impl->n;
  return Box3i::from_bounds({0, 0, 0}, {n[0] / 2, n[1] - 1, n[2] - 1});
}
} // namespace pfc::fft
