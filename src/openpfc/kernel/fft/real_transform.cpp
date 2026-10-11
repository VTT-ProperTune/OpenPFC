// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/real_transform.hpp>

#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace pfc::fft {
namespace {
void require_size(std::size_t n, const BoundedAxis &axis) {
  if (n != static_cast<std::size_t>(axis.n))
    throw std::invalid_argument("real transform: wrong buffer extent");
}
void require_mode(const BoundedAxis &axis, int m) {
  validate(axis);
  if (m < 0 || m >= axis.n)
    throw std::out_of_range("real transform: mode/sample outside axis");
}
} // namespace

bool is_cosine(RealTransformType t) {
  return t == RealTransformType::DCT1 || t == RealTransformType::DCT2;
}

void validate(const BoundedAxis &a) {
  const bool valid =
      (a.type == RealTransformType::DCT1 && a.grid == UniformGrid::Endpoints &&
       a.n >= 2) ||
      (a.type == RealTransformType::DST1 && a.grid == UniformGrid::Interior &&
       a.n >= 1) ||
      ((a.type == RealTransformType::DCT2 || a.type == RealTransformType::DST2) &&
       a.grid == UniformGrid::CellCentered && a.n >= 1);
  if (!valid || !std::isfinite(a.length) || a.length <= 0 ||
      !std::isfinite(a.origin) || a.n > (std::numeric_limits<int>::max() / 2 - 1))
    throw std::invalid_argument("real transform: invalid type/grid/count/geometry");
}

double spacing(const BoundedAxis &a) {
  validate(a);
  return a.length / (a.grid == UniformGrid::Endpoints  ? a.n - 1
                     : a.grid == UniformGrid::Interior ? a.n + 1
                                                       : a.n);
}
double coordinate(const BoundedAxis &a, int j) {
  require_mode(a, j);
  const double shift = a.grid == UniformGrid::CellCentered ? 0.5
                       : a.grid == UniformGrid::Interior   ? 1.0
                                                           : 0.0;
  return a.origin + (j + shift) * spacing(a);
}
int mirrored_size(const BoundedAxis &a) {
  validate(a);
  return 2 * (a.grid == UniformGrid::Endpoints  ? a.n - 1
              : a.grid == UniformGrid::Interior ? a.n + 1
                                                : a.n);
}
double wavenumber(const BoundedAxis &a, int m) {
  require_mode(a, m);
  return std::numbers::pi * (m + (is_cosine(a.type) ? 0 : 1)) / a.length;
}
double eigenvalue(const BoundedAxis &a, int m, LaplacianSymbol s) {
  const double k = wavenumber(a, m);
  if (s == LaplacianSymbol::Spectral) return k * k;
  if (s != LaplacianSymbol::CenteredDifference)
    throw std::invalid_argument("real transform: invalid Laplacian symbol");
  const double h = spacing(a);
  const double q = 2 * std::sin(0.5 * k * h) / h;
  return q * q;
}

struct RealTransform::Impl {
  BoundedAxis axis;
  std::vector<double> data, spectrum;
  fftw_plan forward = nullptr, backward = nullptr;

  explicit Impl(BoundedAxis a) : axis(a) {
    validate(a);
    data.resize(a.n);
    spectrum.resize(a.n);
    fftw_r2r_kind f, b;
    switch (a.type) {
    case RealTransformType::DCT1: f = b = FFTW_REDFT00; break;
    case RealTransformType::DCT2:
      f = FFTW_REDFT10;
      b = FFTW_REDFT01;
      break;
    case RealTransformType::DST1: f = b = FFTW_RODFT00; break;
    case RealTransformType::DST2:
      f = FFTW_RODFT10;
      b = FFTW_RODFT01;
      break;
    default: throw std::invalid_argument("real transform: invalid type");
    }
    forward = fftw_plan_r2r_1d(a.n, data.data(), data.data(), f, FFTW_ESTIMATE);
    backward = fftw_plan_r2r_1d(a.n, data.data(), data.data(), b, FFTW_ESTIMATE);
    if (!forward || !backward) {
      if (forward) fftw_destroy_plan(forward);
      if (backward) fftw_destroy_plan(backward);
      throw std::runtime_error("real transform: FFTW planning failed");
    }
  }
  ~Impl() {
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
  }
};

RealTransform::RealTransform(BoundedAxis a) : m_impl(std::make_unique<Impl>(a)) {}
RealTransform::~RealTransform() = default;
RealTransform::RealTransform(RealTransform &&) noexcept = default;
RealTransform &RealTransform::operator=(RealTransform &&) noexcept = default;
const BoundedAxis &RealTransform::axis() const { return m_impl->axis; }
std::size_t RealTransform::buffer_bytes() const {
  return (m_impl->data.size() + m_impl->spectrum.size()) * sizeof(double);
}
void RealTransform::forward(std::span<const double> in, std::span<double> out) {
  require_size(in.size(), axis());
  require_size(out.size(), axis());
  std::copy(in.begin(), in.end(), m_impl->data.begin());
  fftw_execute(m_impl->forward);
  std::copy(m_impl->data.begin(), m_impl->data.end(), out.begin());
}
void RealTransform::backward(std::span<const double> in, std::span<double> out) {
  require_size(in.size(), axis());
  require_size(out.size(), axis());
  std::copy(in.begin(), in.end(), m_impl->data.begin());
  fftw_execute(m_impl->backward);
  const double divisor = mirrored_size(axis());
  std::transform(m_impl->data.begin(), m_impl->data.end(), out.begin(),
                 [divisor](double v) { return v / divisor; });
}
void RealTransform::laplacian(std::span<const double> in, std::span<double> out,
                              LaplacianSymbol s) {
  forward(in, m_impl->spectrum);
  for (int m = 0; m < axis().n; ++m)
    m_impl->spectrum[m] *= -eigenvalue(axis(), m, s);
  backward(m_impl->spectrum, out);
}
void RealTransform::poisson(std::span<const double> in, std::span<double> out,
                            LaplacianSymbol s, double tol) {
  if (!std::isfinite(tol) || tol < 0 ||
      !std::all_of(in.begin(), in.end(), [](double v) { return std::isfinite(v); }))
    throw std::invalid_argument("real transform: nonfinite forcing or tolerance");
  forward(in, m_impl->spectrum);
  double norm = 0;
  for (double v : m_impl->spectrum) norm = std::max(norm, std::abs(v));
  if (is_cosine(axis().type) && std::abs(m_impl->spectrum[0]) > tol * norm)
    throw std::invalid_argument("real transform: incompatible Neumann forcing");
  for (int m = 0; m < axis().n; ++m) {
    const double lambda = eigenvalue(axis(), m, s);
    m_impl->spectrum[m] = lambda == 0 ? 0 : m_impl->spectrum[m] / lambda;
  }
  backward(m_impl->spectrum, out);
}
double RealTransform::evaluate(std::span<const double> c, double x,
                               int order) const {
  require_size(c.size(), axis());
  if (!std::isfinite(x) || order < 0 || order > 2)
    throw std::invalid_argument("real transform: invalid evaluation point/order");
  const auto &a = axis();
  const bool cosine = is_cosine(a.type);
  const double denominator = mirrored_size(a) / 2.0;
  double sum = 0;
  for (int m = 0; m < a.n; ++m) {
    const double k = wavenumber(a, m), angle = k * (x - a.origin);
    double amplitude = c[m] / denominator;
    if ((cosine && m == 0) || (a.type == RealTransformType::DCT1 && m == a.n - 1) ||
        (a.type == RealTransformType::DST2 && m == a.n - 1))
      amplitude *= 0.5;
    double basis = cosine ? std::cos(angle) : std::sin(angle);
    if (order == 1) basis = k * (cosine ? -std::sin(angle) : std::cos(angle));
    if (order == 2) basis *= -k * k;
    sum += amplitude * basis;
  }
  return sum;
}
} // namespace pfc::fft
