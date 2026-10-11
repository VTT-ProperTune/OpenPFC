// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <heffte.h>
#include <limits>
#include <numbers>
#include <openpfc/kernel/fft/mixed_transform.hpp>
#include <stdexcept>

namespace pfc::fft {
namespace {
void extent(std::size_t a, std::size_t b) {
  if (a != b) throw std::invalid_argument("mixed transform: wrong buffer extent");
}
} // namespace
struct MixedTransform::Impl {
  int nx, ny;
  double lx, ly;
  BoundedAxis z;
  MPI_Comm comm;
  Box3i input_box, output_box;
  std::unique_ptr<heffte::fft3d_r2c<heffte::backend::fftw>> plane;
  std::vector<double> data;
  std::vector<std::complex<double>> spectrum, workspace;
  fftw_plan forward = nullptr, backward = nullptr;

  Impl(int x, int y, double px, double py, BoundedAxis a, MPI_Comm c)
      : nx(x), ny(y), lx(px), ly(py), z(a), comm(c) {
    int rank, ranks;
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &ranks);
    int invalid = 0;
    try {
      validate(a);
    } catch (const std::invalid_argument &) {
      invalid = 1;
    }
    invalid |= x < 1 || y < ranks || !std::isfinite(px) || px <= 0 ||
               !std::isfinite(py) || py <= 0;
    int any_invalid;
    MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, c);
    if (any_invalid)
      throw std::invalid_argument("mixed transform: invalid collective geometry");
    // All peers must agree before heFFTe planning or batched collectives.
    double metadata[8] = {double(x),
                          double(y),
                          px,
                          py,
                          double(a.n),
                          a.length,
                          double(static_cast<int>(a.type)),
                          a.origin};
    double lo[8], hi[8];
    MPI_Allreduce(metadata, lo, 8, MPI_DOUBLE, MPI_MIN, c);
    MPI_Allreduce(metadata, hi, 8, MPI_DOUBLE, MPI_MAX, c);
    if (!std::equal(lo, lo + 8, hi))
      throw std::invalid_argument("mixed transform: geometry differs between ranks");
    auto in = heffte::split_world(heffte::box3d<>({0, 0, 0}, {x - 1, y - 1, 0}),
                                  {1, ranks, 1})[rank];
    auto out = heffte::split_world(heffte::box3d<>({0, 0, 0}, {x / 2, y - 1, 0}),
                                   {1, ranks, 1})[rank];
    input_box = Box3i::from_bounds({0, in.low[1], 0}, {x - 1, in.high[1], a.n - 1});
    output_box =
        Box3i::from_bounds({0, out.low[1], 0}, {x / 2, out.high[1], a.n - 1});
    const auto count = in.count();
    if (count > std::numeric_limits<int>::max()) invalid = 1;
    MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, c);
    if (any_invalid)
      throw std::invalid_argument("mixed transform: local FFTW batch exceeds int");
    auto options = heffte::default_options<heffte::backend::fftw>();
    plane = std::make_unique<heffte::fft3d_r2c<heffte::backend::fftw>>(in, out, 0, c,
                                                                       options);
    data.resize(input_box.count());
    spectrum.resize(output_box.count());
    workspace.resize(static_cast<std::size_t>(a.n) * plane->size_workspace());
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
    default: throw std::invalid_argument("mixed transform: invalid type");
    }
    forward =
        fftw_plan_many_r2r(1, &a.n, static_cast<int>(count), data.data(), nullptr,
                           static_cast<int>(count), 1, data.data(), nullptr,
                           static_cast<int>(count), 1, &f, FFTW_ESTIMATE);
    backward =
        fftw_plan_many_r2r(1, &a.n, static_cast<int>(count), data.data(), nullptr,
                           static_cast<int>(count), 1, data.data(), nullptr,
                           static_cast<int>(count), 1, &b, FFTW_ESTIMATE);
    invalid = !forward || !backward;
    MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX, c);
    if (any_invalid) {
      if (forward) fftw_destroy_plan(forward);
      if (backward) fftw_destroy_plan(backward);
      throw std::runtime_error("mixed transform: FFTW planning failed");
    }
  }
  ~Impl() {
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
  }
};

MixedTransform::MixedTransform(int nx, int ny, double lx, double ly, BoundedAxis z,
                               MPI_Comm c)
    : m_impl(std::make_unique<Impl>(nx, ny, lx, ly, z, c)) {}
MixedTransform::~MixedTransform() = default;
MixedTransform::MixedTransform(MixedTransform &&) noexcept = default;
MixedTransform &MixedTransform::operator=(MixedTransform &&) noexcept = default;
Box3i MixedTransform::inbox() const { return m_impl->input_box; }
Box3i MixedTransform::outbox() const { return m_impl->output_box; }
std::size_t MixedTransform::size_inbox() const { return m_impl->data.size(); }
std::size_t MixedTransform::size_outbox() const { return m_impl->spectrum.size(); }
std::size_t MixedTransform::buffer_bytes() const {
  return m_impl->data.size() * sizeof(double) +
         (m_impl->spectrum.size() + m_impl->workspace.size()) *
             sizeof(std::complex<double>);
}
void MixedTransform::forward(std::span<const double> in,
                             std::span<std::complex<double>> out) {
  extent(in.size(), size_inbox());
  extent(out.size(), size_outbox());
  std::copy(in.begin(), in.end(), m_impl->data.begin());
  fftw_execute(m_impl->forward);
  m_impl->plane->forward(m_impl->z.n, m_impl->data.data(), out.data(),
                         m_impl->workspace.data());
}
void MixedTransform::backward(std::span<const std::complex<double>> in,
                              std::span<double> out) {
  extent(in.size(), size_outbox());
  extent(out.size(), size_inbox());
  std::copy(in.begin(), in.end(), m_impl->spectrum.begin());
  m_impl->plane->backward(m_impl->z.n, m_impl->spectrum.data(), m_impl->data.data(),
                          m_impl->workspace.data(), heffte::scale::full);
  fftw_execute(m_impl->backward);
  double divisor = mirrored_size(m_impl->z);
  std::transform(m_impl->data.begin(), m_impl->data.end(), out.begin(),
                 [divisor](double v) { return v / divisor; });
}
double MixedTransform::eigenvalue(std::array<int, 3> m,
                                  LaplacianSymbol symbol) const {
  if (m[0] < 0 || m[0] > m_impl->nx / 2 || m[1] < 0 || m[1] >= m_impl->ny)
    throw std::out_of_range("mixed transform: mode outside global spectral box");
  double lambda = fft::eigenvalue(m_impl->z, m[2], symbol);
  int modes[2] = {m[0], std::min(m[1], m_impl->ny - m[1])};
  int sizes[2] = {m_impl->nx, m_impl->ny};
  double lengths[2] = {m_impl->lx, m_impl->ly};
  for (int d = 0; d < 2; ++d) {
    double k = 2 * std::numbers::pi * modes[d] / lengths[d];
    if (symbol == LaplacianSymbol::CenteredDifference) {
      double h = lengths[d] / sizes[d];
      k = 2 * std::sin(k * h / 2) / h;
    }
    lambda += k * k;
  }
  return lambda;
}
void MixedTransform::laplacian(std::span<const double> in, std::span<double> out,
                               LaplacianSymbol s) {
  forward(in, m_impl->spectrum);
  pfc::for_each_index(outbox(), [&](auto m) {
    m_impl->spectrum[outbox().to_linear(m)] *= -eigenvalue(m, s);
  });
  backward(m_impl->spectrum, out);
}
void MixedTransform::poisson(std::span<const double> in, std::span<double> out,
                             LaplacianSymbol s, double tol) {
  int invalid =
      !std::isfinite(tol) || tol < 0 || in.size() != size_inbox() ||
      out.size() != size_inbox() ||
      !std::all_of(in.begin(), in.end(), [](double v) { return std::isfinite(v); });
  int any;
  MPI_Allreduce(&invalid, &any, 1, MPI_INT, MPI_MAX, m_impl->comm);
  if (any)
    throw std::invalid_argument(
        "mixed transform: invalid Poisson buffers/forcing/tolerance");
  forward(in, m_impl->spectrum);
  if (is_cosine(m_impl->z.type)) {
    double norm = 0, zero = 0;
    for (auto v : m_impl->spectrum) norm = std::max(norm, std::abs(v));
    if (outbox().contains({0, 0, 0}))
      zero = std::abs(m_impl->spectrum[outbox().to_linear({0, 0, 0})]);
    double local[2] = {norm, zero}, global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, m_impl->comm);
    if (global[1] > tol * global[0])
      throw std::invalid_argument("mixed transform: incompatible Neumann forcing");
  }
  pfc::for_each_index(outbox(), [&](auto m) {
    const double lambda = eigenvalue(m, s);
    auto &v = m_impl->spectrum[outbox().to_linear(m)];
    v = lambda == 0 ? 0 : v / lambda;
  });
  backward(m_impl->spectrum, out);
}
} // namespace pfc::fft
