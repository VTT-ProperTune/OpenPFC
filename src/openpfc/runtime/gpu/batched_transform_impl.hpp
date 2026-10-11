// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Private shared CUDA/HIP implementation; never included by a public header.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <heffte.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <openpfc/kernel/fft/batched_transform.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#if defined(OPENPFC_GPU_API_CUDA)
#include <cufft.h>
#else
#include <rocfft/rocfft.h>
#endif

namespace pfc::fft {
namespace {
#if defined(OPENPFC_GPU_API_CUDA)
using Space = CUDASpace;
using Backend = heffte::backend::cufft;
#else
using Space = HIPSpace;
using Backend = heffte::backend::rocfft;
#endif
constexpr double pi = std::numbers::pi;
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("boundary transform: extent overflow");
  return a * b;
}
void extent(std::size_t a, std::size_t b) {
  if (a != b) throw std::invalid_argument("boundary transform: wrong extent");
}
void axis_check(BoundedAxis a) {
  validate(a);
  if (a.grid != UniformGrid::CellCentered ||
      (a.type != RealTransformType::DCT2 && a.type != RealTransformType::DST2))
    throw std::invalid_argument(
        "batched transforms support only cell-centered type II");
}
void symbol_check(LaplacianSymbol s) {
  if (s != LaplacianSymbol::Spectral && s != LaplacianSymbol::CenteredDifference)
    throw std::invalid_argument("boundary transform: invalid Laplacian symbol");
}
template <class T> struct Array {
  T *p = nullptr;
  std::size_t count = 0;
  Array() = default;
  explicit Array(std::size_t n) { allocate(n); }
  void allocate(std::size_t n) {
    auto bytes = product(n, sizeof(T));
    if (p) throw std::logic_error("boundary transform: workspace allocated twice");
    if (n) GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&p), bytes));
    count = n;
  }
  ~Array() {
    if (p) gpuFree(p);
  }
  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;
  [[nodiscard]] std::size_t bytes() const { return count * sizeof(T); }
};

// Vendor plans own no automatic buffers; forward and inverse reuse one area.
struct Fourier {
  Array<char> work;
#if defined(OPENPFC_GPU_API_CUDA)
  struct Handle {
    cufftHandle p = 0;
    Handle() { check(cufftCreate(&p)); }
    ~Handle() {
      if (p) cufftDestroy(p);
    }
    static void check(cufftResult r) {
      if (r != CUFFT_SUCCESS)
        throw std::runtime_error("boundary cuFFT status " + std::to_string(r));
    }
  } fw, bw;
  Fourier(std::vector<int> shape, int batches) {
    Handle::check(cufftSetAutoAllocation(fw.p, 0));
    Handle::check(cufftSetAutoAllocation(bw.p, 0));
    std::size_t real = 1;
    for (int n : shape) real = product(real, n);
    std::size_t complex = real / shape.back() * (shape.back() / 2 + 1);
    if (real > INT_MAX || complex > INT_MAX)
      throw std::invalid_argument("boundary vendor plan exceeds int distances");
    std::size_t a, b;
    Handle::check(cufftMakePlanMany(fw.p, shape.size(), shape.data(), nullptr, 1,
                                    real, nullptr, 1, complex, CUFFT_D2Z, batches,
                                    &a));
    Handle::check(cufftMakePlanMany(bw.p, shape.size(), shape.data(), nullptr, 1,
                                    complex, nullptr, 1, real, CUFFT_Z2D, batches,
                                    &b));
    work.allocate(std::max(a, b));
    Handle::check(cufftSetWorkArea(fw.p, work.p));
    Handle::check(cufftSetWorkArea(bw.p, work.p));
  }
  void forward(double *in, double2 *out) {
    Handle::check(
        cufftExecD2Z(fw.p, in, reinterpret_cast<cufftDoubleComplex *>(out)));
  }
  void backward(double2 *in, double *out) {
    Handle::check(
        cufftExecZ2D(bw.p, reinterpret_cast<cufftDoubleComplex *>(in), out));
  }
#else
  struct Plan {
    rocfft_plan p = nullptr;
    ~Plan() {
      if (p) rocfft_plan_destroy(p);
    }
  } fw, bw;
  struct Info {
    rocfft_execution_info p = nullptr;
    ~Info() {
      if (p) rocfft_execution_info_destroy(p);
    }
  } info;
  static void check(rocfft_status r) {
    if (r != rocfft_status_success)
      throw std::runtime_error("boundary rocFFT status " + std::to_string(r));
  }
  Fourier(std::vector<int> shape, int batches) {
    // heFFTe's process-wide guard avoids repeated setup/cleanup of rocFFT.
    heffte::rocm::initialize::make();
    std::vector<std::size_t> sizes(shape.rbegin(), shape.rend());
    check(rocfft_plan_create(
        &fw.p, rocfft_placement_notinplace, rocfft_transform_type_real_forward,
        rocfft_precision_double, sizes.size(), sizes.data(), batches, nullptr));
    check(rocfft_plan_create(
        &bw.p, rocfft_placement_notinplace, rocfft_transform_type_real_inverse,
        rocfft_precision_double, sizes.size(), sizes.data(), batches, nullptr));
    std::size_t a, b;
    check(rocfft_plan_get_work_buffer_size(fw.p, &a));
    check(rocfft_plan_get_work_buffer_size(bw.p, &b));
    work.allocate(std::max(a, b));
    check(rocfft_execution_info_create(&info.p));
    check(rocfft_execution_info_set_stream(info.p, nullptr));
    if (work.count)
      check(rocfft_execution_info_set_work_buffer(info.p, work.p, work.count));
  }
  void forward(double *in, double2 *out) {
    void *a = in, *b = out;
    check(rocfft_execute(fw.p, &a, &b, info.p));
  }
  void backward(double2 *in, double *out) {
    void *a = in, *b = out;
    check(rocfft_execute(bw.p, &a, &b, info.p));
  }
#endif
};
__global__ void phases(double2 *p, int n) {
  auto k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < n) sincos(pi * k / (2 * n), &p[k].y, &p[k].x);
}
__device__ int sample(int p, int n) {
  return p < (n + 1) / 2 ? 2 * p : 2 * (n - 1 - p) + 1;
}
__global__ void pack_lines(const double *in, double *out, int n, int batches,
                           int stride, int distance, bool sine, bool inverse) {
  __shared__ double tile[32][33];
  int b = blockIdx.x * 32 + threadIdx.x, p0 = blockIdx.y * 32;
  if (!inverse) {
    for (int t = threadIdx.y; t < 32; t += 8) {
      int p = p0 + t;
      if (b < batches && p < n) {
        int j = sample(p, n);
        tile[t][threadIdx.x] =
            (sine && j % 2 ? -1 : 1) *
            in[std::size_t(b) * distance + std::size_t(j) * stride];
      }
    }
    __syncthreads();
    int p = p0 + threadIdx.x;
    for (int t = threadIdx.y; t < 32; t += 8) {
      int line = blockIdx.x * 32 + t;
      if (line < batches && p < n)
        out[std::size_t(line) * n + p] = tile[threadIdx.x][t];
    }
  } else {
    int p = p0 + threadIdx.x;
    for (int t = threadIdx.y; t < 32; t += 8) {
      int line = blockIdx.x * 32 + t;
      if (line < batches && p < n)
        tile[threadIdx.x][t] = in[std::size_t(line) * n + p];
    }
    __syncthreads();
    for (int t = threadIdx.y; t < 32; t += 8) {
      int p = p0 + t;
      if (b < batches && p < n) {
        int j = sample(p, n);
        out[std::size_t(b) * distance + std::size_t(j) * stride] =
            (sine && j % 2 ? -1 : 1) * tile[t][threadIdx.x];
      }
    }
  }
}
__global__ void line_modes(const double2 *in, double *out, const double2 *phase,
                           int n, int batches, int stride, int distance, bool sine) {
  __shared__ double tile[32][33];
  int k = blockIdx.x * 32 + threadIdx.x, b0 = blockIdx.y * 32;
  for (int t = threadIdx.y; t < 32; t += 8) {
    int b = b0 + t;
    if (k < n && b < batches) {
      double2 a = in[std::size_t(b) * (n / 2 + 1) + (k > n / 2 ? n - k : k)];
      if (k > n / 2) a.y = -a.y;
      tile[t][threadIdx.x] = 2 * (a.x * phase[k].x + a.y * phase[k].y);
    }
  }
  __syncthreads();
  int b = b0 + threadIdx.x;
  for (int t = threadIdx.y; t < 32; t += 8) {
    int q = blockIdx.x * 32 + t;
    if (b < batches && q < n)
      out[std::size_t(b) * distance + std::size_t(sine ? n - 1 - q : q) * stride] =
          tile[threadIdx.x][t];
  }
}
__global__ void inverse_lines(const double *in, double2 *out, const double2 *phase,
                              int n, int batches, int stride, int distance,
                              bool sine) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int h = n / 2 + 1;
  if (i < std::size_t(h) * batches) {
    int k = i % h;
    auto b = i / h;
    double a = in[b * distance + std::size_t(sine ? n - 1 - k : k) * stride];
    if (!k)
      out[i] = {a / (2 * n), 0};
    else {
      double c = in[b * distance + std::size_t(sine ? k - 1 : n - k) * stride];
      double2 p = phase[k];
      out[i] = {(a * p.x + c * p.y) / (2 * n),
                2 * k == n ? 0 : (a * p.y - c * p.x) / (2 * n)};
    }
  }
}
__global__ void reorder(const double *in, double *out, int n, std::size_t plane,
                        bool sine, bool inverse) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < plane * n) {
    int j = sample(i / plane, n);
    auto target = std::size_t(j) * plane + i % plane;
    double sign = sine && j % 2 ? -1 : 1;
    if (inverse)
      out[target] = sign * in[i];
    else
      out[i] = sign * in[target];
  }
}
__device__ double2 cosine_mode(const double2 *v, const double2 *phase, int k, int n,
                               std::size_t plane, std::size_t b) {
  double2 a = v[std::size_t(k) * plane + b];
  if (!k) return {2 * a.x, 2 * a.y};
  double2 c = v[std::size_t(n - k) * plane + b], p = phase[k];
  return {p.x * (a.x + c.x) + p.y * (a.y - c.y),
          p.x * (a.y + c.y) - p.y * (a.x - c.x)};
}
__global__ void expose(const double2 *in, double2 *out, const double2 *phase, int n,
                       std::size_t plane, bool sine) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < plane * n) {
    int k = i / plane;
    auto b = i % plane;
    out[std::size_t(sine ? n - 1 - k : k) * plane + b] =
        cosine_mode(in, phase, k, n, plane, b);
  }
}
__global__ void reconstruct(const double2 *in, double2 *out, const double2 *phase,
                            int n, std::size_t plane, bool sine, double factor) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < plane * n) {
    int k = i / plane;
    auto b = i % plane;
    double2 a = in[std::size_t(sine ? n - 1 - k : k) * plane + b];
    if (!k)
      out[i] = {.5 * factor * a.x, .5 * factor * a.y};
    else {
      double2 c = in[std::size_t(sine ? k - 1 : n - k) * plane + b], p = phase[k];
      double2 u = {a.x + c.y, a.y - c.x};
      out[i] = {.5 * factor * (p.x * u.x - p.y * u.y),
                .5 * factor * (p.y * u.x + p.x * u.y)};
    }
  }
}
__global__ void paired_diagonal(double2 *v, const double *weights,
                                const double2 *phase, int n, std::size_t plane,
                                bool sine) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= plane * (n / 2 + 1)) return;
  int k = i / plane;
  auto b = i % plane;
  double wa = weights[std::size_t(sine ? n - 1 - k : k) * plane + b];
  double2 a = v[i];
  if (!k || 2 * k == n) {
    v[i] = {wa * a.x, wa * a.y};
    return;
  }
  auto j = std::size_t(n - k) * plane + b;
  double2 c = v[j], p = phase[k];
  double2 sum = {a.x + c.x, a.y + c.y}, diff = {a.x - c.x, a.y - c.y};
  double2 ca = {p.x * sum.x + p.y * diff.y, p.x * sum.y - p.y * diff.x};
  double2 cb = {p.y * sum.x - p.x * diff.y, p.y * sum.y + p.x * diff.x};
  double wb = weights[std::size_t(sine ? k - 1 : n - k) * plane + b];
  ca.x *= wa;
  ca.y *= wa;
  cb.x *= wb;
  cb.y *= wb;
  double2 u = {ca.x + cb.y, ca.y - cb.x}, w = {ca.x - cb.y, ca.y + cb.x};
  v[i] = {.5 * (p.x * u.x - p.y * u.y), .5 * (p.y * u.x + p.x * u.y)};
  v[j] = {.5 * (p.x * w.x + p.y * w.y), .5 * (p.x * w.y - p.y * w.x)};
}
__global__ void diagonal_weights(double *out, int nx, int ny, int local_y, int ylow,
                                 int n, double lx, double ly, double lz, bool sine,
                                 bool fd, double alpha, bool laplacian) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t plane = std::size_t(nx / 2 + 1) * local_y;
  if (i < plane * n) {
    int x = i % (nx / 2 + 1), y = i % plane / (nx / 2 + 1) + ylow;
    int q = i / plane + int(sine);
    double wave[3] = {2 * pi * x / lx, 2 * pi * min(y, ny - y) / ly, pi * q / lz};
    double h[3] = {lx / nx, ly / ny, lz / n}, lambda = 0;
    for (int d = 0; d < 3; ++d) {
      double a = fd ? 2 * sin(wave[d] * h[d] / 2) / h[d] : wave[d];
      lambda += a * a;
    }
    double norm = double(nx) * ny * n;
    out[i] = laplacian             ? -lambda / norm
             : lambda + alpha == 0 ? 0
                                   : 1 / ((lambda + alpha) * norm);
  }
}
// Atomic positive-double maximum. Only three scalars cross the device boundary.
__global__ void compatibility(const double2 *v, const double2 *phase, double *stats,
                              int n, std::size_t plane, bool owns_zero) {
  __shared__ double local[256];
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  double a = 0;
  if (i < plane * n) {
    double2 c = cosine_mode(v, phase, i / plane, n, plane, i % plane);
    a = hypot(c.x, c.y);
    if (!isfinite(a))
      atomicExch(reinterpret_cast<unsigned long long *>(stats + 2),
                 __double_as_longlong(1.));
    if (i == 0 && owns_zero) stats[1] = a;
  }
  local[threadIdx.x] = a;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step)
      local[threadIdx.x] = fmax(local[threadIdx.x], local[threadIdx.x + step]);
    __syncthreads();
  }
  if (!threadIdx.x)
    atomicMax(reinterpret_cast<unsigned long long *>(stats),
              static_cast<unsigned long long>(__double_as_longlong(local[0])));
}
unsigned blocks(std::size_t n) {
  if ((n + 255) / 256 > INT_MAX)
    throw std::invalid_argument("boundary launch exceeds grid limit");
  return (n + 255) / 256;
}
} // namespace

template <class M> struct BatchedRealTransform<M>::Impl {
  BoundedAxis z;
  int batches, stride, distance;
  std::size_t extent;
  bool sine;
  Array<double> data;
  Array<double2> modes, phase;
  std::unique_ptr<Fourier> fft;
  Impl(BoundedAxis a, int b, int s, int d)
      : z(a), batches(b), stride(s), distance(d) {
    axis_check(a);
    if (b < 1 || s < 1 || d < 1)
      throw std::invalid_argument("boundary batch/strides must be positive");
    if (double(a.n) * b > INT_MAX || a.n > 65535 * 32 || b > 65535 * 32)
      throw std::invalid_argument("boundary batch exceeds vendor/grid limits");
    auto g = std::gcd(s, d);
    if (d / g < a.n && s / g < b)
      throw std::invalid_argument("boundary lines overlap");
    auto first = product(b - 1, d), last = product(a.n - 1, s);
    if (first > std::numeric_limits<std::size_t>::max() - last - 1)
      throw std::invalid_argument("boundary extent overflow");
    extent = first + last + 1;
    product(extent, sizeof(double));
    sine = a.type == RealTransformType::DST2;
    data.allocate(product(a.n, b));
    modes.allocate(product(a.n / 2 + 1, b));
    phase.allocate(a.n);
    fft = std::make_unique<Fourier>(std::vector<int>{a.n}, b);
    phases<<<blocks(a.n), 256>>>(phase.p, a.n);
    GPU_CHECK(gpuGetLastError());
  }
};
template <class M>
BatchedRealTransform<M>::BatchedRealTransform(BoundedAxis a, int b, int s, int d)
    : m_impl(std::make_unique<Impl>(a, b, s, d)) {}
template <class M> BatchedRealTransform<M>::~BatchedRealTransform() = default;
template <class M>
BatchedRealTransform<M>::BatchedRealTransform(BatchedRealTransform &&) noexcept =
    default;
template <class M>
BatchedRealTransform<M> &
BatchedRealTransform<M>::operator=(BatchedRealTransform &&) noexcept = default;
template <class M> const BoundedAxis &BatchedRealTransform<M>::axis() const {
  return m_impl->z;
}
template <class M> std::size_t BatchedRealTransform<M>::size_buffer() const {
  return m_impl->extent;
}
template <class M> std::size_t BatchedRealTransform<M>::buffer_bytes() const {
  auto &p = *m_impl;
  return p.data.bytes() + p.modes.bytes() + p.phase.bytes() + p.fft->work.bytes();
}
template <class M>
void BatchedRealTransform<M>::forward(std::span<const double> in,
                                      std::span<double> out) {
  auto &p = *m_impl;
  extent(in.size(), p.extent);
  extent(out.size(), p.extent);
  pack_lines<<<dim3((p.batches + 31) / 32, (p.z.n + 31) / 32), dim3(32, 8)>>>(
      in.data(), p.data.p, p.z.n, p.batches, p.stride, p.distance, p.sine, false);
  GPU_CHECK(gpuGetLastError());
  p.fft->forward(p.data.p, p.modes.p);
  line_modes<<<dim3((p.z.n + 31) / 32, (p.batches + 31) / 32), dim3(32, 8)>>>(
      p.modes.p, out.data(), p.phase.p, p.z.n, p.batches, p.stride, p.distance,
      p.sine);
  GPU_CHECK(gpuGetLastError());
}
template <class M>
void BatchedRealTransform<M>::backward(std::span<const double> in,
                                       std::span<double> out) {
  auto &p = *m_impl;
  extent(in.size(), p.extent);
  extent(out.size(), p.extent);
  inverse_lines<<<blocks(p.modes.count), 256>>>(in.data(), p.modes.p, p.phase.p,
                                                p.z.n, p.batches, p.stride,
                                                p.distance, p.sine);
  GPU_CHECK(gpuGetLastError());
  p.fft->backward(p.modes.p, p.data.p);
  pack_lines<<<dim3((p.batches + 31) / 32, (p.z.n + 31) / 32), dim3(32, 8)>>>(
      p.data.p, out.data(), p.z.n, p.batches, p.stride, p.distance, p.sine, true);
  GPU_CHECK(gpuGetLastError());
}

template <class M> struct BatchedMixedTransform<M>::Impl {
  int nx, ny, local_y, ylow;
  double lx, ly, norm;
  BoundedAxis z;
  MPI_Comm comm;
  bool sine;
  Box3i input_box, output_box;
  Array<double> data, weights, stats;
  Array<double2> modes, phase;
  Array<std::complex<double>> work;
  std::unique_ptr<Fourier> local;
  std::unique_ptr<heffte::fft3d_r2c<Backend>> distributed;
  double cached_alpha = -1;
  LaplacianSymbol cached_symbol{};
  bool cached_laplacian = false;
  Impl(int x, int y, double px, double py, BoundedAxis a, MPI_Comm c)
      : nx(x), ny(y), lx(px), ly(py), z(a), comm(c) {
    if (c == MPI_COMM_NULL)
      throw std::invalid_argument("boundary communicator is null");
    int rank, ranks;
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &ranks);
    int invalid = 0, any;
    try {
      axis_check(a);
      if (x < 1 || y < ranks || !std::isfinite(px) || px <= 0 ||
          !std::isfinite(py) || py <= 0)
        invalid = 1;
      product(product(product(std::max(1, x), std::max(1, y)), a.n), 16);
      // heFFTe 2.4.1's vendor executors and this single-rank plan use int sizes.
      if (double(x) * y * a.n > INT_MAX) invalid = 1;
    } catch (const std::invalid_argument &) {
      invalid = 1;
    }
    MPI_Allreduce(&invalid, &any, 1, MPI_INT, MPI_MAX, c);
    if (any)
      throw std::invalid_argument("boundary mixed collective geometry invalid");
    double meta[8] = {double(x), double(y),           px,      py, double(a.n),
                      a.length,  double(int(a.type)), a.origin},
           lo[8], hi[8];
    MPI_Allreduce(meta, lo, 8, MPI_DOUBLE, MPI_MIN, c);
    MPI_Allreduce(meta, hi, 8, MPI_DOUBLE, MPI_MAX, c);
    if (!std::equal(lo, lo + 8, hi))
      throw std::invalid_argument("boundary geometry differs across ranks");
    auto in = heffte::split_world(
        heffte::box3d<>({0, 0, 0}, {x - 1, y - 1, a.n - 1}), {1, ranks, 1})[rank];
    auto out = heffte::split_world(
        heffte::box3d<>({0, 0, 0}, {x / 2, y - 1, a.n - 1}), {1, ranks, 1})[rank];
    input_box = Box3i::from_bounds(in.low, in.high);
    output_box = Box3i::from_bounds(out.low, out.high);
    local_y = in.size[1];
    ylow = out.low[1];
    sine = a.type == RealTransformType::DST2;
    norm = 1 / (double(x) * y * a.n);
    data.allocate(in.count());
    modes.allocate(out.count());
    weights.allocate(out.count());
    phase.allocate(a.n);
    stats.allocate(3);
    if (ranks == 1)
      local = std::make_unique<Fourier>(std::vector<int>{a.n, y, x}, 1);
    else {
      auto options = heffte::default_options<Backend>();
      options.use_reorder = true;
#if (defined(OPENPFC_GPU_API_CUDA) && defined(OpenPFC_MPI_CUDA_AWARE)) ||           \
    (defined(OPENPFC_GPU_API_HIP) && defined(OpenPFC_MPI_HIP_AWARE))
      options.use_gpu_aware = true;
#else
      options.use_gpu_aware = false;
#endif
      distributed =
          std::make_unique<heffte::fft3d_r2c<Backend>>(in, out, 0, c, options);
      work.allocate(distributed->size_workspace());
    }
    phases<<<blocks(a.n), 256>>>(phase.p, a.n);
    GPU_CHECK(gpuGetLastError());
  }
  std::size_t plane() const { return modes.count / z.n; }
  void forward(const double *in) {
    reorder<<<blocks(data.count), 256>>>(in, data.p, z.n, data.count / z.n, sine,
                                         false);
    GPU_CHECK(gpuGetLastError());
    if (local)
      local->forward(data.p, modes.p);
    else
      distributed->forward(data.p, reinterpret_cast<std::complex<double> *>(modes.p),
                           work.p);
  }
  void inverse(double *out) {
    if (local)
      local->backward(modes.p, data.p);
    else
      distributed->backward(reinterpret_cast<std::complex<double> *>(modes.p),
                            data.p, work.p);
    reorder<<<blocks(data.count), 256>>>(data.p, out, z.n, data.count / z.n, sine,
                                         true);
    GPU_CHECK(gpuGetLastError());
  }
  void set_weights(double alpha, LaplacianSymbol s, bool lap) {
    if (alpha == cached_alpha && s == cached_symbol && lap == cached_laplacian)
      return;
    diagonal_weights<<<blocks(modes.count), 256>>>(
        weights.p, nx, ny, local_y, ylow, z.n, lx, ly, z.length, sine,
        s == LaplacianSymbol::CenteredDifference, alpha, lap);
    GPU_CHECK(gpuGetLastError());
    cached_alpha = alpha;
    cached_symbol = s;
    cached_laplacian = lap;
  }
  void apply() {
    paired_diagonal<<<blocks(plane() * (z.n / 2 + 1)), 256>>>(
        modes.p, weights.p, phase.p, z.n, plane(), sine);
    GPU_CHECK(gpuGetLastError());
  }
  void check_compatibility(double tol) {
    double zero[3] = {}, global[3];
    GPU_CHECK(
        gpuMemcpyAsync(stats.p, zero, sizeof(zero), gpuMemcpyHostToDevice, nullptr));
    compatibility<<<blocks(modes.count), 256>>>(
        modes.p, phase.p, stats.p, z.n, plane(), output_box.contains({0, 0, 0}));
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuMemcpy(zero, stats.p, sizeof(zero), gpuMemcpyDeviceToHost));
    MPI_Allreduce(zero, global, 3, MPI_DOUBLE, MPI_MAX, comm);
    if (global[2] || global[1] > tol * global[0])
      throw std::invalid_argument(
          "boundary mixed incompatible/nonfinite Neumann forcing");
  }
};
template <class M>
BatchedMixedTransform<M>::BatchedMixedTransform(int x, int y, double lx, double ly,
                                                BoundedAxis z, MPI_Comm c)
    : m_impl(std::make_unique<Impl>(x, y, lx, ly, z, c)) {}
template <class M> BatchedMixedTransform<M>::~BatchedMixedTransform() = default;
template <class M>
BatchedMixedTransform<M>::BatchedMixedTransform(BatchedMixedTransform &&) noexcept =
    default;
template <class M>
BatchedMixedTransform<M> &
BatchedMixedTransform<M>::operator=(BatchedMixedTransform &&) noexcept = default;
template <class M> Box3i BatchedMixedTransform<M>::inbox() const {
  return m_impl->input_box;
}
template <class M> Box3i BatchedMixedTransform<M>::outbox() const {
  return m_impl->output_box;
}
template <class M> std::size_t BatchedMixedTransform<M>::size_inbox() const {
  return m_impl->data.count;
}
template <class M> std::size_t BatchedMixedTransform<M>::size_outbox() const {
  return m_impl->modes.count;
}
template <class M> std::size_t BatchedMixedTransform<M>::buffer_bytes() const {
  auto &p = *m_impl;
  return p.data.bytes() + p.modes.bytes() + p.weights.bytes() + p.phase.bytes() +
         p.stats.bytes() + p.work.bytes() + (p.local ? p.local->work.bytes() : 0);
}
template <class M>
double BatchedMixedTransform<M>::eigenvalue(std::array<int, 3> m,
                                            LaplacianSymbol s) const {
  auto &p = *m_impl;
  symbol_check(s);
  if (m[0] < 0 || m[0] > p.nx / 2 || m[1] < 0 || m[1] >= p.ny)
    throw std::out_of_range("boundary mode outside box");
  double value = fft::eigenvalue(p.z, m[2], s), len[2] = {p.lx, p.ly};
  int sizes[2] = {p.nx, p.ny}, modes[2] = {m[0], std::min(m[1], p.ny - m[1])};
  for (int d = 0; d < 2; ++d) {
    double k = 2 * pi * modes[d] / len[d];
    if (s == LaplacianSymbol::CenteredDifference)
      k = 2 * std::sin(k * len[d] / (2 * sizes[d])) / (len[d] / sizes[d]);
    value += k * k;
  }
  return value;
}
template <class M>
void BatchedMixedTransform<M>::forward(std::span<const double> in,
                                       std::span<std::complex<double>> out) {
  auto &p = *m_impl;
  extent(in.size(), size_inbox());
  extent(out.size(), size_outbox());
  p.forward(in.data());
  expose<<<blocks(p.modes.count), 256>>>(p.modes.p,
                                         reinterpret_cast<double2 *>(out.data()),
                                         p.phase.p, p.z.n, p.plane(), p.sine);
  GPU_CHECK(gpuGetLastError());
}
template <class M>
void BatchedMixedTransform<M>::backward(std::span<const std::complex<double>> in,
                                        std::span<double> out) {
  auto &p = *m_impl;
  extent(in.size(), size_outbox());
  extent(out.size(), size_inbox());
  reconstruct<<<blocks(p.modes.count), 256>>>(
      reinterpret_cast<const double2 *>(in.data()), p.modes.p, p.phase.p, p.z.n,
      p.plane(), p.sine, p.norm);
  GPU_CHECK(gpuGetLastError());
  p.inverse(out.data());
}
template <class M>
void BatchedMixedTransform<M>::laplacian(std::span<const double> in,
                                         std::span<double> out, LaplacianSymbol s) {
  auto &p = *m_impl;
  extent(in.size(), size_inbox());
  extent(out.size(), size_inbox());
  symbol_check(s);
  p.set_weights(0, s, true);
  p.forward(in.data());
  p.apply();
  p.inverse(out.data());
}
template <class M>
void BatchedMixedTransform<M>::poisson(std::span<const double> in,
                                       std::span<double> out, LaplacianSymbol s,
                                       double tol) {
  helmholtz(in, out, 0, s, tol);
}
template <class M>
void BatchedMixedTransform<M>::helmholtz(std::span<const double> in,
                                         std::span<double> out, double alpha,
                                         LaplacianSymbol s, double tol) {
  auto &p = *m_impl;
  int invalid = in.size() != size_inbox() || out.size() != size_inbox() ||
                !std::isfinite(alpha) || alpha < 0 || !std::isfinite(tol) ||
                tol < 0 ||
                (s != LaplacianSymbol::Spectral &&
                 s != LaplacianSymbol::CenteredDifference),
      any;
  MPI_Allreduce(&invalid, &any, 1, MPI_INT, MPI_MAX, p.comm);
  if (any)
    throw std::invalid_argument(
        "boundary mixed invalid collective solve parameters");
  double meta[3] = {alpha, double(int(s)), tol}, lo[3], hi[3];
  MPI_Allreduce(meta, lo, 3, MPI_DOUBLE, MPI_MIN, p.comm);
  MPI_Allreduce(meta, hi, 3, MPI_DOUBLE, MPI_MAX, p.comm);
  if (!std::equal(lo, lo + 3, hi))
    throw std::invalid_argument("boundary solve parameters differ across ranks");
  p.set_weights(alpha, s, false);
  p.forward(in.data());
  if (!p.sine && alpha == 0) p.check_compatibility(tol);
  p.apply();
  p.inverse(out.data());
}
template class BatchedRealTransform<Space>;
template class BatchedMixedTransform<Space>;
} // namespace pfc::fft
