// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/kernel/field/padded_face_flux.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#include <vector>

namespace {
namespace fd = pfc::field::fd;
struct Current {
  bool periodic;
  OPENPFC_HD int wrap(int x, int n) const { return (x % n + n) % n; }
  OPENPFC_HD double coefficient(int i, int j, int k) const {
    return 1 + 0.1 * wrap(i, 7) + 0.03 * wrap(j, 5) + 0.02 * wrap(k, 3);
  }
  OPENPFC_HD double potential(int i, int j, int k) const {
    const int x = wrap(i, 7), y = wrap(j, 5), z = wrap(k, 3);
    return double(x * x - 3 * y + z * z);
  }
  OPENPFC_HD double operator()(int axis, int i, int j, int k) const {
    if (!periodic)
      return axis == 0   ? 2.0 * (i + 1) * (i + 1)
             : axis == 1 ? -3.0 * (j + 1) * (j + 1)
                         : 4.0 * (k + 1) * (k + 1);
    const int a = i + (axis == 0), b = j + (axis == 1), c = k + (axis == 2);
    return fd::face_flux(fd::average_face(fd::FaceAverage::Arithmetic,
                                          coefficient(i, j, k),
                                          coefficient(a, b, c)),
                         potential(a, b, c), potential(i, j, k), 0.5);
  }
};
template <class Eval>
__global__ void evaluate(Eval eval, double *out, int nx, int ny, int nz) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < nx * ny * nz) out[q] = eval(q % nx, (q / nx) % ny, q / (nx * ny));
}
struct Allocation {
  double *data = nullptr;
  ~Allocation() {
    if (data) (void)pfc::gpuFree(data);
  }
};
} // namespace

TEST_CASE(
    "resident pointwise face current agrees with the independent host operator",
    "[face_current][resident_face_current][integration]") {
  int count = 0;
#if defined(OPENPFC_GPU_API_HIP)
  const auto status = hipGetDeviceCount(&count);
#else
  const auto status = cudaGetDeviceCount(&count);
#endif
  if (status != pfc::gpuSuccess || count == 0) SKIP("No allocated device");
  const int nx = 7, ny = 5, nz = 3, n = nx * ny * nz;
  Allocation device;
  REQUIRE(pfc::gpuMalloc(reinterpret_cast<void **>(&device.data),
                         n * sizeof(double)) == pfc::gpuSuccess);
  std::vector<double> got(n), reference(n), gx(fd::n_xfaces(nx, ny, nz)),
      gy(fd::n_yfaces(nx, ny, nz)), gz(fd::n_zfaces(nx, ny, nz));
  for (int kind = 0; kind < 3; ++kind) {
    const Current current{kind == 0};
    auto walls = kind == 0 ? fd::model_walls() : fd::closed_walls();
    if (kind == 2) {
      walls.xm = {fd::WallPolicy::Prescribed, 2.5};
      walls.zp = {fd::WallPolicy::Prescribed, -1.2};
    }
    fd::fill_faces(current, gx, gy, gz, nx, ny, nz, {true, true, true}, walls);
    fd::face_divergence(gx, gy, gz, reference, nx, ny, nz, {true, true, true}, 0.5,
                        0.5, 0.5);
    const auto eval = fd::face_divergence(current, nx, ny, nz, {true, true, true},
                                          0.5, 0.5, 0.5, walls);
    evaluate<<<(n + 127) / 128, 128>>>(eval, device.data, nx, ny, nz);
    REQUIRE(pfc::gpuGetLastError() == pfc::gpuSuccess);
    REQUIRE(pfc::gpuMemcpy(got.data(), device.data, n * sizeof(double),
                           pfc::gpuMemcpyDeviceToHost) == pfc::gpuSuccess);
    double sum = 0;
    for (int q = 0; q < n; ++q) {
      CHECK(std::abs(got[q] - reference[q]) <= 1e-12 * (1 + std::abs(reference[q])));
      sum += got[q];
    }
    if (kind == 0)
      CHECK(std::abs(sum) < 1e-10);
    else
      CHECK(std::abs(sum * 0.125 - fd::outer_flux(gx, gy, gz, nx, ny, nz,
                                                  {true, true, true}, 0.5, 0.5,
                                                  0.5)) < 1e-10);
  }
}

TEST_CASE("padded device flux matches an independent finite-volume balance",
          "[face_current][padded_face_flux][integration]") {
  int device_count = 0;
#if defined(OPENPFC_GPU_API_HIP)
  const auto status = hipGetDeviceCount(&device_count);
  using Space = pfc::HIPSpace;
#else
  const auto status = cudaGetDeviceCount(&device_count);
  using Space = pfc::CUDASpace;
#endif
  if (status != pfc::gpuSuccess || device_count == 0) SKIP("No allocated device");
  // Thin owned axes and distinct padding widths exercise both field pitches.
  for (int nx : {1, 7}) {
    const int ny = 3, nz = 2, n = nx * ny * nz;
    const auto domain = pfc::domain::with_spacing({nx, ny, nz}, {0.5, 0.7, 0.9},
                                                  {false, true, true});
    const auto box = pfc::Box3i::from_bounds({0, 0, 0}, {nx - 1, ny - 1, nz - 1});
    pfc::data::Field<double, Space> a(domain, box, 1), p(domain, box, 2);
    const auto coefficient = [](int i, int j, int k) {
      return 1.0 + 0.05 * i + 0.03 * j + 0.02 * k;
    };
    const auto potential = [](int i, int j, int k) {
      return double(i * i + 2 * j * j - k * k);
    };
    a.with_host_view([&](double *data, std::size_t) {
      for (int k = -1; k < nz + 1; ++k)
        for (int j = -1; j < ny + 1; ++j)
          for (int i = -1; i < nx + 1; ++i)
            data[a.idx(i, j, k)] = coefficient(i, j, k);
    });
    p.with_host_view([&](double *data, std::size_t) {
      for (int k = -2; k < nz + 2; ++k)
        for (int j = -2; j < ny + 2; ++j)
          for (int i = -2; i < nx + 2; ++i)
            data[p.idx(i, j, k)] = potential(i, j, k);
    });
    Allocation result;
    REQUIRE(pfc::gpuMalloc(reinterpret_cast<void **>(&result.data),
                           n * sizeof(double)) == pfc::gpuSuccess);
    for (auto average : {fd::FaceAverage::Arithmetic, fd::FaceAverage::Harmonic}) {
      auto walls = fd::model_walls();
      walls.xm = walls.xp = {fd::WallPolicy::Zero, 0};
      const auto flux = fd::padded_face_flux(a, p, average);
      const auto div = fd::face_divergence(flux, nx, ny, nz, {true, true, true}, 0.5,
                                           0.7, 0.9, walls);
      evaluate<<<(n + 127) / 128, 128>>>(div, result.data, nx, ny, nz);
      REQUIRE(pfc::gpuGetLastError() == pfc::gpuSuccess);
      std::vector<double> got(n);
      REQUIRE(pfc::gpuMemcpy(got.data(), result.data, n * sizeof(double),
                             pfc::gpuMemcpyDeviceToHost) == pfc::gpuSuccess);
      for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i) {
            double expected = 0;
            const int cell[3] = {i, j, k};
            const int extent[3] = {nx, ny, nz};
            const double h[3] = {0.5, 0.7, 0.9};
            for (int axis = 0; axis < 3; ++axis) {
              double face[2]{};
              for (int side = 0; side < 2; ++side) {
                if (axis == 0 && cell[axis] == (side == 0 ? 0 : extent[axis] - 1))
                  continue;
                const int lo[3] = {i - (axis == 0 && side == 0),
                                   j - (axis == 1 && side == 0),
                                   k - (axis == 2 && side == 0)};
                const int hi[3] = {lo[0] + (axis == 0), lo[1] + (axis == 1),
                                   lo[2] + (axis == 2)};
                const double al = coefficient(lo[0], lo[1], lo[2]);
                const double ah = coefficient(hi[0], hi[1], hi[2]);
                const double af = average == fd::FaceAverage::Arithmetic
                                      ? (al + ah) / 2
                                      : 2 * al * ah / (al + ah);
                face[side] = af *
                             (potential(hi[0], hi[1], hi[2]) -
                              potential(lo[0], lo[1], lo[2])) /
                             h[axis];
              }
              expected += (face[1] - face[0]) / h[axis];
            }
            CHECK(std::abs(got[fd::cell_index(nx, ny, i, j, k)] - expected) < 1e-11);
          }
    }
    pfc::data::Field<double, Space> unpadded(domain, box, 0);
    CHECK_THROWS_AS(fd::padded_face_flux(a, unpadded), std::invalid_argument);
  }
}
