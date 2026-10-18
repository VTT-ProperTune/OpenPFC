// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/face_flux.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
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
