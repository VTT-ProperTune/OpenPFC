// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>
#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/reflection_ghosts_gpu.hpp>
#endif

int main(int argc, char **) {
  try {
    using namespace pfc::communication;
    auto domain = pfc::domain::create(
        pfc::GridSize({8, 6, 5}), pfc::PhysicalOrigin({0., 0., 0.}),
        pfc::GridSpacing({1., 1., 1.}), pfc::Bool3{false, false, false});
    auto box = pfc::domain::index_box(domain);
    pfc::data::Field<double, pfc::HostSpace> u(domain, box, 3);
    const auto n = u.local_size();
    std::fill(u.data(), u.data() + u.size(), -999.);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) u(i, j, k) = 2.;
    const std::vector<double> pristine(u.data(), u.data() + u.size());
    auto walls = reflection_walls<double>(ReflectionParity::Odd,
                                          BoundaryCentering::CellFace, 3.);
    for (auto *s :
         {&walls.xm, &walls.xp, &walls.ym, &walls.yp, &walls.zm, &walls.zp})
      s->extension = BoundaryExtension::Linear;
    apply_reflection_ghosts(u, walls);
    for (int k = -3; k < n[2] + 3; ++k)
      for (int j = -3; j < n[1] + 3; ++j)
        for (int i = -3; i < n[0] + 3; ++i) {
          const int g[] = {i, j, k};
          double a[3], b[3];
          for (int axis = 0; axis < 3; ++axis) {
            const int layer =
                std::abs(g[axis] - std::clamp(g[axis], 0, n[axis] - 1));
            a[axis] = 1 - 2 * layer;
            b[axis] = 6 * layer;
          }
          const double expected =
              2 * a[0] * a[1] * a[2] + b[0] * a[1] * a[2] + b[1] * a[2] + b[2];
          if (u(i, j, k) != expected)
            throw std::runtime_error("host oracle mismatch");
        }
    const std::vector<double> accepted(u.data(), u.data() + u.size());
#if defined(OpenPFC_ENABLE_CUDA)
    if (argc > 1) {
      pfc::data::Field<double, pfc::CUDASpace> device(domain, box, 3);
      device.with_host_view([&](double *dst, std::size_t) {
        std::copy(pristine.begin(), pristine.end(), dst);
      });
      apply_reflection_ghosts(device, walls);
      if (!device.residency().device_valid() || device.residency().host_valid())
        throw std::runtime_error("device result is not resident");
      auto require_accepted = [&] {
        device.with_host_read([&](const double *src, std::size_t count) {
          if (std::vector<double>(src, src + count) != accepted)
            throw std::runtime_error("device oracle or rejection mismatch");
        });
      };
      require_accepted();
      walls.xm.centering = BoundaryCentering::Node;
      bool rejected = false;
      try {
        apply_reflection_ghosts(device, walls);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("invalid extension accepted");
      require_accepted();
    }
#else
    if (argc > 1) throw std::runtime_error("CUDA consumer unavailable");
#endif
    walls.xm.centering = BoundaryCentering::Node;
    bool rejected = false;
    try {
      apply_reflection_ghosts(u, walls);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected || std::vector<double>(u.data(), u.data() + u.size()) != accepted)
      throw std::runtime_error("host rejection mutated accepted field");
    std::cout << "installed linear boundary oracle and rejection PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
