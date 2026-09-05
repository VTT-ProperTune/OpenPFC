// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Compile-only CUDA Field residency API check. Built as an OBJECT library
// (cmake/LibraryConfiguration.cmake); never linked into openpfc-tests. A
// namespace-scope constructor here used to run CUDA at process start, which
// broke Catch2/CTest discovery on GPU-less runners.

#if defined(OpenPFC_ENABLE_CUDA)

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/runtime/cuda/databuffer_cuda.hpp>
#include <openpfc/runtime/cuda/memory_space_cuda.hpp>

namespace {

// Uncalled on purpose: the body must compile; it must not run at load time.
[[maybe_unused]] void compile_cuda_field_residency_api() {
  using namespace pfc;

  auto domain = domain::create(GridSize({10, 1, 1}), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
  auto local_box = Box3i::from_bounds({0, 0, 0}, {9, 0, 0});
  data::Field<double, pfc::CUDASpace> field(domain, local_box, 0);

  field.sync_to_device();
  field.note_device_write();
  const data::Residency &residency = field.residency();
  volatile auto r = &residency;
  (void)r;
}

} // namespace

#endif // OpenPFC_ENABLE_CUDA
