// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/runtime/common/mpi_main.hpp>
#include <stdexcept>

// The ordinary installed host consumer must LINK the vendor launch symbols
// in CUDA/HIP package configurations, but it must never launch GPU kernels
// on GitHub's device-less compile-only runners.
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/runtime/gpu/nodal_multigrid_gpu.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_operator.hpp>
#include <openpfc/solvers/nodal_dirichlet_device_transfers.hpp>
#include <type_traits>

// Verify that the *installed* public CUDA/HIP SDK exposes the rank-owned
// FieldHalo adapter. sizeof instantiates its object layout, but never
// constructs a GPU field or launches a kernel on device-less CI runners.
#if defined(OpenPFC_ENABLE_CUDA)
static_assert(sizeof(pfc::solvers::NodalDirichletDeviceOperator<pfc::CUDASpace>) > 0);
static_assert(sizeof(pfc::solvers::NodalDirichletDeviceTransfers<pfc::CUDASpace>) > 0);
#elif defined(OpenPFC_ENABLE_HIP)
static_assert(sizeof(pfc::solvers::NodalDirichletDeviceOperator<pfc::HIPSpace>) > 0);
static_assert(sizeof(pfc::solvers::NodalDirichletDeviceTransfers<pfc::HIPSpace>) > 0);
#endif

template <typename Function>
void require_device_launcher_link(Function function) {
  // A volatile function pointer forces a link-time symbol reference. This
  // does not dereference device memory or require a GPU at runtime.
  Function volatile resolved = function;
  if (resolved == nullptr)
    throw std::runtime_error("OpenPFC installed GPU launcher missing");
}
#endif

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(argc, argv, [](int, char **, int rank, int ranks) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || ranks < 1 || rank < 0 || rank >= ranks)
      throw std::runtime_error("MPI startup did not initialize the body");
#if defined(OpenPFC_ENABLE_CUDA)
    using namespace pfc::solvers::gpu::nmg;
    require_device_launcher_link(&dirichlet_apply_cuda);
    require_device_launcher_link(&dirichlet_jacobi_cuda);
    require_device_launcher_link(&dirichlet_restrict_cuda);
    require_device_launcher_link(&dirichlet_prolong_cuda);
#elif defined(OpenPFC_ENABLE_HIP)
    using namespace pfc::solvers::gpu::nmg;
    require_device_launcher_link(&dirichlet_apply_hip);
    require_device_launcher_link(&dirichlet_jacobi_hip);
    require_device_launcher_link(&dirichlet_restrict_hip);
    require_device_launcher_link(&dirichlet_prolong_hip);
#endif
    std::cout << "Host startup consumer ranks=" << ranks << '\n';
    return EXIT_SUCCESS;
  });
}
