// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_device_operator.hpp
 * @brief Rank-owned CUDA/HIP Dirichlet Q1 action using OpenPFC FieldHalo.
 *
 * This is the first GPU-resident operator/halo qualification, not a new GMG
 * hierarchy. All decomposition, device storage, MPI transfers and Krylov
 * vector operations belong to existing OpenPFC infrastructure. The numerical
 * point kernel is shared with the CPU Dirichlet V-cycle.
 */

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/solvers/field_operations.hpp>
#include <openpfc/solvers/nodal_dirichlet_kernels.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <mpi.h>

#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/nodal_multigrid_gpu.hpp>

namespace pfc::solvers {

namespace detail {

/// Reuse the existing nodal multigrid CUDA/HIP launcher, never a second
/// numerical kernel or a new GPU memory/transport implementation.
template <class Space> struct DirichletDevicePointLaunch;

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct DirichletDevicePointLaunch<CUDASpace> {
  static void owned_copy(const nmg::Level &g, const double *src, int src_halo,
                         double *dst, int dst_halo) {
    gpu::nmg::copy_owned_cuda(g, src, src_halo, dst, dst_halo);
  }
  static void action(const nmg::Level &g, const nmg::dirichlet::Weights &w,
                     const double *src, double *dst) {
    gpu::nmg::dirichlet_apply_cuda(g, w, src, dst);
  }
  static void restrict_pt(const nmg::Level &fine, const double *src,
                          const nmg::Level &coarse, double *dst) {
    gpu::nmg::dirichlet_restrict_cuda(fine, src, coarse, dst);
  }
  static void prolong_add(const nmg::Level &fine, double *dst,
                          const nmg::Level &coarse, const double *src) {
    gpu::nmg::dirichlet_prolong_cuda(fine, dst, coarse, src);
  }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct DirichletDevicePointLaunch<HIPSpace> {
  static void owned_copy(const nmg::Level &g, const double *src, int src_halo,
                         double *dst, int dst_halo) {
    gpu::nmg::copy_owned_hip(g, src, src_halo, dst, dst_halo);
  }
  static void action(const nmg::Level &g, const nmg::dirichlet::Weights &w,
                     const double *src, double *dst) {
    gpu::nmg::dirichlet_apply_hip(g, w, src, dst);
  }
  static void restrict_pt(const nmg::Level &fine, const double *src,
                          const nmg::Level &coarse, double *dst) {
    gpu::nmg::dirichlet_restrict_hip(fine, src, coarse, dst);
  }
  static void prolong_add(const nmg::Level &fine, double *dst,
                          const nmg::Level &coarse, const double *src) {
    gpu::nmg::dirichlet_prolong_hip(fine, dst, coarse, src);
  }
};
#endif

} // namespace detail

/**
 * @brief Device-resident, constant-coefficient nonperiodic Q1 row operator.
 *
 * The public fields contain **free nodes** only. The caller owns its
 * Decomposition, input and output. The reusable FieldHalo stages any
 * storage halo width (including zero) into a two-cell device scratch,
 * exchanges the complete 26-neighbour nonperiodic halo, then the existing
 * GPU kernel evaluates owned Q1 rows. No full fine-grid array is gathered
 * and no physical field is staged to the host during apply_operator().
 *
 * This operator is intentionally distinct from NodalDirichletMultigrid:
 * the full host V-cycle remains the reference while the device implementation
 * acquires a separately qualified hierarchy under issue #591. Do not use
 * this stage alone as a preconditioner.
 */
template <class MemorySpace> class NodalDirichletDeviceOperator {
public:
  using FieldT = data::Field<double, MemorySpace>;
  using Ops = FieldOperations<MemorySpace>;
  using Halo = FieldHalo<MemorySpace>;

  NodalDirichletDeviceOperator(const decomposition::Decomposition &decomp,
                               int rank, MPI_Comm comm,
                               double diffusion = 1.0, double reaction = 0.0)
      : m_domain(decomposition::domain(decomp)),
        m_decomp(decomp),
        m_box(validate_box(decomp, rank, comm)),
        m_ops(m_domain, m_box, 1, nmg::dirichlet::kHalo, comm),
        m_halo(m_ops, m_decomp, rank, comm, halo_options()),
        m_output(m_domain, m_box, nmg::dirichlet::kHalo) {
    const auto periodic = domain::get_periodic(m_domain);
    const auto global = domain::get_size(m_domain);
    const auto spacing = domain::get_spacing(m_domain);
    if (!(diffusion > 0.0) || !std::isfinite(diffusion) ||
        !(reaction >= 0.0) || !std::isfinite(reaction))
      throw std::invalid_argument(
          "NodalDirichletDeviceOperator: invalid material constants");
    double h[3];
    for (int axis = 0; axis < 3; ++axis) {
      if (periodic[axis] || global[axis] < 3 ||
          !(spacing[axis] > 0.0) || !std::isfinite(spacing[axis]))
        throw std::invalid_argument(
            "NodalDirichletDeviceOperator: expected nonperiodic free-node Q1 domain");
      m_shape.n[axis] = m_box.size[axis];
      m_shape.low[axis] = m_box.low[axis];
      m_shape.gn[axis] = global[axis];
      h[axis] = spacing[axis];
    }
    m_weights = nmg::dirichlet::make_weights(h, diffusion, reaction);
  }

  NodalDirichletDeviceOperator(const NodalDirichletDeviceOperator &) = delete;
  NodalDirichletDeviceOperator &operator=(const NodalDirichletDeviceOperator &) = delete;

  /// Input/output Field storage halos may be 0, 1, 2, or another nonnegative
  /// width. Their owned boxes must agree with the supplied Decomposition.
  /// The input and output may be the same object: owned data is staged first.
  void apply_operator(const FieldT &input, FieldT &output) {
    validate_field(input, "input");
    validate_field(output, "output");

    // FieldHost/View residency may mark a host mirror authoritative. Push it
    // once before reading the device pointer. No device-to-host copy occurs.
    auto &mutable_input = const_cast<FieldT &>(input);
    mutable_input.sync_to_device();

    auto &staged = m_halo.scratch()[0];
    detail::DirichletDevicePointLaunch<MemorySpace>::owned_copy(
        m_shape, input.data(), input.storage_halo(),
        staged.data(), nmg::dirichlet::kHalo);
    staged.note_device_write();
    m_halo.exchange();

    detail::DirichletDevicePointLaunch<MemorySpace>::action(
        m_shape, m_weights, staged.data(), m_output.data());
    m_output.note_device_write();
    detail::DirichletDevicePointLaunch<MemorySpace>::owned_copy(
        m_shape, m_output.data(), nmg::dirichlet::kHalo,
        output.data(), output.storage_halo());
    output.note_device_write();
  }

  [[nodiscard]] const Box3i &owned_box() const noexcept { return m_box; }
  [[nodiscard]] std::size_t local_unknowns() const noexcept {
    return static_cast<std::size_t>(m_box.size[0]) *
           m_box.size[1] * m_box.size[2];
  }

private:
  static comm::HaloExchangeOptions halo_options() {
    comm::HaloExchangeOptions opts;
    opts.connectivity = comm::HaloConnectivity::Full;
    opts.exchange_base = 6000;
    return opts;
  }

  static Box3i validate_box(const decomposition::Decomposition &decomp,
                             int rank, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument(
          "NodalDirichletDeviceOperator: null MPI communicator");
    int world_rank = -1, ranks = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &ranks);
    if (rank != world_rank ||
        decomposition::get_num_domains(decomp) != ranks)
      throw std::invalid_argument(
          "NodalDirichletDeviceOperator: rank/decomposition mismatch");
    // Every MPI rank must reject an unsupported partition before any rank
    // constructs a halo exchanger. Checking only this rank could leave the
    // others inside a collective and turn a simple layout error into a hang.
    for (int r = 0; r < ranks; ++r) {
      const Box3i candidate = decomposition::local_box(decomp, r);
      for (int axis = 0; axis < 3; ++axis)
        if (candidate.size[axis] < nmg::dirichlet::kHalo ||
            candidate.size[axis] > std::numeric_limits<int>::max() - 4)
          throw std::invalid_argument(
              "NodalDirichletDeviceOperator: owned box too thin/large for halo-2");
    }
    return decomposition::local_box(decomp, rank);
  }

  void validate_field(const FieldT &field, const char *role) const {
    if (field.domain() != m_domain ||
        field.box().low != m_box.low ||
        field.box().size != m_box.size ||
        field.storage_halo() < 0)
      throw std::invalid_argument(
          std::string("NodalDirichletDeviceOperator: ") +
          role + " field domain/box mismatch");
  }

  Domain m_domain;
  decomposition::Decomposition m_decomp;
  Box3i m_box;
  Ops m_ops;
  Halo m_halo;
  FieldT m_output;
  nmg::Level m_shape{};
  nmg::dirichlet::Weights m_weights{};
};

} // namespace pfc::solvers

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
