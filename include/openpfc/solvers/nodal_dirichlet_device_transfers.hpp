// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file nodal_dirichlet_device_transfers.hpp
 * @brief Two nested CUDA/HIP Q1 levels with OpenPFC-owned FieldHalo transfers.
 *
 * A small building block towards the device-resident nonperiodic V-cycle:
 * no new partitioner, custom MPI transport, coarse solve or solver hierarchy.
 * OpenPFC creates both decompositions and manages GPU buffers/halos; the
 * portable nmg::dirichlet::restrict_pt/prolong_p implement pure Q1 geometry.
 */
#include <openpfc/solvers/nodal_dirichlet_device_operator.hpp>
#include <openpfc/solvers/nodal_dirichlet_nested_geometry.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <memory>
#include <stdexcept>
#include <string>

namespace pfc::solvers {

/**
 * Rank-owned transfer adapter. Both supplied domains describe free Q1 nodes,
 * related by n_f = 2n_c+1 and h_c = 2h_f per axis. The caller is responsible
 * for creating the OpenPFC fine and coarse decompositions; this class checks
 * their precise Q1 transfer reach before binding HaloExchange.
 *
 * Only unique owned cells are copied into/out of caller Fields (which may have
 * storage halo width 0, 1 or 2). Internal halo-2 fields are persistent and
 * device-resident. The two calls are collective on the supplied communicator:
 * every MPI rank must participate in the same order.
 */
template <class MemorySpace> class NodalDirichletDeviceTransfers {
public:
  using FieldT = data::Field<double, MemorySpace>;
  using Ops = FieldOperations<MemorySpace>;
  using Halo = FieldHalo<MemorySpace>;
  using Point = detail::DirichletDevicePointLaunch<MemorySpace>;

  NodalDirichletDeviceTransfers(
      const decomposition::Decomposition &fine,
      const decomposition::Decomposition &coarse, int rank, MPI_Comm comm)
      : m_fine_domain(decomposition::domain(fine)),
        m_coarse_domain(decomposition::domain(coarse)),
        m_fine_decomp(fine), m_coarse_decomp(coarse),
        m_fine_box(validated_box(fine, coarse, rank, comm)),
        m_coarse_box(decomposition::local_box(coarse, rank)),
        m_fine_ops(m_fine_domain, m_fine_box, 1, nmg::dirichlet::kHalo, comm),
        m_coarse_ops(m_coarse_domain, m_coarse_box, 1,
                     nmg::dirichlet::kHalo, comm),
        m_fine_halo(m_fine_ops, m_fine_decomp, rank, comm, full_halo(8100)),
        m_coarse_halo(m_coarse_ops, m_coarse_decomp, rank, comm,
                      full_halo(8300)),
        m_fine_accumulator(m_fine_domain, m_fine_box, nmg::dirichlet::kHalo),
        m_coarse_restricted(m_coarse_domain, m_coarse_box,
                            nmg::dirichlet::kHalo) {
    init_shape(m_F, m_fine_domain, m_fine_box);
    init_shape(m_C, m_coarse_domain, m_coarse_box);
  }

  NodalDirichletDeviceTransfers(const NodalDirichletDeviceTransfers &) = delete;
  NodalDirichletDeviceTransfers &operator=(
      const NodalDirichletDeviceTransfers &) = delete;

  /// Unscaled R=P^T. The fine residual is halo-exchanged on GPU by OpenPFC;
  /// only coarse owned coefficients are written to caller-owned coarse_out.
  void restrict_pt(const FieldT &fine_residual, FieldT &coarse_out) {
    validate_field(fine_residual, m_fine_domain, m_fine_box, "fine residual");
    validate_field(coarse_out, m_coarse_domain, m_coarse_box, "coarse result");

    const_cast<FieldT &>(fine_residual).sync_to_device();
    auto &f = m_fine_halo.scratch()[0];
    Point::owned_copy(m_F, fine_residual.data(), fine_residual.storage_halo(),
                      f.data(), nmg::dirichlet::kHalo);
    f.note_device_write();
    m_fine_halo.exchange();

    Point::restrict_pt(m_F, f.data(), m_C, m_coarse_restricted.data());
    m_coarse_restricted.note_device_write();
    Point::owned_copy(m_C, m_coarse_restricted.data(), nmg::dirichlet::kHalo,
                      coarse_out.data(), coarse_out.storage_halo());
    coarse_out.note_device_write();
  }

  /// x_f += P x_c with a persistent GPU scratch, even when caller output has
  /// no halo padding. No global fine level is assembled or downloaded.
  void prolong_add(const FieldT &coarse_correction, FieldT &fine_accumulator) {
    validate_field(coarse_correction, m_coarse_domain, m_coarse_box,
                   "coarse correction");
    validate_field(fine_accumulator, m_fine_domain, m_fine_box,
                   "fine accumulator");

    const_cast<FieldT &>(coarse_correction).sync_to_device();
    auto &c = m_coarse_halo.scratch()[0];
    Point::owned_copy(m_C, coarse_correction.data(),
                      coarse_correction.storage_halo(),
                      c.data(), nmg::dirichlet::kHalo);
    c.note_device_write();
    m_coarse_halo.exchange();

    fine_accumulator.sync_to_device();
    Point::owned_copy(m_F, fine_accumulator.data(),
                      fine_accumulator.storage_halo(),
                      m_fine_accumulator.data(), nmg::dirichlet::kHalo);
    m_fine_accumulator.note_device_write();
    Point::prolong_add(m_F, m_fine_accumulator.data(), m_C, c.data());
    m_fine_accumulator.note_device_write();
    Point::owned_copy(m_F, m_fine_accumulator.data(), nmg::dirichlet::kHalo,
                      fine_accumulator.data(), fine_accumulator.storage_halo());
    fine_accumulator.note_device_write();
  }

  [[nodiscard]] const Box3i &fine_owned_box() const noexcept {
    return m_fine_box;
  }
  [[nodiscard]] const Box3i &coarse_owned_box() const noexcept {
    return m_coarse_box;
  }

private:
  static comm::HaloExchangeOptions full_halo(int base_tag) {
    comm::HaloExchangeOptions opt;
    opt.connectivity = comm::HaloConnectivity::Full;
    opt.exchange_base = base_tag;
    return opt;
  }

  static Box3i validated_box(const decomposition::Decomposition &fine,
                              const decomposition::Decomposition &coarse,
                              int rank, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL)
      throw std::invalid_argument(
          "NodalDirichletDeviceTransfers: null MPI communicator");
    int actual_rank = -1, nranks = 0;
    MPI_Comm_rank(comm, &actual_rank);
    MPI_Comm_size(comm, &nranks);
    if (rank != actual_rank ||
        decomposition::get_num_domains(fine) != nranks ||
        decomposition::get_num_domains(coarse) != nranks)
      throw std::invalid_argument(
          "NodalDirichletDeviceTransfers: decomposition/communicator mismatch");
    nmg::dirichlet::validate_nested_transfer_reach(fine, coarse);
    return decomposition::local_box(fine, rank);
  }

  static void init_shape(nmg::Level &g, const Domain &d, const Box3i &b) {
    const auto n = domain::get_size(d);
    for (int axis = 0; axis < 3; ++axis) {
      g.n[axis] = b.size[axis];
      g.low[axis] = b.low[axis];
      g.gn[axis] = n[axis];
    }
  }

  static void validate_field(const FieldT &f, const Domain &d,
                             const Box3i &b, const char *role) {
    if (f.domain() != d || f.box().low != b.low ||
        f.box().size != b.size || f.storage_halo() < 0)
      throw std::invalid_argument(
          std::string("NodalDirichletDeviceTransfers: invalid ") + role +
          " field layout");
  }

  Domain m_fine_domain, m_coarse_domain;
  decomposition::Decomposition m_fine_decomp, m_coarse_decomp;
  Box3i m_fine_box, m_coarse_box;
  Ops m_fine_ops, m_coarse_ops;
  Halo m_fine_halo, m_coarse_halo;
  FieldT m_fine_accumulator, m_coarse_restricted;
  nmg::Level m_F{}, m_C{};
};

} // namespace pfc::solvers

#endif // CUDA || HIP
