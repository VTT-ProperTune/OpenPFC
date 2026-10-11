// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <limits>
#include <openpfc/runtime/gpu/fcc_points.hpp>
#include <openpfc/solvers/finite_strain_fft/gpu/projector.hpp>
#include <openpfc/solvers/newton.hpp>
namespace pfc::finite_strain::gpu {
/// FCC projected-equilibrium consumer of generic Newton. The caller prescribes
/// the mean of F and supplies compatible initial fluctuations. Newton accepts
/// iterates only; material history commits explicitly after independent load
/// admission. Scalar failure checks do not transfer material fields.
template <class MemorySpace> class FccEquilibrium {
public:
  using Projector = CompatibilityProjector<MemorySpace>;
  using Tag = typename Projector::Tag;
  using Ops = solvers::gpu::VectorOperations<Tag>;
  using Vector = typename Ops::Vector;
  Projector &projector;
  Ops &ops;
  FccPoints<Tag> material;
  Vector frozen_tangent;
  bool residual_valid = false;
  FccEquilibrium(Projector &p, Ops &o, FccSlipParameters law)
      : projector(p), ops(o), material(p.count(), law),
        frozen_tangent(81 * p.count()) {}
  void residual(const Vector &f, Vector &r) {
    auto failure = material.evaluate(f);
    int failed = failure.status != FccStatus::Success;
    if (ops.communicator() != MPI_COMM_NULL) {
      int global = 0;
      if (MPI_Allreduce(&failed, &global, 1, MPI_INT, MPI_MAX, ops.communicator()) !=
          MPI_SUCCESS)
        throw std::runtime_error("material admission collective failed");
      failed = global;
    }
    residual_valid = !failed;
    if (!residual_valid) {
      // A successful local update cannot be committed when a peer failed
      // the same physical increment.
      material.reject_increment();
      ops.scale(r, 0);
      ops.scale(r, std::numeric_limits<double>::quiet_NaN());
      return;
    }
    projector.apply(material.piola, r);
  }
  /// Newton calls this at the last successful residual state. Reuse the tangent
  /// already returned by that material integration rather than integrating twice.
  void linearize(const Vector &) {
    if (!residual_valid) throw std::logic_error("no admitted residual to linearize");
    ops.copy(material.tangent, frozen_tangent);
  }
  void jacobian(const Vector &v, Vector &w) {
    projector.tangent(frozen_tangent, v, w);
  }
  void precondition(const Vector &r, Vector &z, int) { ops.copy(r, z); }
  void begin_trial(const Vector &) {}
  void accept_trial(const Vector &) {}
  void reject_trial() {
    residual_valid = false;
    material.reject_increment();
  }
};
} // namespace pfc::finite_strain::gpu
