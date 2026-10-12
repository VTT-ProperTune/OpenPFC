// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <chrono>
#include <mpi.h>
#include <openpfc/solvers/multigrid.hpp>
#include <optional>
namespace pfc::solvers {
/** Small replicated coarse correction for a uniquely owned structured operator.
 * Restriction is the owned rows of P^T and prolongation is those rows of P.
 * The coarse unknown is complete and identical on every rank. It may be
 * replicated only while its dimension stays explicitly small. Setup may build
 * the fine auxiliary; Krylov application does not gather the fine residual
 * and does not keep a host copy of the coarse unknown. The one-device
 * hierarchy and the replicated full-grid adapter stay unchanged.
 */
struct CoarseControls {
  int coarse_nodes = 5;
  CoarseOperator coarse_operator = CoarseOperator::Galerkin;
  int maximum_coarse_dofs = 2048;
};
class SharedCoarseCorrection {
public:
  using Controls = CoarseControls;
  double communication_seconds = 0;
  std::size_t collective_bytes = 0;
  int applications = 0;

  SharedCoarseCorrection(StructuredGrid fine, std::vector<int> owned,
                         Multigrid::Factory factory, MPI_Comm comm,
                         Controls controls = Controls())
      : fine_(fine), owned_(std::move(owned)), comm_(comm), controls_(controls) {
    fine_.validate();
    if (controls_.coarse_nodes < 3 || controls_.maximum_coarse_dofs < 1)
      throw std::invalid_argument("invalid coarse correction controls");
    validate_ownership();
    coarse_ = fine_;
    while (*std::max_element(coarse_.shape.begin(), coarse_.shape.end()) >
           controls_.coarse_nodes)
      for (int d = 0; d < 3; ++d)
        coarse_.shape[d] = std::max(3, (coarse_.shape[d] + 1) / 2);
    if (coarse_.size() > controls_.maximum_coarse_dofs)
      throw std::invalid_argument(
          "coarse correction exceeds the explicit size bound");
    GridTransfer transfer(fine_, coarse_);
    rows_.resize(owned_.size());
    for (std::size_t i = 0; i < owned_.size(); ++i)
      rows_[i] = transfer.weights[owned_[i]];
    SparseOperator coarse_operator =
        controls_.coarse_operator == CoarseOperator::Galerkin
            ? transfer.galerkin(factory(fine_))
            : factory(coarse_);
    MultigridControls hierarchy;
    hierarchy.coarse_nodes =
        *std::max_element(coarse_.shape.begin(), coarse_.shape.end());
    hierarchy.coarse_operator = controls_.coarse_operator;
    coarse_solver_.emplace(
        coarse_,
        [coarse_operator](const StructuredGrid &) { return coarse_operator; },
        hierarchy);
    stored_operator_ = coarse_solver_->operator_at(0);
  }
  const StructuredGrid &fine() const { return fine_; }
  const StructuredGrid &coarse() const { return coarse_; }
  const std::vector<int> &owned() const { return owned_; }
  const std::vector<std::vector<std::pair<int, double>>> &rows() const {
    return rows_;
  }
  const SparseOperator &coarse_operator() const { return stored_operator_; }
  void restrict_owned(const std::vector<double> &r,
                      std::vector<double> &coarse) const {
    if (r.size() != owned_.size())
      throw std::invalid_argument("owned residual size mismatch");
    coarse.assign(coarse_.size(), 0);
    for (std::size_t i = 0; i < owned_.size(); ++i)
      for (auto [j, weight] : rows_[i]) coarse[j] += weight * r[i];
  }
  void prolong_owned(const std::vector<double> &coarse,
                     std::vector<double> &z) const {
    if (coarse.size() != static_cast<std::size_t>(coarse_.size()))
      throw std::invalid_argument("coarse vector size mismatch");
    z.assign(owned_.size(), 0);
    for (std::size_t i = 0; i < owned_.size(); ++i)
      for (auto [j, weight] : rows_[i]) z[i] += weight * coarse[j];
  }
  void gauge(std::vector<double> &z) const { remove_global_mean(z); }
  void operator()(const std::vector<double> &r, std::vector<double> &z, int = 0) {
    require_owned_size(r.size());
    std::vector<double> coarse_residual, reduced;
    restrict_owned(r, coarse_residual);
    auto start = std::chrono::steady_clock::now();
    if (comm_ != MPI_COMM_NULL) {
      reduced.resize(coarse_residual.size());
      if (MPI_Allreduce(coarse_residual.data(), reduced.data(),
                        static_cast<int>(coarse_residual.size()), MPI_DOUBLE,
                        MPI_SUM, comm_) != MPI_SUCCESS)
        throw std::runtime_error("coarse residual reduction failed");
      coarse_residual.swap(reduced);
      collective_bytes += 2 * coarse_residual.size() * sizeof(double);
    }
    communication_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::vector<double> coarse_update;
    (*coarse_solver_)(coarse_residual, coarse_update);
    prolong_owned(coarse_update, z);
    remove_global_mean(z);
    ++applications;
  }

private:
  StructuredGrid fine_, coarse_;
  std::vector<int> owned_;
  std::vector<std::vector<std::pair<int, double>>> rows_;
  SparseOperator stored_operator_;
  std::optional<Multigrid> coarse_solver_;
  MPI_Comm comm_;
  Controls controls_;
  void require_owned_size(std::size_t n) const {
    int valid = n == owned_.size() ? 1 : 0, all_valid = valid;
    if (comm_ != MPI_COMM_NULL) {
      if (MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, comm_) !=
          MPI_SUCCESS)
        throw std::runtime_error("coarse ownership check failed");
    }
    if (!all_valid) throw std::invalid_argument("owned residual size mismatch");
  }
  void validate_ownership() {
    for (int index : owned_)
      if (index < 0 || index >= fine_.size())
        throw std::invalid_argument("owned index outside the structured grid");
    std::vector<int> indices = owned_;
    if (comm_ == MPI_COMM_NULL) {
      if (owned_.empty())
        throw std::invalid_argument("coarse correction needs ownership");
    } else {
      int ranks = 0, local = static_cast<int>(owned_.size());
      MPI_Comm_size(comm_, &ranks);
      std::vector<int> counts(ranks), offsets(ranks);
      if (MPI_Allgather(&local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_) !=
          MPI_SUCCESS)
        throw std::runtime_error("ownership count reduction failed");
      int total = 0;
      for (int rank = 0; rank < ranks; ++rank) {
        if (counts[rank] < 1)
          throw std::invalid_argument("every rank needs a nonempty ownership set");
        offsets[rank] = total;
        total += counts[rank];
      }
      indices.resize(total);
      if (MPI_Allgatherv(owned_.data(), local, MPI_INT, indices.data(),
                         counts.data(), offsets.data(), MPI_INT,
                         comm_) != MPI_SUCCESS)
        throw std::runtime_error("ownership index reduction failed");
    }
    auto sorted = indices;
    std::sort(sorted.begin(), sorted.end());
    if (static_cast<int>(sorted.size()) != fine_.size())
      throw std::invalid_argument("coarse correction needs complete ownership");
    for (int i = 0; i < fine_.size(); ++i)
      if (sorted[i] != i)
        throw std::invalid_argument("coarse correction ownership is not unique");
  }
  void remove_global_mean(std::vector<double> &z) const {
    if (fine_.boundary != Boundary::Periodic) return;
    std::vector<double> local(fine_.components), global(fine_.components);
    for (std::size_t i = 0; i < owned_.size(); ++i)
      local[owned_[i] % fine_.components] += z[i];
    if (comm_ != MPI_COMM_NULL) {
      if (MPI_Allreduce(local.data(), global.data(), fine_.components, MPI_DOUBLE,
                        MPI_SUM, comm_) != MPI_SUCCESS)
        throw std::runtime_error("coarse gauge reduction failed");
    } else
      global = local;
    for (std::size_t i = 0; i < owned_.size(); ++i)
      z[i] -= global[owned_[i] % fine_.components] / fine_.nodes();
  }
};
/** Multiplicative two-level step: Jacobi on the caller action, then one
 * shared coarse correction of the smoothed residual. The action owns any
 * fine halo exchange. This function adds only the coarse collective.
 */
template <class Action>
void two_level(SharedCoarseCorrection &correction, Action &&action,
               const std::vector<double> &diagonal, const std::vector<double> &r,
               std::vector<double> &z, int sweeps = 3, double weight = .6) {
  if (diagonal.size() != r.size() || !(weight > 0 && weight < 1) || sweeps < 1)
    throw std::invalid_argument("invalid two-level smoother");
  z.assign(r.size(), 0);
  std::vector<double> product(r.size());
  auto smooth = [&] {
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      action(z, product);
      for (std::size_t i = 0; i < z.size(); ++i) {
        if (!(diagonal[i] > 0))
          throw std::invalid_argument(
              "two-level smoother needs a positive diagonal");
        z[i] += weight * (r[i] - product[i]) / diagonal[i];
      }
    }
  };
  smooth();
  action(z, product);
  for (std::size_t i = 0; i < product.size(); ++i) product[i] = r[i] - product[i];
  std::vector<double> coarse;
  correction(product, coarse);
  for (std::size_t i = 0; i < z.size(); ++i) z[i] += coarse[i];
  smooth();
  correction.gauge(z);
}
} // namespace pfc::solvers
