// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_preparation.hpp
 * @brief CPU/MPI stage-preparation protocol over existing padded halo exchangers.
 *
 * @details
 * `PaddedHaloExchanger` (and siblings) are **transport**: they move ghost
 * faces given a bound brick. `StagePreparationService` is the **protocol**
 * that interprets stage requirements (`needs_halo_exchange`,
 * `needs_boundary_update`, region kind, boundary/halo ordering) and drives
 * those exchangers plus an injectable boundary hook — without inventing a
 * new MPI transport and without embedding ad-hoc MPI in method/operator
 * layers.
 *
 * Drivers typically obtain requirements from
 * `pfc::integrator::requirements_from(StageContext)` (defined in
 * `stage_context.hpp`, which includes this header). This header does **not**
 * include integrator types, so decomposition stays free of an
 * integrator dependency edge.
 *
 * @note `prepare` is a **pre-evaluation** protocol. It does not apply
 *       post-evaluation boundary conditions after new owned values are
 *       written; that remains a separate driver responsibility outside
 *       `prepare`.
 * @note Rejection / retry: there is no halo-buffer rollback API. After a
 *       rejected attempt, re-prepare from the latest accepted owned core;
 *       ghost rings are recomputed from owned state.
 *
 * @see padded_halo_exchange.hpp
 * @see openpfc/kernel/integrator/stage_context.hpp
 */

#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>

namespace pfc::communication {

/**
 * @brief Field region kind for stage preparation (mirrors integrator values).
 *
 * Independent of `pfc::integrator::StageContext::RegionKind` so this header
 * does not include integrator types. Enumerators and numeric values must stay
 * aligned with the integrator enum for `requirements_from` mapping.
 *
 * When `needs_halo_exchange` is true, padded face exchange refreshes the full
 * 6-face ghost ring regardless of `region_kind`; the kind is recorded for
 * callers and documentation.
 */
enum class RegionKind {
  Interior, ///< Interior cells only
  Boundary, ///< Boundary cells only
  All       ///< All cells (interior + boundary)
};

/**
 * @brief Ordering of injectable boundary preparation vs halo exchange.
 *
 * Default when both flags are true: `BoundaryThenHalo`, so BC writes to owned
 * faces are visible to neighbors after exchange.
 */
enum class BoundaryHaloOrder {
  BoundaryThenHalo, ///< Run boundary hooks, then exchange (default)
  HaloThenBoundary  ///< Exchange first, then boundary hooks
};

/**
 * @brief Stage preparation requirements (StageContext-compatible flags).
 *
 * Compatible with `pfc::integrator::StageContext` via
 * `pfc::integrator::requirements_from`. Does not carry time / stage index;
 * those remain on the descriptor struct for method coordination.
 */
struct StagePreparationRequirements {
  RegionKind region_kind = RegionKind::All;
  bool needs_halo_exchange = false;
  bool needs_boundary_update = false;
  BoundaryHaloOrder ordering = BoundaryHaloOrder::BoundaryThenHalo;
};

/**
 * @brief Blocking CPU/MPI stage preparation over named `PaddedHaloExchanger`s.
 *
 * Bind brick-backed exchangers by name, optionally set a boundary hook, then
 * call `prepare` with requirements and the field name list for this stage.
 * Phase ordering is **global** across the field list (all BC hooks then all
 * exchanges, or reverse) — not interleaved per field.
 *
 * @tparam T Element type of the bound exchangers (default `double`).
 */
template <typename T = double> class StagePreparationService {
public:
  /**
   * @brief Bind a name to a non-owning exchanger pointer.
   *
   * Stores an owned `std::string` key. The exchanger must outlive the service
   * and should use the `PaddedBrick`-binding constructor so
   * `pfc::communication::exchange` is valid.
   *
   * @param name      Field / binding name (owned copy stored).
   * @param exchanger Exchanger to drive when this name is prepared.
   */
  void bind(std::string_view name, PaddedHaloExchanger<T> &exchanger) {
    m_exchangers.insert_or_assign(std::string(name), &exchanger);
  }

  /**
   * @brief Set or clear the injectable boundary-prepare hook.
   *
   * Invoked once per requested field name when `needs_boundary_update` is
   * true. If unset, boundary update is a documented no-op.
   *
   * @param hook Callable receiving the field name, or empty to clear.
   */
  void set_boundary_hook(std::function<void(std::string_view field_name)> hook) {
    m_boundary_hook = std::move(hook);
  }

  /**
   * @brief Run blocking pre-evaluation stage preparation for @p fields.
   *
   * @param req    Halo / boundary / ordering requirements.
   * @param fields Names previously passed to `bind`.
   *
   * @throws std::invalid_argument if any name in @p fields is unbound.
   *
   * @note When `needs_halo_exchange` is false, exchangers are not called.
   * @note When `needs_boundary_update` is true and no hook is set, BC is a
   *       no-op. Post-evaluation BC enforcement is outside this call.
   * @note Reject/retry: poison or discard ghosts and call `prepare` again from
   *       the accepted owned core; no halo rollback API is provided.
   */
  void prepare(const StagePreparationRequirements &req,
               std::span<const std::string_view> fields) {
    for (const std::string_view name : fields) {
      if (!m_exchangers.contains(std::string(name))) {
        throw std::invalid_argument(
            std::string("StagePreparationService::prepare: unbound field \"") +
            std::string(name) + "\"");
      }
    }

    const bool run_boundary = req.needs_boundary_update;
    const bool run_halo = req.needs_halo_exchange;

    if (!run_boundary && !run_halo) {
      return;
    }

    if (req.ordering == BoundaryHaloOrder::BoundaryThenHalo) {
      if (run_boundary) {
        run_boundary_hooks_(fields);
      }
      if (run_halo) {
        run_exchanges_(fields);
      }
    } else {
      if (run_halo) {
        run_exchanges_(fields);
      }
      if (run_boundary) {
        run_boundary_hooks_(fields);
      }
    }

    (void)req.region_kind; // recorded for callers/docs; full face ring on halo
  }

private:
  void run_boundary_hooks_(std::span<const std::string_view> fields) {
    if (!m_boundary_hook) {
      return;
    }
    for (const std::string_view name : fields) {
      m_boundary_hook(name);
    }
  }

  void run_exchanges_(std::span<const std::string_view> fields) {
    for (const std::string_view name : fields) {
      PaddedHaloExchanger<T> *ex = m_exchangers.at(std::string(name));
      exchange(*ex);
    }
  }

  std::unordered_map<std::string, PaddedHaloExchanger<T> *> m_exchangers;
  std::function<void(std::string_view)> m_boundary_hook;
};

} // namespace pfc::communication
