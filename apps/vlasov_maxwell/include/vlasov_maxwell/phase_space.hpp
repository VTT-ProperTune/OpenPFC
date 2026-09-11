// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file phase_space.hpp
 * @brief The `(x, v_x, v_y)` phase space as one OpenPFC 3-D `Domain`, its
 *        `v_y`-only MPI decomposition, and the per-species distributions.
 *
 * @details
 * ## The design claim this file exists to make
 *
 * A 1D2V kinetic phase space **is** a structured 3-D grid. Nothing in
 * `Domain`, `Box3i`, `Field` or the halo machinery knows or cares that axis 0
 * is a length and axes 1 and 2 are velocities. So the whole of `openpfc`'s
 * grid layer applies unmodified, and this header is thin on purpose: it
 * picks the axis order, picks the decomposition, and owns the one piece of
 * communication the scheme needs. If it were thick, the claim would be
 * false.
 *
 * Axis order is fixed, once, here:
 *
 *     axis 0 = x     periodic,     rank-local
 *     axis 1 = v_x   zero inflow,  rank-local
 *     axis 2 = v_y   zero inflow,  **the only distributed axis**
 *
 * ## Why `v_y` and only `v_y`
 *
 * The Strang split of the 1D2V Vlasov equation (issue #84, and the equation
 * block in @ref parameters.hpp) is three constant-coefficient translations,
 * each along a single axis:
 *
 *     A:  d_t f + v_x d_x f = 0                         along x
 *     B:  d_t f + (sigma/mu)(E_x + v_y B_z) d_vx f = 0   along v_x
 *     C:  d_t f + (sigma/mu)(E_y - v_x B_z) d_vy f = 0   along v_y
 *
 * Distributing the *third* axis and nothing else buys three things at once:
 *
 *  1. Step A is a 1-D FFT along a **rank-local** axis. No transpose, no
 *     global communication, no HeFFTe pencil reshuffle -- the single most
 *     expensive thing a spectral kinetic code normally does per step simply
 *     does not happen. This is the reason for the choice; everything else is
 *     a consequence.
 *  2. Step B is a purely local shift, because `v_x` is rank-local too. It
 *     needs no halo at all: a departure point outside `[-v_max, v_max]` is
 *     answered with a literal zero (zero inflow), not with a ghost cell.
 *  3. Only step C communicates, and only along one axis, so the exchange is
 *     a pair of contiguous slabs -- see @ref PhaseSpace::exchange_vy.
 *
 * The price, stated here rather than discovered during a scaling run: the
 * **rank count is capped at `N_{v_y}`**, and in practice at `N_{v_y}` divided
 * by the halo width, since a rank thinner than its own halo cannot be served
 * by a single-neighbour exchange. @ref PhaseSpace::max_ranks reports the cap
 * and the constructor enforces it.
 *
 * ## Why the halo is isotropic even though only one axis needs it
 *
 * `pfc::data::Field` carries one halo width for all three axes. We allocate
 * `hw` on every axis and exchange on exactly one. The waste is
 * `(1 + 2hw/N_x)(1 + 2hw/N_{v_x}) - 1` of the brick, which at the resolutions
 * this application targets (`1024 x 512 x 512`, `hw = 5`) is 3%. Building a
 * per-axis-padded field type to recover it would be a new abstraction used by
 * exactly one application, which the issue's scope limits forbid. The x and
 * `v_x` ghost rings are therefore allocated, never written and never read;
 * they are not a silent boundary condition, because no operator in
 * @ref advect.hpp ever indexes into them.
 *
 * ## Zero inflow is a fill, not a flag
 *
 * The velocity axes are physically truncated: `f` is *assumed* negligible at
 * `|v| = v_max` and the residual there is a reported diagnostic, not an
 * assumption the code is allowed to make quietly. The implementation of that
 * boundary condition is one line -- the outermost `v_y` ghost slab of a rank
 * that has no neighbour is memset to zero before every exchange -- and it
 * means the interpolation stencil in @ref advect.hpp can read straight
 * through the boundary with no special case. A stencil that reaches *past*
 * the ghost slab is a different matter entirely and is an error; see
 * @ref required_halo_width.
 *
 * @see parameters.hpp for the reduction, the normalisation and `SimParams`
 * @see advect.hpp for the three translations that consume this layout
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

#include <vlasov_maxwell/parameters.hpp>

namespace vlasov {

/// The distribution function's storage type. One of these per species.
using PhaseField = pfc::data::Field<double, pfc::HostSpace>;

/// Axis indices of the phase-space `Domain`. Fixed by this header; every
/// other file in the application reads them from here rather than writing
/// `2` and hoping.
inline constexpr int kAxisX = 0;
inline constexpr int kAxisVx = 1;
inline constexpr int kAxisVy = 2;

/**
 * @brief Half-width of the Lagrange stencil used by the semi-Lagrangian steps.
 *
 * `interp_order` is the number of interpolation *points*, which for a
 * Lagrange interpolant through `p` points is also its order of accuracy
 * (`O(h^p)`, because the interpolant is the unique degree-`p-1` polynomial
 * through them). The stencil starts at `k0 - (p-1)/2` and runs `p` points, so
 * the furthest offset from the departure cell `k0` is
 *
 *     below:  (p-1)/2        above:  p - 1 - (p-1)/2  ==  p/2
 *
 * with integer division throughout, and `p/2 >= (p-1)/2` for every `p`. So
 * `p/2` bounds both sides and is the number quoted as `stencil/2` in the
 * issue's halo-width formula.
 */
[[nodiscard]] inline constexpr int lagrange_half_width(int interp_order) noexcept {
  return interp_order / 2;
}

/// First stencil offset relative to the departure cell `floor(k - alpha)`.
/// Odd orders are centred (`p = 5` gives `-2 .. +2`), even orders are the
/// usual one-left-of-centre bias (`p = 4` gives `-1 .. +2`), which is what
/// @ref SimParams::interp_order documents.
[[nodiscard]] inline constexpr int lagrange_first_offset(int interp_order) noexcept {
  return -((interp_order - 1) / 2);
}

/**
 * @brief Ghost width a shift of `|alpha|` cells needs, for a given stencil.
 *
 * A semi-Lagrangian step reads `f` at `k - alpha` for every owned `k`, so the
 * departure cell is `floor(k - alpha)` and the stencil reaches
 * `lagrange_half_width` further in each direction. With `k` in `[0, N)` the
 * extreme indices touched are `-ceil(|alpha|) - (p-1)/2` and
 * `N - 1 + ceil(|alpha|) + p/2`, hence
 *
 *     hw_required = ceil(|alpha|) + p/2 .
 *
 * This is a CFL-like condition on the velocity step and it is **checked at
 * runtime, never assumed**: exceeding the allocated halo on a distributed
 * axis does not produce a large error, it produces a plausible-looking wrong
 * answer built out of a neighbour's ghost cells or out of another rank's data
 * wrapped around. @ref advect_vy throws instead.
 */
[[nodiscard]] inline int required_halo_width(double max_abs_shift_cells,
                                             int interp_order) {
  if (!(max_abs_shift_cells >= 0.0)) { // also catches NaN
    throw std::invalid_argument(
        "required_halo_width: shift must be a non-negative number of cells");
  }
  return static_cast<int>(std::ceil(max_abs_shift_cells)) +
         lagrange_half_width(interp_order);
}

/**
 * @brief The phase-space grid, its decomposition and the species fields.
 *
 * Owns: the global `Domain`, the `v_y`-only `Decomposition`, one padded
 * `PhaseField` per species, and the `v_y` halo exchange. Does **not** own the
 * time stepper, the Maxwell fields or any diagnostic -- those consume it.
 *
 * Coordinates are cell-centred and identical to the closed forms in
 * @ref SimParams (`x_of`, `vx_of`, `vy_of`), so `field.coords(i, j, k)` and
 * `params.x_of(global_i)` agree bit for bit. That is deliberate: the
 * analytic initial conditions and the field's own coordinate accessor must
 * not be two slightly different grids.
 */
class PhaseSpace {
public:
  /**
   * @brief Build the stack.
   *
   * @param params      Validated (or validatable) run parameters.
   * @param halo_width  Ghost width on every axis; only `v_y` is exchanged.
   *                    Size it with @ref required_halo_width from the largest
   *                    `|a_y| dt / dv_y` the run can produce.
   * @param comm        Communicator. Its size is the number of `v_y` slabs.
   *
   * @throws std::invalid_argument if the parameters are inconsistent, if
   *         `halo_width < 1`, or if the rank count exceeds the `N_{v_y}` cap.
   */
  PhaseSpace(const SimParams &params, int halo_width, MPI_Comm comm = MPI_COMM_WORLD)
      // `Decomposition` holds a const member and so cannot be assigned after
      // the fact; every geometry member is therefore built in the
      // initialiser list, in declaration order, and all validation happens
      // inside `checked_domain` which runs first.
      : m_params(params), m_halo(halo_width), m_comm(comm),
        m_rank(comm_rank_of(comm)), m_size(comm_size_of(comm)),
        m_domain(checked_domain(params, halo_width, comm_size_of(comm))),
        // The decomposition is stated, not searched for.
        // `decomposition::create(domain, nparts)` would minimise surface area
        // and split whichever axes it liked, which is exactly the wrong
        // answer here: a split of x would turn step A's rank-local FFT into a
        // global transposing one, and a split of v_x would do the same to
        // step B's local shift.
        m_decomp(pfc::decomposition::create(m_domain,
                                            pfc::Int3{1, 1, comm_size_of(comm)})),
        m_box(pfc::decomposition::local_box(m_decomp, comm_rank_of(comm))) {
    m_fields.reserve(m_params.species.size());
    for (std::size_t s = 0; s < m_params.species.size(); ++s) {
      m_fields.push_back(
          pfc::data::field_from_subdomain<double>(m_decomp, m_rank, m_halo));
    }

    // Slab geometry for the exchange. In the padded, x-fastest layout a
    // constant-k plane is `npx * npy` contiguous doubles, so a `hw`-thick
    // v_y slab is one contiguous block and needs no MPI datatype.
    m_plane = static_cast<std::size_t>(m_params.nx + 2 * m_halo) *
              static_cast<std::size_t>(m_params.nvx + 2 * m_halo);
    m_slab = m_plane * static_cast<std::size_t>(m_halo);
  }

  /// Largest usable rank count: every slab must be at least `halo_width`
  /// thick, or the halo of one rank would have to be served by two.
  [[nodiscard]] static int max_ranks(const SimParams &params,
                                     int halo_width) noexcept {
    if (halo_width < 1) return params.nvy;
    return params.nvy / halo_width;
  }

  // ---- geometry ---------------------------------------------------------

  [[nodiscard]] const SimParams &params() const noexcept { return m_params; }
  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }
  /// The owned index box of this rank, in global phase-space indices.
  [[nodiscard]] const pfc::Box3i &owned_box() const noexcept { return m_box; }
  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int comm_size() const noexcept { return m_size; }
  [[nodiscard]] MPI_Comm comm() const noexcept { return m_comm; }
  [[nodiscard]] int halo_width() const noexcept { return m_halo; }

  [[nodiscard]] int nx() const noexcept { return m_box.size[kAxisX]; }
  [[nodiscard]] int nvx() const noexcept { return m_box.size[kAxisVx]; }
  /// Owned `v_y` cells on this rank. The only extent that differs by rank.
  [[nodiscard]] int nvy_local() const noexcept { return m_box.size[kAxisVy]; }
  [[nodiscard]] int nvy_global() const noexcept { return m_params.nvy; }
  /// Global `v_y` index of local `k = 0`.
  [[nodiscard]] int vy_offset() const noexcept { return m_box.low[kAxisVy]; }

  [[nodiscard]] double dx() const noexcept { return m_params.dx(); }
  [[nodiscard]] double dvx() const noexcept { return m_params.dvx(); }
  [[nodiscard]] double dvy() const noexcept { return m_params.dvy(); }
  /// Phase-space cell volume; the weight that turns a sum into an integral.
  [[nodiscard]] double cell_volume() const noexcept { return dx() * dvx() * dvy(); }

  /// Physical `x` of local index `i` (`x` is never split, so local == global).
  [[nodiscard]] double x(int i) const noexcept { return m_params.x_of(i); }
  /// Physical `v_x` of local index `j` (never split).
  [[nodiscard]] double vx(int j) const noexcept { return m_params.vx_of(j); }
  /// Physical `v_y` of **local** index `k`, which is `k + vy_offset()` global.
  [[nodiscard]] double vy(int k) const noexcept {
    return m_params.vy_of(k + vy_offset());
  }

  /// True when this rank owns the `v_y = -v_max` end of the grid.
  [[nodiscard]] bool at_vy_low_boundary() const noexcept { return vy_offset() == 0; }
  /// True when this rank owns the `v_y = +v_max` end of the grid.
  [[nodiscard]] bool at_vy_high_boundary() const noexcept {
    return vy_offset() + nvy_local() == m_params.nvy;
  }

  // ---- species fields ---------------------------------------------------

  [[nodiscard]] std::size_t n_species() const noexcept { return m_fields.size(); }
  [[nodiscard]] const Species &species(std::size_t s) const {
    return m_params.species.at(s);
  }
  [[nodiscard]] PhaseField &f(std::size_t s) { return m_fields.at(s); }
  [[nodiscard]] const PhaseField &f(std::size_t s) const { return m_fields.at(s); }

  /**
   * @brief Fill species `s` from an analytic `fn(x, vx, vy)`.
   *
   * The callable is handed **physical** coordinates, cell-centred, in the
   * normalisation of @ref parameters.hpp. Only owned cells are written; the
   * ghost ring stays zero, which is already the zero-inflow state.
   */
  template <typename Fn> void initialise(std::size_t s, Fn &&fn) {
    initialise_field(f(s), std::forward<Fn>(fn));
  }

  /// Same, for a field that is not one of the species (a scratch or a
  /// reference solution). Geometry comes from the field itself, so this is
  /// correct for any field built from this stack's decomposition.
  template <typename Fn> void initialise_field(PhaseField &field, Fn &&fn) const {
    const int lnx = field.local_size()[kAxisX];
    const int lnvx = field.local_size()[kAxisVx];
    const int lnvy = field.local_size()[kAxisVy];
    for (int k = 0; k < lnvy; ++k) {
      for (int j = 0; j < lnvx; ++j) {
        for (int i = 0; i < lnx; ++i) {
          field(i, j, k) = fn(x(i), vx(j), vy(k));
        }
      }
    }
  }

  /// Allocate another field on this stack's geometry (scratch, reference,
  /// a second copy for a splitting-order study).
  [[nodiscard]] PhaseField make_field() const {
    return pfc::data::field_from_subdomain<double>(m_decomp, m_rank, m_halo);
  }

  // ---- diagnostics ------------------------------------------------------

  /// Unweighted sum of the owned cells of `field` on this rank. The building
  /// block of every conservation diagnostic; kept separate from the volume
  /// weight and the reduction so a test can assert on the rank-local number.
  [[nodiscard]] static double local_sum(const PhaseField &field) {
    const int lnx = field.local_size()[kAxisX];
    const int lnvx = field.local_size()[kAxisVx];
    const int lnvy = field.local_size()[kAxisVy];
    double total = 0.0;
    for (int k = 0; k < lnvy; ++k) {
      for (int j = 0; j < lnvx; ++j) {
        for (int i = 0; i < lnx; ++i) {
          total += field(i, j, k);
        }
      }
    }
    return total;
  }

  /// `int f dx dv_x dv_y` over the whole phase space, reduced over ranks.
  [[nodiscard]] double total_mass(const PhaseField &field) const {
    double local = local_sum(field) * cell_volume();
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, m_comm);
    return global;
  }

  // ---- communication ----------------------------------------------------

  /**
   * @brief Fill the `v_y` ghost slabs of `field` from the neighbouring ranks,
   *        and with zeros where there is no neighbour.
   *
   * Two `MPI_Sendrecv` calls on contiguous blocks. There is deliberately no
   * `pfc::comm::HaloExchange` here, for two reasons:
   *
   *  - we want **one** axis, and the generic exchanger's non-periodic face
   *    handling returns a neighbour rank of `-1` rather than `MPI_PROC_NULL`,
   *    which is a valid no-op rank under MPICH but not under Open MPI. A
   *    kinetic boundary condition is not the place to rely on that;
   *  - a `v_y` slab is contiguous in the padded, x-fastest layout, so the
   *    whole exchange is two blocking calls on plain `MPI_DOUBLE` buffers and
   *    needs no derived datatype at all.
   *
   * The zero fill at the ends is the zero-inflow boundary condition. It is
   * repeated on every call rather than done once at construction, because
   * "nobody ever writes there" is exactly the kind of invariant that a later
   * GPU port or an in-place operator quietly breaks.
   */
  void exchange_vy(PhaseField &field) const {
    if (field.storage_halo() != m_halo) {
      throw std::invalid_argument("PhaseSpace::exchange_vy: field halo " +
                                  std::to_string(field.storage_halo()) +
                                  " does not match the stack's " +
                                  std::to_string(m_halo));
    }
    double *buf = field.data();
    const std::size_t nz = static_cast<std::size_t>(nvy_local());

    // Padded k index of owned k is k + hw, so:
    //   lower ghost slab (owned k = -hw..-1)  -> padded 0 .. hw-1
    //   first owned slab (owned k = 0..hw-1)  -> padded hw .. 2hw-1
    //   last owned slab  (owned k = nz-hw..)  -> padded nz .. nz+hw-1
    //   upper ghost slab (owned k = nz..)     -> padded nz+hw .. nz+2hw-1
    double *ghost_lo = buf;
    double *send_lo = buf + static_cast<std::size_t>(m_halo) * m_plane;
    double *send_hi = buf + nz * m_plane;
    double *ghost_hi = buf + (nz + static_cast<std::size_t>(m_halo)) * m_plane;

    const int below = (m_rank > 0) ? m_rank - 1 : MPI_PROC_NULL;
    const int above = (m_rank + 1 < m_size) ? m_rank + 1 : MPI_PROC_NULL;

    // Zero inflow: a face with no neighbour sees vacuum. Done before the
    // exchange so that a partially connected rank still gets both halves
    // right.
    if (below == MPI_PROC_NULL) {
      for (std::size_t n = 0; n < m_slab; ++n) ghost_lo[n] = 0.0;
    }
    if (above == MPI_PROC_NULL) {
      for (std::size_t n = 0; n < m_slab; ++n) ghost_hi[n] = 0.0;
    }
    if (m_size == 1) return;

    const int count = static_cast<int>(m_slab);
    // Up: my top owned slab becomes my upper neighbour's lower ghost.
    MPI_Sendrecv(send_hi, count, MPI_DOUBLE, above, kTagUp, ghost_lo, count,
                 MPI_DOUBLE, below, kTagUp, m_comm, MPI_STATUS_IGNORE);
    // Down: my bottom owned slab becomes my lower neighbour's upper ghost.
    MPI_Sendrecv(send_lo, count, MPI_DOUBLE, below, kTagDown, ghost_hi, count,
                 MPI_DOUBLE, above, kTagDown, m_comm, MPI_STATUS_IGNORE);
  }

  /// Exchange every species field. Convenience for the stepper.
  void exchange_vy_all() {
    for (auto &field : m_fields) exchange_vy(field);
  }

private:
  static constexpr int kTagUp = 0x5601;
  static constexpr int kTagDown = 0x5602;

  static int comm_rank_of(MPI_Comm comm) {
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
  }
  static int comm_size_of(MPI_Comm comm) {
    int n = 1;
    MPI_Comm_size(comm, &n);
    return n;
  }

  /// Validate everything, then build the global `Domain`. Called from the
  /// initialiser list so that no half-built stack can exist.
  static pfc::Domain checked_domain(const SimParams &p, int halo_width, int nranks) {
    p.validate();
    if (halo_width < 1) {
      throw std::invalid_argument(
          "PhaseSpace: halo_width must be at least 1; the v_y exchange and "
          "the zero-inflow ghost fill both need a ring to write into");
    }
    if (nranks > max_ranks(p, halo_width)) {
      throw std::invalid_argument(
          "PhaseSpace: " + std::to_string(nranks) +
          " ranks exceeds the v_y decomposition cap of " +
          std::to_string(max_ranks(p, halo_width)) + " for nvy=" +
          std::to_string(p.nvy) + " with halo " + std::to_string(halo_width) +
          ". The cap is N_vy/halo, not N_vy, because a slab thinner than its "
          "own halo cannot be filled from one neighbour on each side. Add a "
          "second decomposition axis (and a transpose) only if this is "
          "genuinely the limit you have hit.");
    }
    // Cell-centred coordinates: the origin sits at the centre of cell 0, so
    // `origin + i*spacing` reproduces SimParams::x_of / vx_of / vy_of exactly.
    return pfc::domain::create(
        pfc::GridSize({p.nx, p.nvx, p.nvy}),
        pfc::PhysicalOrigin(
            {0.5 * p.dx(), -p.v_max + 0.5 * p.dvx(), -p.v_max + 0.5 * p.dvy()}),
        pfc::GridSpacing({p.dx(), p.dvx(), p.dvy()}),
        // x is physically periodic; the velocity axes are truncated. These
        // flags are consumed by the neighbour arithmetic, so getting them
        // wrong would silently wrap v_y onto itself -- a plasma whose
        // fastest particles reappear as its slowest.
        pfc::Bool3{true, false, false});
  }

  SimParams m_params;
  int m_halo{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
  int m_rank{0};
  int m_size{1};
  pfc::Domain m_domain{};
  pfc::decomposition::Decomposition m_decomp;
  pfc::Box3i m_box{};
  std::vector<PhaseField> m_fields{};
  std::size_t m_plane{0}; ///< doubles in one padded constant-`v_y` plane
  std::size_t m_slab{0};  ///< doubles in one `hw`-thick `v_y` slab
};

} // namespace vlasov
