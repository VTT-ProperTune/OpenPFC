// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file device_step_hip.hpp
 * @brief The HIP half of the 1D2V Vlasov-Maxwell step: the two
 *        semi-Lagrangian velocity gathers and the velocity-space moment
 *        reduction, plus the host glue that drives them.
 *
 * @details
 * ## What is on the device and why exactly that
 *
 * The Strang step of `step.hpp` has five phases. Three of them are
 * `O(N_x N_vx N_vy)` -- the size of the distribution itself -- and two are
 * `O(N_x)`:
 *
 *  | phase                         | work                  | where it runs |
 *  |-------------------------------|-----------------------|---------------|
 *  | A  `advect_x`, spectral shift | `N_x log N_x` per line, `N_vx N_vy` lines | **host** |
 *  | B  `advect_vx`, gather        | `p` FMA per cell      | **device**    |
 *  | C  `advect_vy`, gather        | `p` FMA per cell      | **device**    |
 *  | D  moments, `rho`/`J`         | one pass over the brick | **device**  |
 *  | E  the 1-D Maxwell solve      | `N_x` (kilobytes)     | **host**     |
 *
 * Phase E is not a judgement call: the transverse pair, Gauss and Ampere all
 * live on a 1-D line of `N_x` doubles. At the resolution this application is
 * meant to reach that is 8 kB against a 2 GB brick, so a kernel launch costs
 * more than the arithmetic. It stays on the host, and so -- for the same
 * reason -- does the `O(N_x N_v)` Lagrange coefficient setup the gathers
 * need, so the fields themselves never have to cross the bus at all.
 *
 * Phases B, C and D are the port, and they are the port because they are
 * *embarrassingly parallel over the phase space and bandwidth-bound*: B and
 * C each read `p` cells and write one, D reads every cell once. There is no
 * dependence between output cells and no communication except, for C, the
 * `v_y` halo. An MI250X GCD moves about 1.3 TB/s against a LUMI-C socket's
 * ~200 GB/s, and that ratio -- not a flop count -- is the entire argument.
 *
 * Phase A is the honest problem, and `vlasov_hip_cost.cpp` is where it is
 * measured rather than argued about. It is left on the host in this header,
 * which means every step pays two device-to-host and two host-to-device
 * copies of the whole brick. Whether that is a sensible trade or a reason
 * not to port anything is a *measurement*, and the measurement and its
 * consequence are written down in `docs/hpc/vlasov_gpu.md`. Nothing in this
 * header assumes the answer.
 *
 * ## The host code stays the reference
 *
 * Nothing here replaces anything. `advect.hpp`, `moments.hpp` and `step.hpp`
 * are untouched and remain the definition of what the application computes;
 * @ref vlasov::hip::DeviceStepper is an *alternative* driver of the same
 * `vlasov::Stepper` state that happens to run three phases somewhere else.
 * Both drivers can be instantiated in the same process against the same
 * initial condition, which is exactly what `vlasov_hip_parity.cpp` does.
 *
 * ## Why the interpolation coefficients are computed on the host
 *
 * A semi-Lagrangian shift needs, per advected line, an integer departure
 * cell and `p` Lagrange weights. That is `O(N_x N_vy)` numbers for step B
 * and `O(N_x N_vx)` for step C -- one *plane*, not a brick, so between 0.4%
 * and 4% of the data the gather itself touches. Computing them on the host
 * with the unmodified @ref vlasov::lagrange_weights and uploading them buys
 * two things worth more than that bandwidth:
 *
 *  1. the weights are then **bitwise** the host's weights, so any difference
 *     the parity driver sees is a difference in the *gather*, not in the
 *     coefficients, and
 *  2. the `v_y` halo guard -- which must throw, not clamp, and must throw on
 *     every rank before anybody communicates (@ref vlasov::advect_vy) --
 *     stays in one place, in host code, with its `MPI_Allreduce` intact.
 *
 * ## Bitwise parity of the gathers, and why the reduction cannot have it
 *
 * The gather kernels accumulate in the same order as the host loops: stencil
 * point `m = 0 .. p-1`, ascending, skipping the same out-of-range points.
 * Floating-point addition is deterministic given an order, the weights are
 * bitwise equal, and `fma` contraction is disabled for these kernels, so
 * steps B and C are expected to agree with the host **bitwise** and the
 * parity driver asserts that rather than a tolerance.
 *
 * The moment reduction cannot make that claim and should not pretend to. A
 * sum of `N_vx N_vy` terms in host order and the same sum in a tree order
 * differ by at most `(n_host + n_device) u * sum|f_i|` with `u = 2^-53`, and
 * by `O(sqrt(N) u)` in practice. That bound, not an observed number, is what
 * `vlasov_hip_parity.cpp` uses as its tolerance; see the driver for the
 * arithmetic. The entropy column additionally passes every cell through
 * `log`, where the device and host libm may differ by one ulp, which adds a
 * further `u` per term.
 *
 * ## Layout
 *
 * The brick is `pfc::data::Field`'s padded, x-fastest storage, copied to the
 * device verbatim: index `(i + hw) + npx (j + hw) + npx npy (k + hw)` with
 * `npx = N_x + 2hw`, `npy = N_vx + 2hw`. Keeping the padding on the device
 * costs `(1+2hw/N_x)(1+2hw/N_vx)-1` of the allocation (3% at the target
 * resolution) and buys a `hipMemcpy` of the whole buffer with no packing,
 * and a `v_y` ghost slab that is still one contiguous block -- which is what
 * makes the halo exchange two `MPI_Sendrecv` calls here exactly as it is on
 * the host.
 *
 * @see advect.hpp for the host reference of steps B and C
 * @see moments.hpp for the host reference of the reduction
 * @see step.hpp for the Strang composition this header drives
 * @see docs/hpc/vlasov_gpu.md for the measurement and what it decided
 */

#if !defined(OpenPFC_ENABLE_HIP)
#error "vlasov_maxwell/device_step_hip.hpp requires HIP (configure with -DOpenPFC_ENABLE_HIP=ON)"
#endif

#include <cstddef>

namespace vlasov::hip {

/**
 * @brief The padded brick's shape, as the kernels see it.
 *
 * `nvy` is this rank's **owned** `v_y` extent, not the global one: the
 * decomposition splits that axis and nothing else. `nvy_global` and
 * `vy_offset` are carried separately by the reduction, which needs global
 * `v_y` indices to know which cells sit on a velocity boundary face.
 */
struct DeviceGeometry {
  int nx{0};    ///< owned `x` cells (never split, so also global)
  int nvx{0};   ///< owned `v_x` cells (never split, so also global)
  int nvy{0};   ///< owned `v_y` cells on this rank
  int halo{0};  ///< ghost width on every axis, as allocated

  [[nodiscard]] constexpr int npx() const noexcept { return nx + 2 * halo; }
  [[nodiscard]] constexpr int npy() const noexcept { return nvx + 2 * halo; }
  [[nodiscard]] constexpr int npz() const noexcept { return nvy + 2 * halo; }

  /// Doubles in one padded constant-`v_y` plane; also the `v_y` stride.
  [[nodiscard]] constexpr std::size_t plane() const noexcept {
    return static_cast<std::size_t>(npx()) * static_cast<std::size_t>(npy());
  }
  /// Doubles in the whole padded allocation.
  [[nodiscard]] constexpr std::size_t padded_cells() const noexcept {
    return plane() * static_cast<std::size_t>(npz());
  }
  /// Doubles this rank actually owns.
  [[nodiscard]] constexpr std::size_t owned_cells() const noexcept {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(nvx) *
           static_cast<std::size_t>(nvy);
  }
  /// Offset of owned cell `(0,0,0)` inside the padded allocation.
  [[nodiscard]] constexpr std::size_t first_owned() const noexcept {
    const std::size_t h = static_cast<std::size_t>(halo);
    return h + static_cast<std::size_t>(npx()) * h + plane() * h;
  }
};

/**
 * @brief Quantities the moment kernel reduces, one profile of `N_x` each.
 *
 * The first four are the `x` profiles `moments.hpp` calls `n`, `flux_x`,
 * `flux_y` and `v2`, *unweighted* by the cell volume -- the host applies
 * `dv_x dv_y` afterwards, so that the device and host multiply by the same
 * constant at the same point and the comparison is not polluted by a
 * different scaling order.
 *
 * The next four are the phase-space scalars, also reduced per `x` so that
 * the final sum over `x` can be done on the host in index order and is
 * therefore reproducible from run to run. The last three are extrema, whose
 * value does not depend on the reduction order at all.
 */
enum : int {
  kMomN = 0,       ///< `sum f`
  kMomFluxX = 1,   ///< `sum v_x f`
  kMomFluxY = 2,   ///< `sum v_y f`
  kMomV2 = 3,      ///< `sum |v|^2 f`
  kMomShell = 4,   ///< `sum f` over the boundary shell
  kMomL1 = 5,      ///< `sum |f|`
  kMomL2Sq = 6,    ///< `sum f^2`
  kMomEntropy = 7, ///< `-sum f ln f` over `f > 0`
  kMomFMin = 8,    ///< `min f`
  kMomFMax = 9,    ///< `max f`
  kMomFFace = 10,  ///< `max |f|` on the velocity boundary faces
  kMomQuantities = 11
};

// ---------------------------------------------------------------------------
// Device memory and control. Thin wrappers so that the driver translation
// units never include <hip/hip_runtime.h> and can be compiled by the ordinary
// host compiler; only the .hip file is fed to hipcc.
// ---------------------------------------------------------------------------

/// Number of HIP devices visible to this process. Zero is not an error here.
[[nodiscard]] int device_count();
/// Bind this process to `local_rank % device_count()`. Throws if none.
void bind_local_device(int local_rank);
/// Name of the bound device, for the provenance line of a measurement.
[[nodiscard]] const char *device_name();

/// `hipMalloc`, in doubles. Throws `std::runtime_error` on failure.
[[nodiscard]] double *device_alloc(std::size_t n_doubles);
/// `hipMalloc`, in ints.
[[nodiscard]] int *device_alloc_int(std::size_t n_ints);
/// `hipFree`. Null is a no-op; errors are swallowed so this is safe in a
/// destructor.
void device_free(void *p) noexcept;

void device_upload(double *dst_dev, const double *src_host, std::size_t n);
void device_upload_int(int *dst_dev, const int *src_host, std::size_t n);
void device_download(double *dst_host, const double *src_dev, std::size_t n);
void device_zero(double *dst_dev, std::size_t n);
/// Block until every queued kernel and copy has retired. Every timing in
/// `vlasov_hip_cost.cpp` brackets its phase with this.
void device_synchronize();

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * @brief Step B on the device: `f_out(i, j, k) = sum_m w f_in(i, j+s_m, k)`.
 *
 * The exact operator of @ref vlasov::advect_vx, with the coefficients handed
 * in rather than recomputed. `base` and `wts` are indexed by
 * `line = k * N_x + i` -- the shift of step B depends on `(x, v_y)` and not
 * on `v_x`, which is the property that makes it a rigid translation of each
 * `v_x` line -- with `p` consecutive weights per line.
 *
 * Out-of-range stencil points contribute nothing, which is the zero-inflow
 * boundary condition on the rank-local `v_x` axis: no halo is read and none
 * is needed. Accumulation is in ascending `m`, skipping the same points the
 * host skips, so the result is bitwise the host's.
 *
 * @param f_dev   padded input brick; only owned cells are read
 * @param out_dev padded output brick; only owned cells are written, so the
 *                caller may ping-pong without touching the ghost ring
 * @param first   `lagrange_first_offset(p)`, passed rather than derived so
 *                that the stencil convention lives in exactly one place
 */
void advect_vx_gather_hip(const double *f_dev, double *out_dev,
                          const DeviceGeometry &g, const int *base_dev,
                          const double *wts_dev, int interp_order, int first);

/**
 * @brief Step C on the device: the same gather along the distributed axis.
 *
 * `base` and `wts` are indexed by `line = j * N_x + i`, because the shift of
 * step C depends on `(x, v_x)`. Unlike step B this reads the `v_y` ghost
 * slabs, so the caller must have exchanged them (@ref exchange_vy_device)
 * *and* must have checked that the shift plus the stencil fits inside the
 * allocated halo. The kernel does not check: the host check in
 * @ref vlasov::advect_vy is a collective and has to stay there, and a second
 * per-cell test would cost more than the arithmetic it guards.
 */
void advect_vy_gather_hip(const double *f_dev, double *out_dev,
                          const DeviceGeometry &g, const int *base_dev,
                          const double *wts_dev, int interp_order, int first);

/// Scratch, in doubles, that @ref moments_hip needs for this geometry.
[[nodiscard]] std::size_t moments_scratch_doubles(const DeviceGeometry &g);

/**
 * @brief Phase D on the device: every velocity-space reduction, in one pass.
 *
 * Reduces the eleven quantities of @ref kMomQuantities per `x` index, into
 * `out_dev` laid out as `out[q * N_x + i]`. The host completes the sums --
 * over `x` for the scalars, over ranks with `MPI_Allreduce` -- and applies
 * the quadrature weights, exactly as @ref vlasov::reduce_velocity does.
 *
 * One pass, because `f` is the largest object in the application and does
 * not fit in any cache: a second traversal is a second trip through HBM for
 * every byte of it.
 *
 * @param v_max, dvx, dvy   velocity geometry; the kernel reconstructs
 *                          `v = -v_max + (index + 1/2) dv` with the same
 *                          expression as @ref vlasov::SimParams::vx_of, so
 *                          the coordinates are bitwise the host's
 * @param vy_offset         global `v_y` index of local `k = 0`
 * @param nvy_global        global `v_y` extent, for the boundary-face test
 * @param edge              `v_max - v_thermal`; the shell test is
 *                          `|v| > edge`, empty when `v_thermal` is zero
 */
void moments_hip(const double *f_dev, const DeviceGeometry &g, double v_max,
                 double dvx, double dvy, int vy_offset, int nvy_global,
                 double edge, double *scratch_dev, double *out_dev);

} // namespace vlasov::hip

// ===========================================================================
// Host glue.
//
// Everything below is ordinary host C++ and pulls in the CPU application
// headers (MPI, FFTW through advect.hpp). The .hip translation unit defines
// VLASOV_HIP_KERNELS_ONLY before including this file so that hipcc is asked
// to compile the kernel declarations and nothing else -- it needs
// DeviceGeometry and the prototypes above, and none of the machinery below.
// ===========================================================================

#ifndef VLASOV_HIP_KERNELS_ONLY

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <vlasov_maxwell/advect.hpp>
#include <vlasov_maxwell/moments.hpp>
#include <vlasov_maxwell/parameters.hpp>
#include <vlasov_maxwell/phase_space.hpp>
#include <vlasov_maxwell/step.hpp>

namespace vlasov::hip {

/**
 * @brief Wall-clock seconds spent in each phase, and how many steps of each.
 *
 * Accumulated rather than averaged, so that a caller can divide by whatever
 * it considers a step. Every entry is measured with the device *drained*
 * (@ref device_synchronize) at both ends, because an asynchronous launch
 * that has not run yet is not a measurement of anything.
 */
struct PhaseTimings {
  double advect_x{0.0};   ///< host, spectral shift along x (both halves)
  double advect_vx{0.0};  ///< device gather, step B (both halves)
  double advect_vy{0.0};  ///< device gather, step C
  double moments{0.0};    ///< device reduction + host completion
  double fields{0.0};     ///< host, the 1-D Maxwell solve
  double coeffs{0.0};     ///< host weight setup + its upload
  double h2d{0.0};        ///< host-to-device brick copies
  double d2h{0.0};        ///< device-to-host brick copies
  double halo{0.0};       ///< v_y ghost exchange (device zero fill or MPI)
  int steps{0};

  [[nodiscard]] double total() const noexcept {
    return advect_x + advect_vx + advect_vy + moments + fields + coeffs + h2d +
           d2h + halo;
  }
  /// Round-trip share of the step: the number the port stands or falls on.
  [[nodiscard]] double transfer_fraction() const noexcept {
    const double t = total();
    return t > 0.0 ? (h2d + d2h) / t : 0.0;
  }
  void clear() { *this = PhaseTimings{}; }
};

/// Seconds since an arbitrary epoch, monotonic. One place, so every phase is
/// timed with the same clock.
[[nodiscard]] inline double wall_seconds() {
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1.0e-9 * static_cast<double>(ts.tv_nsec);
}

/**
 * @brief One species' distribution, resident on the device.
 *
 * Two padded bricks, ping-ponged. The gathers cannot be done in place -- the
 * departure stencil reads cells a same-brick write would already have
 * destroyed -- and the alternative to a second brick is a copy back, which
 * costs two more passes through HBM out of the `p + 1` the gather already
 * pays. Ping-ponging is safe here for a reason worth stating: the gather
 * writes only owned cells, and of the ghost ring only the `v_y` slabs are
 * ever *read*, and those are refilled by @ref exchange_vy_device immediately
 * before the one step that reads them. The `x` and `v_x` ghost rings are
 * allocated and never touched, exactly as on the host.
 *
 * Memory is therefore two padded bricks per species on the device, matching
 * the host's own high-water mark of two (`f` plus
 * @ref vlasov::TransportWorkspace::brick).
 */
class DeviceBrick {
public:
  DeviceBrick(const DeviceGeometry &g) : m_g(g) {
    m_a = device_alloc(g.padded_cells());
    try {
      m_b = device_alloc(g.padded_cells());
    } catch (...) {
      device_free(m_a);
      throw;
    }
    device_zero(m_a, g.padded_cells());
    device_zero(m_b, g.padded_cells());
  }
  ~DeviceBrick() {
    device_free(m_a);
    device_free(m_b);
  }
  DeviceBrick(const DeviceBrick &) = delete;
  DeviceBrick &operator=(const DeviceBrick &) = delete;
  DeviceBrick(DeviceBrick &&) = delete;
  DeviceBrick &operator=(DeviceBrick &&) = delete;

  [[nodiscard]] double *current() noexcept { return m_a; }
  [[nodiscard]] const double *current() const noexcept { return m_a; }
  [[nodiscard]] double *spare() noexcept { return m_b; }
  /// Make the gather's output the live brick.
  void swap() noexcept { std::swap(m_a, m_b); }
  [[nodiscard]] const DeviceGeometry &geometry() const noexcept { return m_g; }

  /// Copy the whole padded buffer up, ghosts and all. One `hipMemcpy`: the
  /// host and device layouts are identical by construction, so there is
  /// nothing to pack.
  void upload(const PhaseField &f) {
    require_same_size(f);
    device_upload(m_a, f.data(), m_g.padded_cells());
  }
  void download(PhaseField &f) const {
    require_same_size(f);
    device_download(f.data(), m_a, m_g.padded_cells());
    f.note_host_write();
  }

private:
  void require_same_size(const PhaseField &f) const {
    if (f.size() != m_g.padded_cells()) {
      throw std::runtime_error(
          "DeviceBrick: host field holds " + std::to_string(f.size()) +
          " doubles but the device geometry says " +
          std::to_string(m_g.padded_cells()) +
          ". The two layouts must agree exactly; a mismatch here would copy a "
          "brick into the wrong shape and still run.");
    }
  }

  DeviceGeometry m_g{};
  double *m_a{nullptr};
  double *m_b{nullptr};
};

/**
 * @brief The Lagrange coefficients of one velocity step, on both sides.
 *
 * Held as host vectors plus their device copies, resized once and refilled
 * every step. The host arrays are filled by the unmodified
 * @ref vlasov::lagrange_weights, so the device sees bitwise the same numbers
 * the host reference uses; see the file comment for why that matters more
 * than the bandwidth it costs.
 */
class CoefficientBuffer {
public:
  CoefficientBuffer() = default;
  ~CoefficientBuffer() {
    device_free(m_base_dev);
    device_free(m_wts_dev);
  }
  CoefficientBuffer(const CoefficientBuffer &) = delete;
  CoefficientBuffer &operator=(const CoefficientBuffer &) = delete;

  void resize(std::size_t n_lines, int p) {
    if (n_lines == m_lines && p == m_p) return;
    device_free(m_base_dev);
    device_free(m_wts_dev);
    m_base_dev = nullptr;
    m_wts_dev = nullptr;
    m_lines = n_lines;
    m_p = p;
    m_base.assign(n_lines, 0);
    m_wts.assign(n_lines * static_cast<std::size_t>(p), 0.0);
    m_base_dev = device_alloc_int(n_lines);
    m_wts_dev = device_alloc(n_lines * static_cast<std::size_t>(p));
  }

  [[nodiscard]] std::vector<int> &base() noexcept { return m_base; }
  [[nodiscard]] std::vector<double> &wts() noexcept { return m_wts; }
  [[nodiscard]] const int *base_dev() const noexcept { return m_base_dev; }
  [[nodiscard]] const double *wts_dev() const noexcept { return m_wts_dev; }
  /// Bytes moved by one refresh; reported so the reader can weigh it against
  /// the brick the gather reads.
  [[nodiscard]] std::size_t upload_bytes() const noexcept {
    return m_lines * (sizeof(int) + static_cast<std::size_t>(m_p) * sizeof(double));
  }

  void upload() {
    device_upload_int(m_base_dev, m_base.data(), m_lines);
    device_upload(m_wts_dev, m_wts.data(), m_wts.size());
  }

private:
  std::size_t m_lines{0};
  int m_p{0};
  std::vector<int> m_base{};
  std::vector<double> m_wts{};
  int *m_base_dev{nullptr};
  double *m_wts_dev{nullptr};
};

/**
 * @brief Fill the `v_y` ghost slabs of a device-resident brick.
 *
 * The host's @ref vlasov::PhaseSpace::exchange_vy, with the buffers on the
 * device. A slab is `hw` padded planes and is contiguous, so the exchange is
 * still two `MPI_Sendrecv` calls on plain `MPI_DOUBLE`; the only difference
 * is that the payload is staged through host memory rather than sent from
 * where it lives. Staging rather than handing MPI a device pointer is
 * deliberate: GPU-aware MPI is a property of the site's MPI build, not of
 * this application, and a halo exchange that silently memcpy's a device
 * pointer as host memory is a segfault on a good day and wrong numbers on a
 * bad one. The slab is `hw N_x N_vx` doubles -- kilobytes next to the brick
 * -- so the staging is not where the time goes; the measurement in
 * `vlasov_hip_cost.cpp` reports it separately so that claim is checkable.
 *
 * The zero fill at a face with no neighbour is the zero-inflow boundary
 * condition and is done on the device, before the exchange, exactly as the
 * host does it and for the same reason: "nobody ever writes there" is the
 * kind of invariant a ping-pong buffer quietly breaks.
 */
inline void exchange_vy_device(const PhaseSpace &ps, const DeviceGeometry &g,
                               double *brick_dev, std::vector<double> &stage_send,
                               std::vector<double> &stage_recv) {
  const std::size_t plane = g.plane();
  const std::size_t slab = plane * static_cast<std::size_t>(g.halo);
  const std::size_t nz = static_cast<std::size_t>(g.nvy);

  double *ghost_lo = brick_dev;
  double *send_lo = brick_dev + static_cast<std::size_t>(g.halo) * plane;
  double *send_hi = brick_dev + nz * plane;
  double *ghost_hi = brick_dev + (nz + static_cast<std::size_t>(g.halo)) * plane;

  const int rank = ps.rank();
  const int size = ps.comm_size();
  const int below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  const int above = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

  if (below == MPI_PROC_NULL) device_zero(ghost_lo, slab);
  if (above == MPI_PROC_NULL) device_zero(ghost_hi, slab);
  if (size == 1) {
    device_synchronize();
    return;
  }

  stage_send.resize(slab);
  stage_recv.resize(slab);
  const int count = static_cast<int>(slab);

  // Up: my top owned slab becomes my upper neighbour's lower ghost.
  device_download(stage_send.data(), send_hi, slab);
  MPI_Sendrecv(stage_send.data(), count, MPI_DOUBLE, above, 0x5611,
               stage_recv.data(), count, MPI_DOUBLE, below, 0x5611, ps.comm(),
               MPI_STATUS_IGNORE);
  if (below != MPI_PROC_NULL) device_upload(ghost_lo, stage_recv.data(), slab);

  // Down: my bottom owned slab becomes my lower neighbour's upper ghost.
  device_download(stage_send.data(), send_lo, slab);
  MPI_Sendrecv(stage_send.data(), count, MPI_DOUBLE, below, 0x5612,
               stage_recv.data(), count, MPI_DOUBLE, above, 0x5612, ps.comm(),
               MPI_STATUS_IGNORE);
  if (above != MPI_PROC_NULL) device_upload(ghost_hi, stage_recv.data(), slab);
  device_synchronize();
}

/**
 * @brief The same Strang step as @ref vlasov::Stepper, with phases B, C and
 *        D on the device.
 *
 * Holds a reference to a host @ref vlasov::Stepper and *drives its state*:
 * the fields, the sources, the moments and the ledger inputs all live in the
 * host object and are updated by the host code, so a device run and a host
 * run produce the same objects and can be compared field by field. What the
 * device owns is the distribution, between the two `advect_x` phases.
 *
 * The residency pattern of one step is
 *
 *     D2H ; advect_x(dt/2) ; H2D ; deposit ; fields(dt)
 *         ; advect_vx(dt/2) ; advect_vy(dt) ; advect_vx(dt/2)
 *         ; D2H ; advect_x(dt/2) ; H2D ; deposit
 *
 * -- four whole-brick copies per species per step, because phase A was not
 * ported. That is the cost the measurement has to justify, and it is
 * reported separately in @ref PhaseTimings so it can be.
 */
class DeviceStepper {
public:
  DeviceStepper(Stepper &st, PhaseSpace &ps)
      : m_st(&st), m_ps(&ps), m_p(&ps.params()) {
    m_g.nx = ps.nx();
    m_g.nvx = ps.nvx();
    m_g.nvy = ps.nvy_local();
    m_g.halo = ps.halo_width();
    m_bricks.reserve(ps.n_species());
    for (std::size_t s = 0; s < ps.n_species(); ++s) {
      m_bricks.push_back(std::make_unique<DeviceBrick>(m_g));
    }
    m_scratch = device_alloc(moments_scratch_doubles(m_g));
    m_out = device_alloc(static_cast<std::size_t>(kMomQuantities) *
                         static_cast<std::size_t>(m_g.nx));
    m_host_out.assign(static_cast<std::size_t>(kMomQuantities) *
                          static_cast<std::size_t>(m_g.nx),
                      0.0);
  }
  ~DeviceStepper() {
    device_free(m_scratch);
    device_free(m_out);
  }
  DeviceStepper(const DeviceStepper &) = delete;
  DeviceStepper &operator=(const DeviceStepper &) = delete;

  [[nodiscard]] const DeviceGeometry &geometry() const noexcept { return m_g; }
  [[nodiscard]] PhaseTimings &timings() noexcept { return m_t; }
  [[nodiscard]] const PhaseTimings &timings() const noexcept { return m_t; }
  /// Largest `v_y` halo width any step has needed. Same meaning, same
  /// reporting, as @ref vlasov::Stepper::peak_halo_used.
  [[nodiscard]] int peak_halo_used() const noexcept { return m_peak_halo; }

  /// Push every species' distribution to the device. Call once, after the
  /// initial condition, and again only if the host state was changed behind
  /// this object's back.
  void upload_all() {
    for (std::size_t s = 0; s < m_bricks.size(); ++s) {
      m_bricks[s]->upload(m_ps->f(s));
    }
    device_synchronize();
  }
  /// Pull every species back, so the host ledger, the field output and the
  /// parity comparison all see the device's answer.
  void download_all() {
    for (std::size_t s = 0; s < m_bricks.size(); ++s) {
      m_bricks[s]->download(m_ps->f(s));
    }
    device_synchronize();
  }

  /// Phase D for every species, followed by the host's own `add_species` and
  /// `apply_background`. The result lands in the host @ref vlasov::Stepper,
  /// so the field solve and the ledger are unchanged code.
  void deposit_all() {
    const double t0 = wall_seconds();
    ReductionOptions opt;
    opt.v_thermal = m_p->v_thermal;
    opt.comm = m_ps->comm();
    m_st->sources = Sources::zeros(m_p->nx);
    for (std::size_t s = 0; s < m_bricks.size(); ++s) {
      m_st->moments[s] = reduce_device(*m_bricks[s], opt);
      add_species(m_p->species[s], m_st->moments[s], m_st->sources);
    }
    apply_background(*m_p, m_st->sources);
    m_t.moments += wall_seconds() - t0;
  }

  /// One Strang step, phase for phase the composition of
  /// @ref vlasov::Stepper::advance.
  void advance(double dt) {
    const double h = 0.5 * dt;
    half_x_step(h);
    deposit_all();
    {
      const double t0 = wall_seconds();
      m_st->update_fields(dt);
      m_t.fields += wall_seconds() - t0;
    }
    for (std::size_t s = 0; s < m_bricks.size(); ++s) {
      const double qm = m_p->species[s].qm();
      const auto Bz = m_st->effective_bz();
      device_advect_vx(*m_bricks[s], qm, h, m_st->fields.Ex, Bz);
      device_advect_vy(*m_bricks[s], qm, dt, m_st->fields.Ey, Bz);
      device_advect_vx(*m_bricks[s], qm, h, m_st->fields.Ex, Bz);
    }
    half_x_step(h);
    deposit_all();
    ++m_t.steps;
  }

  // ---- the individual device phases, exposed for the cost driver --------

  /**
   * @brief Step B on the device, including its host coefficient setup.
   *
   * The coefficients are the same arithmetic as @ref vlasov::advect_vx in
   * the same order, written into a compact `p`-wide array instead of the
   * host's `kMaxInterpOrder`-wide one -- fewer bytes to upload, bitwise the
   * same values.
   */
  void device_advect_vx(DeviceBrick &brick, double qm, double dt,
                        std::span<const double> Ex, std::span<const double> Bz) {
    const int nx = m_g.nx;
    const int nvy = m_g.nvy;
    const int p = m_p->interp_order;
    detail::check_interp_order(p);
    detail::check_field_extent(Ex, nx, "E_x");
    detail::check_field_extent(Bz, nx, "B_z");

    double t0 = wall_seconds();
    m_cx.resize(static_cast<std::size_t>(nvy) * static_cast<std::size_t>(nx), p);
    const double inv_dvx = 1.0 / m_ps->dvx();
    auto &base = m_cx.base();
    auto &wts = m_cx.wts();
    for (int k = 0; k < nvy; ++k) {
      const double vy_k = m_ps->vy(k);
      for (int i = 0; i < nx; ++i) {
        const double ax = qm * (Ex[static_cast<std::size_t>(i)] +
                                vy_k * Bz[static_cast<std::size_t>(i)]);
        const double alpha = ax * dt * inv_dvx;
        const double departure = -alpha;
        const double cell = std::floor(departure);
        const std::size_t line =
            static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(i);
        base[line] = static_cast<int>(cell);
        lagrange_weights(p, departure - cell,
                         &wts[line * static_cast<std::size_t>(p)]);
      }
    }
    m_cx.upload();
    device_synchronize();
    m_t.coeffs += wall_seconds() - t0;

    t0 = wall_seconds();
    advect_vx_gather_hip(brick.current(), brick.spare(), m_g, m_cx.base_dev(),
                         m_cx.wts_dev(), p, lagrange_first_offset(p));
    device_synchronize();
    brick.swap();
    m_t.advect_vx += wall_seconds() - t0;
  }

  /**
   * @brief Step C on the device, with the host's collective halo guard.
   *
   * The guard is deliberately unchanged and deliberately still a collective:
   * the required width is reduced with `MPI_MAX` and compared *before* any
   * rank communicates, so an overrun throws everywhere instead of hanging on
   * one rank and producing a plausible answer on the others. Moving that
   * check into a kernel would turn it into a per-cell branch that cannot
   * throw and cannot be collective, which is a worse check in every respect.
   */
  void device_advect_vy(DeviceBrick &brick, double qm, double dt,
                        std::span<const double> Ey, std::span<const double> Bz) {
    const int nx = m_g.nx;
    const int nvx = m_g.nvx;
    const int p = m_p->interp_order;
    detail::check_interp_order(p);
    detail::check_field_extent(Ey, nx, "E_y");
    detail::check_field_extent(Bz, nx, "B_z");

    double t0 = wall_seconds();
    m_cy.resize(static_cast<std::size_t>(nvx) * static_cast<std::size_t>(nx), p);
    const double inv_dvy = 1.0 / m_ps->dvy();
    auto &base = m_cy.base();
    auto &wts = m_cy.wts();
    double local_max = 0.0;
    for (int j = 0; j < nvx; ++j) {
      const double vx_j = m_ps->vx(j);
      for (int i = 0; i < nx; ++i) {
        const double ay = qm * (Ey[static_cast<std::size_t>(i)] -
                                vx_j * Bz[static_cast<std::size_t>(i)]);
        const double alpha = ay * dt * inv_dvy;
        local_max = std::max(local_max, std::abs(alpha));
        const double departure = -alpha;
        const double cell = std::floor(departure);
        const std::size_t line =
            static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(i);
        base[line] = static_cast<int>(cell);
        lagrange_weights(p, departure - cell,
                         &wts[line * static_cast<std::size_t>(p)]);
      }
    }
    double global_max = local_max;
    if (m_ps->comm_size() > 1) {
      MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, m_ps->comm());
    }
    const int need = required_halo_width(global_max, p);
    m_peak_halo = std::max(m_peak_halo, need);
    if (need > m_ps->halo_width()) {
      throw std::runtime_error(
          "device_advect_vy: a shift of " + std::to_string(global_max) +
          " v_y cells with a " + std::to_string(p) +
          "-point stencil needs a halo of " + std::to_string(need) +
          " but only " + std::to_string(m_ps->halo_width()) +
          " is allocated. Same condition, same reason and same refusal to "
          "clamp as the host advect_vy.");
    }
    m_cy.upload();
    device_synchronize();
    m_t.coeffs += wall_seconds() - t0;

    t0 = wall_seconds();
    exchange_vy_device(*m_ps, m_g, brick.current(), m_stage_send, m_stage_recv);
    m_t.halo += wall_seconds() - t0;

    t0 = wall_seconds();
    advect_vy_gather_hip(brick.current(), brick.spare(), m_g, m_cy.base_dev(),
                         m_cy.wts_dev(), p, lagrange_first_offset(p));
    device_synchronize();
    brick.swap();
    m_t.advect_vy += wall_seconds() - t0;
  }

  /**
   * @brief Phase D for one brick: the device pass plus the host completion.
   *
   * The host half is deliberately the same arithmetic as
   * @ref vlasov::reduce_velocity after its own per-cell loop: sum the `x`
   * profiles into the scalars in index order, `MPI_Allreduce` the packed
   * buffer, then apply `dv` and `dV`. Only the per-cell loop moved.
   */
  [[nodiscard]] VelocityMoments reduce_device(DeviceBrick &brick,
                                              const ReductionOptions &opt) {
    const int nx = m_g.nx;
    const double edge = m_p->v_max - opt.v_thermal;
    moments_hip(brick.current(), m_g, m_p->v_max, m_ps->dvx(), m_ps->dvy(),
                m_ps->vy_offset(), m_p->nvy, edge, m_scratch, m_out);
    device_synchronize();
    device_download(m_host_out.data(), m_out, m_host_out.size());

    const std::size_t nxs = static_cast<std::size_t>(nx);
    constexpr int kScalars = 5;
    std::vector<double> acc(4 * nxs + kScalars, 0.0);
    double *an = acc.data();
    double *ax = an + nxs;
    double *ay = ax + nxs;
    double *ae = ay + nxs;
    double *sc = ae + nxs; // total, shell, l1, l2sq, entropy
    // The kernel seeds the extrema with +/-infinity for an empty slab, and
    // the host guards on that at the end exactly as reduce_velocity does.
    double fmin = std::numeric_limits<double>::infinity();
    double fmax = -std::numeric_limits<double>::infinity();
    double fface = 0.0;
    for (std::size_t i = 0; i < nxs; ++i) {
      an[i] = m_host_out[static_cast<std::size_t>(kMomN) * nxs + i];
      ax[i] = m_host_out[static_cast<std::size_t>(kMomFluxX) * nxs + i];
      ay[i] = m_host_out[static_cast<std::size_t>(kMomFluxY) * nxs + i];
      ae[i] = m_host_out[static_cast<std::size_t>(kMomV2) * nxs + i];
      sc[0] += an[i];
      sc[1] += m_host_out[static_cast<std::size_t>(kMomShell) * nxs + i];
      sc[2] += m_host_out[static_cast<std::size_t>(kMomL1) * nxs + i];
      sc[3] += m_host_out[static_cast<std::size_t>(kMomL2Sq) * nxs + i];
      sc[4] += m_host_out[static_cast<std::size_t>(kMomEntropy) * nxs + i];
      fmin = std::min(fmin, m_host_out[static_cast<std::size_t>(kMomFMin) * nxs + i]);
      fmax = std::max(fmax, m_host_out[static_cast<std::size_t>(kMomFMax) * nxs + i]);
      fface = std::max(fface, m_host_out[static_cast<std::size_t>(kMomFFace) * nxs + i]);
    }

    int inited = 0;
    MPI_Initialized(&inited);
    if (inited != 0 && opt.comm != MPI_COMM_NULL) {
      MPI_Allreduce(MPI_IN_PLACE, acc.data(), static_cast<int>(acc.size()),
                    MPI_DOUBLE, MPI_SUM, opt.comm);
      MPI_Allreduce(MPI_IN_PLACE, &fmin, 1, MPI_DOUBLE, MPI_MIN, opt.comm);
      std::array<double, 2> hi{fmax, fface};
      MPI_Allreduce(MPI_IN_PLACE, hi.data(), 2, MPI_DOUBLE, MPI_MAX, opt.comm);
      fmax = hi[0];
      fface = hi[1];
    }

    const double dv = m_p->dvx() * m_p->dvy();
    const double dV = m_p->dx() * dv;
    VelocityMoments m;
    m.nx = nx;
    m.n.resize(nxs);
    m.flux_x.resize(nxs);
    m.flux_y.resize(nxs);
    m.v2.resize(nxs);
    for (std::size_t i = 0; i < nxs; ++i) {
      m.n[i] = an[i] * dv;
      m.flux_x[i] = ax[i] * dv;
      m.flux_y[i] = ay[i] * dv;
      m.v2[i] = ae[i] * dv;
    }
    m.number = sc[0] * dV;
    m.l1 = sc[2] * dV;
    m.l2 = std::sqrt(std::max(sc[3], 0.0) * dV);
    m.entropy = sc[4] * dV;
    m.f_min = std::isfinite(fmin) ? fmin : 0.0;
    m.f_max = std::isfinite(fmax) ? fmax : 0.0;
    m.f_face_max = fface;
    m.boundary_fraction = sc[1] / (std::fabs(sc[0]) > kTiny ? sc[0] : kTiny);
    return m;
  }

  [[nodiscard]] DeviceBrick &brick(std::size_t s) { return *m_bricks.at(s); }

private:
  /// Phase A, on the host, with the two brick copies it forces.
  void half_x_step(double h) {
    for (std::size_t s = 0; s < m_bricks.size(); ++s) {
      double t0 = wall_seconds();
      m_bricks[s]->download(m_ps->f(s));
      device_synchronize();
      m_t.d2h += wall_seconds() - t0;

      t0 = wall_seconds();
      advect_x(*m_ps, m_ps->f(s), h, m_st->xplan, m_st->work);
      m_t.advect_x += wall_seconds() - t0;

      t0 = wall_seconds();
      m_bricks[s]->upload(m_ps->f(s));
      device_synchronize();
      m_t.h2d += wall_seconds() - t0;
    }
  }

  Stepper *m_st{nullptr};
  PhaseSpace *m_ps{nullptr};
  const SimParams *m_p{nullptr};
  DeviceGeometry m_g{};
  std::vector<std::unique_ptr<DeviceBrick>> m_bricks{};
  CoefficientBuffer m_cx{};
  CoefficientBuffer m_cy{};
  double *m_scratch{nullptr};
  double *m_out{nullptr};
  std::vector<double> m_host_out{};
  std::vector<double> m_stage_send{};
  std::vector<double> m_stage_recv{};
  PhaseTimings m_t{};
  int m_peak_halo{0};
};

} // namespace vlasov::hip

#endif // VLASOV_HIP_KERNELS_ONLY
