// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file classify.hpp
 * @brief Separate interior, exterior, and cut Cartesian cells.
 *
 * Analytic families use the exact range of φ on the cell. Equal corner
 * signs are not that test: a sphere sitting inside one cell is positive
 * at every corner and still cut. `classify_general` is the contract for
 * an arbitrary implicit function. Without a Lipschitz constant it will
 * not promote an unresolved leaf to exterior.
 */

#include <openpfc/kernel/geometry/cells.hpp>
#include <openpfc/kernel/geometry/implicit.hpp>

#include <vector>

namespace pfc::geometry {

struct Verdict {
  CellKind kind = CellKind::Exterior;
  int tangent = 0;
  int unresolved = 0;
  int degenerate_gradient = 0;
  double phi_min = 0;
  double phi_max = 0;
};

struct CornerSample {
  int inside = 0;
  int outside = 0;
  int on_interface = 0;
};

OPENPFC_HD inline Verdict
verdict_from_range(const Range &range, const Policy &policy, double length_scale) {
  Verdict out;
  out.phi_min = range.min_phi;
  out.phi_max = range.max_phi;
  const double band_min = phi_band(policy, range.grad_at_min, length_scale);
  const double band_max = phi_band(policy, range.grad_at_max, length_scale);
  if (range.grad_at_min <= policy.gradient_absolute ||
      range.grad_at_max <= policy.gradient_absolute)
    out.degenerate_gradient = 1;
  if (range.max_phi < -band_max) {
    out.kind = CellKind::Interior;
    return out;
  }
  if (range.min_phi > band_min) {
    out.kind = CellKind::Exterior;
    return out;
  }
  out.kind = CellKind::Cut;
  const bool touch_out = range.min_phi <= band_min && range.min_phi >= -band_min &&
                         range.max_phi > band_min;
  const bool touch_in = range.max_phi >= -band_max && range.max_phi <= band_max &&
                        range.min_phi < -band_max;
  out.tangent = touch_out || touch_in;
  return out;
}

OPENPFC_HD inline double box_diagonal(Box box) { return norm(box.hi - box.lo); }

OPENPFC_HD inline Verdict classify_box(const Implicit &shape, Box box,
                                       const Policy &policy) {
  Range range;
  Verdict out;
  if (!shape.range_on_box(box.lo, box.hi, range)) {
    out.kind = CellKind::Cut;
    out.unresolved = 1;
    return out;
  }
  return verdict_from_range(range, policy, box_diagonal(box));
}

inline CornerSample sample_corners(const Implicit &shape, Box box,
                                   const Policy &policy) {
  CornerSample sample;
  const double scale = box_diagonal(box);
  const Vec3 c[8] = {{box.lo.x, box.lo.y, box.lo.z}, {box.hi.x, box.lo.y, box.lo.z},
                     {box.lo.x, box.hi.y, box.lo.z}, {box.hi.x, box.hi.y, box.lo.z},
                     {box.lo.x, box.lo.y, box.hi.z}, {box.hi.x, box.lo.y, box.hi.z},
                     {box.lo.x, box.hi.y, box.hi.z}, {box.hi.x, box.hi.y, box.hi.z}};
  for (int i = 0; i < 8; ++i) {
    const Side side = side_of(shape, c[i], policy, scale);
    if (side == Side::Inside)
      ++sample.inside;
    else if (side == Side::Outside)
      ++sample.outside;
    else
      ++sample.on_interface;
  }
  return sample;
}

/// `lipschitz(box)` must be a Lipschitz constant of φ on that box, or a
/// negative value when none is known. An unresolved leaf is cut.
template <class Phi, class Lip>
Verdict classify_general(const Phi &phi, const Lip &lipschitz, Box box,
                         const Policy &policy, int depth) {
  struct Job {
    Box box;
    int depth;
  };
  Job stack[96];
  int top = 0;
  stack[top++] = {box, depth};
  bool saw_inside = false;
  bool saw_outside = false;
  bool unresolved = false;
  while (top > 0) {
    const Job job = stack[--top];
    const Vec3 mid = (job.box.lo + job.box.hi) * 0.5;
    const double value = phi(mid);
    const double radius = 0.5 * box_diagonal(job.box);
    const double L = lipschitz(job.box);
    const double band = length_tolerance(policy, box_diagonal(job.box));
    if (L >= 0 && absd(value) > L * radius + band) {
      if (value < 0)
        saw_inside = true;
      else
        saw_outside = true;
      continue;
    }
    Vec3 corners[8];
    detail::corners_of(job.box.lo, job.box.hi, corners);
    for (int i = 0; i < 8; ++i) {
      const double v = phi(corners[i]);
      if (v < -band)
        saw_inside = true;
      else if (v > band)
        saw_outside = true;
    }
    if (job.depth <= 0 || top > 80) {
      if (!(L >= 0 && absd(value) > L * radius + band)) unresolved = true;
      continue;
    }
    const Vec3 c0 = job.box.lo;
    const Vec3 c1 = mid;
    const Vec3 c2 = job.box.hi;
    const Vec3 xs[2] = {c0, c1};
    const Vec3 xe[2] = {c1, c2};
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          if (top >= 96) {
            unresolved = true;
            continue;
          }
          stack[top++] = {
              Box{{xs[i].x, xs[j].y, xs[k].z}, {xe[i].x, xe[j].y, xe[k].z}},
              job.depth - 1};
        }
  }
  Verdict out;
  if (saw_inside && saw_outside)
    out.kind = CellKind::Cut;
  else if (unresolved) {
    out.kind = CellKind::Cut;
    out.unresolved = 1;
  } else if (saw_inside)
    out.kind = CellKind::Interior;
  else
    out.kind = CellKind::Exterior;
  return out;
}

struct CutList {
  std::vector<int> index;
  int interior = 0;
  int exterior = 0;
  int cut = 0;
  int tangent = 0;
};

inline CutList classify_layout(const Implicit &shape, const CellLayout &layout,
                               const Policy &policy = {}) {
  CutList list;
  for (int k = 0; k < layout.count[2]; ++k)
    for (int j = 0; j < layout.count[1]; ++j)
      for (int i = 0; i < layout.count[0]; ++i) {
        const Verdict verdict = classify_box(shape, layout.box(i, j, k), policy);
        if (verdict.kind == CellKind::Interior)
          ++list.interior;
        else if (verdict.kind == CellKind::Exterior)
          ++list.exterior;
        else {
          ++list.cut;
          list.tangent += verdict.tangent;
          list.index.push_back(layout.linear(i, j, k));
        }
      }
  return list;
}

} // namespace pfc::geometry
