// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace pfc::solvers {
namespace transmission_detail {

inline double segment_resistance(const geometry::Implicit &shape, geometry::Vec3 a,
                                 geometry::Vec3 b, double kappa_minus,
                                 double kappa_plus, int depth) {
  const double phi_a = shape.value(a);
  const double phi_b = shape.value(b);
  const double length = geometry::norm(b - a);
  auto kappa_of = [&](double phi) { return phi < 0 ? kappa_minus : kappa_plus; };
  if (shape.family == geometry::Family::Plane) {
    if (phi_a == 0 && phi_b == 0) return length / kappa_plus;
    if (phi_a * phi_b < 0) {
      const double t = phi_a / (phi_a - phi_b);
      return t * length / kappa_of(phi_a) + (1.0 - t) * length / kappa_of(phi_b);
    }
    return length / kappa_of(phi_a);
  }
  const geometry::Vec3 mid = (a + b) * 0.5;
  const double phi_m = shape.value(mid);
  const double scale = 1e-12 * (1.0 + geometry::absd(phi_a) + geometry::absd(phi_b) +
                                geometry::absd(phi_m));
  const bool inside = phi_a < -scale && phi_m < -scale && phi_b < -scale;
  const bool outside = phi_a > scale && phi_m > scale && phi_b > scale;
  if (inside) return length / kappa_minus;
  if (outside) return length / kappa_plus;
  if (depth <= 0) return length / (phi_m < 0 ? kappa_minus : kappa_plus);
  return segment_resistance(shape, a, mid, kappa_minus, kappa_plus, depth - 1) +
         segment_resistance(shape, mid, b, kappa_minus, kappa_plus, depth - 1);
}

inline void tangent_basis(geometry::Vec3 normal, geometry::Vec3 &t1,
                          geometry::Vec3 &t2) {
  const geometry::Vec3 hint = geometry::absd(normal.z) < 0.9
                                  ? geometry::Vec3{0, 0, 1}
                                  : geometry::Vec3{0, 1, 0};
  t1 = geometry::cross(normal, hint);
  const double n1 = geometry::norm(t1);
  t1 = n1 > 0 ? t1 / n1 : geometry::Vec3{1, 0, 0};
  t2 = geometry::cross(normal, t1);
}

inline void project_orthogonal(double *coeff, int width, double modes[4][7],
                               int nmodes) {
  double basis[4][7]{};
  int rank = 0;
  for (int mode = 0; mode < nmodes; ++mode) {
    double v[7]{};
    for (int i = 0; i < width; ++i) v[i] = modes[mode][i];
    for (int k = 0; k < rank; ++k) {
      double dot = 0;
      for (int i = 0; i < width; ++i) dot += v[i] * basis[k][i];
      for (int i = 0; i < width; ++i) v[i] -= dot * basis[k][i];
    }
    double norm = 0;
    for (int i = 0; i < width; ++i) norm += v[i] * v[i];
    norm = std::sqrt(norm);
    if (!(norm > 1e-14)) continue;
    for (int i = 0; i < width; ++i) basis[rank][i] = v[i] / norm;
    ++rank;
  }
  for (int k = 0; k < rank; ++k) {
    double dot = 0;
    for (int i = 0; i < width; ++i) dot += coeff[i] * basis[k][i];
    for (int i = 0; i < width; ++i) coeff[i] -= dot * basis[k][i];
  }
}

} // namespace transmission_detail

inline void TransmissionOperator::apply(const std::vector<double> &u,
                                        std::vector<double> &y) const {
  y.assign(u.size(), 0.0);
  std::vector<char> done(static_cast<std::size_t>(node_count), 0);
  for (const TransmissionRow &row : interface_row) {
    double acc = 0;
    for (int s = 0; s < row.n; ++s)
      acc += row.value[s] * u[static_cast<std::size_t>(row.column[s])];
    y[static_cast<std::size_t>(row.node)] = acc;
    done[static_cast<std::size_t>(row.node)] = 1;
  }
  const double invh2 = 1.0 / (spacing * spacing);
  const int step[3] = {1, nodes_per_axis, nodes_per_axis * nodes_per_axis};
  for (int k = 0; k < nodes_per_axis; ++k)
    for (int j = 0; j < nodes_per_axis; ++j)
      for (int i = 0; i < nodes_per_axis; ++i) {
        const int id = index(i, j, k);
        if (done[static_cast<std::size_t>(id)]) continue;
        if (boundary(i, j, k)) {
          y[static_cast<std::size_t>(id)] = u[static_cast<std::size_t>(id)];
          continue;
        }
        const double kappa =
            shape.value(point(i, j, k)) < 0 ? kappa_minus : kappa_plus;
        const int neighbor[6] = {id - step[0], id + step[0], id - step[1],
                                 id + step[1], id - step[2], id + step[2]};
        double acc = 6.0 * kappa * invh2 * u[static_cast<std::size_t>(id)];
        for (int n = 0; n < 6; ++n)
          acc -= kappa * invh2 * u[static_cast<std::size_t>(neighbor[n])];
        y[static_cast<std::size_t>(id)] = acc;
      }
}

inline double TransmissionOperator::diagonal(int node) const {
  for (const TransmissionRow &row : interface_row)
    if (row.node == node) {
      for (int s = 0; s < row.n; ++s)
        if (row.column[s] == node) return row.value[s];
    }
  const int n = nodes_per_axis;
  const int i = node % n;
  const int jk = node / n;
  const int j = jk % n;
  const int k = jk / n;
  if (boundary(i, j, k)) return 1;
  const double kappa = shape.value(point(i, j, k)) < 0 ? kappa_minus : kappa_plus;
  return 6.0 * kappa / (spacing * spacing);
}

inline TransmissionOperator build_transmission(const geometry::Implicit &shape,
                                               const TransmissionRequest &request) {
  using geometry::Vec3;
  TransmissionOperator op;
  op.cells = request.cells;
  op.spacing = 1.0 / request.cells;
  op.nodes_per_axis = request.cells + 1;
  op.node_count = op.nodes_per_axis * op.nodes_per_axis * op.nodes_per_axis;
  op.kappa_minus = request.kappa_minus;
  op.kappa_plus = request.kappa_plus;
  op.closure = request.closure;
  op.shape = shape;
  op.interface_at.assign(static_cast<std::size_t>(op.node_count), -1);
  const int step[3] = {1, op.nodes_per_axis, op.nodes_per_axis * op.nodes_per_axis};
  const int sign[6][3] = {{-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
                          {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
  for (int k = 1; k < request.cells; ++k)
    for (int j = 1; j < request.cells; ++j)
      for (int i = 1; i < request.cells; ++i) {
        const int self = op.index(i, j, k);
        int column[7] = {self};
        double coeff[7] = {};
        Vec3 location[7] = {op.point(i, j, k)};
        int width = 1;
        bool touches = false;
        Vec3 anchor{};
        bool have_anchor = false;
        for (int d = 0; d < 6; ++d) {
          const int ni = i + sign[d][0];
          const int nj = j + sign[d][1];
          const int nk = k + sign[d][2];
          const int nb = self + sign[d][0] * step[0] + sign[d][1] * step[1] +
                         sign[d][2] * step[2];
          const Vec3 end = op.point(ni, nj, nk);
          const double plain =
              op.spacing / (shape.value(location[0]) < 0 ? request.kappa_minus
                                                         : request.kappa_plus);
          const double resistance = transmission_detail::segment_resistance(
              shape, location[0], end, request.kappa_minus, request.kappa_plus, 24);
          const double floor_r =
              request.theta_floor > 0 ? request.theta_floor * plain : 0;
          const double used = resistance < floor_r ? floor_r : resistance;
          if (std::abs(used - plain) > 1e-12 * plain) {
            touches = true;
            if (!have_anchor) {
              geometry::Vec3 left = location[0];
              geometry::Vec3 right = end;
              for (int it = 0; it < 40; ++it) {
                const Vec3 mid = (left + right) * 0.5;
                if (shape.value(left) * shape.value(mid) <= 0)
                  right = mid;
                else
                  left = mid;
              }
              anchor = (left + right) * 0.5;
              have_anchor = true;
            }
            ++op.cut_edges;
          }
          const double weight = 1.0 / (used * op.spacing);
          coeff[0] += weight;
          column[width] = nb;
          location[width] = end;
          coeff[width] = -weight;
          ++width;
        }
        if (touches) op.cut_edges -= 0; // counted once per directed edge below
        bool calibrate = touches &&
                         request.closure == TransmissionClosure::Calibrated &&
                         have_anchor;
        if (calibrate) {
          const geometry::Normal normal = shape.unit_normal(anchor, {});
          Vec3 nrm = normal.degenerate ? Vec3{1, 0, 0} : normal.direction;
          Vec3 t1, t2;
          transmission_detail::tangent_basis(nrm, t1, t2);
          double saved[7];
          for (int s = 0; s < width; ++s) saved[s] = coeff[s];
          double modes[4][7]{};
          for (int s = 0; s < width; ++s) {
            const Vec3 dx = location[s] - anchor;
            const double sdist = geometry::dot(nrm, dx);
            modes[0][s] = 1;
            modes[1][s] = geometry::dot(t1, dx);
            modes[2][s] = geometry::dot(t2, dx);
            const double factor =
                sdist < 0 ? request.kappa_plus / request.kappa_minus : 1.0;
            modes[3][s] = factor * sdist;
          }
          transmission_detail::project_orthogonal(coeff, width, modes, 4);
          double diag = 0;
          for (int s = 0; s < width; ++s)
            if (column[s] == self) diag = coeff[s];
          if (!(diag > 0) || !std::isfinite(diag)) {
            for (int s = 0; s < width; ++s) coeff[s] = saved[s];
            ++op.calibration_fallbacks;
            calibrate = false;
          }
        }
        if (touches && request.closure != TransmissionClosure::Nodal) {
          TransmissionRow row;
          row.node = self;
          row.n = width;
          for (int s = 0; s < width; ++s) {
            row.column[s] = column[s];
            row.value[s] = coeff[s];
          }
          op.interface_at[static_cast<std::size_t>(self)] =
              static_cast<int>(op.interface_row.size());
          op.interface_row.push_back(row);
        }
        (void)calibrate;
      }
  // Each interior edge is visited from both ends. Report unique cut edges.
  op.cut_edges = 0;
  for (int k = 0; k < op.nodes_per_axis; ++k)
    for (int j = 0; j < op.nodes_per_axis; ++j)
      for (int i = 0; i < op.nodes_per_axis; ++i) {
        const Vec3 start = op.point(i, j, k);
        const int next[3][3] = {{i + 1, j, k}, {i, j + 1, k}, {i, j, k + 1}};
        const bool room[3] = {i < request.cells, j < request.cells,
                              k < request.cells};
        for (int axis = 0; axis < 3; ++axis) {
          if (!room[axis]) continue;
          const Vec3 end = op.point(next[axis][0], next[axis][1], next[axis][2]);
          const double plain =
              op.spacing /
              (shape.value(start) < 0 ? request.kappa_minus : request.kappa_plus);
          const double resistance = transmission_detail::segment_resistance(
              shape, start, end, request.kappa_minus, request.kappa_plus, 12);
          if (std::abs(resistance - plain) > 1e-10 * plain) ++op.cut_edges;
        }
      }
  return op;
}

} // namespace pfc::solvers
