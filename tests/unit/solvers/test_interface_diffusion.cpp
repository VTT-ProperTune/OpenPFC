// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/solvers/interface_diffusion.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Gmres {
  int iterations = 0;
  double residual = 0;
  bool converged = false;
};

Gmres gmres(const pfc::solvers::TransmissionOperator &op, std::vector<double> &x,
            const std::vector<double> &b, int restart, int max_iter, double rtol) {
  const int n = static_cast<int>(b.size());
  std::vector<double> diag(static_cast<std::size_t>(n), 1);
  for (int i = 0; i < n; ++i) {
    const double d = op.diagonal(i);
    diag[static_cast<std::size_t>(i)] = std::abs(d) > 1e-14 ? d : 1;
  }
  auto apply_prec = [&](const std::vector<double> &v, std::vector<double> &w) {
    op.apply(v, w);
    for (int i = 0; i < n; ++i) w[i] /= diag[static_cast<std::size_t>(i)];
  };
  std::vector<double> bscale = b;
  for (int i = 0; i < n; ++i) bscale[i] /= diag[static_cast<std::size_t>(i)];
  auto dot = [&](const std::vector<double> &a, const std::vector<double> &c) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * c[i];
    return s;
  };
  auto axpy = [&](std::vector<double> &y, double alpha,
                  const std::vector<double> &v) {
    for (int i = 0; i < n; ++i) y[i] += alpha * v[i];
  };
  std::vector<double> r(n), w(n);
  op.apply(x, w);
  for (int i = 0; i < n; ++i) {
    w[i] /= diag[static_cast<std::size_t>(i)];
    r[i] = bscale[i] - w[i];
  }
  const double bnorm = std::sqrt(dot(bscale, bscale));
  const double stop = rtol * (bnorm > 0 ? bnorm : 1);
  Gmres out;
  for (int iter = 0; iter < max_iter && std::sqrt(dot(r, r)) > stop;) {
    const int m = std::min(restart, max_iter - iter);
    std::vector<std::vector<double>> v(static_cast<std::size_t>(m + 1),
                                       std::vector<double>(n));
    const double beta = std::sqrt(dot(r, r));
    for (int i = 0; i < n; ++i) v[0][i] = r[i] / beta;
    std::vector<double> h((m + 1) * m, 0), cs(m), sn(m), g(m + 1, 0);
    g[0] = beta;
    int used = 0;
    for (int j = 0; j < m; ++j) {
      op.apply(v[static_cast<std::size_t>(j)], w);
      for (int i = 0; i < n; ++i) w[i] /= diag[static_cast<std::size_t>(i)];
      for (int i = 0; i <= j; ++i) {
        h[i * m + j] = dot(w, v[static_cast<std::size_t>(i)]);
        axpy(w, -h[i * m + j], v[static_cast<std::size_t>(i)]);
      }
      h[(j + 1) * m + j] = std::sqrt(dot(w, w));
      if (h[(j + 1) * m + j] > 0)
        for (int i = 0; i < n; ++i)
          v[static_cast<std::size_t>(j + 1)][i] = w[i] / h[(j + 1) * m + j];
      for (int i = 0; i < j; ++i) {
        const double tmp = cs[i] * h[i * m + j] + sn[i] * h[(i + 1) * m + j];
        h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
        h[i * m + j] = tmp;
      }
      const double rho = std::hypot(h[j * m + j], h[(j + 1) * m + j]);
      cs[j] = h[j * m + j] / rho;
      sn[j] = h[(j + 1) * m + j] / rho;
      h[j * m + j] = rho;
      h[(j + 1) * m + j] = 0;
      const double gamma = cs[j] * g[j] + sn[j] * g[j + 1];
      g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
      g[j] = gamma;
      ++used;
      ++iter;
      if (std::abs(g[j + 1]) <= stop) break;
    }
    for (int j = used - 1; j >= 0; --j) {
      for (int i = j + 1; i < used; ++i) g[j] -= h[j * m + i] * g[i];
      g[j] /= h[j * m + j];
    }
    for (int j = 0; j < used; ++j) axpy(x, g[j], v[static_cast<std::size_t>(j)]);
    op.apply(x, w);
    for (int i = 0; i < n; ++i) {
      w[i] /= diag[static_cast<std::size_t>(i)];
      r[i] = bscale[i] - w[i];
    }
    out.iterations = iter;
    out.residual = std::sqrt(dot(r, r));
    if (out.residual <= stop) {
      out.converged = true;
      return out;
    }
  }
  out.residual = std::sqrt(dot(r, r));
  out.converged = out.residual <= stop;
  return out;
}

double manufactured(const pfc::geometry::Implicit &shape, double kappa_minus,
                    double kappa_plus, pfc::geometry::Vec3 tangential, double slope,
                    pfc::geometry::Vec3 x) {
  const pfc::geometry::Vec3 n = shape.normal;
  const double s = shape.value(x);
  const double factor = s < 0 ? kappa_plus / kappa_minus : 1.0;
  return pfc::geometry::dot(tangential, x) + slope * factor * s;
}

double nodal_l2(const pfc::solvers::TransmissionOperator &op,
                const std::vector<double> &u, const std::vector<double> &exact) {
  double acc = 0;
  int count = 0;
  for (int k = 1; k < op.cells; ++k)
    for (int j = 1; j < op.cells; ++j)
      for (int i = 1; i < op.cells; ++i) {
        const int id = op.index(i, j, k);
        const double d =
            u[static_cast<std::size_t>(id)] - exact[static_cast<std::size_t>(id)];
        acc += d * d;
        ++count;
      }
  return std::sqrt(acc / count);
}

std::vector<double> solve_plane(int cells, pfc::solvers::TransmissionClosure closure,
                                pfc::geometry::Vec3 normal,
                                pfc::geometry::Vec3 point, double kappa_minus,
                                double kappa_plus, Gmres &record) {
  const auto shape = pfc::geometry::make_plane(point, normal);
  pfc::geometry::Vec3 tangent = pfc::geometry::cross(shape.normal, {0, 0, 1});
  if (pfc::geometry::norm(tangent) < 1e-8)
    tangent = pfc::geometry::cross(shape.normal, {0, 1, 0});
  tangent = tangent / pfc::geometry::norm(tangent);
  const double slope = 0.7;
  pfc::solvers::TransmissionRequest request;
  request.cells = cells;
  request.kappa_minus = kappa_minus;
  request.kappa_plus = kappa_plus;
  request.closure = closure;
  const auto op = pfc::solvers::build_transmission(shape, request);
  std::vector<double> exact(static_cast<std::size_t>(op.node_count));
  std::vector<double> rhs(exact.size(), 0);
  std::vector<double> u(exact.size(), 0);
  for (int k = 0; k < op.nodes_per_axis; ++k)
    for (int j = 0; j < op.nodes_per_axis; ++j)
      for (int i = 0; i < op.nodes_per_axis; ++i) {
        const int id = op.index(i, j, k);
        exact[static_cast<std::size_t>(id)] = manufactured(
            shape, kappa_minus, kappa_plus, tangent, slope, op.point(i, j, k));
        if (op.boundary(i, j, k))
          rhs[static_cast<std::size_t>(id)] = exact[static_cast<std::size_t>(id)];
      }
  record = gmres(op, u, rhs, 40, 4000, 1e-10);
  std::cout << (closure == pfc::solvers::TransmissionClosure::Series ? "series"
                                                                     : "calibrated")
            << " n=" << cells << " l2=" << nodal_l2(op, u, exact)
            << " iters=" << record.iterations << " cut=" << op.cut_edges
            << " rows=" << op.interface_row.size()
            << " fallback=" << op.calibration_fallbacks
            << " converged=" << record.converged << "\n";
  return u;
}

int fail(const std::string &why) {
  std::cerr << why << "\n";
  return 1;
}

} // namespace

int main() {
  {
    const pfc::geometry::Vec3 aligned{1, 0, 0};
    const auto shape = pfc::geometry::make_plane({0.37, 0, 0}, aligned);
    pfc::solvers::TransmissionRequest request;
    request.cells = 16;
    request.closure = pfc::solvers::TransmissionClosure::Series;
    request.kappa_minus = 1;
    request.kappa_plus = 10;
    const auto op = pfc::solvers::build_transmission(shape, request);
    pfc::geometry::Vec3 tangent{0, 1, 0};
    std::vector<double> exact(static_cast<std::size_t>(op.node_count)),
        rhs(exact.size()), u(exact.size());
    for (int k = 0; k < op.nodes_per_axis; ++k)
      for (int j = 0; j < op.nodes_per_axis; ++j)
        for (int i = 0; i < op.nodes_per_axis; ++i) {
          const int id = op.index(i, j, k);
          exact[id] = manufactured(shape, 1, 10, tangent, 0.7, op.point(i, j, k));
          if (op.boundary(i, j, k)) rhs[id] = exact[id];
        }
    std::vector<double> y(exact.size());
    op.apply(exact, y);
    double interior = 0;
    double boundary_miss = 0;
    for (int k = 0; k < op.nodes_per_axis; ++k)
      for (int j = 0; j < op.nodes_per_axis; ++j)
        for (int i = 0; i < op.nodes_per_axis; ++i) {
          const int id = op.index(i, j, k);
          const double r =
              y[static_cast<std::size_t>(id)] - rhs[static_cast<std::size_t>(id)];
          if (op.boundary(i, j, k))
            boundary_miss = std::max(boundary_miss, std::abs(r));
          else
            interior = std::max(interior, std::abs(r));
        }
    std::cout << "aligned residual interior=" << interior
              << " boundary=" << boundary_miss << "\n";
    const auto solved = gmres(op, u, rhs, 40, 4000, 1e-10);
    const double err = nodal_l2(op, u, exact);
    std::cout << "aligned-series l2=" << err << " iters=" << solved.iterations
              << "\n";
    if (!(interior < 1e-8) || boundary_miss != 0)
      return fail("aligned series flux is not exact");
    if (!(err < 1e-8)) return fail("aligned series solution is not exact");
  }
  {
    const pfc::geometry::Vec3 oblique = {1.0, 1.3, 0.4};
    const auto shape = pfc::geometry::make_plane({0.46, 0.52, 0.48}, oblique);
    pfc::geometry::Vec3 tangent = pfc::geometry::cross(shape.normal, {0, 0, 1});
    tangent = tangent / pfc::geometry::norm(tangent);
    auto run = [&](pfc::solvers::TransmissionClosure closure) {
      pfc::solvers::TransmissionRequest request;
      request.cells = 16;
      request.closure = closure;
      request.kappa_minus = 1;
      request.kappa_plus = 10;
      const auto op = pfc::solvers::build_transmission(shape, request);
      std::vector<double> exact(static_cast<std::size_t>(op.node_count)),
          rhs(exact.size()), u(exact.size());
      for (int k = 0; k < op.nodes_per_axis; ++k)
        for (int j = 0; j < op.nodes_per_axis; ++j)
          for (int i = 0; i < op.nodes_per_axis; ++i) {
            const int id = op.index(i, j, k);
            exact[id] = manufactured(shape, 1, 10, tangent, 0.7, op.point(i, j, k));
            if (op.boundary(i, j, k)) rhs[id] = exact[id];
          }
      std::vector<double> y(exact.size());
      op.apply(exact, y);
      double interior = 0;
      for (int k = 1; k < op.cells; ++k)
        for (int j = 1; j < op.cells; ++j)
          for (int i = 1; i < op.cells; ++i) {
            const int id = op.index(i, j, k);
            interior = std::max(interior, std::abs(y[static_cast<std::size_t>(id)]));
          }
      const auto solved = gmres(op, u, rhs, 40, 2000, 1e-8);
      std::cout << (closure == pfc::solvers::TransmissionClosure::Series
                        ? "series"
                        : "calibrated")
                << " residual=" << interior << " l2=" << nodal_l2(op, u, exact)
                << " iters=" << solved.iterations << "\n";
      return std::pair<double, int>{nodal_l2(op, u, exact), solved.iterations};
    };
    const auto series = run(pfc::solvers::TransmissionClosure::Series);
    const auto calibrated = run(pfc::solvers::TransmissionClosure::Calibrated);
    std::cout << "oblique-series l2=" << series.first << " iters=" << series.second
              << "\n";
    std::cout << "oblique-calibrated l2=" << calibrated.first
              << " iters=" << calibrated.second << "\n";
    if (!(series.first > 1e-4))
      return fail("series flux unexpectedly solved the oblique jump");
    if (!(calibrated.first < 1e-6))
      return fail("calibrated transmission missed the oblique field");
  }
  {
    const auto shape = pfc::geometry::make_plane({0.5 + 1e-6, 0, 0}, {1, 0, 0});
    pfc::solvers::TransmissionRequest request;
    request.cells = 16;
    request.closure = pfc::solvers::TransmissionClosure::Calibrated;
    request.kappa_minus = 1;
    request.kappa_plus = 100;
    const auto op = pfc::solvers::build_transmission(shape, request);
    std::vector<double> exact(static_cast<std::size_t>(op.node_count)),
        rhs(exact.size()), u(exact.size());
    for (int k = 0; k < op.nodes_per_axis; ++k)
      for (int j = 0; j < op.nodes_per_axis; ++j)
        for (int i = 0; i < op.nodes_per_axis; ++i) {
          const int id = op.index(i, j, k);
          exact[static_cast<std::size_t>(id)] =
              manufactured(shape, 1, 100, {0, 1, 0}, 0.4, op.point(i, j, k));
          if (op.boundary(i, j, k))
            rhs[static_cast<std::size_t>(id)] = exact[static_cast<std::size_t>(id)];
        }
    const auto solved = gmres(op, u, rhs, 40, 4000, 1e-8);
    const double err = nodal_l2(op, u, exact);
    std::cout << "near-node contrast-100 l2=" << err
              << " iters=" << solved.iterations
              << " fallback=" << op.calibration_fallbacks << "\n";
    if (!(err < 1e-5)) return fail("near-node transmission lost the planar field");
  }
  {
    const double radius = 0.25;
    const double center = 0.5;
    const auto sphere = pfc::geometry::make_sphere({center, center, center}, radius);
    auto eshelby = [&](double kin, double kout, pfc::geometry::Vec3 p) {
      const double beta = 3 * kout / (kin + 2 * kout);
      const double gamma = (kout - kin) / (kin + 2 * kout);
      const double X = p.x - center, Y = p.y - center, Z = p.z - center;
      const double r2 = X * X + Y * Y + Z * Z;
      const double r = std::sqrt(r2);
      if (r <= radius) return beta * X;
      return X * (1.0 + gamma * radius * radius * radius / (r2 * r));
    };
    for (int cells : {8, 16}) {
      for (double kin : {10.0, 100.0}) {
        for (auto closure : {pfc::solvers::TransmissionClosure::Nodal,
                             pfc::solvers::TransmissionClosure::Series,
                             pfc::solvers::TransmissionClosure::Calibrated}) {
          pfc::solvers::TransmissionRequest request;
          request.cells = cells;
          request.kappa_minus = kin;
          request.kappa_plus = 1;
          request.closure = closure;
          const auto op = pfc::solvers::build_transmission(sphere, request);
          std::vector<double> exact(static_cast<std::size_t>(op.node_count)),
              rhs(exact.size()), u(exact.size());
          for (int k = 0; k < op.nodes_per_axis; ++k)
            for (int j = 0; j < op.nodes_per_axis; ++j)
              for (int i = 0; i < op.nodes_per_axis; ++i) {
                const int id = op.index(i, j, k);
                exact[static_cast<std::size_t>(id)] =
                    eshelby(kin, 1, op.point(i, j, k));
                if (op.boundary(i, j, k))
                  rhs[static_cast<std::size_t>(id)] =
                      exact[static_cast<std::size_t>(id)];
              }
          const auto solved = gmres(op, u, rhs, 30, 1500, 1e-8);
          const char *name =
              closure == pfc::solvers::TransmissionClosure::Nodal    ? "nodal"
              : closure == pfc::solvers::TransmissionClosure::Series ? "series"
                                                                     : "calibrated";
          std::cout << "sphere " << name << " n=" << cells << " contrast=" << kin
                    << " l2=" << nodal_l2(op, u, exact)
                    << " iters=" << solved.iterations << "\n";
          if (closure != pfc::solvers::TransmissionClosure::Nodal &&
              nodal_l2(op, u, exact) > 0.01)
            return fail("interface flux did not improve on the sphere");
        }
      }
    }
  }
  std::cout << "transmission checks passed\n";
  return 0;
}
