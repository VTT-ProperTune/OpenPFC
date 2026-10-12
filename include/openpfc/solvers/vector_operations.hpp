// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace pfc::solvers {
struct ReductionMetrics {
  std::size_t calls = 0;
  double seconds = 0;
  /// Time inside MPI collectives. The rest of `seconds` is local arithmetic
  /// and, on a device, the scalar transfer that feeds the collective.
  double collective_seconds = 0;
};

/// Algebra for scalar entries; tensor consumers supply inner/axpy/scale.
template <class Value> struct ScalarAlgebra {
  static double inner(const Value &a, const Value &b) { return a * b; }
  static void axpy(Value &y, double a, const Value &x) { y += a * x; }
  static void scale(Value &y, double a) { y *= a; }
};

/**
 * Default std::vector operations with optional communicator-wide inner products.
 * MPI_COMM_NULL selects purely local arithmetic without requiring MPI_Init.
 * Algorithms only depend on this operations contract, so another vector/storage
 * implementation can provide make, copy, scale, axpy, dot and norm instead.
 * All collective consumers must use the same communicator and call sequence.
 */
template <class Value = double, class Algebra = ScalarAlgebra<Value>>
class VectorOperations {
  MPI_Comm comm_;

public:
  using Vector = std::vector<Value>;
  ReductionMetrics reductions;
  explicit VectorOperations(MPI_Comm comm = MPI_COMM_NULL) : comm_(comm) {}
  Vector make(std::size_t n) const { return Vector(n); }
  void copy(const Vector &x, Vector &y) const { y = x; }
  void scale(Vector &x, double a) const {
    for (auto &v : x) Algebra::scale(v, a);
  }
  void axpy(Vector &y, double a, const Vector &x) const {
    if (x.size() != y.size()) throw std::invalid_argument("vector size mismatch");
    for (std::size_t i = 0; i < x.size(); ++i) Algebra::axpy(y[i], a, x[i]);
  }
  double dot(const Vector &x, const Vector &y) {
    double out = 0;
    dots(&x, 1, y, &out);
    return out;
  }
  /// One collective for several inner products against the same vector.
  /// Coefficients are taken before any of those columns are subtracted, so a
  /// caller that wants modified Gram-Schmidt must still pass one column.
  void dots(const Vector *columns, int count, const Vector &y, double *out) {
    if (count < 1 || columns == nullptr || out == nullptr)
      throw std::invalid_argument("empty inner-product batch");
    double stack[64];
    std::vector<double> heap;
    double *local = stack;
    if (count > 64) {
      heap.resize(count);
      local = heap.data();
    }
    for (int column = 0; column < count; ++column) {
      if (columns[column].size() != y.size())
        throw std::invalid_argument("vector size mismatch");
      double sum = 0;
      for (std::size_t i = 0; i < y.size(); ++i)
        sum += Algebra::inner(columns[column][i], y[i]);
      local[column] = sum;
    }
    if (comm_ == MPI_COMM_NULL) {
      for (int column = 0; column < count; ++column) out[column] = local[column];
      return;
    }
    const auto t = std::chrono::steady_clock::now();
    if (MPI_Allreduce(local, out, count, MPI_DOUBLE, MPI_SUM, comm_) != MPI_SUCCESS)
      throw std::runtime_error("solver MPI reduction failed");
    const double waited =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    ++reductions.calls;
    reductions.seconds += waited;
    reductions.collective_seconds += waited;
  }
  double norm(const Vector &x) { return std::sqrt(dot(x, x)); }
};

/// Identity inverse-preconditioner; the iteration argument is for flexible M_k.
struct IdentityPreconditioner {
  template <class Vector> void operator()(const Vector &r, Vector &z, int) const {
    z = r;
  }
};
} // namespace pfc::solvers
