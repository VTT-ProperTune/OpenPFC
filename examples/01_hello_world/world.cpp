// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <openpfc/core/world.hpp>

/** \example world.cpp
 *
 * ## Overview
 *
 * This tutorial-style example introduces the `World` object, which plays a
 * foundational role in OpenPFC. It represents the global simulation domain \( \Omega
 * \), encoding the size, lower and upper bounds, spacing, coordinate system, and
 * periodicity.
 *
 * The core purpose of `World` is to provide a **geometric abstraction** of the
 * domain used in simulations. It does not perform computational steps, but rather
 * serves as a foundational definition of the simulation space, essential for
 * defining fields, applying boundary conditions, and performing discretization.
 *
 * In the most common case of Cartesian grids, the relation between grid indices
 * and physical coordinates is defined as:
 *
 * \[
 *   x(i) = x_0 + i \cdot \Delta x, \quad i = 0, 1, \dots, L_x - 1
 * \]
 *
 * This maps each grid point (i, j, k) to a point in physical space (x, y, z).
 *
 * In this set of examples, we explore several different ways of constructing `World`
 * objects and using their features, showcasing the flexibility and design philosophy
 * of the OpenPFC framework.
 */

/**
 * ### Example 1: Creating a Basic Cartesian World
 *
 * In this example, we construct a `World` object using dimensions, origin,
 * and spacing values in a traditional, positional-parameter style.
 *
 * This interface is fully supported for compatibility, but we anticipate that
 * it may become deprecated in the future due to the potential for **silent bugs**
 * caused by incorrect parameter ordering. Since C++ does not support keyword
 * arguments or dictionary-style construction natively, there is no language-level
 * safeguard to prevent users from passing spacing in the position of origin, or vice
 * versa. To address this, we introduce *strong typedefs* in later examples which
 * make each argument semantically explicit and enable validation at compile-time or
 * construction-time.
 *
 * #### Coordinate System and Grid Definition
 *
 * By default, the coordinate system is assumed to be `Cartesian`, which corresponds
 * to a uniform grid in three orthogonal directions. This is suitable for simulations
 * using finite differences or pseudo-spectral methods on a rectangular grid.
 *
 * The relation between grid index \( i \) and physical coordinate \( x \) is given
 * by:
 *
 * \[
 *   x(i) = x_0 + i \cdot \Delta x, \quad i = 0, 1, \dots, N - 1
 * \]
 *
 * The same pattern applies to \( y(j) \) and \( z(k) \). This correspondence allows
 * us to directly associate physical and grid spaces in both directions.
 *
 * #### Boundary Conditions and Periodicity
 *
 * This basic constructor implicitly assumes **periodic boundary conditions**
 * in all directions. This design choice reflects a common use case in materials
 * physics where periodicity enables efficient spectral methods using the FFT.
 *
 * However, we emphasize that OpenPFC also supports **non-periodic boundaries**,
 * and our roadmap includes support for polynomial-based spectral interpolation using
 * **Chebyshev polynomials**. These allow *non-periodic differentiation* to be
 * performed using FFT-style algorithms in \( \mathcal{O}(n \log n) \) time while
 * retaining high accuracy.
 *
 * Thus, while this example uses default periodic boundaries and Cartesian geometry,
 * the OpenPFC framework is not limited to this setup and will evolve toward
 * greater flexibility in geometry and boundary treatments.
 */
void example1_basic_cartesian_world() {

  cout << "=== Example 1: Basic Cartesian World ===\n";

  std::array<int, 3> dimensions = {10, 20, 30};
  std::array<double, 3> origin = {0.0, 0.0, 0.0};
  std::array<double, 3> discretization = {0.1, 0.1, 0.1};

  pfc::World w = pfc::world::create(dimensions, origin, discretization);

  std::cout << "World created:\n" << w << endl;

  std::cout << "Size: " << get_size(w)[0] << ", " << get_size(w)[1] << ", "
            << get_size(w)[2] << endl;
  std::cout << "Origin: (" << get_origin(w)[0] << ", " << get_origin(w)[1] << ", "
            << get_origin(w)[2] << ")\n";
  std::cout << "Spacing: " << get_spacing(w)[0] << ", " << get_spacing(w)[1] << ", "
            << get_spacing(w)[2] << endl;
  std::cout << "Coordinate System: " << static_cast<int>(get_coordinate_system(w))
            << endl;

  std::cout << "Periodicity in dimensions (x, y, z):\n";
  for (int i = 0; i < 3; ++i) {
    std::cout << "  Dimension " << i << ": "
              << (is_periodic(w, i) ? "true" : "false") << std::endl;
  }
}

void example2_minimal_world() {
  /**
   * ### Example 2: Creating a Minimalistic World
   *
   * In this example, we show how to define a simulation domain using only the
   * number of grid points. This is useful for prototyping or testing when precise
   * geometric units are not needed.
   *
   * The default origin is assumed to be (0.0, 0.0, 0.0), and the spacing is
   * assumed to be 1.0 in each direction. This results in a cube of side length
   * equal to the number of grid points in each direction.
   */

  cout << "\n=== Example 2: Minimal Definition ===\n";

  World w = world::create({64, 64, 64});
  cout << w << endl;
}

void example3_strong_typedef_construction() {
  /**
   * ### Example 3: Construction with Strong Typedefs
   *
   * In this example, we demonstrate the use of strong typedef wrappers to
   * construct a `World` object. This method ensures correctness by making
   * it impossible to accidentally swap parameters like spacing and bounds.
   *
   * Each wrapper (e.g. `Size3`, `LowerBounds3`, `Spacing3`) validates its values
   * during construction, and improves the readability and maintainability of code
   * by documenting intent explicitly.
   */

  cout << "\n=== Example 3: Strong Typedefs ===\n";

  using namespace world;

  Size3 size({16, 16, 1});
  LowerBounds3 lower({-1.0, -1.0, 0.0});
  UpperBounds3 upper({1.0, 1.0, 0.0});
  Spacing3 spacing({0.125, 0.125, 1.0});
  Periodic3 periodic({true, true, false});

  World w = world::create(size, lower, upper, spacing, periodic,
                          CoordinateSystemTag::Plane);
  cout << w << endl;
}

void example4_coordinate_conversion() {
  /**
   * ### Example 4: Coordinate Transformation in the Native System
   *
   * In this example, we show how to convert between grid indices and physical
   * coordinates in the **native coordinate system** of the `World`. For Cartesian
   * coordinates, this is straightforward multiplication and offsetting.
   *
   * OpenPFC does not lock the user into one coordinate system. We support
   * several coordinate systems including:
   *
   * - `Line` (1D Cartesian)
   * - `Plane` (2D Cartesian)
   * - `Cartesian` (3D Cartesian)
   * - `Polar` (2D Polar)
   * - `Cylindrical` (3D Cylindrical)
   * - `Spherical` (3D Spherical)
   *
   * All computations in this example are performed in the coordinate system’s
   * native basis. In future OpenPFC versions, we will support *inter-coordinate*
   * conversions (e.g. cylindrical → Cartesian), Jacobian evaluation, and
   * even user-defined coordinate warping such as bending a grid along a spline
   * or twisting a cylindrical volume.
   *
   * This enables highly flexible domain geometries without sacrificing the
   * simplicity of structured grid indexing.
   */

  cout << "\n=== Example 4: Grid ↔ Coordinates ===\n";

  World w = world::create({10, 10, 1});

  Int3 idx = {3, 7, 0};
  Real3 coords = to_coords(w, idx);

  cout << "Grid index: (" << idx[0] << ", " << idx[1] << ", " << idx[2]
       << ") maps to: ";
  cout << "(" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";

  Int3 roundtrip = to_indices(w, coords);
  cout << "Back to index: (" << roundtrip[0] << ", " << roundtrip[1] << ", "
       << roundtrip[2] << ")\n";
}

int main() {
  example1_basic_cartesian_world();
  example2_minimal_world();
  example3_strong_typedef_construction();
  example4_coordinate_conversion();
  return 0;
}
