// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file discrete_field.hpp
 * @brief Discrete field representation with physical coordinate mapping
 *
 * @details
 * This file defines the DiscreteField template class, which represents a
 * discretized field with:
 * - N-dimensional array storage (via Array<T, D>)
 * - Physical coordinate system (origin, discretization, bounds)
 * - Interpolation capabilities (nearest-neighbor)
 * - Coordinate ↔ index transformations
 * - Field operations (apply functions to all points)
 *
 * DiscreteField bridges the gap between:
 * - Discrete grid indices (integer coordinates)
 * - Physical space coordinates (real-valued positions)
 *
 * Typical usage:
 * @code
 * // Create 3D field with 64³ points
 * pfc::DiscreteField<double, 3> field(
 *     {64, 64, 64},           // dimensions
 *     {0, 0, 0},              // offset
 *     {0.0, 0.0, 0.0},        // origin
 *     {1.0, 1.0, 1.0}         // discretization (spacing)
 * );
 *
 * // Apply function to all points
 * field.apply([](double x, double y, double z) {
 *     return std::sin(x) * std::cos(y);
 * });
 *
 * // Interpolate at physical coordinates
 * double value = field.interpolate({0.5, 1.2, 0.8});
 * @endcode
 *
 * This file is part of the Utilities module, providing convenient field
 * manipulation for initial conditions, post-processing, and analysis.
 *
 * @see array.hpp for underlying storage implementation
 * @see core/world.hpp for coordinate system abstraction
 * @see initial_conditions/ for usage examples in ICs
 *
 * @author OpenPFC Contributors
 * @date 2025
 */

#ifndef PFC_DISCRETE_FIELD_HPP
#define PFC_DISCRETE_FIELD_HPP

#include "array.hpp"
#include "openpfc/core/world.hpp"
#include "utils/show.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ostream>

namespace pfc {

/**
 * @brief Discrete field with physical coordinate mapping and interpolation
 *
 * DiscreteField provides a N-dimensional array with physical coordinate system
 * awareness, bridging discrete grid indices and continuous physical space. It
 * supports coordinate transformations, interpolation, and functional operations.
 *
 * **Core Concept:**
 * - Storage: N-dimensional array (via Array<T, D>)
 * - Geometry: Origin, discretization (spacing), bounding box
 * - Transformations: Indices ↔ physical coordinates
 * - Operations: Apply functions, interpolate, access by index/coordinate
 *
 * **Design Philosophy:**
 * - Immutable geometry (const origin, discretization, bounds)
 * - Direct data access (operator[], get_data())
 * - Functional operations (apply lambda to all points)
 * - Simple nearest-neighbor interpolation
 *
 * @tparam T Element type (typically double or std::complex<double>)
 * @tparam D Dimensionality (typically 3)
 *
 * @example Creating and initializing a field
 * @code
 * // 3D field: 32³ points, spacing 0.5 in each direction
 * pfc::DiscreteField<double, 3> field(
 *     {32, 32, 32},           // dimensions
 *     {0, 0, 0},              // offset (for subdomains)
 *     {0.0, 0.0, 0.0},        // origin
 *     {0.5, 0.5, 0.5}         // discretization (spacing)
 * );
 *
 * // Initialize with mathematical function
 * field.apply([](double x, double y, double z) {
 *     return std::sin(x) * std::cos(y) * std::exp(-z/10.0);
 * });
 * @endcode
 *
 * @example Array-style indexing
 * @code
 * pfc::DiscreteField<double, 3> field({64, 64, 64}, {0,0,0},
 *                                      {0.0,0.0,0.0}, {1.0,1.0,1.0});
 *
 * // Access by 3D index
 * field[{10, 20, 30}] = 1.0;
 *
 * // Access by linear index
 * field[0] = 0.5;  // First element
 *
 * // Iterate over all elements
 * for (size_t i = 0; i < field.get_data().size(); i++) {
 *     field[i] *= 2.0;
 * }
 * @endcode
 *
 * @example Coordinate-space operations
 * @code
 * pfc::DiscreteField<double, 3> field({32, 32, 32}, {0,0,0},
 *                                      {0.0,0.0,0.0}, {0.5,0.5,0.5});
 *
 * // Map index to physical coordinate
 * auto coords = field.map_indices_to_coordinates({10, 10, 10});
 * // coords = {5.0, 5.0, 5.0}  (10 * 0.5 spacing)
 *
 * // Map coordinate to nearest index
 * auto indices = field.map_coordinates_to_indices({7.3, 8.1, 9.8});
 * // indices = {15, 16, 20}  (rounded to nearest)
 *
 * // Check if coordinate is in bounds
 * bool valid = field.inbounds({12.0, 8.0, 4.0});
 * @endcode
 *
 * @example Interpolation (nearest-neighbor)
 * @code
 * pfc::DiscreteField<double, 3> field({64, 64, 64}, {0,0,0},
 *                                      {0.0,0.0,0.0}, {1.0,1.0,1.0});
 * field.apply([](double x, double y, double z) {
 *     return x*x + y*y + z*z;
 * });
 *
 * // Interpolate at arbitrary physical coordinate
 * double value = field.interpolate({5.3, 10.7, 20.2});
 * // Returns field value at nearest grid point
 *
 * // Note: Currently only nearest-neighbor (no higher-order interpolation)
 * @endcode
 *
 * @example Functional initialization (multiple overloads)
 * @code
 * pfc::DiscreteField<double, 3> field({64, 64, 64}, {0,0,0},
 *                                      {0.0,0.0,0.0}, {1.0,1.0,1.0});
 *
 * // 3D function: f(x, y, z)
 * field.apply([](double x, double y, double z) {
 *     return std::sin(x) * std::cos(y);
 * });
 *
 * // 2D function (for 2D fields): f(x, y)
 * pfc::DiscreteField<double, 2> field2d({64, 64}, {0,0},
 *                                        {0.0,0.0}, {1.0,1.0});
 * field2d.apply([](double x, double y) {
 *     return std::exp(-(x*x + y*y)/100.0);
 * });
 *
 * // 1D function: f(x) - uses first coordinate only
 * field.apply([](double x) {
 *     return std::tanh(x - 32.0);
 * });
 *
 * // N-D function: f(std::array<double, D>)
 * field.apply([](std::array<double, 3> coords) {
 *     return coords[0] + 2.0*coords[1] + 3.0*coords[2];
 * });
 * @endcode
 *
 * @example Integration with Model fields
 * @code
 * // Model stores fields as std::vector<T> (Field = std::vector<double>)
 * pfc::Model model(world, std::move(fft));
 * model.add_real_field("density");
 *
 * // Get field data
 * auto& field_data = model.get_real_field("density");
 *
 * // Create DiscreteField wrapper for coordinate-aware operations
 * auto inbox = pfc::fft::get_inbox(model.get_fft());
 * pfc::DiscreteField<double, 3> discrete_field(
 *     {inbox.high[0] - inbox.low[0] + 1,
 *      inbox.high[1] - inbox.low[1] + 1,
 *      inbox.high[2] - inbox.low[2] + 1},
 *     {inbox.low[0], inbox.low[1], inbox.low[2]},
 *     pfc::world::get_origin(world),
 *     pfc::world::get_spacing(world)
 * );
 *
 * // Initialize using DiscreteField's apply()
 * discrete_field.apply([](double x, double y, double z) {
 *     return std::sin(x);
 * });
 *
 * // Copy data back to Model
 * field_data = discrete_field.get_data();
 * @endcode
 *
 * @example Complex fields (for FFT operations)
 * @code
 * using Complex = std::complex<double>;
 * pfc::DiscreteField<Complex, 3> kspace_field(
 *     {64, 64, 33},  // FFT size (nz/2+1 for real-to-complex)
 *     {0, 0, 0},
 *     {0.0, 0.0, 0.0},
 *     {1.0, 1.0, 1.0}
 * );
 *
 * // Initialize complex field
 * kspace_field.apply([](double kx, double ky, double kz) {
 *     double k2 = kx*kx + ky*ky + kz*kz;
 *     return Complex(std::exp(-k2/10.0), 0.0);
 * });
 * @endcode
 *
 * **Performance Considerations:**
 * - Direct data access via get_data() for maximum performance
 * - apply() has iterator overhead but ensures coordinate correctness
 * - Nearest-neighbor interpolation is O(1)
 * - Coordinate transformations are cheap (simple arithmetic)
 *
 * **Memory Layout:**
 * - Underlying storage is std::vector<T> (contiguous, cache-friendly)
 * - Row-major order (last index varies fastest)
 * - Compatible with MPI subdomain data
 *
 * **Common Use Cases:**
 * - Initial condition setup with coordinate-space functions
 * - Post-processing and analysis with interpolation
 * - Coordinate-aware field manipulation
 * - Testing and validation (analytical comparisons)
 *
 * @note Geometry (origin, discretization, bounds) is immutable after construction
 * @note Only nearest-neighbor interpolation supported (no linear/cubic)
 * @note For MPI subdomains, use offset parameter to specify global position
 *
 * @warning interpolate() returns reference - ensure coordinate is in bounds!
 *          Use inbounds() first if uncertain
 *
 * @see Array<T,D> for underlying storage
 * @see MultiIndex<D> for index iteration
 * @see world.hpp for coordinate system abstraction
 * @see Model::get_real_field() for integration with simulation fields
 */
template <typename T, size_t D> class DiscreteField {
private:
  Array<T, D> m_array; /**< Multidimensional array containing data. */
  const std::array<double, D> m_origin; /**< The origin of the field. */
  const std::array<double, D>
      m_discretization;                      /**< The discretization of the field. */
  const std::array<double, D> m_coords_low;  /**< The lower bound of coordinates. */
  const std::array<double, D> m_coords_high; /**< The upper bound of coordinates. */

  /**
   * @brief Calculate lower bounding box of this field.
   *
   */
  std::array<double, D> calculate_coords_low(const std::array<int, D> &offset) {
    std::array<double, D> coords_low;
    for (size_t i = 0; i < D; i++)
      coords_low[i] = m_origin[i] + offset[i] * m_discretization[i];
    return coords_low;
  }

  /**
   * @brief Calculate upper bounding box of this field.
   *
   */
  std::array<double, D> calculate_coords_high(const std::array<int, D> &offset,
                                              const std::array<int, D> &size) {
    std::array<double, D> coords_high;
    for (size_t i = 0; i < D; i++)
      coords_high[i] = m_origin[i] + (offset[i] + size[i]) * m_discretization[i];
    return coords_high;
  }

public:
  /**
   * @brief Construct discrete field with specified geometry
   *
   * Creates a field with N-dimensional array storage and physical coordinate
   * system mapping. The geometry (origin, discretization, bounds) is immutable
   * after construction.
   *
   * @param dimensions Size in each dimension [nx, ny, nz]
   * @param offsets Global offset for subdomains (typically {0,0,0} for full domain)
   * @param origin Physical coordinates of the first grid point
   * @param discretization Spacing between grid points in each direction
   *
   * @example Full domain field
   * @code
   * // 64³ domain, unit spacing, origin at (0,0,0)
   * pfc::DiscreteField<double, 3> field(
   *     {64, 64, 64},           // dimensions
   *     {0, 0, 0},              // no offset (full domain)
   *     {0.0, 0.0, 0.0},        // origin
   *     {1.0, 1.0, 1.0}         // unit spacing
   * );
   * // Physical domain: [0, 64) x [0, 64) x [0, 64)
   * @endcode
   *
   * @example MPI subdomain with offset
   * @code
   * // Subdomain starting at global index (10, 20, 0)
   * pfc::DiscreteField<double, 3> subdomain(
   *     {32, 32, 64},           // local size
   *     {10, 20, 0},            // global offset
   *     {0.0, 0.0, 0.0},        // global origin
   *     {0.5, 0.5, 0.5}         // spacing
   * );
   * // Physical subdomain: [5,21) x [10,26) x [0,32)
   * @endcode
   *
   * @note Bounds are calculated as: [origin + offset*dx, origin + (offset+size)*dx)
   * @note Use consistent origin/discretization across all subdomains for correctness
   */
  DiscreteField(const std::array<int, D> &dimensions,
                const std::array<int, D> &offsets,
                const std::array<double, D> &origin,
                const std::array<double, D> &discretization)
      : m_array(dimensions, offsets), m_origin(origin),
        m_discretization(discretization),
        m_coords_low(calculate_coords_low(offsets)),
        m_coords_high(calculate_coords_high(offsets, dimensions)) {}

  /**
   * @brief Constructs a DiscreteField from an Decomposition object.
   *
   * @param decomp The Decomposition object.
   */
  /* TODO: Make free function for this
 DiscreteField(const Decomposition &decomp)
     : DiscreteField(get_inbox_size(decomp), get_inbox_offset(decomp),
                     get_origin(decomp.get_world()),
                     get_spacing(decomp.get_world())) {}
 */

  const std::array<double, D> &get_origin() const { return m_origin; }
  const std::array<double, D> &get_discretization() const {
    return m_discretization;
  }
  const std::array<double, D> &get_coords_low() const { return m_coords_low; }
  const std::array<double, D> &get_coords_high() const { return m_coords_high; }

  Array<T, D> &get_array() { return m_array; }
  const Array<T, D> &get_array() const { return m_array; }
  const MultiIndex<D> &get_index() { return get_array().get_index(); }
  const MultiIndex<D> &get_index() const { return get_array().get_index(); }
  std::vector<T> &get_data() { return get_array().get_data(); }

  /**
   * @brief Returns the element at the specified index.
   *
   * @param indices multi-dimensional indices
   * @return T&
   */
  T &operator[](const std::array<int, D> &indices) { return get_array()[indices]; }

  /**
   * @brief Returns the element at the specified index.
   *
   * @param idx
   * @return T&
   */
  T &operator[](size_t idx) { return get_array()[idx]; }

  /**
   * @brief Maps indices to coordinates in the field.
   *
   * @param indices The indices to map.
   * @return The corresponding coordinates.
   */
  std::array<double, D>
  map_indices_to_coordinates(const std::array<int, D> &indices) const {
    std::array<double, D> coordinates;
    for (size_t i = 0; i < D; ++i) {
      coordinates[i] = m_origin[i] + indices[i] * m_discretization[i];
    }
    return coordinates;
  }

  /**
   * @brief Maps coordinates to indices in the field.
   *
   * @param coordinates The coordinates to map.
   * @return The corresponding indices.
   */
  std::array<int, D>
  map_coordinates_to_indices(const std::array<double, D> &coordinates) const {
    std::array<int, D> indices;
    for (size_t i = 0; i < D; ++i) {
      indices[i] = static_cast<int>(
          std::round((coordinates[i] - m_origin[i]) / m_discretization[i]));
    }
    return indices;
  }

  /**
   * @brief Checks if the given coordinates are within the bounds of the field.
   *
   * @param coords The coordinates to check.
   * @return True if the coordinates are within the bounds, false otherwise.
   */
  bool inbounds(const std::array<double, D> &coords) {
    for (size_t i = 0; i < D; i++) {
      if (m_coords_low[i] > coords[i] || coords[i] >= m_coords_high[i]) return false;
    }
    return true;
  }

  /**
   * @brief Interpolate field value at physical coordinates (nearest-neighbor)
   *
   * Returns the field value at the grid point nearest to the given physical
   * coordinates. This uses simple rounding (not linear or higher-order).
   *
   * @param coordinates Physical coordinates [x, y, z] to interpolate at
   * @return Reference to the field value at the nearest grid point
   *
   * @example Basic interpolation
   * @code
   * pfc::DiscreteField<double, 3> field({64,64,64}, {0,0,0},
   *                                      {0.0,0.0,0.0}, {1.0,1.0,1.0});
   * field.apply([](double x, double y, double z) { return x + y + z; });
   *
   * // Query at arbitrary coordinate
   * double val = field.interpolate({5.7, 10.2, 20.8});
   * // Returns field value at grid point (6, 10, 21) - nearest neighbor
   * @endcode
   *
   * @example Safe interpolation with bounds checking
   * @code
   * std::array<double, 3> query_point = {15.3, 22.1, 8.9};
   * if (field.inbounds(query_point)) {
   *     double value = field.interpolate(query_point);
   * } else {
   *     // Handle out-of-bounds case
   * }
   * @endcode
   *
   * @warning Returns reference - coordinate must be in bounds!
   *          Use inbounds() first if uncertain. Out-of-bounds access is undefined.
   * @note Nearest-neighbor only (no linear/cubic interpolation)
   * @note Coordinates are rounded (not truncated) to nearest index
   *
   * @see map_coordinates_to_indices() for the index computation
   * @see inbounds() for bounds checking
   */
  T &interpolate(const std::array<double, D> &coordinates) {
    return get_array()[(map_coordinates_to_indices(coordinates))];
  }

  using FunctionND = std::function<T(std::array<double, D>)>;
  using Function1D = std::function<T(double)>;
  using Function2D = std::function<T(double, double)>;
  using Function3D = std::function<T(double, double, double)>;

  /**
   * @brief Applies the given function to each element of the field.
   *
   * The function must be invocable with std::array<double, D> and return a type
   * convertible to T.
   *
   * @tparam FunctionND The type of the function.
   * @param func The function to apply.
   */
  void apply(FunctionND &&func) {
    static_assert(
        std::is_convertible_v<
            std::invoke_result_t<FunctionND, std::array<double, D>>, T>,
        "Func must be invocable with std::array<double, D> and return a type "
        "convertible to T");
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      element = std::invoke(std::forward<FunctionND>(func),
                            map_indices_to_coordinates(*index_iterator));
      ++index_iterator;
    }
  }

  /**
   * @brief Applies the given 1D function to each element of the field.
   *
   * The function must be invocable with (double) and return a type convertible
   * toT.
   *
   * @tparam Function1D The type of the 1D function.
   * @param func The 1D function to apply.
   */
  void apply(Function1D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto coords = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function1D>(func), coords[0]);
      ++index_iterator;
    }
  }

  /**
   * @brief Applies the given 2D function to each element of the field.
   *
   * The function must be invocable with (double, double) and return a type
   * convertible toT.
   *
   * @tparam Function2D The type of the 2D function.
   * @param func The 2D function to apply.
   */
  void apply(Function2D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function2D>(func), x, y);
      ++index_iterator;
    }
  }

  /**
   * @brief Apply 3D function f(x,y,z) to all field points
   *
   * Evaluates the function at each grid point's physical coordinates and
   * assigns the result to that point. This is the most common way to initialize
   * fields with analytical functions.
   *
   * @tparam Function3D Function type callable with (double, double, double)
   * @param func Function to apply, must return type convertible to T
   *
   * @example Simple mathematical functions
   * @code
   * pfc::DiscreteField<double, 3> field({64,64,64}, {0,0,0},
   *                                      {0.0,0.0,0.0}, {1.0,1.0,1.0});
   *
   * // Sine wave in x, constant in y,z
   * field.apply([](double x, double y, double z) {
   *     return std::sin(2.0 * M_PI * x / 64.0);
   * });
   *
   * // Radial Gaussian
   * field.apply([](double x, double y, double z) {
   *     double r2 = (x-32)*(x-32) + (y-32)*(y-32) + (z-32)*(z-32);
   *     return std::exp(-r2 / 100.0);
   * });
   * @endcode
   *
   * @example Coordinate-dependent initialization
   * @code
   * // Different behavior in different regions
   * field.apply([](double x, double y, double z) {
   *     if (x < 32.0) {
   *         return 0.3;  // Left half: low density
   *     } else {
   *         return 0.7;  // Right half: high density
   *     }
   * });
   * @endcode
   *
   * @note Function is called once per grid point with physical coordinates
   * @note For large fields, this may be expensive - avoid in hot loops
   * @note Structured bindings used: auto [x, y, z] = coords;
   *
   * @see apply(Function1D) for 1D functions
   * @see apply(Function2D) for 2D functions
   * @see apply(FunctionND) for general N-D functions
   */
  void apply(Function3D &&func) {
    auto index_iterator = get_index().begin();
    for (T &element : get_data()) {
      const auto [x, y, z] = map_indices_to_coordinates(*index_iterator);
      element = std::invoke(std::forward<Function3D>(func), x, y, z);
      ++index_iterator;
    }
  }

  /**
   * @brief Convert DiscreteField<T, D> to std::vector<T>
   *
   * @return A reference to the underlying data.
   */
  operator std::vector<T> &() { return get_data(); }
  operator std::vector<T> &() const { return get_data(); }

  /**
   * @brief Get the size of the field.
   *
   * @return const std::array<int, D>&
   */
  const std::array<int, D> &get_size() const { return get_index().get_size(); }

  const std::array<int, D> &get_offset() const { return get_index().get_begin(); }

  void set_data(const std::vector<T> &data) { get_array().set_data(data); }

  /**
   * @brief Outputs the discrete field to the specified output stream.
   *
   * @param os The output stream.
   * @param field The discrete field to output.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os,
                                  const DiscreteField<T, D> &field) {
    const Array<T, D> &array = field.get_array();
    const MultiIndex<D> &index = array.get_index();
    os << "DiscreteField<" << TypeName<T>::get() << "," << D
       << ">(begin = " << utils::array_to_string(index.get_begin())
       << ", end = " << utils::array_to_string(index.get_end())
       << ", size = " << utils::array_to_string(index.get_size())
       << ", linear_size = " << index.get_linear_size()
       << ", origin = " << utils::array_to_string(field.get_origin())
       << ", discretization = " << utils::array_to_string(field.get_discretization())
       << ", coords_low = " << utils::array_to_string(field.get_coords_low())
       << ", coords_high = " << utils::array_to_string(field.get_coords_high());
    return os;
  }
};

/**
 * @brief Apply function to discrete field.
 *
 * @param field The discrete field.
 * @param func The function to apply.
 *
 */
template <typename T, size_t D, typename Function>
void apply(DiscreteField<T, D> &field, Function &&func) {
  field.apply(std::forward<Function>(func));
}

template <typename T, size_t D> void show(DiscreteField<T, D> &field) {
  utils::show(field.get_array().get_data(), field.get_index().get_size(),
              field.get_index().get_begin());
}

} // namespace pfc

#endif
