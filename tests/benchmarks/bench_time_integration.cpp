// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file bench_time_integration.cpp
 * @brief Performance benchmarks for time integration steppers
 *
 * These benchmarks measure the performance characteristics of explicit
 * time integration steppers (Euler, RK2, RK4) on representative PDE problems
 * (heat equation, wave equation) with different backend types (finite
 * difference, spectral).
 *
 * Expected performance characteristics:
 * - Single-step timing: Euler ~10-100 μs (32³), RK2 ~20-200 μs, RK4 ~40-400 μs
 * - Full simulation: RK4 fastest overall despite higher per-step cost
 * - Memory usage: Euler ~O(n), RK2 ~O(3n), RK4 ~O(5n) for buffer storage
 * - Scaling: Linear O(N) scaling across grid sizes
 *
 * Run with:
 *   ./build/tests/openpfc-tests "[time_integration][benchmark]"
 *
 * For accurate results:
 *   - Use Release build: cmake -B build -DCMAKE_BUILD_TYPE=Release
 *   - Run on dedicated system (no background processes)
 *   - Ensure OpenPFC_BUILD_BENCHMARKS=ON is set
 */

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>
#include <openpfc/kernel/simulation/for_each_interior.hpp>

using namespace pfc;
using namespace pfc::sim::steppers;

// ============================================================================
// Test Models
// ============================================================================

namespace test_models {

/**
 * @brief Heat equation model: ∂u/∂t = α∇²u
 *
 * Uses second derivatives (xx, yy, zz) from the gradient evaluator
 * to compute the Laplacian.
 */
struct HeatGrads {
  double xx{}, yy{}, zz{};
};

class HeatEquation {
public:
  explicit HeatEquation(double alpha = 1.0) : m_alpha(alpha) {}

  template <class G>
  double rhs(double /*t*/, const G& g) const {
    return m_alpha * (g.xx + g.yy + g.zz);
  }

  double alpha() const noexcept { return m_alpha; }

private:
  double m_alpha;
};

/**
 * @brief Wave equation model: ∂²u/∂t² = c²∇²u
 *
 * First-order system formulation using velocity and displacement fields.
 * This model provides the velocity increment for the displacement field.
 */
struct WaveGrads {
  double xx{}, yy{}, zz{};
};

class WaveEquation {
public:
  explicit WaveEquation(double c = 1.0) : m_c(c) {}

  template <class G>
  double rhs(double /*t*/, const G& g) const {
    // First-order system: dv/dt = c²∇²u, du/dt = v
    // Returns velocity increment for displacement field
    return m_c * m_c * (g.xx + g.yy + g.zz);
  }

  double wave_speed() const noexcept { return m_c; }

private:
  double m_c;
};

} // namespace test_models

// ============================================================================
// Single-Step Timing Benchmarks
// ============================================================================

TEST_CASE("Time integration - single-step timing", "[time_integration][benchmark]") {
  using namespace test_models;

  const Int3 size_32 = {32, 32, 32};
  const Int3 size_64 = {64, 64, 64};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {1.0, 1.0, 1.0};

  SECTION("Euler stepper - FD gradient - heat equation - 32³") {
    auto world = world::create(GridSize(size_32), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);
    auto stepper = create(field, eval, model, 0.01);

    BENCHMARK("Euler single step") {
      return stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~10-100 μs per step (32³, FD, 2nd order)");
  }

  SECTION("RK2 midpoint stepper - FD gradient - heat equation - 32³") {
    auto world = world::create(GridSize(size_32), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);

    auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
      pfc::sim::for_each_interior(model, eval, du.data(), t);
    };
    auto rk2_tableau = make_rk2_midpoint<double>();
    ExplicitRKStepper<decltype(rk2_rhs)> rk2_stepper(0.01, field.size(), rk2_tableau, rk2_rhs);

    BENCHMARK("RK2 midpoint single step") {
      return rk2_stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~20-200 μs per step (32³, FD, 2nd order, 2 RHS evals)");
  }

  SECTION("RK4 classical stepper - FD gradient - heat equation - 32³") {
    auto world = world::create(GridSize(size_32), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);
    auto stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("RK4 classical single step") {
      return stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~40-400 μs per step (32³, FD, 4th order, 4 RHS evals)");
  }

  SECTION("Euler stepper - FD gradient - heat equation - 64³") {
    auto world = world::create(GridSize(size_64), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);
    auto stepper = create(field, eval, model, 0.01);

    BENCHMARK("Euler single step (64³)") {
      return stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~80-800 μs per step (64³, FD, 2nd order)");
  }

  SECTION("RK2 midpoint stepper - FD gradient - heat equation - 64³") {
    auto world = world::create(GridSize(size_64), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);

    auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
      pfc::sim::for_each_interior(model, eval, du.data(), t);
    };
    auto rk2_tableau = make_rk2_midpoint<double>();
    ExplicitRKStepper<decltype(rk2_rhs)> rk2_stepper(0.01, field.size(), rk2_tableau, rk2_rhs);

    BENCHMARK("RK2 midpoint single step (64³)") {
      return rk2_stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~160-1600 μs per step (64³, FD, 2nd order, 2 RHS evals)");
  }

  SECTION("RK4 classical stepper - FD gradient - heat equation - 64³") {
    auto world = world::create(GridSize(size_64), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);
    auto stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("RK4 classical single step (64³)") {
      return stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~320-3200 μs per step (64³, FD, 4th order, 4 RHS evals)");
  }

  SECTION("RK4 classical stepper - FD gradient - wave equation - 32³") {
    auto world = world::create(GridSize(size_32), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<WaveGrads>(field, /*order=*/2);
    WaveEquation model(1.0);
    auto stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("RK4 wave equation single step") {
      return stepper.step(0.0, field.vec());
    };

    INFO("Expected: ~40-400 μs per step (32³, FD, wave equation)");
  }
}

// ============================================================================
// Full Simulation Timing Benchmarks
// ============================================================================

TEST_CASE("Time integration - full simulation timing", "[time_integration][benchmark]") {
  using namespace test_models;

  const Int3 size = {32, 32, 32};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {1.0, 1.0, 1.0};

  SECTION("Euler vs RK2 vs RK4 - heat equation - FD gradient") {
    auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);

    const double dt_euler = 0.001;
    const double dt_rk2 = 0.01;   // Larger due to higher order
    const double dt_rk4 = 0.02;   // Larger due to higher order
    const int steps_euler = 1000;
    const int steps_rk2 = 100;
    const int steps_rk4 = 50;

    BENCHMARK("Euler full simulation") {
      auto field_copy = field;
      auto stepper = create(field_copy, eval, model, dt_euler);
      double t = 0.0;
      for (int i = 0; i < steps_euler; ++i) {
        t = stepper.step(t, field_copy.vec());
      }
      return t;
    };

    BENCHMARK("RK2 midpoint full simulation") {
      auto field_copy = field;
      auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
        pfc::sim::for_each_interior(model, eval, du.data(), t);
      };
      auto rk2_tableau = make_rk2_midpoint<double>();
      ExplicitRKStepper<decltype(rk2_rhs)> stepper(dt_rk2, field_copy.size(), rk2_tableau, rk2_rhs);
      double t = 0.0;
      for (int i = 0; i < steps_rk2; ++i) {
        t = stepper.step(t, field_copy.vec());
      }
      return t;
    };

    BENCHMARK("RK4 classical full simulation") {
      auto field_copy = field;
      auto stepper = create(field_copy, eval, model, dt_rk4, make_rk4_classical<double>());
      double t = 0.0;
      for (int i = 0; i < steps_rk4; ++i) {
        t = stepper.step(t, field_copy.vec());
      }
      return t;
    };

    INFO("Expected: RK4 fastest overall despite higher per-step cost");
    INFO("Higher order methods allow larger time steps for same accuracy");
  }
}

// ============================================================================
// Memory Usage Benchmarks
// ============================================================================

TEST_CASE("Time integration - memory usage", "[time_integration][benchmark]") {
  using namespace test_models;

  const std::size_t local_size = 32 * 32 * 32;

  SECTION("Stepper object sizes") {
    auto world = world::create(GridSize({32, 32, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                               GridSpacing({1.0, 1.0, 1.0}));
    auto decomp = decomposition::create(world, 1);
    auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
    HeatEquation model(1.0);

    auto euler_stepper = create(field, eval, model, 0.01);

    auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
      pfc::sim::for_each_interior(model, eval, du.data(), t);
    };
    auto rk2_tableau = make_rk2_midpoint<double>();
    auto rk2_stepper = ExplicitRKStepper<decltype(rk2_rhs)>(0.01, field.size(), rk2_tableau, rk2_rhs);

    auto rk4_stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("EulerStepper sizeof") { return sizeof(euler_stepper); };
    BENCHMARK("ExplicitRKStepper (RK2 midpoint) sizeof") { return sizeof(rk2_stepper); };
    BENCHMARK("ExplicitRKStepper sizeof") { return sizeof(rk4_stepper); };

    INFO("Expected: Euler ~O(n), RK2 ~O(3n), RK4 ~O(5n) for buffer storage");
    INFO("Object overhead should be minimal (captured lambdas + tableau)");
  }

  SECTION("Buffer size scaling") {
    const std::vector<std::size_t> sizes = {32*32*32, 64*64*64};

    for (auto sz : sizes) {
      const Int3 size = {static_cast<int>(std::cbrt(sz)),
                         static_cast<int>(std::cbrt(sz)),
                         static_cast<int>(std::cbrt(sz))};
      auto world = world::create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                                 GridSpacing({1.0, 1.0, 1.0}));
      auto decomp = decomposition::create(world, 1);
      auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
      auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
      HeatEquation model(1.0);

      const std::string label_euler = "Euler buffer size " + std::to_string(static_cast<int>(std::cbrt(sz))) + "³";
      BENCHMARK(label_euler.c_str()) {
        auto stepper = create(field, eval, model, 0.01);
        return field.size(); // Approximate buffer size
      };

      const std::string label_rk2 = "RK2 buffer size " + std::to_string(static_cast<int>(std::cbrt(sz))) + "³";
      BENCHMARK(label_rk2.c_str()) {
        auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
          pfc::sim::for_each_interior(model, eval, du.data(), t);
        };
        auto rk2_tableau = make_rk2_midpoint<double>();
        ExplicitRKStepper<decltype(rk2_rhs)> stepper(0.01, field.size(), rk2_tableau, rk2_rhs);
        return field.size() * 3; // RK2 stores ~3 buffers
      };

      const std::string label_rk4 = "RK4 buffer size " + std::to_string(static_cast<int>(std::cbrt(sz))) + "³";
      BENCHMARK(label_rk4.c_str()) {
        auto stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());
        return field.size() * 5; // RK4 stores ~5 buffers
      };
    }

    INFO("Expected: O(sz) scaling with constant factors 1x, 3x, 5x");
  }
}

// ============================================================================
// Scaling with Problem Size Benchmarks
// ============================================================================

TEST_CASE("Time integration - scaling with problem size", "[time_integration][benchmark]") {
  using namespace test_models;

  const std::vector<Int3> grid_sizes = {{32, 32, 32}, {64, 64, 64}, {128, 128, 128}};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {1.0, 1.0, 1.0};

  SECTION("Euler stepper - FD gradient - heat equation - scaling") {
    for (const auto& size : grid_sizes) {
      auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
      auto decomp = decomposition::create(world, 1);
      auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
      auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
      HeatEquation model(1.0);
      auto stepper = create(field, eval, model, 0.01);

      const std::string label = "Euler " + std::to_string(size[0]) + "³";
      BENCHMARK(label.c_str()) {
        return stepper.step(0.0, field.vec());
      };
    }

    INFO("Expected: Linear O(N) scaling across grid sizes");
    INFO("64³ should be ~8x slower than 32³, 128³ ~64x slower than 32³");
  }

  SECTION("RK2 midpoint stepper - FD gradient - heat equation - scaling") {
    for (const auto& size : grid_sizes) {
      auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
      auto decomp = decomposition::create(world, 1);
      auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
      auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
      HeatEquation model(1.0);

      auto rk2_rhs = [&eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
        pfc::sim::for_each_interior(model, eval, du.data(), t);
      };
      auto rk2_tableau = make_rk2_midpoint<double>();
      ExplicitRKStepper<decltype(rk2_rhs)> stepper(0.01, field.size(), rk2_tableau, rk2_rhs);

      const std::string label = "RK2 midpoint " + std::to_string(size[0]) + "³";
      BENCHMARK(label.c_str()) {
        return stepper.step(0.0, field.vec());
      };
    }

    INFO("Expected: Linear O(N) scaling across grid sizes");
    INFO("64³ should be ~8x slower than 32³, 128³ ~64x slower than 32³");
  }

  SECTION("RK4 classical stepper - FD gradient - heat equation - scaling") {
    for (const auto& size : grid_sizes) {
      auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
      auto decomp = decomposition::create(world, 1);
      auto field = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
      auto eval = pfc::field::create<HeatGrads>(field, /*order=*/2);
      HeatEquation model(1.0);
      auto stepper = create(field, eval, model, 0.01, make_rk4_classical<double>());

      const std::string label = "RK4 classical " + std::to_string(size[0]) + "³";
      BENCHMARK(label.c_str()) {
        return stepper.step(0.0, field.vec());
      };
    }

    INFO("Expected: Linear O(N) scaling across grid sizes");
    INFO("64³ should be ~8x slower than 32³, 128³ ~64x slower than 32³");
  }
}

// ============================================================================
// Spectral Gradient Backend Benchmarks
// ============================================================================

TEST_CASE("Time integration - spectral gradient backend", "[time_integration][benchmark]") {
  using namespace test_models;

  const Int3 size = {32, 32, 32};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {1.0, 1.0, 1.0};

  SECTION("Spectral vs FD gradient - RK4 stepper - heat equation") {
    // Spectral gradient setup with FFT
    auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto fft = fft::create(decomp);

    auto field = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());
    auto spectral_eval = pfc::field::create<HeatGrads>(field, fft);

    HeatEquation model(1.0);
    auto spectral_stepper = create(field, spectral_eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("Spectral gradient - RK4 single step") {
      return spectral_stepper.step(0.0, field.vec());
    };

    // FD gradient benchmark for comparison
    auto field_fd = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto fd_eval = pfc::field::create<HeatGrads>(field_fd, /*order=*/2);
    auto fd_stepper = create(field_fd, fd_eval, model, 0.01, make_rk4_classical<double>());

    BENCHMARK("FD gradient - RK4 single step") {
      return fd_stepper.step(0.0, field_fd.vec());
    };

    INFO("Expected: Spectral has higher per-step cost due to FFT overhead");
    INFO("Spectral advantage appears for high-accuracy requirements or smooth solutions");
  }

  SECTION("Spectral vs FD gradient - Euler stepper - heat equation") {
    auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto fft = fft::create(decomp);

    auto field = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());
    auto spectral_eval = pfc::field::create<HeatGrads>(field, fft);

    HeatEquation model(1.0);
    auto spectral_stepper = create(field, spectral_eval, model, 0.01);

    BENCHMARK("Spectral gradient - Euler single step") {
      return spectral_stepper.step(0.0, field.vec());
    };

    auto field_fd = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto fd_eval = pfc::field::create<HeatGrads>(field_fd, /*order=*/2);
    auto fd_stepper = create(field_fd, fd_eval, model, 0.01);

    BENCHMARK("FD gradient - Euler single step") {
      return fd_stepper.step(0.0, field_fd.vec());
    };

    INFO("Expected: Euler shows similar backend cost pattern as RK4");
  }

  SECTION("Spectral vs FD gradient - RK2 midpoint stepper - heat equation") {
    auto world = world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));
    auto decomp = decomposition::create(world, 1);
    auto fft = fft::create(decomp);

    auto field = field::LocalField<double>::from_inbox(world, fft.get_inbox_bounds());
    auto spectral_eval = pfc::field::create<HeatGrads>(field, fft);

    HeatEquation model(1.0);

    auto spectral_rk2_rhs = [&spectral_eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
      pfc::sim::for_each_interior(model, spectral_eval, du.data(), t);
    };
    auto rk2_tableau = make_rk2_midpoint<double>();
    ExplicitRKStepper<decltype(spectral_rk2_rhs)> spectral_stepper(0.01, field.size(), rk2_tableau, spectral_rk2_rhs);

    BENCHMARK("Spectral gradient - RK2 midpoint single step") {
      return spectral_stepper.step(0.0, field.vec());
    };

    auto field_fd = field::LocalField<double>::from_subdomain(decomp, 0, /*halo_width=*/1);
    auto fd_eval = pfc::field::create<HeatGrads>(field_fd, /*order=*/2);

    auto fd_rk2_rhs = [&fd_eval, &model](double t, const std::vector<double>& u, std::vector<double>& du) {
      pfc::sim::for_each_interior(model, fd_eval, du.data(), t);
    };
    ExplicitRKStepper<decltype(fd_rk2_rhs)> fd_stepper(0.01, field_fd.size(), rk2_tableau, fd_rk2_rhs);

    BENCHMARK("FD gradient - RK2 midpoint single step") {
      return fd_stepper.step(0.0, field_fd.vec());
    };

    INFO("Expected: RK2 shows similar backend cost pattern as RK4");
  }
}
