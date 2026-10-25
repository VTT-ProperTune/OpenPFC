// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/complex_outbox.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/fft_layout.hpp>

using namespace pfc;
using pfc::fft::layout::complex_proc_grid_for_r2c;
using pfc::fft::layout::complex_proc_grid_is_legal;
using pfc::fft::layout::ComplexOutboxCost;
using pfc::fft::layout::effective_r2c_plan_options;
using pfc::fft::layout::legacy_complex_proc_grid_for_r2c;
using pfc::fft::layout::legal_complex_proc_grids;
using pfc::fft::layout::plan_r2c_outbox_cost;
using pfc::fft::layout::r2c_complex_world_box;
using pfc::fft::layout::real_world_box;
using pfc::fft::layout::select_min_distributed_reshape_complex_grid;
using pfc::fft::layout::split_boxes_are_legal;

namespace {

struct EnvGuard {
  ~EnvGuard() {
    unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID");
    unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX");
  }
};

heffte::plan_options fftw_r2c_options() {
  return effective_r2c_plan_options<heffte::backend::fftw>(
      heffte::default_options<heffte::backend::fftw>());
}

/// tungsten_hip TOML after off-node comm_scale: slabs, reorder, p2p_plined.
heffte::plan_options production_slab_r2c_options() {
  return heffte::plan_options(true, heffte::reshape_algorithm::p2p_plined, false);
}

ComplexOutboxCost cost_for(const Int3 &real_size, const Int3 &real_grid,
                           const Int3 &complex_grid, int r2c,
                           const heffte::plan_options &opts) {
  return plan_r2c_outbox_cost(real_world_box(real_size), real_grid,
                              r2c_complex_world_box(real_size, r2c), complex_grid,
                              r2c, opts);
}

Int3 select_for(const Int3 &real_size, const Int3 &real_grid, int r2c,
                const heffte::plan_options &opts) {
  return select_min_distributed_reshape_complex_grid(
      real_world_box(real_size), real_grid, r2c_complex_world_box(real_size, r2c),
      r2c, opts);
}

void require_exact_cover(const heffte::box3d<int> &world, const Int3 &grid) {
  const auto boxes = heffte::split_world(world, grid);
  REQUIRE(boxes.size() == static_cast<std::size_t>(grid[0] * grid[1] * grid[2]));
  REQUIRE(split_boxes_are_legal(boxes, world));
  long long vol = 0;
  for (const auto &b : boxes) {
    REQUIRE_FALSE(b.empty());
    vol += b.count();
  }
  REQUIRE(vol == world.count());
}

} // namespace

TEST_CASE("heFFTe split_world covers uneven complex boxes exactly",
          "[fft][layout][unit][complex_outbox]") {
  const auto world = r2c_complex_world_box({1200, 1200, 1152}, 0);
  REQUIRE(world.size[0] == 601);
  REQUIRE(world.size[1] == 1200);
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 32, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{2, 16, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{4, 8, 1}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 16, 2}, 32));
  REQUIRE(complex_proc_grid_is_legal(world, Int3{1, 1, 32}, 32));
  REQUIRE_FALSE(complex_proc_grid_is_legal(world, Int3{1, 31, 1}, 32));
  REQUIRE_FALSE(complex_proc_grid_is_legal(world, Int3{0, 32, 1}, 32));

  const auto yslab = heffte::split_world(world, Int3{1, 32, 1});
  REQUIRE(yslab.front().size[1] == 38);
  REQUIRE(yslab.back().size[1] == 37);
  require_exact_cover(world, Int3{1, 32, 1});

  const auto g216 = heffte::split_world(world, Int3{2, 16, 1});
  REQUIRE(g216.front().size[0] == 301);
  REQUIRE(g216.back().size[0] == 300);
  require_exact_cover(world, Int3{2, 16, 1});
}

TEST_CASE("r2c planner reproduces 1 vs 2 MPI reshapes",
          "[fft][layout][unit][complex_outbox]") {
  const Int3 z32{1, 1, 32};
  const auto opts = production_slab_r2c_options();
  {
    const Int3 n{1152, 1152, 1152};
    const auto yslab = cost_for(n, z32, Int3{1, 32, 1}, 0, opts);
    REQUIRE(yslab.n_mpi_reshapes == 1);
    REQUIRE(yslab.predicted_all_peer_rounds_per_cycle == 2);
    REQUIRE(legacy_complex_proc_grid_for_r2c(z32, r2c_complex_world_box(n, 0), 0) ==
            Int3{1, 32, 1});
  }
  {
    const Int3 n{1216, 1216, 1216};
    REQUIRE(cost_for(n, z32, Int3{1, 32, 1}, 0, opts).n_mpi_reshapes == 1);
  }
  {
    const Int3 n{1200, 1200, 1152};
    const auto current = cost_for(n, z32, Int3{1, 1, 32}, 0, opts);
    REQUIRE(current.n_mpi_reshapes == 2);
    REQUIRE(current.predicted_all_peer_rounds_per_cycle == 4);
    REQUIRE(legacy_complex_proc_grid_for_r2c(z32, r2c_complex_world_box(n, 0), 0) ==
            z32);
  }
}

TEST_CASE("selector picks a 1-MPI-reshape outbox on 1200×1200×1152",
          "[fft][layout][unit][complex_outbox]") {
  const Int3 n{1200, 1200, 1152};
  const Int3 real_g{1, 1, 32};
  const auto opts = production_slab_r2c_options();
  REQUIRE(cost_for(n, real_g, Int3{1, 1, 32}, 0, opts).n_mpi_reshapes == 2);
  for (const Int3 g : {Int3{1, 32, 1}, Int3{2, 16, 1}, Int3{4, 8, 1}, Int3{8, 4, 1},
                       Int3{16, 2, 1}, Int3{32, 1, 1}}) {
    REQUIRE(cost_for(n, real_g, g, 0, opts).n_mpi_reshapes == 1);
  }
  for (const Int3 g : {Int3{1, 16, 2}, Int3{1, 8, 4}, Int3{1, 4, 8}, Int3{1, 2, 16},
                       Int3{2, 8, 2}}) {
    REQUIRE(cost_for(n, real_g, g, 0, opts).n_mpi_reshapes == 2);
  }
  const Int3 selected = select_for(n, real_g, 0, opts);
  REQUIRE(selected == Int3{2, 16, 1});
  REQUIRE(cost_for(n, real_g, selected, 0, opts).n_mpi_reshapes == 1);
}

TEST_CASE("min_reshape keeps already-good even y-slabs",
          "[fft][layout][unit][complex_outbox]") {
  const auto opts = production_slab_r2c_options();
  REQUIRE(select_for({960, 960, 960}, {1, 1, 16}, 0, opts) == Int3{1, 16, 1});
  REQUIRE(select_for({1152, 1152, 1152}, {1, 1, 32}, 0, opts) == Int3{1, 32, 1});
  REQUIRE(select_for({1216, 1216, 1216}, {1, 1, 32}, 0, opts) == Int3{1, 32, 1});
  REQUIRE(select_for({1536, 1536, 1536}, {1, 1, 64}, 0, opts) == Int3{1, 64, 1});
  REQUIRE(select_for({1920, 1920, 1920}, {1, 1, 128}, 0, opts) == Int3{1, 128, 1});
}

TEST_CASE("reshape_algorithm and gpu_aware do not change the logic plan",
          "[fft][layout][unit][complex_outbox][plan_options]") {
  const Int3 n{1200, 1200, 1152};
  const Int3 real_g{1, 1, 32};
  heffte::plan_options a(true, heffte::reshape_algorithm::alltoallv, false);
  heffte::plan_options b(true, heffte::reshape_algorithm::p2p_plined, false);
  b.use_gpu_aware = !a.use_gpu_aware;
  const auto ca = cost_for(n, real_g, Int3{2, 16, 1}, 0, a);
  const auto cb = cost_for(n, real_g, Int3{2, 16, 1}, 0, b);
  REQUIRE(ca.n_mpi_reshapes == cb.n_mpi_reshapes);
  REQUIRE(ca.n_local_reshapes == cb.n_local_reshapes);
  REQUIRE(ca.n_identity == cb.n_identity);
  REQUIRE(select_for(n, real_g, 0, a) == select_for(n, real_g, 0, b));
  REQUIRE(select_for(n, real_g, 0, a) == Int3{2, 16, 1});
}

TEST_CASE("use_reorder does not change MPI reshape count for this pair",
          "[fft][layout][unit][complex_outbox][plan_options]") {
  const Int3 n{1200, 1200, 1152};
  const Int3 real_g{1, 1, 32};
  heffte::plan_options reorder(true, heffte::reshape_algorithm::p2p_plined, false);
  heffte::plan_options no_reorder(false, heffte::reshape_algorithm::p2p_plined,
                                  false);
  REQUIRE(cost_for(n, real_g, Int3{1, 1, 32}, 0, reorder).n_mpi_reshapes ==
          cost_for(n, real_g, Int3{1, 1, 32}, 0, no_reorder).n_mpi_reshapes);
  REQUIRE(select_for(n, real_g, 0, reorder) == select_for(n, real_g, 0, no_reorder));
}

TEST_CASE("planner uses the caller's use_pencils flag",
          "[fft][layout][unit][complex_outbox][plan_options]") {
  const Int3 n{64, 64, 64};
  const Int3 real_g{2, 2, 2};
  heffte::plan_options slabs(true, heffte::reshape_algorithm::alltoallv, false);
  heffte::plan_options pencils(true, heffte::reshape_algorithm::alltoallv, true);
  const Int3 slab_sel = select_for(n, real_g, 0, slabs);
  const Int3 pencil_sel = select_for(n, real_g, 0, pencils);
  REQUIRE(cost_for(n, real_g, slab_sel, 0, slabs).n_mpi_reshapes <=
          cost_for(n, real_g, real_g, 0, slabs).n_mpi_reshapes);
  REQUIRE(cost_for(n, real_g, pencil_sel, 0, pencils).n_mpi_reshapes <=
          cost_for(n, real_g, real_g, 0, pencils).n_mpi_reshapes);
}

TEST_CASE("FFTW defaults and production slab options agree on known grids",
          "[fft][layout][unit][complex_outbox][plan_options]") {
  const auto fftw = fftw_r2c_options();
  const auto hip = production_slab_r2c_options();
  REQUIRE(select_for({1152, 1152, 1152}, {1, 1, 32}, 0, fftw) == Int3{1, 32, 1});
  REQUIRE(select_for({1152, 1152, 1152}, {1, 1, 32}, 0, hip) == Int3{1, 32, 1});
  REQUIRE(select_for({1200, 1200, 1152}, {1, 1, 32}, 0, fftw) == Int3{2, 16, 1});
  REQUIRE(select_for({1200, 1200, 1152}, {1, 1, 32}, 0, hip) == Int3{2, 16, 1});
}

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
TEST_CASE("rocFFT r2c effective options match the HIP factory path",
          "[fft][layout][unit][complex_outbox][plan_options][hip]") {
  auto opts = heffte::default_options<heffte::backend::rocfft>();
  opts.algorithm = heffte::reshape_algorithm::p2p_plined;
  opts.use_pencils = false;
  const auto effective = effective_r2c_plan_options<heffte::backend::rocfft>(opts);
  REQUIRE(effective.use_reorder == true);
  REQUIRE(select_for({1200, 1200, 1152}, {1, 1, 32}, 0, effective) ==
          Int3{2, 16, 1});
}
#endif

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
TEST_CASE("cuFFT r2c defaults keep use_reorder unless the caller sets it",
          "[fft][layout][unit][complex_outbox][plan_options][cuda]") {
  const auto raw = heffte::default_options<heffte::backend::cufft>();
  const auto effective = effective_r2c_plan_options<heffte::backend::cufft>(raw);
  REQUIRE(effective.use_reorder == raw.use_reorder);
  REQUIRE(select_for({1200, 1200, 1152}, {1, 1, 32}, 0, effective) ==
          Int3{2, 16, 1});
}
#endif

TEST_CASE("selector matrix does not regress n_mpi versus the legacy heuristic",
          "[fft][layout][unit][complex_outbox][matrix]") {
  const auto opts = production_slab_r2c_options();
  struct Case {
    Int3 size;
    Int3 real_grid;
    int r2c;
  };
  const Case cases[] = {
      {{8, 8, 8}, {1, 1, 1}, 0},     {{16, 16, 16}, {1, 1, 2}, 0},
      {{16, 16, 16}, {1, 2, 1}, 0},  {{16, 16, 16}, {2, 1, 1}, 0},
      {{32, 16, 8}, {1, 1, 4}, 0},   {{32, 16, 8}, {1, 1, 4}, 1},
      {{32, 16, 8}, {1, 1, 4}, 2},   {{64, 32, 16}, {1, 2, 2}, 0},
      {{64, 64, 64}, {2, 2, 2}, 0},  {{48, 48, 48}, {1, 1, 8}, 0},
      {{96, 96, 96}, {1, 1, 16}, 0}, {{96, 80, 64}, {1, 1, 8}, 0},
      {{128, 96, 80}, {1, 4, 2}, 0}, {{64, 64, 64}, {1, 8, 1}, 0},
  };
  for (const auto &c : cases) {
    const auto world = r2c_complex_world_box(c.size, c.r2c);
    const Int3 legacy = legacy_complex_proc_grid_for_r2c(c.real_grid, world, c.r2c);
    const Int3 selected = select_for(c.size, c.real_grid, c.r2c, opts);
    const auto sel_cost = cost_for(c.size, c.real_grid, selected, c.r2c, opts);
    const auto old_cost = cost_for(c.size, c.real_grid, legacy, c.r2c, opts);
    REQUIRE(sel_cost.n_mpi_reshapes <= old_cost.n_mpi_reshapes);
    require_exact_cover(world, selected);
  }
}

TEST_CASE("selector fails closed on unplannable worlds",
          "[fft][layout][unit][complex_outbox]") {
  const auto opts = production_slab_r2c_options();
  const auto tiny = r2c_complex_world_box({2, 2, 2}, 0);
  REQUIRE(legal_complex_proc_grids(tiny, 5).empty());
  REQUIRE_THROWS_AS(select_min_distributed_reshape_complex_grid(
                        real_world_box({2, 2, 2}), Int3{1, 1, 5}, tiny, 0, opts),
                    std::invalid_argument);
}

TEST_CASE("OPENPFC_FFT_COMPLEX_PROC_GRID overrides fail closed",
          "[fft][layout][unit][complex_outbox]") {
  EnvGuard guard;
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID") == 0);
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  auto domain = domain::create(Int3{1200, 1200, 1152});
  auto decomp = decomposition::create(domain, Int3{1, 1, 32});
  const auto opts = production_slab_r2c_options();

  {
    auto layout = fft::layout::create(decomp, 0, opts);
    REQUIRE(fft::layout::get_complex_proc_grid(layout) == Int3{2, 16, 1});
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[0] == 301);
    REQUIRE(c0.size[1] == 75);
    REQUIRE(c0.size[2] == 1152);
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_OUTBOX", "legacy", 1) == 0);
  {
    auto layout = fft::layout::create(decomp, 0, opts);
    REQUIRE(fft::layout::get_complex_proc_grid(layout) == Int3{1, 1, 32});
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[1] == 1200);
    REQUIRE(c0.size[2] == 36);
  }

  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "2,16,1", 1) == 0);
  {
    auto layout = fft::layout::create(decomp, 0, opts);
    REQUIRE(fft::layout::get_complex_proc_grid(layout) == Int3{2, 16, 1});
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,32,1", 1) == 0);
  {
    auto layout = fft::layout::create(decomp, 0, opts);
    const auto &c0 = fft::layout::get_complex_box(layout, 0);
    REQUIRE(c0.size[1] == 38);
    REQUIRE(c0.size[2] == 1152);
  }

  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,31,1", 1) == 0);
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0, opts), std::invalid_argument);
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "nope", 1) == 0);
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0, opts), std::invalid_argument);
}

TEST_CASE("OPENPFC_FFT_COMPLEX_OUTBOX rejects unknown modes",
          "[fft][layout][unit][complex_outbox]") {
  EnvGuard guard;
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID") == 0);
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_OUTBOX", "not-a-mode", 1) == 0);
  auto domain = domain::create(Int3{32, 32, 32});
  auto decomp = decomposition::create(domain, Int3{1, 1, 4});
  REQUIRE_THROWS_AS(fft::layout::create(decomp, 0), std::invalid_argument);
}

TEST_CASE("default layout path uses FFTW options and min_reshape",
          "[fft][layout][unit][complex_outbox]") {
  EnvGuard guard;
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_PROC_GRID") == 0);
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  auto d1152 = domain::create(Int3{1152, 1152, 1152});
  auto de1152 = decomposition::create(d1152, Int3{1, 1, 32});
  auto l1152 = fft::layout::create(de1152, 0);
  REQUIRE(fft::layout::get_complex_proc_grid(l1152) == Int3{1, 32, 1});
  const auto &c1152 = fft::layout::get_complex_box(l1152, 0);
  REQUIRE(c1152.size[1] == 36);
  REQUIRE(c1152.size[2] == 1152);
}

TEST_CASE("uneven y-slab outbox forward/backward roundtrip",
          "[fft][MPI][grid][complex_outbox]") {
  EnvGuard guard;
  int nproc = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (nproc != 4) {
    SKIP("needs 4 ranks");
  }
  REQUIRE(unsetenv("OPENPFC_FFT_COMPLEX_OUTBOX") == 0);
  auto domain = domain::create(Int3{12, 6, 16});
  auto decomp = decomposition::create(domain, Int3{1, 1, 4});
  REQUIRE(setenv("OPENPFC_FFT_COMPLEX_PROC_GRID", "1,4,1", 1) == 0);
  auto options = heffte::default_options<heffte::backend::fftw>();
  options.use_pencils = false;
  options.use_reorder = true;
  auto layout = fft::layout::create(decomp, 0, options);
  const auto &c0 = fft::layout::get_complex_box(layout, 0);
  REQUIRE(c0.size[2] == 16);
  REQUIRE(c0.size[1] != 6);

  auto fft = fft::create(layout, rank, options, MPI_COMM_WORLD);
  std::vector<double> in(fft.size_inbox());
  for (std::size_t i = 0; i < in.size(); ++i) {
    in[i] = 0.25 * std::sin(0.1 * static_cast<double>(i + 17 * rank)) + 0.5;
  }
  std::vector<std::complex<double>> hat(fft.size_outbox());
  std::vector<double> out(fft.size_inbox(), 0.0);
  fft.forward(in, hat);
  fft.backward(hat, out);
  for (std::size_t i = 0; i < in.size(); ++i) {
    REQUIRE_THAT(out[i], Catch::Matchers::WithinAbs(in[i], 1.0e-10));
  }
}
