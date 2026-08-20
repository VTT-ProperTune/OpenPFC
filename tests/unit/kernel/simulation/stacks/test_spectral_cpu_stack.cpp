// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>

using pfc::Int3;
using pfc::Real3;
namespace pfc::sim::stacks {

TEST_CASE("SpectralCPUStack constructs from Domain", "[cpu_stack][stacks][unit]") {
  const Int3 size{32, 32, 32};
  const Real3 origin{0.0, 0.0, 0.0};
  const Real3 spacing{1.0, 1.0, 1.0};
  constexpr int rank = 0;
  constexpr int nproc = 1;

  const auto domain = pfc::domain::create(pfc::GridSize(size),
                                          pfc::PhysicalOrigin(origin),
                                          pfc::GridSpacing(spacing));

  SpectralCPUStack stack(domain, rank, nproc);

  SECTION("geometry stores Domain values") {
    const auto &geom = stack.geometry();
    REQUIRE(geom.size == size);
    REQUIRE(geom.spacing == spacing);
    REQUIRE(geom.origin == origin);
    REQUIRE(geom.periodic == pfc::Bool3{true, true, true});
  }

  SECTION("world() returns valid adapter") {
    const auto &world = stack.world();
    REQUIRE(pfc::world::get_size(world) == size);
    REQUIRE(pfc::world::get_spacing(world) == spacing);
    REQUIRE(pfc::world::get_origin(world) == origin);
  }

  SECTION("rank() and nproc() return correct values") {
    REQUIRE(stack.rank() == rank);
    REQUIRE(stack.nproc() == nproc);
  }

  SECTION("decomposition is valid") {
    const auto &decomp = stack.decomposition();
    const auto local_box = pfc::decomposition::local_box(decomp, rank);
    REQUIRE(local_box.size[0] == size[0]);
    REQUIRE(local_box.size[1] == size[1]);
    REQUIRE(local_box.size[2] == size[2]);
  }

  SECTION("fft() is valid") {
    const auto &fft = stack.fft();
    REQUIRE(fft.size_inbox() > 0);
    REQUIRE(fft.size_inbox() <= static_cast<size_t>(size[0] * size[1] * size[2]));
  }

  SECTION("u() field is valid") {
    const auto &u = stack.u();
    const auto &global_size = u.global_size();
    REQUIRE(global_size[0] == size[0]);
    REQUIRE(global_size[1] == size[1]);
    REQUIRE(global_size[2] == size[2]);
  }
}

TEST_CASE("SpectralCPUStack with non-default Domain parameters",
          "[cpu_stack][stacks][unit]") {
  const Int3 size{48, 48, 32};
  const Real3 origin{-5.0, -5.0, 0.0};
  const Real3 spacing{0.5, 0.5, 1.0};
  constexpr int rank = 0;
  constexpr int nproc = 1;

  const auto domain = pfc::domain::create(pfc::GridSize(size),
                                          pfc::PhysicalOrigin(origin),
                                          pfc::GridSpacing(spacing));

  SpectralCPUStack stack(domain, rank, nproc);

  const auto &geom = stack.geometry();
  REQUIRE(geom.size == size);
  REQUIRE(geom.spacing == spacing);
  REQUIRE(geom.origin == origin);
}

TEST_CASE("SpectralCPUStack with non-periodic Domain", "[cpu_stack][stacks][unit]") {
  const Int3 size{16, 16, 16};
  const Real3 origin{0.0, 0.0, 0.0};
  const Real3 spacing{1.0, 1.0, 1.0};
  const pfc::Bool3 periodic{false, true, false};
  constexpr int rank = 0;
  constexpr int nproc = 1;

  const auto domain = pfc::domain::create(pfc::GridSize(size),
                                          pfc::PhysicalOrigin(origin),
                                          pfc::GridSpacing(spacing),
                                          periodic);

  SpectralCPUStack stack(domain, rank, nproc);

  const auto &geom = stack.geometry();
  REQUIRE(geom.periodic == periodic);

  // Validate world() adapter returns consistent geometry
  const auto &world = stack.world();
  REQUIRE(pfc::world::get_size(world) == size);
  REQUIRE(pfc::world::get_origin(world) == origin);
  REQUIRE(pfc::world::get_periodic(world) == periodic);
}

TEST_CASE("SpectralCPUStack is non-copyable and non-movable", "[cpu_stack][stacks][unit]") {
  const auto domain = pfc::domain::create(Int3{16, 16, 16});
  SpectralCPUStack stack(domain, 0, 1);

  SECTION("cannot copy construct") {
    REQUIRE_FALSE(std::is_copy_constructible_v<SpectralCPUStack>);
  }

  SECTION("cannot copy assign") {
    REQUIRE_FALSE(std::is_copy_assignable_v<SpectralCPUStack>);
  }

  SECTION("cannot move construct") {
    REQUIRE_FALSE(std::is_move_constructible_v<SpectralCPUStack>);
  }

  SECTION("cannot move assign") {
    REQUIRE_FALSE(std::is_move_assignable_v<SpectralCPUStack>);
  }
}

} // namespace pfc::sim::stacks
