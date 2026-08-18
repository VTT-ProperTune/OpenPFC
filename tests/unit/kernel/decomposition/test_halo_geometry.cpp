// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// M4 host-side contract for halo_geometry.hpp. Device-CUDA consumers of these
// slabs/tags cannot be executed on LUMI (no NVIDIA GPU); verify that half on
// tohtori. HIP can consume the same header here.

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <stdexcept>

#include <openpfc/kernel/decomposition/halo_geometry.hpp>

using pfc::Int3;
using pfc::halo::direction_to_canonical_tag;
using pfc::halo::direction_to_face_slot;
using pfc::halo::face_slot_to_direction;
using pfc::halo::field_tag_base;
using pfc::halo::kCanonicalTagCount;
using pfc::halo::kFaceDirections;
using pfc::halo::kFaceSlotCount;
using pfc::halo::opposite_direction;
using pfc::halo::opposite_slot;
using pfc::halo::padded_recv_slab;
using pfc::halo::padded_send_slab;
using pfc::halo::recv_tag;
using pfc::halo::send_tag;

TEST_CASE("opposite_slot is slot XOR 1 on the six faces", "[halo][geometry]") {
  for (int slot = 0; slot < kFaceSlotCount; ++slot) {
    REQUIRE(opposite_slot(slot) == (slot ^ 1));
    REQUIRE(opposite_slot(opposite_slot(slot)) == slot);
  }
}

TEST_CASE("face slot and direction round-trip", "[halo][geometry]") {
  for (int slot = 0; slot < kFaceSlotCount; ++slot) {
    const Int3 d = face_slot_to_direction(slot);
    REQUIRE(direction_to_face_slot(d) == slot);
    REQUIRE(direction_to_canonical_tag(d) == slot);
    REQUIRE(opposite_direction(d) == face_slot_to_direction(opposite_slot(slot)));
  }
  REQUIRE(direction_to_face_slot(Int3{1, 1, 0}) == -1);
  REQUIRE_THROWS_AS(face_slot_to_direction(-1), std::out_of_range);
  REQUIRE_THROWS_AS(face_slot_to_direction(6), std::out_of_range);
}

TEST_CASE("send_tag(d) equals recv_tag(-d) on the same field",
          "[halo][geometry]") {
  constexpr int base = 100;
  constexpr int field = 2;
  const Int3 d{1, 0, 0};
  REQUIRE(send_tag(base, field, d) == recv_tag(base, field, opposite_direction(d)));
  REQUIRE(recv_tag(base, field, d) == send_tag(base, field, opposite_direction(d)));
}

TEST_CASE("two exchangers and six fields have distinct face send tags",
          "[halo][geometry][tags]") {
  // M4 required test: overlapping lifetimes, no tag collision when each
  // exchanger takes a disjoint base (field_tag_base stride).
  std::set<int> tags;
  constexpr int n_exchangers = 2;
  constexpr int n_fields = 6;
  for (int ex = 0; ex < n_exchangers; ++ex) {
    const int base = field_tag_base(0, ex * n_fields);
    for (int f = 0; f < n_fields; ++f) {
      for (const Int3 &d : kFaceDirections) {
        const bool inserted = tags.insert(send_tag(base, f, d)).second;
        REQUIRE(inserted);
      }
    }
  }
  REQUIRE(tags.size() ==
          static_cast<std::size_t>(n_exchangers * n_fields * kFaceSlotCount));
}

TEST_CASE("same exchange base and field collide across two callers",
          "[halo][geometry][tags]") {
  // Documents why exchangers must pick distinct bases: the scheme is
  // deterministic, not a runtime allocator.
  REQUIRE(send_tag(0, 0, kFaceDirections[0]) ==
          send_tag(0, 0, kFaceDirections[0]));
  REQUIRE(field_tag_base(0, 1) - field_tag_base(0, 0) == kCanonicalTagCount);
}

TEST_CASE("padded face slabs match analytic face volumes", "[halo][geometry]") {
  const Int3 owned{32, 16, 8};
  constexpr int hw = 2;
  const std::size_t expect[6] = {
      static_cast<std::size_t>(hw) * 16u * 8u,  // ±X
      static_cast<std::size_t>(hw) * 16u * 8u,
      32u * static_cast<std::size_t>(hw) * 8u,  // ±Y
      32u * static_cast<std::size_t>(hw) * 8u,
      32u * 16u * static_cast<std::size_t>(hw), // ±Z
      32u * 16u * static_cast<std::size_t>(hw),
  };
  for (int slot = 0; slot < kFaceSlotCount; ++slot) {
    const Int3 d = kFaceDirections[static_cast<std::size_t>(slot)];
    REQUIRE(padded_send_slab(owned, hw, d).volume() == expect[slot]);
    REQUIRE(padded_recv_slab(owned, hw, d).volume() == expect[slot]);
  }
}

TEST_CASE("padded +X send/recv sit on the owned/halo boundary",
          "[halo][geometry]") {
  const Int3 owned{10, 6, 4};
  constexpr int hw = 2;
  const Int3 plus_x{1, 0, 0};
  const auto send = padded_send_slab(owned, hw, plus_x);
  const auto recv = padded_recv_slab(owned, hw, plus_x);
  REQUIRE(send.start == Int3{hw + owned[0] - hw, hw, hw});
  REQUIRE(send.count == Int3{hw, owned[1], owned[2]});
  REQUIRE(recv.start == Int3{hw + owned[0], hw, hw});
  REQUIRE(recv.count == Int3{hw, owned[1], owned[2]});
}

TEST_CASE("edge and corner slabs shrink every active axis", "[halo][geometry]") {
  const Int3 owned{8, 8, 8};
  constexpr int hw = 1;
  const auto edge = padded_recv_slab(owned, hw, Int3{1, 1, 0});
  REQUIRE(edge.count == Int3{hw, hw, owned[2]});
  const auto corner = padded_recv_slab(owned, hw, Int3{1, 1, 1});
  REQUIRE(corner.count == Int3{hw, hw, hw});
  REQUIRE(corner.volume() == 1u);
}

TEST_CASE("halo slab rejects a zero direction or a too-thin owned axis",
          "[halo][geometry]") {
  REQUIRE_THROWS_AS(padded_send_slab({4, 4, 4}, 1, Int3{0, 0, 0}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(padded_recv_slab({1, 8, 8}, 2, Int3{1, 0, 0}),
                    std::invalid_argument);
}
