// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_state_capture.cpp
 * @brief Catch2 coverage for pfc::checkpoint capture/restore.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <openpfc/kernel/checkpoint/state_capture.hpp>

using pfc::checkpoint::capture_component;
using pfc::checkpoint::capture_field;
using pfc::checkpoint::CoordinateOrder;
using pfc::checkpoint::DecompositionMeta;
using pfc::checkpoint::empty_component_payload;
using pfc::checkpoint::field_expected_nbytes;
using pfc::checkpoint::FieldDtype;
using pfc::checkpoint::FieldPayload;
using pfc::checkpoint::PersistentState;
using pfc::checkpoint::restore_component;
using pfc::checkpoint::restore_field;
using pfc::checkpoint::RestoreError;

namespace {

std::vector<double> make_ramp(std::size_t n, double scale) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = scale * static_cast<double>(i + 1);
  }
  return v;
}

bool all_equal_bytes(std::span<const std::byte> a, std::byte sentinel) {
  for (auto b : a) {
    if (b != sentinel) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST_CASE("scalar field payload round-trip", "[checkpoint][state_capture]") {
  const pfc::types::Int3 extents{2, 3, 4};
  const std::size_t n = 2 * 3 * 4;
  const auto src = make_ramp(n, 1.0);

  const auto payload = capture_field<double>("scalar.u", extents, src);
  REQUIRE(payload.field_id == "scalar.u");
  REQUIRE(payload.dtype == FieldDtype::Float64);
  REQUIRE(payload.extents == extents);
  REQUIRE(payload.coordinate_order == CoordinateOrder::XFastest);
  REQUIRE(payload.version == pfc::checkpoint::kFieldPayloadFormatVersion);
  REQUIRE(payload.bytes.size() == n * sizeof(double));

  std::vector<double> dest(n, 0.0);
  const auto outcome =
      restore_field(payload, "scalar.u", FieldDtype::Float64, extents,
                    std::as_writable_bytes(std::span<double>(dest)));
  REQUIRE(outcome.ok);
  REQUIRE(dest == src);
}

TEST_CASE("two-field PersistentState round-trip", "[checkpoint][state_capture]") {
  const pfc::types::Int3 extents{2, 2, 2};
  const std::size_t n = 8;
  const auto u = make_ramp(n, 1.0);
  const auto v = make_ramp(n, 10.0);

  PersistentState state;
  state.fields.push_back(capture_field<double>("wave2d.u", extents, u));
  state.fields.push_back(capture_field<double>("wave2d.v", extents, v));

  std::vector<double> u_dest(n, 0.0);
  std::vector<double> v_dest(n, 0.0);
  REQUIRE(restore_field(state.fields[0], "wave2d.u", FieldDtype::Float64, extents,
                        std::as_writable_bytes(std::span<double>(u_dest)))
              .ok);
  REQUIRE(restore_field(state.fields[1], "wave2d.v", FieldDtype::Float64, extents,
                        std::as_writable_bytes(std::span<double>(v_dest)))
              .ok);
  REQUIRE(u_dest == u);
  REQUIRE(v_dest == v);
}

TEST_CASE("restore rejects metadata mismatch without mutating destination",
          "[checkpoint][state_capture]") {
  const pfc::types::Int3 extents{2, 2, 1};
  const std::size_t n = 4;
  const auto src = make_ramp(n, 1.0);
  auto payload = capture_field<double>("field.a", extents, src);

  const std::byte sentinel{0x2A};
  std::vector<std::byte> dest(n * sizeof(double), sentinel);

  SECTION("wrong version") {
    payload.version = 99;
    const auto outcome =
        restore_field(payload, "field.a", FieldDtype::Float64, extents, dest);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::VersionMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }

  SECTION("wrong field id") {
    const auto outcome =
        restore_field(payload, "field.other", FieldDtype::Float64, extents, dest);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::FieldIdMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }

  SECTION("wrong extents") {
    const pfc::types::Int3 wrong{3, 2, 1};
    std::vector<std::byte> big(3 * 2 * 1 * sizeof(double), sentinel);
    const auto outcome =
        restore_field(payload, "field.a", FieldDtype::Float64, wrong, big);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::ShapeMismatch);
    REQUIRE(all_equal_bytes(big, sentinel));
  }

  SECTION("mismatched DecompositionMeta") {
    payload.decomposition = DecompositionMeta{
        .rank_count = 2,
        .rank = 0,
        .global_extents = {4, 2, 1},
        .local_extents = extents,
        .local_offset = {0, 0, 0},
    };
    DecompositionMeta expected = *payload.decomposition;
    expected.rank = 1;
    const auto outcome = restore_field(payload, "field.a", FieldDtype::Float64,
                                       extents, dest, expected);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::DecompositionMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }
}

TEST_CASE("restore rejects bytes.size mismatch without mutating destination",
          "[checkpoint][state_capture]") {
  const pfc::types::Int3 extents{2, 2, 2};
  const std::size_t n = 8;
  const auto src = make_ramp(n, 1.0);
  auto payload = capture_field<double>("field.b", extents, src);
  const auto expected = field_expected_nbytes(FieldDtype::Float64, extents);
  REQUIRE(payload.bytes.size() == expected);

  const std::byte sentinel{0x7E};
  std::vector<std::byte> dest(expected, sentinel);

  SECTION("truncated bytes") {
    payload.bytes.resize(expected / 2);
    const auto outcome =
        restore_field(payload, "field.b", FieldDtype::Float64, extents, dest);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::BytesSizeMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }

  SECTION("oversized bytes") {
    payload.bytes.push_back(std::byte{0});
    const auto outcome =
        restore_field(payload, "field.b", FieldDtype::Float64, extents, dest);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::BytesSizeMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }
}

TEST_CASE("empty component round-trip and reject", "[checkpoint][state_capture]") {
  const auto empty = empty_component_payload("euler");
  REQUIRE(empty.component_id == "euler");
  REQUIRE(empty.bytes.empty());

  SECTION("empty restore succeeds with empty dest") {
    std::span<std::byte> dest;
    const auto outcome = restore_component(empty, "euler", 0, dest);
    REQUIRE(outcome.ok);
  }

  SECTION("non-empty bytes vs expected_nbytes==0 rejects without mutate") {
    auto bad = empty;
    bad.bytes.push_back(std::byte{1});
    const std::byte sentinel{0x11};
    std::vector<std::byte> dest(4, sentinel);
    const auto outcome = restore_component(bad, "euler", 0, dest);
    REQUIRE_FALSE(outcome.ok);
    REQUIRE(outcome.error == RestoreError::BytesSizeMismatch);
    REQUIRE(all_equal_bytes(dest, sentinel));
  }

  SECTION("capture_component with bytes round-trips") {
    const std::byte raw[] = {std::byte{1}, std::byte{2}, std::byte{3}};
    const auto payload = capture_component("ctrl", raw);
    std::vector<std::byte> dest(3, std::byte{0});
    REQUIRE(restore_component(payload, "ctrl", 3, dest).ok);
    REQUIRE(dest[0] == std::byte{1});
    REQUIRE(dest[1] == std::byte{2});
    REQUIRE(dest[2] == std::byte{3});
  }
}
