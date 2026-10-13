// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <openpfc/runtime/common/grain_remapping.hpp>
#include <openpfc/runtime/cpu/grain_tracking.hpp>
#include <type_traits>
using namespace pfc::grain;

TEST_CASE("public signed transfer plan stages a UID cycle without mutation",
          "[grain][public]") {
  static_assert(std::is_trivially_copyable_v<Assignment>);
  const Id first = (Id(1) << 60) + 1, second = first + 9;
  const std::vector<Grain> grains{{first, 0, true}, {second, 1, true}};
  const std::vector<Transfer> moves{{first, 0, 1}, {second, 1, 0}};
  const auto plan = prepare_transfer(2, grains, moves);
  REQUIRE(plan.status == TransferStatus::Success);
  REQUIRE(find_assignment(plan.assignments.data(), plan.assignments.size(), first) ==
          0);
  REQUIRE(find_assignment(plan.assignments.data(), plan.assignments.size(),
                          second) == 1);
  REQUIRE(find_assignment(plan.assignments.data(), plan.assignments.size(),
                          first + 1) == 2);
  const std::vector<double> values{-2, 0, -7, -4};
  const std::vector<Id> labels{first, first, second, second};
  std::vector<double> staged(4);
  std::vector<Id> staged_labels(4);
  for (std::size_t cell = 0; cell < 2; ++cell)
    REQUIRE(stage_signed_cell(cell, 2, 2, values.data(), labels.data(),
                              plan.assignments.data(), plan.assignments.size(), -9,
                              staged.data(),
                              staged_labels.data()) == TransferStatus::Success);
  REQUIRE(staged == std::vector<double>{-7, -4, -2, 0});
  REQUIRE(staged_labels == std::vector<Id>{second, second, first, first});
  REQUIRE(values == std::vector<double>{-2, 0, -7, -4});
  REQUIRE(labels == std::vector<Id>{first, first, second, second});
  REQUIRE(plan.grains[0].slot == 1);
  REQUIRE(plan.grains[1].slot == 0);
}
TEST_CASE("public transfer rejects invalid plans and aliasing before publication",
          "[grain][public]") {
  std::vector<Grain> grains{{11, 0, true}, {22, 1, true}};
  auto invalid = prepare_transfer(2, grains, std::vector<Transfer>{{11, 1, 0}});
  REQUIRE(invalid.status != TransferStatus::Success);
  REQUIRE(invalid.assignments.empty());
  REQUIRE(invalid.grains.empty());
  auto plan = prepare_transfer(2, grains, {});
  std::vector<double> values{1, 2};
  const std::vector<Id> labels{11, 22};
  std::vector<Id> scratch_labels(2);
  REQUIRE(stage_signed_cell(0, 1, 2, values.data(), labels.data(),
                            plan.assignments.data(), 2, -1, values.data(),
                            scratch_labels.data()) == TransferStatus::InvalidLayout);
  REQUIRE(values == std::vector<double>{1, 2});
  std::vector<double> scratch(2);
  REQUIRE(stage_signed_cell(1, 1, 2, values.data(), labels.data(),
                            plan.assignments.data(), 2, -1, scratch.data(),
                            scratch_labels.data()) == TransferStatus::InvalidLayout);
}
TEST_CASE("public graph decisions preserve identities and reject malformed input",
          "[grain][public]") {
  const std::vector<Grain> grains{{11, 0, true}, {22, 0, true}};
  const auto graph = make_contact_graph({11, 22}, {{11, 22}});
  const std::vector<double> weights{1, 2};
  const auto decision = remapping::decide(graph, grains, 2, weights);
  REQUIRE(decision.status == remapping::Status::Success);
  REQUIRE_FALSE(decision.moves.empty());
  auto plan = prepare_transfer(2, grains, decision.moves);
  REQUIRE(plan.status == TransferStatus::Success);
  REQUIRE(plan.grains[0].id == 11);
  REQUIRE(plan.grains[1].id == 22);
  REQUIRE(plan.grains[0].slot != plan.grains[1].slot);
  REQUIRE(grains[0].slot == 0);
  REQUIRE(grains[1].slot == 0);
  auto malformed = graph;
  malformed.edges[0].second = 33;
  REQUIRE(remapping::decide(malformed, grains, 2, weights).status ==
          remapping::Status::InvalidInput);
  REQUIRE(remapping::decide(graph, grains, 2, std::vector<double>{1}).status ==
          remapping::Status::InvalidInput);
  REQUIRE(remapping::decide(graph, grains, 2, std::vector<double>{1, -1}).status ==
          remapping::Status::InvalidInput);
  REQUIRE(remapping::decide(make_contact_graph({33}, {}), grains, 2,
                            std::vector<double>{1})
              .status == remapping::Status::UnknownIdentity);
}
TEST_CASE("public CPU tracking facade propagates existing UID support",
          "[grain][public]") {
  const Grid2D grid{3, 1, false, false, Connectivity::Four};
  const std::vector<std::uint8_t> support{1, 1, 1};
  const std::vector<Id> seeds{11, 0, 0};
  const auto result = tracking::reference::propagate(grid, 1, support, seeds);
  REQUIRE(result.complete);
  REQUIRE(result.labels == std::vector<Id>{11, 11, 11});
}

TEST_CASE("public signed staging rejects byte overlap without any write",
          "[grain][public]") {
  auto plan =
      prepare_transfer(2, std::vector<Grain>{{11, 0, true}, {22, 1, true}}, {});
  std::vector<double> values{-99, -98, -2, 0, -7, -4, -97, -96};
  std::vector<Id> labels{91, 92, 11, 11, 22, 22, 93, 94};
  std::vector<double> scratch(4, -123);
  std::vector<Id> scratch_labels(4, 123);
  const auto original_values = values;
  const auto original_labels = labels;
  const auto original_table = plan.assignments;
  const auto original_scratch = scratch;
  const auto original_scratch_labels = scratch_labels;
  double *output = scratch.data();
  Id *output_labels = scratch_labels.data();
  SECTION("forward value overlap") { output = values.data() + 3; }
  SECTION("reverse value overlap") { output = values.data() + 1; }
  SECTION("forward identity overlap") { output_labels = labels.data() + 3; }
  SECTION("reverse identity overlap") { output_labels = labels.data() + 1; }
  SECTION("values overlap accepted identities") {
    output = reinterpret_cast<double *>(labels.data() + 3);
  }
  SECTION("identities overlap accepted values") {
    output_labels = reinterpret_cast<Id *>(values.data() + 3);
  }
  SECTION("values overlap assignment table") {
    output = reinterpret_cast<double *>(plan.assignments.data());
  }
  SECTION("identities overlap assignment table") {
    output_labels = reinterpret_cast<Id *>(plan.assignments.data());
  }
  SECTION("staging buffers overlap") {
    output_labels = reinterpret_cast<Id *>(scratch.data() + 1);
  }
  REQUIRE(stage_signed_cell(0, 2, 2, values.data() + 2, labels.data() + 2,
                            plan.assignments.data(), 2, -9, output,
                            output_labels) == TransferStatus::InvalidLayout);
  REQUIRE(values == original_values);
  REQUIRE(labels == original_labels);
  REQUIRE(std::memcmp(plan.assignments.data(), original_table.data(),
                      original_table.size() * sizeof(Assignment)) == 0);
  REQUIRE(scratch == original_scratch);
  REQUIRE(scratch_labels == original_scratch_labels);
}

TEST_CASE("public signed staging rejects overflow before reading or writing",
          "[grain][public]") {
  std::vector<double> values{-2, 0, -7, -4}, scratch(4, -123);
  std::vector<Id> labels{11, 11, 22, 22}, scratch_labels(4, 123);
  auto plan =
      prepare_transfer(2, std::vector<Grain>{{11, 0, true}, {22, 1, true}}, {});
  const auto original_values = values;
  const auto original_labels = labels;
  const auto original_table = plan.assignments;
  std::size_t cells = 2, count = 2;
  Slot slots = 2;
  const double *input = values.data();
  double *output = scratch.data();
  SECTION("reserved slot count") { slots = unassigned; }
  SECTION("cell count times slots") {
    cells = std::numeric_limits<std::size_t>::max() / 2 + 1;
  }
  SECTION("field byte extent") {
    slots = 1;
    cells = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
  }
  SECTION("assignment byte extent") {
    count = std::numeric_limits<std::size_t>::max() / sizeof(Assignment) + 1;
  }
  SECTION("input end address") {
    input = reinterpret_cast<const double *>(
        std::numeric_limits<std::uintptr_t>::max() - 7);
  }
  SECTION("output end address") {
    output =
        reinterpret_cast<double *>(std::numeric_limits<std::uintptr_t>::max() - 7);
  }
  REQUIRE(stage_signed_cell(0, cells, slots, input, labels.data(),
                            plan.assignments.data(), count, -9, output,
                            scratch_labels.data()) == TransferStatus::InvalidLayout);
  REQUIRE(values == original_values);
  REQUIRE(labels == original_labels);
  REQUIRE(std::memcmp(plan.assignments.data(), original_table.data(),
                      original_table.size() * sizeof(Assignment)) == 0);
  REQUIRE(scratch == std::vector<double>(4, -123));
  REQUIRE(scratch_labels == std::vector<Id>(4, 123));
}

TEST_CASE("public signed staging admits adjacent disjoint buffers",
          "[grain][public]") {
  auto plan = prepare_transfer(2, std::vector<Grain>{{11, 0, true}, {22, 1, true}},
                               std::vector<Transfer>{{11, 0, 1}, {22, 1, 0}});
  std::vector<double> values{-2, 0, -7, -4, -123, -123, -123, -123};
  std::vector<Id> labels{11, 11, 22, 22, 123, 123, 123, 123};
  for (std::size_t cell = 0; cell < 2; ++cell)
    REQUIRE(stage_signed_cell(cell, 2, 2, values.data(), labels.data(),
                              plan.assignments.data(), 2, -9, values.data() + 4,
                              labels.data() + 4) == TransferStatus::Success);
  REQUIRE(values == std::vector<double>{-2, 0, -7, -4, -7, -4, -2, 0});
  REQUIRE(labels == std::vector<Id>{11, 11, 22, 22, 22, 22, 11, 11});
}

TEST_CASE("failed support staging preserves accepted inputs", "[grain][public]") {
  auto plan = prepare_transfer(2, std::vector<Grain>{{11, 0, true}}, {});
  const std::vector<double> values{-2, 0, -7, -4};
  const std::vector<Id> labels{11, 11, 22, 22};
  std::vector<double> scratch(4);
  std::vector<Id> scratch_labels(4);
  REQUIRE(stage_signed_cell(0, 2, 2, values.data(), labels.data(),
                            plan.assignments.data(), 1, -9, scratch.data(),
                            scratch_labels.data()) ==
          TransferStatus::InvalidSupport);
  REQUIRE(values == std::vector<double>{-2, 0, -7, -4});
  REQUIRE(labels == std::vector<Id>{11, 11, 22, 22});
}
