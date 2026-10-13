// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <openpfc/runtime/common/grain_remapping.hpp>
#include <openpfc/runtime/cpu/grain_tracking.hpp>
#include <stdexcept>
#define CHECK(expr)                                                                 \
  do {                                                                              \
    if (!(expr)) throw std::runtime_error(#expr);                                   \
  } while (false)
int main() {
  namespace g = pfc::grain;
  std::vector<g::Grain> grains{{11, 0, true}, {22, 0, true}};
  auto graph = g::make_contact_graph({11, 22}, {{11, 22}});
  auto decision = g::remapping::decide(graph, grains, 2, std::vector<double>{1, 2});
  CHECK(decision.status == g::remapping::Status::Success);
  auto plan = g::prepare_transfer(2, grains, decision.moves);
  CHECK(plan.status == g::TransferStatus::Success);
  CHECK(g::find_assignment(plan.assignments.data(), 2, 22) == 1);
  g::Grid2D grid{3, 1, false, false, g::Connectivity::Four};
  auto labels = g::tracking::reference::propagate(
      grid, 1, std::vector<std::uint8_t>{1, 1, 1}, std::vector<g::Id>{11, 0, 0});
  CHECK(labels.complete);
  double values[5]{-2, 0, -7, -4, -99};
  g::Id support[5]{11, 11, 22, 22, 99}, scratch[4]{};
  auto cycle =
      g::prepare_transfer(2, std::vector<g::Grain>{{11, 0, true}, {22, 1, true}},
                          std::vector<g::Transfer>{{11, 0, 1}, {22, 1, 0}});
  CHECK(g::stage_signed_cell(0, 2, 2, values, support, cycle.assignments.data(), 2,
                             -9, values + 1,
                             scratch) == g::TransferStatus::InvalidLayout);
  CHECK(values[0] == -2 && values[1] == 0 && values[2] == -7 && values[3] == -4 &&
        values[4] == -99);
  double output[4]{-1, -1, -1, -1};
  CHECK(g::stage_signed_cell(
            0, std::numeric_limits<std::size_t>::max() / sizeof(double) + 1, 1,
            values, support, cycle.assignments.data(), 2, -9, output,
            scratch) == g::TransferStatus::InvalidLayout);
  CHECK(output[0] == -1 && output[1] == -1 && output[2] == -1 && output[3] == -1);
}
