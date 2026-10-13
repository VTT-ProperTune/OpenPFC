// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <cstring>
#include <cuda_runtime.h>
#include <openpfc/kernel/grain/signed_transfer.hpp>
#include <stdexcept>
#define CHECK(expr)                                                                 \
  do {                                                                              \
    if (!(expr)) throw std::runtime_error(#expr);                                   \
  } while (false)
using namespace pfc::grain;
__global__ void stage(const double *values, const Id *labels,
                      const Assignment *table, double *out, Id *ids,
                      unsigned *status) {
  auto i = static_cast<std::size_t>(threadIdx.x);
  status[i] = static_cast<unsigned>(pfc::grain::stage_signed_cell(
      i, 2, 2, values, labels, table, 2, -9, out, ids));
}
// Deliberately invalid layouts exercise the installed public guard on device.
// No invalid pointer is dereferenced: each call must reject before any write.
__global__ void reject_stage(unsigned mode, double *values, Id *labels,
                             Assignment *table, double *out, Id *ids,
                             unsigned *status) {
  std::size_t cells = 2, count = 2;
  Slot slots = 2;
  const double *input = values;
  const Id *input_labels = labels;
  switch (mode) {
  case 0: out = values + 1; break;
  case 1:
    input = values + 1;
    out = values;
    break;
  case 2: ids = labels + 1; break;
  case 3:
    input_labels = labels + 1;
    ids = labels;
    break;
  case 4: out = reinterpret_cast<double *>(labels); break;
  case 5: ids = reinterpret_cast<Id *>(values); break;
  case 6: out = reinterpret_cast<double *>(table); break;
  case 7: ids = reinterpret_cast<Id *>(table); break;
  case 8: ids = reinterpret_cast<Id *>(out + 1); break;
  case 9: cells = std::numeric_limits<std::size_t>::max() / 2 + 1; break;
  case 10:
    slots = 1;
    cells = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
    break;
  case 11:
    count = std::numeric_limits<std::size_t>::max() / sizeof(Assignment) + 1;
    break;
  case 12:
    input = reinterpret_cast<const double *>(
        std::numeric_limits<std::uintptr_t>::max() - 7);
    break;
  case 13:
    out = reinterpret_cast<double *>(std::numeric_limits<std::uintptr_t>::max() - 7);
    break;
  case 14: out = values; break;
  case 15: ids = labels; break;
  case 16: slots = unassigned; break;
  case 17: count = 1; break;
  }
  status[0] = static_cast<unsigned>(stage_signed_cell(
      0, cells, slots, input, input_labels, table, count, -9, out, ids));
}
int main() {
  const Id a = (Id(1) << 60) + 1, b = a + 9;
  auto plan = prepare_transfer(2, std::vector<Grain>{{a, 0, true}, {b, 1, true}},
                               std::vector<Transfer>{{a, 0, 1}, {b, 1, 0}});
  CHECK(plan.status == TransferStatus::Success);
  double host[8]{-2, 0, -7, -4, -91, -92, -93, -94}, result[4];
  Id labels[8]{a, a, b, b, 91, 92, 93, 94}, ids[4];
  unsigned statuses[2];
  double *in, *out;
  Id *lab, *dst;
  Assignment *table;
  unsigned *status;
  CHECK(cudaMalloc(&in, sizeof(host)) == cudaSuccess);
  CHECK(cudaMalloc(&out, sizeof(host)) == cudaSuccess);
  CHECK(cudaMalloc(&lab, sizeof(labels)) == cudaSuccess);
  CHECK(cudaMalloc(&dst, sizeof(labels)) == cudaSuccess);
  CHECK(cudaMalloc(&table, 2 * sizeof(Assignment)) == cudaSuccess);
  CHECK(cudaMalloc(&status, sizeof(statuses)) == cudaSuccess);
  CHECK(cudaMemcpy(in, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaMemcpy(lab, labels, sizeof(labels), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(cudaMemcpy(table, plan.assignments.data(), 2 * sizeof(Assignment),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  stage<<<1, 2>>>(in, lab, table, out, dst, status);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  CHECK(cudaMemcpy(ids, dst, sizeof(ids), cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(cudaMemcpy(statuses, status, sizeof(statuses), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  for (unsigned s : statuses)
    CHECK(s == static_cast<unsigned>(TransferStatus::Success));
  CHECK(result[0] == -7 && result[1] == -4 && result[2] == -2 && result[3] == 0);
  CHECK(ids[0] == b && ids[1] == b && ids[2] == a && ids[3] == a);
  const double sentinel[4]{-101, -102, -103, -104};
  const Id sentinel_ids[4]{101, 102, 103, 104};
  Assignment saved_table[2];
  double saved_input[8];
  Id saved_labels[8];
  for (unsigned mode = 0; mode < 18; ++mode) {
    CHECK(cudaMemcpy(out, sentinel, sizeof(sentinel), cudaMemcpyHostToDevice) ==
          cudaSuccess);
    CHECK(cudaMemcpy(dst, sentinel_ids, sizeof(sentinel_ids),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    reject_stage<<<1, 1>>>(mode, in, lab, table, out, dst, status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(statuses, status, sizeof(unsigned), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    CHECK(statuses[0] == static_cast<unsigned>(mode == 17
                                                   ? TransferStatus::InvalidSupport
                                                   : TransferStatus::InvalidLayout));
    CHECK(cudaMemcpy(saved_input, in, sizeof(host), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    CHECK(cudaMemcpy(saved_labels, lab, sizeof(labels), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    CHECK(cudaMemcpy(saved_table, table, sizeof(saved_table),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(std::memcmp(saved_input, host, sizeof(host)) == 0);
    CHECK(std::memcmp(saved_labels, labels, sizeof(labels)) == 0);
    CHECK(std::memcmp(saved_table, plan.assignments.data(), sizeof(saved_table)) ==
          0);
    CHECK(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    CHECK(cudaMemcpy(ids, dst, sizeof(ids), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (mode != 17) {
      CHECK(std::memcmp(result, sentinel, sizeof(result)) == 0);
      CHECK(std::memcmp(ids, sentinel_ids, sizeof(ids)) == 0);
    }
  }
  // One allocation can hold adjacent, disjoint accepted and staging slices.
  stage<<<1, 2>>>(in, lab, table, in + 4, lab + 4, status);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(statuses, status, sizeof(statuses), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  for (unsigned s : statuses)
    CHECK(s == static_cast<unsigned>(TransferStatus::Success));
  CHECK(cudaMemcpy(saved_input, in, sizeof(host), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  CHECK(cudaMemcpy(saved_labels, lab, sizeof(labels), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  const double expected[8]{-2, 0, -7, -4, -7, -4, -2, 0};
  const Id expected_ids[8]{a, a, b, b, b, b, a, a};
  CHECK(std::memcmp(saved_input, expected, sizeof(expected)) == 0);
  CHECK(std::memcmp(saved_labels, expected_ids, sizeof(expected_ids)) == 0);
  cudaFree(in);
  cudaFree(out);
  cudaFree(lab);
  cudaFree(dst);
  cudaFree(table);
  cudaFree(status);
}
