// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "test_helpers.hpp"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/fcc_batch.hpp>
#include <vector>
using namespace pfc::finite_strain;
#if defined(OPENPFC_TEST_FCC_HIP)
using Tag = pfc::backend::HIPTag;
#else
using Tag = pfc::backend::CUDATag;
#endif
template <class T> using Buffer = pfc::core::DataBuffer<Tag, T>;
void batch(FccSlipParameters p, FccBatchView v) {
#if defined(OPENPFC_TEST_FCC_HIP)
  evaluate_fcc_batch_hip(p, v);
#else
  evaluate_fcc_batch_cuda(p, v);
#endif
}
FccBatchFailure check(const Buffer<FccStatus> &s, Buffer<unsigned long long> &w) {
#if defined(OPENPFC_TEST_FCC_HIP)
  return check_fcc_batch_hip(s.data(), s.size(), w.data());
#else
  return check_fcc_batch_cuda(s.data(), s.size(), w.data());
#endif
}
bool available() {
#if defined(OPENPFC_TEST_FCC_HIP)
  return pfc::gpu::test::is_hip_available();
#else
  return pfc::gpu::test::is_cuda_available();
#endif
}
TEST_CASE("FCC device batches preserve stress tangent and transactional history",
          "[constitutive][fcc][device]") {
  if (!available()) SKIP("GPU unavailable");
  FccSlipParameters p{2, 1, .5, .001, 5, 2.25, .1, .02, .2, .05, 1, 1.4};
  constexpr int n = 37;
  Buffer<double> q(9 * n), f(9 * n), committed(42 * n), trial(42 * n), stress(9 * n),
      tangent(81 * n), tau(12 * n), rate(12 * n);
  Buffer<FccStatus> status(n);
  Buffer<unsigned long long> workspace(1);
  std::vector<Tensor2> orientation(n);
  std::vector<FccSlipState> states(n, fcc_identity_state(p));
  std::vector<double> host_q(9 * n), host_f(9 * n), history(42 * n);
  for (int i = 0; i < n; ++i) {
    auto &r = orientation[i];
    const double a = .07 * i;
    r = identity2();
    r(0, 0) = r(1, 1) = std::cos(a);
    r(0, 1) = -std::sin(a);
    r(1, 0) = std::sin(a);
    for (int c = 0; c < 9; ++c) host_q[c * n + i] = r(c / 3, c % 3);
    for (int c = 0; c < 9; ++c) history[c * n + i] = states[i].plastic(c / 3, c % 3);
    for (int c = 0; c < 12; ++c) history[(18 + c) * n + i] = states[i].resistance[c];
  }
  q.copy_from_host(host_q);
  committed.copy_from_host(history);
  FccBatchView v{n,
                 q.data(),
                 f.data(),
                 committed.data(),
                 trial.data(),
                 stress.data(),
                 tangent.data(),
                 status.data(),
                 tau.data(),
                 rate.data()};
  for (double shear : {.001, .025, .05, .10, .15, .10, .05, 0.}) {
    std::vector<FccEvaluation> oracle;
    for (int i = 0; i < n; ++i) {
      auto deformation = identity2();
      deformation(0, 1) = shear;
      for (int c = 0; c < 9; ++c) host_f[c * n + i] = deformation(c / 3, c % 3);
      oracle.push_back(evaluate_fcc(p, orientation[i], states[i], deformation));
    }
    f.copy_from_host(host_f);
    batch(p, v);
    REQUIRE(check(status, workspace).status == FccStatus::Success);
    auto s = status.to_host();
    auto t = trial.to_host(), k = tangent.to_host();
    auto sig = stress.to_host(), tr = tau.to_host(), sr = rate.to_host();
    REQUIRE(committed.to_host() == history); // Krylov/trials cannot commit.
    for (int i = 0; i < n; ++i) {
      REQUIRE(s[i] == FccStatus::Success);
      for (int c = 0; c < 9; ++c) {
        REQUIRE(std::abs(sig[c * n + i] - oracle[i].update.piola(c / 3, c % 3)) <
                2e-10);
        REQUIRE(std::abs(t[c * n + i] -
                         oracle[i].update.state.plastic(c / 3, c % 3)) < 2e-10);
        REQUIRE(std::abs(t[(9 + c) * n + i] -
                         oracle[i].update.state.velocity(c / 3, c % 3)) < 2e-10);
      }
      for (int c = 0; c < 81; ++c)
        REQUIRE(std::abs(k[c * n + i] - oracle[i].tangent(c / 27, (c / 9) % 3,
                                                          (c / 3) % 3, c % 3)) <
                2e-8);
      for (int c = 0; c < 12; ++c) {
        REQUIRE(std::abs(t[(30 + c) * n + i] - oracle[i].update.state.shear[c]) <
                2e-10);
        REQUIRE(std::abs(t[(18 + c) * n + i] -
                         oracle[i].update.state.resistance[c]) < 2e-10);
        REQUIRE(std::abs(tr[c * n + i] - oracle[i].update.tau[c]) < 2e-10);
        REQUIRE(std::abs(sr[c * n + i] - oracle[i].update.shear_rate[c]) < 2e-10);
      }
      states[i] = oracle[i].update.state;
    }
    // Accept the trial. A rejected trial simply leaves committed untouched.
    history = t;
    committed.copy_from_host(history);
  }
  for (auto expected :
       {FccStatus::Parameters, FccStatus::Orientation, FccStatus::RateRange}) {
    auto bad = p;
    auto hq = host_q, hf = host_f;
    if (expected == FccStatus::Parameters) bad.time_step = 0;
    if (expected == FccStatus::Orientation) hq[0] = 2;
    if (expected == FccStatus::RateRange) hf[n] = 100;
    q.copy_from_host(hq);
    f.copy_from_host(hf);
    batch(bad, v);
    REQUIRE(status.to_host()[0] == expected);
    REQUIRE(check(status, workspace).status == expected);
    REQUIRE(check(status, workspace).point == 0);
    REQUIRE(committed.to_host() == history);
  }
  auto alias = v;
  alias.trial = const_cast<double *>(alias.committed);
  REQUIRE_THROWS_AS(batch(p, alias), std::invalid_argument);

  // Refuse the established large-shear resistance-cap case from a fresh
  // state, not just malformed parameters. Mixed batches may contain both
  // accepted and refused points; failed points cannot publish partial data.
  std::fill(history.begin(), history.end(), 0.);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < 9; ++c) history[c * n + i] = identity2()(c / 3, c % 3);
    for (int c = 0; c < 12; ++c) history[(18 + c) * n + i] = p.initial_resistance;
  }
  auto large = identity2();
  large(0, 1) = .3;
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < 9; ++c) host_f[c * n + i] = large(c / 3, c % 3);
  constexpr double sentinel = -12345.;
  trial.copy_from_host(std::vector<double>(42 * n, sentinel));
  stress.copy_from_host(std::vector<double>(9 * n, sentinel));
  tangent.copy_from_host(std::vector<double>(81 * n, sentinel));
  committed.copy_from_host(history);
  q.copy_from_host(host_q);
  f.copy_from_host(host_f);
  batch(p, v);
  auto failures = status.to_host();
  auto values = trial.to_host();
  auto sig = stress.to_host(), k = tangent.to_host();
  for (int i = 0; i < n; ++i) {
    auto oracle = try_evaluate_fcc(p, orientation[i], fcc_identity_state(p), large);
    REQUIRE(failures[i] == oracle.status);
    if (i == 0) REQUIRE(oracle.status == FccStatus::ResistanceIterations);
    if (oracle.status == FccStatus::Success) continue;
    for (int c = 0; c < 42; ++c) REQUIRE(values[c * n + i] == sentinel);
    for (int c = 0; c < 9; ++c) REQUIRE(sig[c * n + i] == sentinel);
    for (int c = 0; c < 81; ++c) REQUIRE(k[c * n + i] == sentinel);
  }
  REQUIRE(check(status, workspace).status == FccStatus::ResistanceIterations);
  REQUIRE(check(status, workspace).point == 0);
  REQUIRE(committed.to_host() == history);
}
int main(int argc, char **argv) { return Catch::Session().run(argc, argv); }
