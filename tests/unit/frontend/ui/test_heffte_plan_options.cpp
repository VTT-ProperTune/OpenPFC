// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_heffte.hpp>
#include <openpfc/frontend/ui/spectral_cpu_stack_detail.hpp>
#include <openpfc/frontend/ui/spectral_fft_stack_factory.hpp>
#include <stdexcept>
#include <type_traits>

using json = nlohmann::json;
using pfc::ui::from_json;

TEST_CASE("from_json parses HeFFTe reshape algorithm", "[ui][heffte]") {
  const json config = {{"reshape_algorithm", "p2p"}};

  const auto options = from_json<heffte::plan_options>(config);

  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(static_cast<AlgorithmType>(options.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p));
}

TEST_CASE("from_json rejects unknown HeFFTe reshape algorithm", "[ui][heffte]") {
  const json config = {{"reshape_algorithm", "typo"}};

  REQUIRE_THROWS_AS(from_json<heffte::plan_options>(config), std::invalid_argument);
}

TEST_CASE("CPU spectral plan rejects cuda backend", "[ui][heffte][spectral_cpu]") {
  const json settings = {{"plan_options", {{"backend", "cuda"}}}};
  REQUIRE_THROWS_AS(pfc::ui::cpu_spectral_plan_options_from_json(settings),
                    std::invalid_argument);
}

TEST_CASE("CPU spectral plan merges root backend into plan_options",
          "[ui][heffte][spectral_cpu]") {
  const json settings = {{"backend", "fftw"},
                         {"plan_options", {{"use_pencils", true}}}};
  const auto opts = pfc::ui::cpu_spectral_plan_options_from_json(settings);
  REQUIRE(opts.use_pencils == true);
}

TEST_CASE("merged_spectral_plan_options_json merges root backend",
          "[ui][heffte][spectral]") {
  const json settings = {{"backend", "cuda"},
                         {"plan_options", {{"use_pencils", true}}}};
  const json merged = pfc::ui::merged_spectral_plan_options_json(settings);
  REQUIRE(merged["backend"] == "cuda");
  REQUIRE(merged["use_pencils"] == true);
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("cuda_spectral_plan_options_from_json overlays plan_options",
          "[ui][heffte][spectral_gpu]") {
  const json settings = {{"plan_options", {{"use_pencils", true}}}};
  const auto opts = pfc::ui::cuda_spectral_plan_options_from_json(settings);
  REQUIRE(opts.use_pencils == true);
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
TEST_CASE("hip_spectral_plan_options_from_json overlays plan_options",
          "[ui][heffte][spectral_gpu]") {
  const json settings = {{"plan_options", {{"use_pencils", true}}}};
  const auto opts = pfc::ui::hip_spectral_plan_options_from_json(settings);
  REQUIRE(opts.use_pencils == true);
}
#endif
