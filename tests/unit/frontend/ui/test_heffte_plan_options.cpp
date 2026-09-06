// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <heffte.h>
#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_heffte.hpp>
#include <openpfc/frontend/ui/spectral_fft_stack_factory.hpp>
#include <stdexcept>
#include <type_traits>

using json = nlohmann::json;
using pfc::ui::from_json;

TEST_CASE("from_json parses HeFFTe num_subranks", "[ui][heffte]") {
  const json config = {{"num_subranks", 2}};
  const auto options = from_json<heffte::plan_options>(config);
  REQUIRE(options.get_subranks() == 2);
}

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

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
TEST_CASE("cuda_spectral_plan_options_from_json overlays plan_options",
          "[ui][heffte][spectral_gpu]") {
  const json settings = {{"plan_options",
                          {{"use_pencils", true},
                           {"use_gpu_aware", true},
                           {"reshape_algorithm", "p2p_plined"}}}};
  const auto opts = pfc::ui::cuda_spectral_plan_options_from_json(settings);
  REQUIRE(opts.use_pencils == true);
  REQUIRE(opts.use_gpu_aware == true);
  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(static_cast<AlgorithmType>(opts.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p_plined));
  const auto via_space =
      pfc::ui::gpu_spectral_plan_options_from_json<pfc::CUDASpace>(settings);
  REQUIRE(via_space.use_pencils == opts.use_pencils);
  REQUIRE(via_space.use_gpu_aware == opts.use_gpu_aware);
  REQUIRE(static_cast<AlgorithmType>(via_space.algorithm) ==
          static_cast<AlgorithmType>(opts.algorithm));
}
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
TEST_CASE("hip_spectral_plan_options_from_json overlays plan_options",
          "[ui][heffte][spectral_gpu]") {
  const json settings = {{"plan_options", {{"use_pencils", true}}}};
  const auto opts = pfc::ui::hip_spectral_plan_options_from_json(settings);
  REQUIRE(opts.use_pencils == true);
}

TEST_CASE("hip_spectral_plan_options_from_json uses slabs off-node",
          "[ui][heffte][spectral_gpu][comm_scale]") {
  const json settings = {
      {"plan_options",
       {{"reshape_algorithm", "p2p_plined"}, {"use_pencils", true}}}};
  const auto one_node = pfc::ui::hip_spectral_plan_options_from_json(settings, 8);
  const auto two_nodes = pfc::ui::hip_spectral_plan_options_from_json(settings, 16);
  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(one_node.use_pencils == true);
  REQUIRE(two_nodes.use_pencils == false);
  REQUIRE(static_cast<AlgorithmType>(two_nodes.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p_plined));
}
#endif

TEST_CASE("apply_heffte_comm_scale keeps pencils on one LUMI-G node",
          "[ui][heffte][comm_scale]") {
  heffte::plan_options opts = heffte::default_options<heffte::backend::fftw>();
  opts.use_pencils = true;
  opts.algorithm = heffte::reshape_algorithm::p2p_plined;
  pfc::ui::apply_heffte_comm_scale(opts, 8);
  REQUIRE(opts.use_pencils == true);
  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(static_cast<AlgorithmType>(opts.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p_plined));
}

TEST_CASE("apply_heffte_comm_scale switches to slabs above 8 ranks",
          "[ui][heffte][comm_scale]") {
  heffte::plan_options opts = heffte::default_options<heffte::backend::fftw>();
  opts.use_pencils = true;
  opts.algorithm = heffte::reshape_algorithm::p2p_plined;
  pfc::ui::apply_heffte_comm_scale(opts, 16);
  REQUIRE(opts.use_pencils == false);
  using AlgorithmType = std::underlying_type_t<heffte::reshape_algorithm>;
  REQUIRE(static_cast<AlgorithmType>(opts.algorithm) ==
          static_cast<AlgorithmType>(heffte::reshape_algorithm::p2p_plined));
}

TEST_CASE("cpu_spectral_plan_options_from_json uses slabs at 16 ranks",
          "[ui][heffte][spectral_cpu][comm_scale]") {
  const json settings = {
      {"plan_options",
       {{"use_pencils", true}, {"reshape_algorithm", "p2p_plined"}}}};
  const auto eight = pfc::ui::cpu_spectral_plan_options_from_json(settings, 8);
  const auto sixteen = pfc::ui::cpu_spectral_plan_options_from_json(settings, 16);
  REQUIRE(eight.use_pencils == true);
  REQUIRE(sixteen.use_pencils == false);
}
