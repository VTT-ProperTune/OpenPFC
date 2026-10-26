# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(outward_diffusion_flux outward-flux/model.cpp)
target_link_libraries(outward_diffusion_flux PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(outward_diffusion_flux PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

if(OpenPFC_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME example-outward-diffusion-flux
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../render_outward_flux.py"
      --execute $<TARGET_FILE:outward_diffusion_flux>
      --run "${PROJECT_SOURCE_DIR}/results/example-checks/outward-flux")
  set_tests_properties(example-outward-diffusion-flux PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
