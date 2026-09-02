# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Kernel simulation layer unit tests
target_sources(openpfc-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_time.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_adaptive_control_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_adaptive_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_steppers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_euler_stepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_imex_euler_stepper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_wave_model_multifield_integration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/state_concepts_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_solver_contract.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_diagonal_solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_state_field_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_physics_concepts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_parameter_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_observable_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_session_selection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_results_writer_domain.cpp
)
if(OpenPFC_ENABLE_HEFFTE)
  target_sources(openpfc-tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/test_stepper_equivalence.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral_etd_system.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_moving_frame_mean_field_etd.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_swift_hohenberg_three_way.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_session.cpp
  )
endif()

# Add wave2d include directory for WaveModel access
target_include_directories(openpfc-tests PRIVATE
    ${CMAKE_SOURCE_DIR}/apps/wave2d/include
)

add_subdirectory(steppers)
add_subdirectory(stacks)
