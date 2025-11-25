# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

#===============================================================================
# OpenPFC API Documentation Examples
#===============================================================================
#
# This directory contains comprehensive standalone examples demonstrating
# the top 10 most-used OpenPFC APIs.
#
# Each example is:
#   - Standalone (can be compiled independently)
#   - Production-quality (realistic usage patterns)
#   - Well-documented (inline comments, output messages)
#   - Runnable (demonstrates actual execution)
#
# To build these examples:
#   cmake -B build -DBUILD_API_EXAMPLES=ON
#   cmake --build build
#
# To run (example):
#   mpirun -np 4 ./build/docs/api/examples/10_complete_pfc_model
#
#===============================================================================

option(BUILD_API_EXAMPLES "Build API documentation examples" OFF)

if(BUILD_API_EXAMPLES)
    message(STATUS "Building API documentation examples")
    
    #===========================================================================
    # Example 01: World API - Domain geometry and coordinate systems
    #===========================================================================
    add_executable(example_01_world_basics 01_world_basics.cpp)
    target_link_libraries(example_01_world_basics PRIVATE OpenPFC)
    set_target_properties(example_01_world_basics PROPERTIES
        OUTPUT_NAME "01_world_basics"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 02: FFT API - Spectral transforms and k-space operations
    #===========================================================================
    add_executable(example_02_fft_transforms 02_fft_transforms.cpp)
    target_link_libraries(example_02_fft_transforms PRIVATE OpenPFC)
    set_target_properties(example_02_fft_transforms PROPERTIES
        OUTPUT_NAME "02_fft_transforms"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 03: Simulator API - Simulation orchestration
    #===========================================================================
    add_executable(example_03_simulator_workflow 03_simulator_workflow.cpp)
    target_link_libraries(example_03_simulator_workflow PRIVATE OpenPFC)
    set_target_properties(example_03_simulator_workflow PROPERTIES
        OUTPUT_NAME "03_simulator_workflow"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 04: Time API - Time stepping and output scheduling
    #===========================================================================
    add_executable(example_04_time_stepping 04_time_stepping.cpp)
    target_link_libraries(example_04_time_stepping PRIVATE OpenPFC)
    set_target_properties(example_04_time_stepping PROPERTIES
        OUTPUT_NAME "04_time_stepping"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 05: Decomposition API - MPI domain decomposition
    #===========================================================================
    add_executable(example_05_decomposition_parallel 05_decomposition_parallel.cpp)
    target_link_libraries(example_05_decomposition_parallel PRIVATE OpenPFC)
    set_target_properties(example_05_decomposition_parallel PROPERTIES
        OUTPUT_NAME "05_decomposition_parallel"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 06: ResultsWriter API - Parallel output to files
    #===========================================================================
    add_executable(example_06_results_writers 06_results_writers.cpp)
    target_link_libraries(example_06_results_writers PRIVATE OpenPFC)
    set_target_properties(example_06_results_writers PROPERTIES
        OUTPUT_NAME "06_results_writers"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 07: FieldModifier API - Custom initial and boundary conditions
    #===========================================================================
    add_executable(example_07_field_modifiers 07_field_modifiers.cpp)
    target_link_libraries(example_07_field_modifiers PRIVATE OpenPFC)
    set_target_properties(example_07_field_modifiers PROPERTIES
        OUTPUT_NAME "07_field_modifiers"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 08: DiscreteField API - Fields with coordinate mapping
    #===========================================================================
    add_executable(example_08_discrete_field 08_discrete_field.cpp)
    target_link_libraries(example_08_discrete_field PRIVATE OpenPFC)
    set_target_properties(example_08_discrete_field PROPERTIES
        OUTPUT_NAME "08_discrete_field"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 09: Initial Conditions - Built-in IC classes
    #===========================================================================
    add_executable(example_09_initial_conditions 09_initial_conditions.cpp)
    target_link_libraries(example_09_initial_conditions PRIVATE OpenPFC)
    set_target_properties(example_09_initial_conditions PROPERTIES
        OUTPUT_NAME "09_initial_conditions"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Example 10: Complete PFC Model - Full simulation integrating all APIs
    #===========================================================================
    add_executable(example_10_complete_pfc_model 10_complete_pfc_model.cpp)
    target_link_libraries(example_10_complete_pfc_model PRIVATE OpenPFC)
    set_target_properties(example_10_complete_pfc_model PROPERTIES
        OUTPUT_NAME "10_complete_pfc_model"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/api/examples"
    )
    
    #===========================================================================
    # Summary message
    #===========================================================================
    message(STATUS "API examples will be built in: ${CMAKE_BINARY_DIR}/docs/api/examples")
    message(STATUS "To run examples: mpirun -np 4 ./build/docs/api/examples/<example_name>")
    
else()
    message(STATUS "API documentation examples disabled (use -DBUILD_API_EXAMPLES=ON to enable)")
endif()
