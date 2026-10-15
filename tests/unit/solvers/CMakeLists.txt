# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_subdirectory(microelasticity)
add_subdirectory(finite_strain_fft)

target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_krylov.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_field_operations.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_newton.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_root.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_multigrid.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_field_multigrid.cpp)
target_sources(openpfc-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_nodal_multigrid.cpp)

add_executable(test_interface_diffusion test_interface_diffusion.cpp)
target_include_directories(test_interface_diffusion PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME InterfaceDiffusion COMMAND test_interface_diffusion)
