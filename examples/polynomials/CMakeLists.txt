# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(polynomial_interpolation interpolation/model.cpp)
target_link_libraries(polynomial_interpolation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_interpolation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_nodal_polynomials nodal-polynomials/model.cpp)
target_link_libraries(polynomial_nodal_polynomials PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_nodal_polynomials PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_differentiation differentiation/model.cpp)
target_link_libraries(polynomial_differentiation PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_differentiation PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_regularity regularity/model.cpp)
target_link_libraries(polynomial_regularity PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_regularity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_integration integration/model.cpp)
target_link_libraries(polynomial_integration PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_integration PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_boundary_values boundary-values/model.cpp)
target_link_libraries(polynomial_boundary_values PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_boundary_values PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(polynomial_nonlinear_boundary nonlinear-boundary/model.cpp)
target_link_libraries(polynomial_nonlinear_boundary PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(polynomial_nonlinear_boundary PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(chebyshev_eigenmodes eigenmodes/model.cpp)
target_link_libraries(chebyshev_eigenmodes PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(chebyshev_eigenmodes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(airy_eigenproblem airy/model.cpp)
target_link_libraries(airy_eigenproblem PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(airy_eigenproblem PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(membrane_eigenmodes membrane/model.cpp)
target_link_libraries(membrane_eigenmodes PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(membrane_eigenmodes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(square_poisson poisson-square/model.cpp)
target_link_libraries(square_poisson PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(square_poisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(square_helmholtz helmholtz-square/model.cpp)
target_link_libraries(square_helmholtz PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(square_helmholtz PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(square_laplace laplace-square/model.cpp)
target_link_libraries(square_laplace PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(square_laplace PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(clamped_beam clamped-beam/model.cpp)
target_link_libraries(clamped_beam PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(clamped_beam PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(clamped_plate clamped-plate/model.cpp)
target_link_libraries(clamped_plate PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(clamped_plate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(davies_pseudospectra pseudospectra/model.cpp)
target_link_libraries(davies_pseudospectra PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(davies_pseudospectra PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(orr_sommerfeld orr-sommerfeld/model.cpp)
target_link_libraries(orr_sommerfeld PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(orr_sommerfeld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(disk_eigenmodes disk-modes/model.cpp)
target_link_libraries(disk_eigenmodes PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(disk_eigenmodes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(disk_poisson poisson-disk/model.cpp)
target_link_libraries(disk_poisson PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(disk_poisson PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(allen_cahn_fixed allen-cahn-fixed/model.cpp)
target_link_libraries(allen_cahn_fixed PRIVATE OpenPFC::openpfc nlohmann_json::nlohmann_json)
set_target_properties(allen_cahn_fixed PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
