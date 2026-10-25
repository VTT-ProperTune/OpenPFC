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
