# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
find_package(OpenSSL 1.1 REQUIRED COMPONENTS Crypto)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/catalog.json" OPENPFC_CATALOG_JSON)
configure_file(catalog.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/catalog.hpp" @ONLY)
add_executable(openpfc_cli main.cpp support.cpp cases.cpp model.cpp help.cpp starter.cpp
  elf.cpp bundle.cpp)
target_compile_features(openpfc_cli PRIVATE cxx_std_20)
target_include_directories(openpfc_cli PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(openpfc_cli PRIVATE OpenSSL::Crypto nlohmann_json::nlohmann_json)
target_compile_definitions(openpfc_cli PRIVATE
  "OPENPFC_CLI_VERSION=\"${PROJECT_VERSION}${OpenPFC_VERSION_SUFFIX}\""
  "OPENPFC_CLI_SOURCE_HINT=\"${CMAKE_SOURCE_DIR}\"")
set_target_properties(openpfc_cli PROPERTIES OUTPUT_NAME openpfc
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${config}" upper)
  set_target_properties(openpfc_cli PROPERTIES
    "RUNTIME_OUTPUT_DIRECTORY_${upper}" "${CMAKE_BINARY_DIR}/bin")
endforeach()
configure_file(model.cmake "${CMAKE_BINARY_DIR}/share/openpfc/templates/model.cmake" COPYONLY)
install(TARGETS openpfc_cli RUNTIME DESTINATION bin)
install(FILES model.cmake DESTINATION share/openpfc/templates)
foreach(name model.cpp input.json README.md input.json.license)
  configure_file("templates/diffusion/${name}"
    "${CMAKE_BINARY_DIR}/share/openpfc/templates/diffusion/${name}" COPYONLY)
  install(FILES "templates/diffusion/${name}" DESTINATION share/openpfc/templates/diffusion)
endforeach()
