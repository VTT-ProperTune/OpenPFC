# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Minimal downstream consumer used by the packaging smoke test (audit 11 / PN):
# proves an installed OpenPFC is usable via find_package(OpenPFC) with all
# transitive dependencies resolved by the installed OpenPFCConfig.cmake.
cmake_minimum_required(VERSION 3.21)
# C is enabled because OpenPFC links MPI::MPI_C (its config find_package(MPI)
# requests the C component); an MPI-based consumer must enable C as well as CXX.
project(openpfc_consumer LANGUAGES C CXX)

find_package(OpenPFC 0.4 CONFIG REQUIRED)

if(NOT OpenPFC_SOURCE_STATE MATCHES "^(clean|dirty|unknown)$")
  message(FATAL_ERROR "Missing or invalid OpenPFC source state")
endif()
if(NOT OpenPFC_SOURCE_REVISION STREQUAL "unknown")
  string(LENGTH "${OpenPFC_SOURCE_REVISION}" revision_length)
  if(NOT revision_length EQUAL 40 OR NOT OpenPFC_SOURCE_REVISION MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Invalid OpenPFC source revision")
  endif()
endif()
if(DEFINED EXPECTED_OPENPFC_REVISION AND
    NOT OpenPFC_SOURCE_REVISION STREQUAL EXPECTED_OPENPFC_REVISION)
  message(FATAL_ERROR "OpenPFC source revision does not match the built source")
endif()
if(DEFINED EXPECTED_OPENPFC_STATE AND
    NOT OpenPFC_SOURCE_STATE STREQUAL EXPECTED_OPENPFC_STATE)
  message(FATAL_ERROR "OpenPFC source state does not match the built source")
endif()

add_executable(consumer main.cpp spectral_state.cpp)
target_link_libraries(consumer PRIVATE OpenPFC::openpfc)
