# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Inverse homogenization (issue #161). This directory starts with the
# forward periodic homogenizer; the inverse loop is later work on the
# same header in apps/common.

option(INVERSE_HOMOGENIZATION_ENABLE_TESTS
       "Enable building inverse-homogenization smoke tests" ON)

set(_invhom_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_invhom_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_invhom_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_invhom_heffte_tgt heffte)
endif()

add_executable(openpfc_homogenize src/homogenize.cpp)
target_link_libraries(openpfc_homogenize PRIVATE OpenPFC openpfc_apps_common
                      ${_invhom_heffte_tgt})
install(TARGETS openpfc_homogenize DESTINATION bin)

if(INVERSE_HOMOGENIZATION_ENABLE_TESTS AND OpenPFC_BUILD_TESTS)
  include(CTest)
  add_test(NAME inverse-homogenization-homogeneous-smoke
           COMMAND $<TARGET_FILE:openpfc_homogenize>
                   --shape=homogeneous --nx=8 --ny=8 --nz=8 --volume=1)
  set_tests_properties(inverse-homogenization-homogeneous-smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "HOMOGENIZATION_CHECKSUM"
    TIMEOUT 180)
endif()
