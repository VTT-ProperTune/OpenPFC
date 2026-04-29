# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(HEAT3D_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(_heat3d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_heat3d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_heat3d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_heat3d_heffte_tgt heffte)
endif()

add_executable(heat3d src/cpu/heat3d.cpp)
target_include_directories(heat3d PRIVATE ${HEAT3D_INCLUDE_DIR})
target_link_libraries(heat3d PRIVATE OpenPFC ${_heat3d_heffte_tgt})
install(TARGETS heat3d DESTINATION bin)
