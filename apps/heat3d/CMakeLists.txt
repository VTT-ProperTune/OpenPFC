# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(_heat3d_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_heat3d_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_heat3d_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_heat3d_heffte_tgt heffte)
endif()

add_executable(heat3d src/cpu/heat3d.cpp)
target_link_libraries(heat3d PRIVATE OpenPFC ${_heat3d_heffte_tgt})
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(heat3d PRIVATE OpenMP::OpenMP_CXX)
endif()
install(TARGETS heat3d DESTINATION bin)
