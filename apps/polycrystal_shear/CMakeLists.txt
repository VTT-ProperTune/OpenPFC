# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

set(_shear_heffte_tgt "")
if(TARGET Heffte::Heffte)
  set(_shear_heffte_tgt Heffte::Heffte)
elseif(TARGET Heffte)
  set(_shear_heffte_tgt Heffte)
elseif(TARGET heffte)
  set(_shear_heffte_tgt heffte)
endif()

add_executable(polycrystal_shear src/polycrystal_shear.cpp)
target_link_libraries(polycrystal_shear PRIVATE OpenPFC ${_shear_heffte_tgt})
# AVX2 and FMA, which Skylake, Zen 3, and Emerald Rapids all implement.
target_compile_options(polycrystal_shear PRIVATE -march=x86-64-v3)
install(TARGETS polycrystal_shear DESTINATION bin)

if(OpenPFC_BUILD_TESTS)
  # The driver calls MPI_Init itself. Do not wrap it with srun.
  add_test(NAME polycrystal-shear-smoke
           COMMAND $<TARGET_FILE:polycrystal_shear>
                   --size 4
                   --steps 2
                   --shear 0.002
                   --outdir ${CMAKE_CURRENT_BINARY_DIR}/smoke
                   --commit test)
  set_tests_properties(polycrystal-shear-smoke PROPERTIES
                       PASS_REGULAR_EXPRESSION
                       "polycrystal_shear status=ok grid=4x4x4 steps_completed=2"
                       TIMEOUT 180)
endif()
