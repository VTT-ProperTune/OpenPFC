# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

include(${PROJECT_SOURCE_DIR}/cmake/OpenPFCAppRuntime.cmake)

# The tests of apps/<name> carry the CTest label `app:<name>`, so
# `ctest -L '^app:heat3d$'` runs one application and
# `cmake --build build --target apps/heat3d/all` builds it. Every application
# that registers tests also names one quick CPU smoke test with
# openpfc_app_smoke_test(); shared C++ changes run `ctest -L '^smoke$'`.
# See docs/development/testing.md.
function(openpfc_add_app name)
  add_subdirectory(${name})
  set_property(DIRECTORY ${name} PROPERTY LABELS "app:${name}")
  # The smoke test is a host test under OpenPFC_BUILD_TESTS. Device builds
  # register device-only checks outside that guard (vlasov_maxwell's HIP
  # cases) or lose their host test on images without OpenMP (kobayashi), so
  # the requirement is enforced only where it is meaningful: host builds
  # with tests, which is what the pull-request tier of CI configures.
  if(NOT OpenPFC_BUILD_TESTS OR OpenPFC_ENABLE_CUDA OR OpenPFC_ENABLE_HIP)
    return()
  endif()
  get_directory_property(_tests DIRECTORY ${name} TESTS)
  if(NOT _tests)
    return()
  endif()
  get_property(_smokes GLOBAL PROPERTY OPENPFC_APP_SMOKE_TESTS)
  set(_count 0)
  foreach(_entry IN LISTS _smokes)
    if(_entry MATCHES "^${name}=")
      math(EXPR _count "${_count} + 1")
    endif()
  endforeach()
  if(NOT _count EQUAL 1)
    message(FATAL_ERROR "apps/${name} registers tests but ${_count} smoke tests; "
      "name exactly one with openpfc_app_smoke_test(<test>)")
  endif()
endfunction()

if(OpenPFC_ENABLE_HEFFTE)
  openpfc_add_app(tungsten)
  openpfc_add_app(aluminumNew)
  openpfc_add_app(cahn_hilliard)
  openpfc_add_app(thin_film)
  openpfc_add_app(surface_diffusion)
  openpfc_add_app(kawahara)
  openpfc_add_app(ehd_film)
  openpfc_add_app(gradient_elasticity)
  openpfc_add_app(higher_order_pfc)
  openpfc_add_app(inverse_homogenization)
  openpfc_add_app(incompressible_flow)
  openpfc_add_app(polycrystal_shear)
endif()
openpfc_add_app(allen_cahn)
openpfc_add_app(alloy_dendrite_elastic)
openpfc_add_app(heat3d)
openpfc_add_app(wave2d)
openpfc_add_app(kobayashi)
openpfc_add_app(vlasov_maxwell)
openpfc_add_app(burgers)
