# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Normally the configured C++/CUDA/HIP compiler selects its matching runtime.
# A mixed site toolchain may opt into a specific runtime in a preset/cache.
set(OpenPFC_APP_CXX_RUNTIME "" CACHE FILEPATH
    "Optional C++ runtime library for application links using a mixed toolchain")
if(OpenPFC_APP_CXX_RUNTIME AND NOT EXISTS "${OpenPFC_APP_CXX_RUNTIME}")
  message(FATAL_ERROR "OpenPFC_APP_CXX_RUNTIME does not exist: ${OpenPFC_APP_CXX_RUNTIME}")
endif()

function(openpfc_app_link_cxx_runtime target)
  if(OpenPFC_APP_CXX_RUNTIME)
    target_link_libraries(${target} PRIVATE "${OpenPFC_APP_CXX_RUNTIME}")
    get_filename_component(runtime_dir "${OpenPFC_APP_CXX_RUNTIME}" DIRECTORY)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${runtime_dir}")
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${runtime_dir}")
  endif()
endfunction()

# Each application names exactly one quick CPU test as its smoke check.
# `ctest -L '^smoke$'` is what a shared C++ change runs for every consumer;
# apps/CMakeLists.txt refuses an application that registers tests without
# one (or with several).
function(openpfc_app_smoke_test test)
  if(NOT TEST ${test})
    message(FATAL_ERROR "openpfc_app_smoke_test: no test named ${test}")
  endif()
  set_property(TEST ${test} APPEND PROPERTY LABELS smoke)
  # Applications may call project() themselves; the directory name is the key.
  get_filename_component(_app "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  set_property(GLOBAL APPEND PROPERTY OPENPFC_APP_SMOKE_TESTS "${_app}=${test}")
endfunction()
