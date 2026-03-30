# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Try well-known HeFFTe install prefixes and set Heffte_DIR so find_package(Heffte CONFIG)
# succeeds without manual CMAKE_PREFIX_PATH. See INSTALL.md §3.

function(_openpfc_heffte_config_dir_from_prefix _prefix _out)
  set(${_out} "" PARENT_SCOPE)
  if(NOT IS_DIRECTORY "${_prefix}")
    return()
  endif()
  if(EXISTS "${_prefix}/lib64/cmake/Heffte/HeffteConfig.cmake")
    set(${_out} "${_prefix}/lib64/cmake/Heffte" PARENT_SCOPE)
  elseif(EXISTS "${_prefix}/lib/cmake/Heffte/HeffteConfig.cmake")
    set(${_out} "${_prefix}/lib/cmake/Heffte" PARENT_SCOPE)
  endif()
endfunction()

# Sets ${_out_list} in parent scope to a list of install prefix directories to probe.
function(openpfc_heffte_collect_hint_prefixes _out_list)
  set(_roots "")

  # Spack / EasyBuild: install prefix (contains lib64/cmake/Heffte)
  foreach(_env EBROOTHEFFTE HEFFTE_ROOT)
    if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
      list(APPEND _roots "$ENV{${_env}}")
    endif()
  endforeach()

  if(DEFINED ENV{HOME})
    foreach(_ver IN ITEMS 2.4.1-cpu 2.4.1-cuda 2.4.1-rocm 2.4.1-openpfc-verify 2.4.1)
      list(APPEND _roots "$ENV{HOME}/opt/heffte/${_ver}")
    endforeach()
    file(GLOB _home_glob LIST_DIRECTORIES true "$ENV{HOME}/opt/heffte/*")
    if(_home_glob)
      list(APPEND _roots ${_home_glob})
    endif()
  endif()

  list(APPEND _roots
    "/opt/heffte/2.4.1-cpu"
    "/opt/heffte/2.4.1-cuda"
    "/opt/heffte/2.4.1-rocm"
    "/opt/heffte/2.4.1"
    "/share/apps/heffte/2.4.1-cpu"
    "/share/apps/heffte/2.4.1-cuda"
    "/share/apps/heffte/2.4.1-rocm"
    "/share/apps/heffte/2.4.1"
    "/share/apps/opt/heffte/2.4.1-cpu"
    "/usr/local/heffte/2.4.1-cpu"
  )

  set(${_out_list} "${_roots}" PARENT_SCOPE)
endfunction()

# Call after an initial find_package(Heffte) failed. Sets CACHE Heffte_DIR when a
# HeffteConfig.cmake is found under a known prefix (overrides invalid/stale Heffte_DIR).
function(openpfc_heffte_autodetect_from_hints)
  if(TARGET Heffte::Heffte OR TARGET Heffte OR TARGET heffte)
    return()
  endif()

  if(Heffte_FOUND)
    return()
  endif()

  # Valid user-supplied Heffte_DIR: nothing to do
  if(DEFINED Heffte_DIR AND Heffte_DIR AND EXISTS "${Heffte_DIR}/HeffteConfig.cmake")
    return()
  endif()

  # Some sites set HEFFTE_DIR to the CMake package directory itself
  if(DEFINED ENV{HEFFTE_DIR} AND EXISTS "$ENV{HEFFTE_DIR}/HeffteConfig.cmake")
    set(Heffte_DIR "$ENV{HEFFTE_DIR}" CACHE PATH
        "Directory containing HeffteConfig.cmake (from HEFFTE_DIR environment)" FORCE)
    message(STATUS "OpenPFC: HeFFTe from environment HEFFTE_DIR=${Heffte_DIR}")
    return()
  endif()

  openpfc_heffte_collect_hint_prefixes(_roots)
  set(OPENPFC_HEFFTE_HINT_PATHS "${_roots}" PARENT_SCOPE)

  foreach(_root IN LISTS _roots)
    _openpfc_heffte_config_dir_from_prefix("${_root}" _cfg_dir)
    if(_cfg_dir)
      set(Heffte_DIR "${_cfg_dir}" CACHE PATH
          "Directory containing HeffteConfig.cmake (auto-detected by OpenPFC)" FORCE)
      message(STATUS "OpenPFC: HeFFTe CMake package auto-detected: ${Heffte_DIR}")
      return()
    endif()
  endforeach()
endfunction()
