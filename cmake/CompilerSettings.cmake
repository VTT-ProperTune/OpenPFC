# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compiler flags, warnings, and C++ standard configuration

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings belong to project targets, never CMAKE_CXX_FLAGS or directory scope:
# FetchContent dependencies keep their own policy, and installed consumers do
# not inherit a PRIVATE development policy. Apply after all subdirectories exist.
# Retain the established Wall/Wextra, format-security and null/logic checks.
# Debug adds unused/shadow/pedantic. Wconversion/Wsign-conversion remain disabled because
# index/size conversions need a separate reviewed migration; Wfloat-equal would
# flag intentional exact comparisons. Pedantic warnings stay local to project
# Debug code; dependencies retain their own policy. No CI gate is changed here.
function(openpfc_project_warnings directory)
  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(kind ${target} TYPE)
    get_target_property(source_dir ${target} SOURCE_DIR)
    file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source_dir}")
    if(kind MATCHES "^(INTERFACE_LIBRARY|UTILITY)$" OR
       relative MATCHES "^[.][.]/" OR
       relative MATCHES "(^|/)(external|third_party|_deps)(/|$)")
      continue()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX,HIP>:-Wall;-Wextra;-Wfatal-errors;-Werror=format-security;-Wnull-dereference>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wduplicated-cond;-Wduplicated-branches;-Wlogical-op>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX,HIP>,$<CONFIG:Debug>>:-Wunused;-Wshadow;-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wfatal-errors,-Werror=format-security,-Wnull-dereference>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=-Wunused,-Wshadow,-Wpedantic>")
    endif()
  endforeach()
  get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS children)
    openpfc_project_warnings("${child}")
  endforeach()
endfunction()

# Preserve build/install interface expressions while marking dependency headers
# as system includes. Resolve aliases before changing the underlying target.
function(openpfc_system_dependency target)
  if(NOT TARGET ${target})
    return()
  endif()
  get_target_property(real_target ${target} ALIASED_TARGET)
  if(NOT real_target)
    set(real_target ${target})
  endif()
  get_target_property(includes ${real_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(includes)
    set_property(TARGET ${real_target} APPEND PROPERTY
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${includes}")
  endif()
endfunction()

function(openpfc_apply_warning_policy)
  foreach(dependency IN ITEMS Catch2::Catch2 Catch2::Catch2WithMain
      nlohmann_json::nlohmann_json tomlplusplus::tomlplusplus
      tomlplusplus_tomlplusplus argparse::argparse argparse
      Heffte::Heffte Heffte heffte)
    openpfc_system_dependency(${dependency})
  endforeach()
  openpfc_project_warnings("${PROJECT_SOURCE_DIR}")
endfunction()

# Build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# NaN checks are enabled automatically in Debug builds and can be forced on for
# other build types with -DOpenPFC_ENABLE_NAN_CHECK=ON.
option(OpenPFC_ENABLE_NAN_CHECK "Enable NaN runtime checks in all build types" OFF)
if(DEFINED NAN_CHECK_ENABLED AND NAN_CHECK_ENABLED)
    message(WARNING "NAN_CHECK_ENABLED is deprecated; use OpenPFC_ENABLE_NAN_CHECK instead.")
    set(OpenPFC_ENABLE_NAN_CHECK ON CACHE BOOL "Enable NaN runtime checks in all build types" FORCE)
endif()

set(OpenPFC_NAN_CHECK_ACTIVE OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR OpenPFC_ENABLE_NAN_CHECK)
    set(OpenPFC_NAN_CHECK_ACTIVE ON)
endif()
# NAN_CHECK_ENABLED is applied as a PUBLIC usage requirement on openpfc in
# LibraryConfiguration.cmake (directory-scope add_compile_definitions is not
# part of the installed INTERFACE).

# Enable clang-tidy if USE_CLANG_TIDY is ON
option(USE_CLANG_TIDY "Enable clang-tidy static code analysis" OFF)

if(USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy")
    if(CLANG_TIDY_EXECUTABLE)
        message(STATUS "Enabling clang-tidy")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
    else()
        message(WARNING "clang-tidy executable not found. Please install clang-tidy and ensure it is in your system PATH.")
        message(WARNING "Refer to the installation instructions at: https://clang.llvm.org/extra/clang-tidy/")
    endif()
endif()

# Optional AddressSanitizer (CI manual job / local debugging). HeFFTe must be
# built with compatible -fsanitize=address flags when linking against a
# sanitized OpenPFC (see scripts/install-heffte-ci.sh and .github/workflows/asan.yml).
option(OpenPFC_ENABLE_ADDRESS_SANITIZER
       "Build with AddressSanitizer (-fsanitize=address); use Debug + matching HeFFTe"
       OFF)
if(OpenPFC_ENABLE_ADDRESS_SANITIZER)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
    message(STATUS "OpenPFC_ENABLE_ADDRESS_SANITIZER: AddressSanitizer enabled")
  else()
    message(
      FATAL_ERROR
        "OpenPFC_ENABLE_ADDRESS_SANITIZER requires GNU or Clang (got ${CMAKE_CXX_COMPILER_ID})"
    )
  endif()
endif()
