# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compiler flags, warnings, and C++ standard configuration

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Base compiler flags for all build types
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -Werror=format-security")

# Additional warnings for better code quality
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wnull-dereference    # Warn about potential null pointer dereferences
  )
endif()

# GCC-specific warnings (not supported by Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(
    -Wduplicated-cond     # Warn about duplicated conditions
    -Wduplicated-branches # Warn about duplicated branches
    -Wlogical-op          # Warn about logical operator issues
  )
endif()

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
    add_compile_definitions(NAN_CHECK_ENABLED)
    set(OpenPFC_NAN_CHECK_ACTIVE ON)
endif()

# Enable debug macros and stricter warnings for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Additional warning flags for Debug mode to catch more issues
    # Note: We enable useful warnings but avoid overly pedantic ones
    add_compile_options(
        -Wunused                # Warn on unused entities (variables, functions, etc.)
        -Wshadow                # Warn on variable shadowing
        # Disabled overly pedantic warnings:
        # -Wconversion          # Too many false positives with int/size_t conversions
        # -Wsign-conversion     # Too pedantic for array indexing with int
        # -Wmissing-braces      # C++ aggregate initialization is fine without double braces
        # -Wpedantic            # Enforces strict ISO C++, sometimes too restrictive
        # -Wfloat-equal         # Sometimes we do need exact floating point comparisons
        # Future consideration:
        # -Werror               # Treat all warnings as errors (good for CI/CD)
    )
endif()

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
