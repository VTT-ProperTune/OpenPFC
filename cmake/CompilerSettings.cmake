# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Compiler flags, warnings, and C++ standard configuration

set(CMAKE_CXX_STANDARD 17)
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

# Enable debug macros and stricter warnings for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(NAN_CHECK_ENABLED)
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
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}" "-extra-arg=-stdlib=libc++")
    else()
        message(WARNING "clang-tidy executable not found. Please install clang-tidy and ensure it is in your system PATH.")
        message(WARNING "Refer to the installation instructions at: https://clang.llvm.org/extra/clang-tidy/")
    endif()
endif()
