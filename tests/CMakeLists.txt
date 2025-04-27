# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top

find_package(Catch2 REQUIRED)
include_directories(${Catch2_INCLUDE_DIRS})  # Add Catch2 include directories

add_executable(openpfc-tests
               runtests.cpp
               test_world.cpp
               test_decomposition.cpp
               test_fft.cpp
               test_arraynd.cpp
               test_discrete_field.cpp
               test_field_modifier.cpp
               test_ic_constant.cpp
               test_model.cpp
               test_multi_index.cpp
               test_simulator.cpp
               test_time.cpp)

target_link_libraries(openpfc-tests PRIVATE OpenPFC Catch2::Catch2)

if(OpenPFC_ENABLE_CODE_COVERAGE)
  target_compile_options(openpfc-tests PUBLIC --coverage)
  target_link_options(openpfc-tests PUBLIC --coverage)
endif()

# Disable automatic test discovery during the build
# Comment out or remove the following line:
# catch_discover_tests(openpfc-tests)

# Optionally, add a custom target to run tests manually
add_custom_target(run-tests COMMAND openpfc-tests)
