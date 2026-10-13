# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(StartupConsumer LANGUAGES C CXX)
find_package(OpenPFC CONFIG REQUIRED)
add_executable(startup_consumer main.cpp)
target_link_libraries(startup_consumer PRIVATE OpenPFC::openpfc)
# No direct CUDA/HIP target: this is an ordinary host MPI application.
enable_testing()
add_test(NAME startup_consumer COMMAND startup_consumer)
