# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(vtk_consumer LANGUAGES C CXX)
find_package(OpenPFC 0.5 CONFIG REQUIRED)
add_executable(vtk_consumer main.cpp)
target_link_libraries(vtk_consumer PRIVATE OpenPFC::openpfc)
target_compile_features(vtk_consumer PRIVATE cxx_std_20)
