# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
cmake_minimum_required(VERSION 3.21)
project(local_fft_consumer LANGUAGES C CXX)
find_package(OpenPFC CONFIG REQUIRED)
add_executable(local_fft_consumer main.cpp)
target_link_libraries(local_fft_consumer PRIVATE OpenPFC::openpfc)
