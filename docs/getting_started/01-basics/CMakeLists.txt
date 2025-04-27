# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)
project(OpenPFC-basics)
find_package(OpenPFC REQUIRED)

# a simple "hello world" app to make sure environment is working
add_executable(hello hello.cpp)
target_link_libraries(hello OpenPFC)

# an example how to utilize fft to calculate derivative of a function
add_executable(derivative derivative.cpp)
target_link_libraries(derivative OpenPFC)
