# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Minimal downstream consumer used by the packaging smoke test (audit 11 / PN):
# proves an installed OpenPFC is usable via find_package(OpenPFC) with all
# transitive dependencies resolved by the installed OpenPFCConfig.cmake.
cmake_minimum_required(VERSION 3.21)
# C is enabled because OpenPFC links MPI::MPI_C (its config find_package(MPI)
# requests the C component); an MPI-based consumer must enable C as well as CXX.
project(openpfc_consumer LANGUAGES C CXX)

find_package(OpenPFC REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE OpenPFC::openpfc)
