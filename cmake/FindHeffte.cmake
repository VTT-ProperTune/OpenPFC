# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Deprecated: OpenPFC no longer fetches HeFFTe via FetchContent.
# HeFFTe must be installed beforehand; see INSTALL.md and cmake/Dependencies.cmake.

message(FATAL_ERROR
  "cmake/FindHeffte.cmake is deprecated and must not be included.\n"
  "Install HeFFTe and use find_package(Heffte CONFIG) with CMAKE_PREFIX_PATH or Heffte_DIR.\n"
  "See INSTALL.md in the repository root."
)
