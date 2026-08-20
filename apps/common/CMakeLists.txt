# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Header-only helpers shared by FD apps and PFC solidification BCs.
add_library(openpfc_apps_common INTERFACE)
target_include_directories(openpfc_apps_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(openpfc_apps_common INTERFACE OpenPFC)
