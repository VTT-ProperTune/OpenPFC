# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# Unit tests organized by kernel / runtime / frontend
add_subdirectory(kernel)
add_subdirectory(frontend)
add_subdirectory(runtime)
# add_subdirectory(operators)  # Requires diffop API (pfc/diffop.hpp)
