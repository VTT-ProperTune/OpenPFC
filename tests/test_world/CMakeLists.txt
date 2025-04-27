# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

add_executable(test_world test_world.cpp)
target_link_libraries(test_world PRIVATE OpenPFC)
add_test(NAME test_world COMMAND test_world)

add_test(NAME 01_hello_world COMMAND 01_hello_world)
