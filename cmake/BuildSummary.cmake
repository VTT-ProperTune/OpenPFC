# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Final build configuration summary

message(STATUS "-------------------------------------------------------------")
message(STATUS " ✅ OpenPFC Build Configuration Summary")
message(STATUS "-------------------------------------------------------------")

message(STATUS " CMake Version                : ${CMAKE_VERSION}")
message(STATUS " C++ Compiler                 : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " C++ Standard                 : ${CMAKE_CXX_STANDARD}")
message(STATUS " Build Type                   : ${CMAKE_BUILD_TYPE}")
message(STATUS " Install RPATH                : ${CMAKE_INSTALL_RPATH_USE_LINK_PATH}")

# development mode or not?
if(OpenPFC_DEVELOPMENT)
  message(STATUS " Development mode             : ON")
else()
  message(STATUS " Development mode             : OFF")
endif()

# shared or static library?
if(BUILD_SHARED_LIBS)
  message(STATUS " Build type                   : SHARED")
else()
  message(STATUS " Build type                   : STATIC")
endif()

# build system: ninja or make?
if(CMAKE_GENERATOR MATCHES "Ninja")
  message(STATUS " Build system                 : Ninja")
else()
  message(STATUS " Build system                 : Make")
endif()

# export compile_commands.json or not?
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  message(STATUS " Export compile_commands.json : YES")
else()
  message(STATUS " Export compile_commands.json : NO")
endif()

message(STATUS "-------------------------------------------------------------")
message(STATUS " 📦 Third-Party Packages:")
message(STATUS " MPI                    : ${MPI_CXX_COMPILER}")
message(STATUS " Heffte_DIR             : ${Heffte_DIR}")
message(STATUS " nlohmann_json          : ${nlohmann_json_DIR}")
if(Doxygen_FOUND)
  message(STATUS " Doxygen                : ${DOXYGEN_EXECUTABLE} (version ${DOXYGEN_VERSION})")
else()
  message(STATUS " Doxygen                : NOT FOUND")
endif()
if(Catch2_FOUND)
  message(STATUS " Catch2                 : ${Catch2_DIR} (version ${Catch2_VERSION})")
else()
  message(STATUS " Catch2                 : NOT FOUND")
endif()

message(STATUS "-------------------------------------------------------------")
message(STATUS " 🛠 Build Options:")
message(STATUS " OpenPFC_DEVELOPMENT              = ${OpenPFC_DEVELOPMENT}")
message(STATUS " OpenPFC_BUILD_APPS               = ${OpenPFC_BUILD_APPS}")
message(STATUS " OpenPFC_BUILD_EXAMPLES           = ${OpenPFC_BUILD_EXAMPLES}")
message(STATUS " OpenPFC_BUILD_TESTS              = ${OpenPFC_BUILD_TESTS}")
message(STATUS " OpenPFC_BUILD_BENCHMARKS         = ${OpenPFC_BUILD_BENCHMARKS}")
message(STATUS " OpenPFC_BUILD_DOCUMENTATION      = ${OpenPFC_BUILD_DOCUMENTATION}")
message(STATUS " OpenPFC_ENABLE_CODE_COVERAGE     = ${OpenPFC_ENABLE_CODE_COVERAGE}")
message(STATUS " USE_CLANG_TIDY                   = ${USE_CLANG_TIDY}")
if(OpenPFC_ENABLE_CUDA AND OpenPFC_CUDA_AVAILABLE)
  message(STATUS " OpenPFC_ENABLE_CUDA             = ON (✅ CUDA available)")
elseif(OpenPFC_ENABLE_CUDA)
  message(STATUS " OpenPFC_ENABLE_CUDA             = ON (⚠️  CUDA not found)")
else()
  message(STATUS " OpenPFC_ENABLE_CUDA             = OFF")
endif()
message(STATUS "-------------------------------------------------------------")
message(STATUS " 📂 Install prefix       : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------------------------------------------")
message(STATUS " 🎉 Ready to build OpenPFC!")
message(STATUS "-------------------------------------------------------------")
message(STATUS "To build OpenPFC, run:")
message(STATUS "  cmake --build build")
message(STATUS "-------------------------------------------------------------")
