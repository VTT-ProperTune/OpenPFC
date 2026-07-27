# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.21)

# Support both `add_subdirectory(docs)` from the OpenPFC build and a lightweight
# standalone documentation configure that does not require MPI or HeFFTe.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(OpenPFCDocumentation NONE)
  find_package(Doxygen REQUIRED)
  set(OPENPFC_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
else()
  set(OPENPFC_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
endif()

set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

# The existing public API is only partially documented. Requiring prose for
# every member and parameter would turn the migration into thousands of
# unrelated warnings. Keep warnings for malformed documentation and unresolved
# references; expand curated API coverage incrementally.
set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_NO_PARAMDOC NO)
set(DOXYGEN_WARN_AS_ERROR NO)
set(DOXYGEN_WARN_LOGFILE "${CMAKE_CURRENT_BINARY_DIR}/doxygen.log")
set(DOXYGEN_QUIET YES)
set(DOXYGEN_INCLUDE_PATH "${OPENPFC_SOURCE_DIR}/include")

# Source annotations that are meaningful to the compiler but should disappear
# before Breathe asks Sphinx to parse the resulting C++ declarations.
set(DOXYGEN_PREDEFINED
  "OPENPFC_DEPRECATED_API="
)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_MAX_INITIALIZER_LINES 0)
set(DOXYGEN_EXCLUDE_SYMBOLS
  "*::comm_deleter"
)

# Doxygen is the C++ parser in the unified documentation pipeline. Sphinx and
# Breathe own the rendered HTML product.
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_XML_OUTPUT xml)
set(DOXYGEN_CALL_GRAPH NO)
set(DOXYGEN_CALLER_GRAPH NO)

# The generated API represents the installed source-level contract. Runnable
# examples remain linked from prose but are not parsed as API declarations;
# several intentionally reuse small demo class names across separate programs.
file(GLOB_RECURSE OPENPFC_HEADER_FILES
  "${OPENPFC_SOURCE_DIR}/include/*.hpp"
)

doxygen_add_docs(openpfc-doxygen-xml
  "${OPENPFC_HEADER_FILES}"
  WORKING_DIRECTORY "${OPENPFC_SOURCE_DIR}"
  COMMENT "Extracting OpenPFC public C++ API as Doxygen XML"
)

# Preserve the historical target name for existing local scripts and users.
add_custom_target(docs DEPENDS openpfc-doxygen-xml)

# API examples are only buildable when this directory is part of the full
# OpenPFC project and the required library targets exist.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_subdirectory(api/examples)
endif()
