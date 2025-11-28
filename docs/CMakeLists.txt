# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

# ============================================================================
# Core Doxygen Settings
# ============================================================================

# Only document explicitly documented items (encourages better documentation)
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)

# Warnings - encourage good documentation practices
set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_NO_PARAMDOC YES)
set(DOXYGEN_WARN_AS_ERROR NO)  # Don't fail build on warnings, but show them

# Math support - needed for LaTeX formulas in code (e.g., world.hpp)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_RELPATH "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")

# Output formats
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_GENERATE_XML YES)  # For tool integration (e.g., Sphinx with Breathe)
set(DOXYGEN_USE_PDFLATEX YES)
set(DOXYGEN_COMPACT_LATEX YES)

# ============================================================================
# Navigation and UI
# ============================================================================

set(DOXYGEN_AWESOME "${CMAKE_CURRENT_SOURCE_DIR}/doxygen-awesome-css")

set(DOXYGEN_GENERATE_TREEVIEW YES) # optional. Also works without treeview
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_TOC_INCLUDE_HEADINGS 5)

# Custom styling
set(DOXYGEN_HTML_EXTRA_STYLESHEET
  ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-custom/custom.css
  ${DOXYGEN_AWESOME}/doxygen-awesome.css
  ${DOXYGEN_AWESOME}/doxygen-awesome-sidebar-only.css
  ${DOXYGEN_AWESOME}/doxygen-awesome-sidebar-only-darkmode-toggle.css
)

set(DOXYGEN_HTML_EXTRA_FILES
  ${DOXYGEN_AWESOME}/doxygen-awesome-darkmode-toggle.js
  ${DOXYGEN_AWESOME}/doxygen-awesome-fragment-copy-button.js
  ${DOXYGEN_AWESOME}/doxygen-awesome-paragraph-link.js
  ${DOXYGEN_AWESOME}/doxygen-awesome-interactive-toc.js
)

set(DOXYGEN_PROJECT_LOGO ${CMAKE_CURRENT_SOURCE_DIR}/img/logo.png)

# Custom header and footer
set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-custom/header.html)
set(DOXYGEN_HTML_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-custom/footer.html)

# ============================================================================
# Examples Configuration
# ============================================================================

set(DOXYGEN_EXAMPLE_PATH 
  "${PROJECT_SOURCE_DIR}/examples"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/examples"
)
set(DOXYGEN_EXAMPLE_RECURSIVE YES)  # Include examples in subdirectories

# ============================================================================
# Call Graphs (requires Graphviz/dot)
# ============================================================================

set(DOXYGEN_CALL_GRAPH YES)      # Show what functions call
set(DOXYGEN_CALLER_GRAPH YES)    # Show what calls this function

# ============================================================================
# File Collection
# ============================================================================

# NOTE: Using GLOB_RECURSE for documentation file collection.
# This is acceptable here because:
# 1. Documentation generation is not part of the build dependency graph
# 2. Files are collected at configure time, not build time
# 3. The alternative (explicit file lists) would be very maintenance-heavy
#    for a project with many header files
# If you prefer explicit lists for better CMake dependency tracking,
# replace these with explicit file() calls listing each file.

file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE EXAMPLE_FILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
file(GLOB_RECURSE API_EXAMPLE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/api/examples/*.cpp")

# ============================================================================
# Include API examples subdirectory
# ============================================================================

add_subdirectory(api/examples)

# ============================================================================
# Copy images for documentation front page
# ============================================================================

# to make img in front page to show up also in documentation
file(COPY img DESTINATION html/docs)

# ============================================================================
# Generate Documentation
# ============================================================================

doxygen_add_docs(docs
  "${PROJECT_SOURCE_DIR}/README.md"
  "${HEADER_FILES}"
  "${EXAMPLE_FILES}"
  "${API_EXAMPLE_FILES}"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")
