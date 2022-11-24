set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_USE_PDFLATEX YES)
set(DOXYGEN_COMPACT_LATEX YES)

set(DOXYGEN_AWESOME "${CMAKE_CURRENT_SOURCE_DIR}/doxygen-awesome-css")

set(DOXYGEN_GENERATE_TREEVIEW YES) # optional. Also works without treeview
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
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

set(DOXYGEN_TOC_INCLUDE_HEADINGS 5)

set(DOXYGEN_HTML_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-custom/header.html)

# to make img in front page to show up also in documentation
file(COPY img DESTINATION html/docs)

doxygen_add_docs(docs
  "${PROJECT_SOURCE_DIR}/README.md"
  "${CMAKE_CURRENT_SOURCE_DIR}/tutorial.md"
  pfc/constants.hpp
  pfc/decomposition.hpp
  pfc/fft.hpp
  pfc/field_modifier.hpp
  pfc/model.hpp
  pfc/mpi.hpp
  pfc/results_writer.hpp
  pfc/simulator.hpp
  pfc/time.hpp
  pfc/types.hpp
  pfc/utils.hpp
  pfc/world.hpp
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")
