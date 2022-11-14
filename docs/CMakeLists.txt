set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
set(DOXYGEN_GENERATE_LATEX YES)
set(DOXYGEN_USE_PDFLATEX YES)
set(DOXYGEN_COMPACT_LATEX YES)

doxygen_add_docs(docs
  "${PROJECT_SOURCE_DIR}/README.md"
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
  ALL
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include")
