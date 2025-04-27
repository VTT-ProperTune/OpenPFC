# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

macro(heffte_find_fftw_libraries)
# Usage:
#   heffte_find_fftw_libraries(PREFIX <fftw-root>
#                              VAR <list-name>
#                              REQUIRED <list-names, e.g., "fftw3" "fftw3f">
#                              OPTIONAL <list-names, e.g., "fftw3_threads">)
#  will append the result from find_library() to the <list-name>
#  both REQUIRED and OPTIONAL libraries will be searched
#  if PREFIX is true, then it will be searched exclusively
#                     otherwise standard paths will be used in the search
#  if a library listed in REQUIRED is not found, a FATAL_ERROR will be raised
#
    cmake_parse_arguments(heffte_fftw "" "PREFIX;VAR" "REQUIRED;OPTIONAL" ${ARGN})
    foreach(heffte_lib ${heffte_fftw_REQUIRED} ${heffte_fftw_OPTIONAL})
        if (heffte_fftw_PREFIX)
            find_library(
                heffte_fftw_lib
                NAMES ${heffte_lib}
                PATHS ${heffte_fftw_PREFIX}
                PATH_SUFFIXES lib
                              lib64
                              ${CMAKE_LIBRARY_ARCHITECTURE}/lib
                              ${CMAKE_LIBRARY_ARCHITECTURE}/lib64
                              lib/${CMAKE_LIBRARY_ARCHITECTURE}
                              lib64/${CMAKE_LIBRARY_ARCHITECTURE}
                NO_DEFAULT_PATH
                        )
        else()
            find_library(
                heffte_fftw_lib
                NAMES ${heffte_lib}
            )
        endif()
        message(STATUS "HeFFTe: found FFTW library ${heffte_fftw_lib}")
        if (heffte_fftw_lib)
            list(APPEND ${heffte_fftw_VAR} ${heffte_fftw_lib})
        elseif (${heffte_lib} IN_LIST "${heffte_fftw_REQUIRED}")
            message(FATAL_ERROR "Could not find required fftw3 component: ${heffte_lib}")
        endif()
        unset(heffte_fftw_lib CACHE)
    endforeach()
    unset(heffte_lib)

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_package(OpenMP REQUIRED)
        list(APPEND FFTW_LIBRARIES ${OpenMP_CXX_LIBRARIES})
    else()
        if ("fftw3_omp" IN_LIST FFTW_LIBRARIES)
            list(APPEND FFTW_LIBRARIES "-lgomp")
        endif()
    endif()

endmacro(heffte_find_fftw_libraries)
