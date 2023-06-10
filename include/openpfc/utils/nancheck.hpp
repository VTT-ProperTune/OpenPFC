#pragma once

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#if defined(NAN_CHECK_ENABLED)
/**
 * @def CHECK_AND_ABORT_IF_NAN(value)
 *
 * Macro for checking if a single float or double value is NaN and aborting
 * the application if NaN is detected.
 *
 * To enable NaN checks, define the preprocessor flag NAN_CHECK_ENABLED
 * during the build process, typically in the CMake build system, for example:
 *
 *     cmake -DNAN_CHECK_ENABLED=ON path/to/source
 *
 * When enabled, the CHECK_AND_ABORT_IF_NAN macro will call the abortIfNaN
 * function to check the provided value for NaN. If NaN is detected, the
 * application will be aborted and an error message will be displayed,
 * indicating the process rank, file name, and line number where the NaN
 * was detected.
 *
 * Note: NaN checks may introduce a performance overhead and are typically
 * used for debugging and validation purposes. It is recommended to enable
 * NaN checks in debug builds and disable them in release builds to optimize
 * performance.
 */
#define CHECK_AND_ABORT_IF_NAN(value) abortIfNaN(value, __FILE__, __LINE__)

/**
 * @def CHECK_AND_ABORT_IF_NANS(vec)
 *
 * Macro for checking if any NaNs are present in a vector of float or double
 * values and aborting the application if NaNs are detected.
 *
 * To enable NaN checks, define the preprocessor flag NAN_CHECK_ENABLED
 * during the build process, typically in the CMake build system, for example:
 *
 *     cmake -DNAN_CHECK_ENABLED=ON path/to/source
 *
 * When enabled, the CHECK_AND_ABORT_IF_NANS macro will call the abortIfNaNs
 * function to check the vector for NaNs. If NaNs are detected, the
 * application will be aborted and an error message will be displayed,
 * indicating the process rank, file name, and line number where the NaNs
 * were detected.
 *
 * Note: NaN checks may introduce a performance overhead and are typically
 * used for debugging and validation purposes. It is recommended to enable
 * NaN checks in debug builds and disable them in release builds to optimize
 * performance.
 */
#define CHECK_AND_ABORT_IF_NANS(vec) abortIfNaNs(vec, __FILE__, __LINE__)
#else
#define CHECK_AND_ABORT_IF_NAN(value)
#define CHECK_AND_ABORT_IF_NANS(vec)
#endif

namespace pfc {
namespace utils {

/**
 * Checks if there are any NaNs in a vector of floats.
 *
 * @param vec The vector of floats to check.
 * @return True if NaNs are found, false otherwise.
 */
template <typename T> bool hasNaNs(const std::vector<T> &vec) {
  for (float value : vec) {
    if (std::isnan(value)) {
      return true;
    }
  }
  return false;
}

template <typename T> void abortIfNaN(T value, const char *filename, int line) {
  if (std::isnan(value)) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "NaN detected on process " << rank << " at " << filename << ":"
              << line << ". Aborting MPI application." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

/**
 * Checks if there are any NaNs in a vector of floats and aborts the MPI
 * application if NaNs are detected. Prints an error message indicating the
 * process rank where NaNs were found before aborting.
 *
 * @param vec The vector of floats to check.
 */
template <typename T>
void abortIfNaNs(const std::vector<T> &vec, const char *filename, int line) {
  if (hasNaNs(vec)) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "NaNs detected on process " << rank << " at " << filename
              << ":" << line << ". Aborting MPI application." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

} // namespace utils
} // namespace pfc
