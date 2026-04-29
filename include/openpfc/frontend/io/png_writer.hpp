// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file png_writer.hpp
 * @brief Grayscale PNG export for quick 2D field visualization (e.g. Allen–Cahn)
 */

#pragma once

#include <mpi.h>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <string>
#include <vector>

namespace pfc::io {

/**
 * @brief Write a row-major WxH grayscale image (8-bit) to a PNG file.
 *
 * @param path Output path (`.png` recommended)
 * @param width Number of columns (x)
 * @param height Number of rows (y)
 * @param row_major One byte per pixel, length width*height, index x + y*width
 */
void write_png_grayscale_8(const std::string &path, int width, int height,
                           const unsigned char *row_major);

/**
 * @brief Map doubles to [0,255] using [vmin, vmax] and write PNG.
 */
void write_png_grayscale_from_doubles(const std::string &path, int width, int height,
                                      const double *row_major, double vmin,
                                      double vmax);

/**
 * @brief Gather a distributed scalar field (one z-layer, nz==1 per rank) onto rank 0
 *        and write a grayscale PNG. No file is written on other ranks.
 *
 * Values are scaled to the global min/max across the gathered field.
 *
 * @throws std::invalid_argument if global nz != 1 or any local patch has nz != 1
 */
void write_mpi_scalar_field_png_xy(MPI_Comm comm,
                                   const pfc::decomposition::Decomposition &decomp,
                                   int rank, const std::vector<double> &local_field,
                                   const std::string &path);

} // namespace pfc::io
