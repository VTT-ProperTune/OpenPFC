#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read an OpenPFC HDF5Writer file (`/field`, C-order nz,ny,nx) with h5py."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="HDF5 file written by HDF5Writer")
    parser.add_argument(
        "--expect-shape",
        metavar="NZ,NY,NX",
        help="Require /field.shape == (nz, ny, nx) (C-order)",
    )
    args = parser.parse_args()
    try:
        import h5py
    except ImportError:
        print("h5py is not installed", file=sys.stderr)
        return 2
    with h5py.File(args.path, "r") as handle:
        if "field" not in handle:
            print("missing dataset /field", file=sys.stderr)
            return 1
        data = handle["field"]
        if data.ndim != 3:
            print(f"expected 3D /field, got ndim={data.ndim}", file=sys.stderr)
            return 1
        if args.expect_shape:
            parts = [int(p) for p in args.expect_shape.split(",")]
            if len(parts) != 3 or tuple(data.shape) != tuple(parts):
                print(
                    f"shape mismatch: got {tuple(data.shape)} expected {tuple(parts)}",
                    file=sys.stderr,
                )
                return 1
        print(f"shape={tuple(data.shape)} dtype={data.dtype} sum={data[...].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
