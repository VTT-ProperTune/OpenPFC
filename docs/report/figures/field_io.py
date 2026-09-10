#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read OpenPFC field output for the applications report figures.

OpenPFC applications write fields in two formats (see
`docs/reference/binary_field_io_spec.md` and
`include/openpfc/frontend/io/vtk_writer.hpp`):

- **VTK ImageData (`.vti`)**: an XML header (whole extent, origin, spacing,
  one `<DataArray>` per field) followed by the payload, either an
  `<AppendedData encoding="raw">` block (what `pfc::VTKWriter` emits: a
  `header_type`-sized byte-length prefix then the raw little-endian
  `Float64` payload) or a base64-encoded appended/inline block (not emitted
  by OpenPFC today, but valid VTI written by other tools, e.g. ParaView).
  Both are supported here so this module keeps working if a writer changes.
- **Raw binary (`.bin`)**: a headerless Fortran-ordered `double` brick
  (`pfc::BinaryWriter` / MPI-IO). The caller must supply the grid shape and
  spacing out of band (there is no metadata in the file).
- **8-bit grayscale PNG**: the *only* field output of the command-line
  applications that never touch the JSON session pipeline (`apps/allen_cahn`,
  `apps/kobayashi`). `pfc::io::write_mpi_scalar_field_png_xy` applies a
  fixed affine map with caller-supplied clip bounds, so the file is an
  invertible (if quantised) record of the field, not merely a picture of it
  -- see `read_gray_png`.

Only single-piece (single-MPI-rank) `.vti` files are handled: the figures in
this report are all rendered from `-n 1` runs. A `.pvti` master plus
per-rank `.vti` pieces would need stitching first; that is not implemented
here because it is not needed by any figure in this report.
"""

from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HEADER_STRUCT = {"UInt32": "<I", "UInt64": "<Q"}
_VTK_DTYPE = {"Float32": "<f4", "Float64": "<f8"}


@dataclass
class Field2D:
    """A 2D scalar field snapshot, ready to plot.

    `data` is indexed `[iy, ix]` (row-major, matplotlib `imshow` convention).
    `extent` is `(xmin, xmax, ymin, ymax)` in physical units for
    `imshow(..., extent=...)`. `time` is the simulation time if known.
    """

    data: np.ndarray
    extent: tuple[float, float, float, float]
    name: str
    time: Optional[float] = None
    units: str = "grid units"


@dataclass
class GridSpec:
    """Out-of-band geometry needed to interpret a headerless `.bin` dump.

    `nx, ny, nz` are grid point counts (as in the run's JSON `domain.Lx`
    etc. -- `Lx` there is a grid point count, not a physical length).
    `dx, dy, dz` are the grid spacings and `origin` is `"corner"` (domain
    starts at 0) or `"center"` (domain is centred on 0), matching the JSON
    session's `domain.origin`.
    """

    nx: int
    ny: int
    nz: int = 1
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    origin: str = "corner"

    def axis_extent(self, n: int, d: float) -> tuple[float, float]:
        length = n * d
        if self.origin == "center":
            return (-0.5 * length, 0.5 * length)
        return (0.0, length)


# --------------------------------------------------------------------------
# .vti reader
# --------------------------------------------------------------------------

_HEADER_TYPE_RE = re.compile(rb'header_type="([^"]+)"')
_WHOLE_EXTENT_RE = re.compile(rb'WholeExtent="([^"]+)"')
_ORIGIN_RE = re.compile(rb'Origin="([^"]+)"')
_SPACING_RE = re.compile(rb'Spacing="([^"]+)"')
_DATA_ARRAY_RE = re.compile(
    rb'<DataArray\s+([^>]*?)/?>(.*?)(?:</DataArray>)?', re.DOTALL
)
_ATTR_RE = re.compile(rb'(\w+)="([^"]*)"')
_APPENDED_DATA_RE = re.compile(rb'<AppendedData\s+encoding="([^"]+)"[^>]*>')


def _parse_attrs(blob: bytes) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in _ATTR_RE.findall(blob)}


def read_vti(path: Path | str, *, time: Optional[float] = None) -> Field2D:
    """Read a single-piece `.vti` (VTK ImageData) scalar field.

    Assumes exactly one `<DataArray>` (one scalar field per file, which is
    how every OpenPFC `fields[]` writer entry is configured in this repo).
    Handles both `format="appended"` (raw or base64 `<AppendedData>`) and a
    fully inline `format="binary"` `<DataArray>` (base64, header + payload
    together). Z is squeezed if `Lz == 1`.
    """
    raw = Path(path).read_bytes()

    m = _HEADER_TYPE_RE.search(raw)
    header_type = m.group(1).decode() if m else "UInt64"
    header_fmt = _HEADER_STRUCT[header_type]
    header_size = struct.calcsize(header_fmt)

    m = _WHOLE_EXTENT_RE.search(raw)
    if not m:
        raise ValueError(f"{path}: no WholeExtent found in VTI header")
    e = [int(v) for v in m.group(1).split()]
    nx, ny, nz = e[1] - e[0] + 1, e[3] - e[2] + 1, e[5] - e[4] + 1

    m = _ORIGIN_RE.search(raw)
    origin = tuple(float(v) for v in m.group(1).split()) if m else (0.0, 0.0, 0.0)
    m = _SPACING_RE.search(raw)
    spacing = tuple(float(v) for v in m.group(1).split()) if m else (1.0, 1.0, 1.0)

    # Locate the <DataArray ...> tag for the (single) field. We search only
    # the header portion (before <AppendedData>, if any) so a raw-binary
    # payload embedded later in the file cannot confuse the regex.
    appended_match = _APPENDED_DATA_RE.search(raw)
    header_end = appended_match.start() if appended_match else len(raw)
    header_blob = raw[:header_end]

    da_match = re.search(rb'<DataArray\s+([^>]*?)/?>', header_blob)
    if not da_match:
        raise ValueError(f"{path}: no <DataArray> found in VTI header")
    attrs = _parse_attrs(da_match.group(1))
    vtk_dtype = np.dtype(_VTK_DTYPE[attrs.get("type", "Float64")])
    field_name = attrs.get("Name", "Field")
    fmt = attrs.get("format", "appended")

    if fmt == "appended":
        if not appended_match:
            raise ValueError(f"{path}: DataArray format=appended but no <AppendedData>")
        encoding = appended_match.group(1).decode()
        body_start = appended_match.end()
        # VTK allows whitespace (typically a newline) between the opening
        # tag and the "_" marker that starts the raw/base64 blob.
        marker_end = body_start
        while raw[marker_end:marker_end + 1].isspace():
            marker_end += 1
        if raw[marker_end:marker_end + 1] == b"_":
            marker_end += 1
        if encoding == "base64":
            # Bounding the text is safe here (unlike the raw case below):
            # "<" never occurs in the base64 alphabet, so this cannot match
            # inside the encoded payload itself.
            close_idx = raw.find(b"</AppendedData>", marker_end)
            text = raw[marker_end:close_idx if close_idx != -1 else len(raw)]
            blob = base64.b64decode(b"".join(text.split()))
        else:
            # Raw binary payload: do not text-search inside it (it could
            # coincidentally contain "</AppendedData>"). The length prefix
            # at `offset` tells us exactly how many payload bytes to take.
            blob = raw[marker_end:]
        offset = int(attrs.get("offset", "0"))
        length, = struct.unpack_from(header_fmt, blob, offset)
        payload = blob[offset + header_size: offset + header_size + length]
    elif fmt in ("binary", "ascii"):
        text = da_match.group(2) if da_match.lastindex and da_match.group(2) else b""
        text = text.strip()
        if fmt == "binary":
            blob = base64.b64decode(text)
            length, = struct.unpack_from(header_fmt, blob, 0)
            payload = blob[header_size: header_size + length]
        else:  # pragma: no cover - ascii VTI not produced by OpenPFC
            values = np.array(text.split(), dtype=float)
            payload = values.astype(vtk_dtype).tobytes()
    else:
        raise ValueError(f"{path}: unsupported DataArray format {fmt!r}")

    values = np.frombuffer(payload, dtype=vtk_dtype)
    # VTK ImageData point order is X-fastest (Fortran / column-major over
    # x,y,z), matching pfc::VTKWriter's write of the local Fortran-ordered
    # field brick.
    grid_xyz = values.reshape((nx, ny, nz), order="F")
    grid = np.transpose(grid_xyz, (2, 1, 0))  # -> [nz, ny, nx]
    if nz == 1:
        grid2d = grid[0]
    else:
        grid2d = grid[nz // 2]  # mid-depth slice, matches slice_bin default

    extent = (
        origin[0],
        origin[0] + nx * spacing[0],
        origin[1],
        origin[1] + ny * spacing[1],
    )
    return Field2D(data=grid2d.astype(np.float64), extent=extent, name=field_name, time=time)


# --------------------------------------------------------------------------
# .bin reader
# --------------------------------------------------------------------------


def read_bin(path: Path | str, grid: GridSpec) -> np.ndarray:
    """Read a headerless `pfc::BinaryWriter` dump as a 3D array `[nz, ny, nx]`.

    The on-disk layout is a single global Fortran-ordered (`x` fastest)
    brick of native `double` (see `docs/reference/binary_field_io_spec.md`);
    no header, no rank splitting for a single-rank run.
    """
    raw = Path(path).read_bytes()
    expected = grid.nx * grid.ny * grid.nz * 8
    if len(raw) != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes for a {grid.nx}x{grid.ny}x{grid.nz} "
            f"double brick, got {len(raw)}"
        )
    values = np.frombuffer(raw, dtype="<f8")
    cube = values.reshape((grid.nx, grid.ny, grid.nz), order="F")
    return np.transpose(cube, (2, 1, 0))  # -> [nz, ny, nx]


def slice_bin(path: Path | str, grid: GridSpec, *, axis: str = "z", index: Optional[int] = None,
              name: str = "Field", time: Optional[float] = None) -> Field2D:
    """Read a `.bin` dump and take a 2D slice, as a `Field2D` ready to plot.

    `axis` is the axis normal to the slice plane (`"x"`, `"y"`, or `"z"`);
    `index` defaults to the mid-plane. This is the "take a slice" path for
    3D fields such as the tungsten PFC density.
    """
    cube = read_bin(path, grid)  # [nz, ny, nx]
    if axis == "z":
        idx = grid.nz // 2 if index is None else index
        plane = cube[idx, :, :]
        ex = grid.axis_extent(grid.nx, grid.dx) + grid.axis_extent(grid.ny, grid.dy)
    elif axis == "y":
        idx = grid.ny // 2 if index is None else index
        plane = cube[:, idx, :]
        ex = grid.axis_extent(grid.nx, grid.dx) + grid.axis_extent(grid.nz, grid.dz)
    elif axis == "x":
        idx = grid.nx // 2 if index is None else index
        plane = cube[:, :, idx]
        ex = grid.axis_extent(grid.ny, grid.dy) + grid.axis_extent(grid.nz, grid.dz)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    return Field2D(data=plane.astype(np.float64), extent=ex, name=name, time=time)


def read_series(paths_and_times: list[tuple[Path, float]], *, reader="vti") -> list[Field2D]:
    """Read a time series of `.vti` files, tagging each with its time."""
    if reader != "vti":
        raise ValueError("read_series only supports the .vti reader; build .bin series by hand")
    return [read_vti(p, time=t) for p, t in paths_and_times]


# --------------------------------------------------------------------------
# grayscale-PNG reader
# --------------------------------------------------------------------------


def read_gray_png(
    path: Path | str,
    *,
    vmin: float,
    vmax: float,
    grid: Optional[GridSpec] = None,
    name: str = "Field",
    time: Optional[float] = None,
) -> Field2D:
    """Recover a scalar field from an 8-bit grayscale PNG written by OpenPFC.

    `apps/allen_cahn` and `apps/kobayashi` are command-line applications that
    do not use the JSON session pipeline, so they emit no `.vti`/`.bin` at
    all: their only field output is `pfc::io::write_mpi_scalar_field_png_xy`
    (`src/openpfc/frontend/io/png_writer.cpp`). That writer applies a *fixed,
    documented* affine map, so the PNG is an invertible record of the run
    rather than a picture of one:

        g = round(255 * clamp((f - vmin) / (vmax - vmin), 0, 1))

    with the clip bounds passed by the application -- `(-1, 1)` for
    Allen-Cahn's `phi` and `(0, 1)` for Kobayashi's. This function inverts
    it, `f = vmin + (vmax - vmin) * g / 255`, so the caller MUST pass the
    same bounds the run used or the recovered values are wrong.

    Two honest limitations, which the figures using this reader state in
    their captions. First, the recovered field is quantised to 1/255 of
    `vmax - vmin` -- negligible for a figure, and small enough that the
    Allen-Cahn superlevel-set areas recovered here come out as the same
    integers the program prints for its own exit criterion. Second, any
    value the run pushed outside `[vmin, vmax]` was *saturated* by the
    writer and cannot be recovered. That one is not hypothetical: Kobayashi
    keeps `phi` in `[0, 1]` by construction, but Allen-Cahn's constant
    driving force shifts both wells, so its `phi` really does reach `+1.15`
    and the grain interior comes back flattened to `+1`. Check the run's own
    printed min/max against the clip bounds before trusting a figure that
    depends on interior values.

    Row order needs no flip: the writer indexes the gathered image as
    `global[gx + gy * nx_glob]` and PNG stores row 0 first, so row `iy` of
    the file is grid row `iy`, which is exactly what `imshow(origin="lower")`
    wants. `grid` supplies physical extent and origin; without it the extent
    is the pixel index range.
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))
    if img.ndim == 3:
        # Pillow hands back RGB(A) for some PNG flavours; the writer emits a
        # single 8-bit channel, so all channels are identical -- take one.
        img = img[..., 0]
    # matplotlib returns float in [0, 1] for 8-bit PNGs; be tolerant of a
    # uint8 array in case a future matplotlib/Pillow changes that.
    if img.dtype == np.uint8:
        level = img.astype(np.float64) / 255.0
    else:
        level = img.astype(np.float64)
    data = vmin + (vmax - vmin) * level

    ny, nx = data.shape
    if grid is None:
        extent = (0.0, float(nx), 0.0, float(ny))
    else:
        if (grid.nx, grid.ny) != (nx, ny):
            raise ValueError(
                f"{path}: PNG is {nx}x{ny} but GridSpec says {grid.nx}x{grid.ny}"
            )
        extent = grid.axis_extent(grid.nx, grid.dx) + grid.axis_extent(grid.ny, grid.dy)
    return Field2D(data=data, extent=extent, name=name, time=time)


def with_spacing(
    field: Field2D,
    *,
    dx: float,
    dy: float = 1.0,
    origin: str = "corner",
) -> Field2D:
    """Restate a `.vti` field's extent in physical units.

    **Why this is needed.** The JSON session pipeline configures its writers
    through `pfc::apply_writer_domain`
    (`include/openpfc/kernel/simulation/results_writer_domain.hpp`), which
    calls only the three-argument `set_domain(global, local, offset)` and
    never `VTKWriter::set_spacing` / `set_origin`. Every `.vti` it emits
    therefore carries `Origin="0 0 0" Spacing="1 1 1"` no matter what the
    run's `domain.dx` and `domain.origin` say, so `read_vti`'s extent is in
    *grid indices*, not physical length. That is invisible for the shipped
    presets with `dx = 1` (Cahn-Hilliard, thin film) and wrong by a factor
    of four for, say, `apps/kawahara`'s `dx = 0.25` solitary-wave inputs,
    whose chapter quotes positions and wavelengths in physical units.

    Rather than silently plotting index units under a physical axis label,
    a figure that needs physical `x` passes the run's own `domain.dx`
    through here. Fixing the writer is the real cure; this keeps the
    figures honest until then.
    """
    ny, nx = field.data.shape  # not derived from `extent`: that is the bug
    grid = GridSpec(nx=nx, ny=ny, dx=dx, dy=dy, origin=origin)
    extent = grid.axis_extent(nx, dx) + grid.axis_extent(ny, dy)
    return Field2D(
        data=field.data, extent=extent, name=field.name, time=field.time,
        units=field.units,
    )
