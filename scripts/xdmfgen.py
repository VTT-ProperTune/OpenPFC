#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate XDMF so OpenPFC Fortran-ordered ``.bin`` bricks open in ParaView.

Headerless MPI-IO dumps need an XDMF sidecar (see
``docs/reference/binary_field_io_spec.md``). Pass a showcase
``*_manifest.json`` with ``--manifest``; File → Open the ``.xdmf`` in
ParaView. The legacy JSON/CLI path still prints XDMF to stdout.
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import re
import sys


header = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
    <Domain>
        <Topology name="topo" TopologyType="3DCoRectMesh" Dimensions="{Lz} {Ly} {Lx}"></Topology>
        <Geometry name="geo" Type="ORIGIN_DXDYDZ">
            <DataItem Format="XML" Dimensions="3">{z0} {y0} {x0}</DataItem>
            <DataItem Format="XML" Dimensions="3">{dz} {dy} {dx}</DataItem>
        </Geometry>

        <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
            <Time TimeType="HyperSlab">
                <DataItem Format="XML" NumberType="Float" Dimensions="{ntimesteps}">{timesteps}</DataItem>
            </Time>
"""

content = """            <Grid Name="{grid_name}" GridType="Uniform"><Topology Reference="/Xdmf/Domain/Topology[1]" /><Geometry Reference="/Xdmf/Domain/Geometry[1]" /><Attribute Name="u" Center="Node"><DataItem Format="Binary" DataType="Float" Precision="8" Endian="Little" Dimensions="{Lz} {Ly} {Lx}">{f}</DataItem></Attribute></Grid>"""

footer = """
        </Grid>
    </Domain>
</Xdmf>"""


def extract_number(s):
    numbers = re.findall(r"\d+", s)
    assert len(numbers) == 1
    return int(numbers[0])


def main_old(args):
    if args["json"] is not None and os.path.exists(args["json"]):
        args.update(json.load(open(args["json"])))
    apply_domain_block(args)
    if args["results_dir"] != "" and not os.path.isdir(args["results_dir"]):
        print("Directory %s does not exist!" % args["results_dir"])
        return
    files = glob.glob(os.path.join(args["results_dir"], "*.bin"))
    if len(files) == 0:
        print("No data files found!")
        return
    files = sorted(files, key=lambda f: extract_number(os.path.basename(f)))
    expected_size = 8 * args["Lx"] * args["Ly"] * args["Lz"]
    if os.path.getsize(files[0]) != expected_size:
        print("File size mismatch, check lx, ly, lz!")
        return
    args["ntimesteps"] = len(files)
    args["timesteps"] = " ".join("%0.3f" % (i*args["saveat"]) for i in range(len(files)))
    print(header.format(**args))
    for f in files:
        print(content.format(**args, grid_name=os.path.splitext(os.path.basename(f))[0], f=os.path.relpath(f)))
    print(footer.format(**args))


def apply_domain_block(args):
    """Take grid size, spacing and origin from a ``domain`` block with
    ``size``/``lower``/``upper``; flat legacy ``Lx``/``dx``/``origo`` keys
    from older inputs are left as they are."""
    d = args.get("domain")
    if not isinstance(d, dict) or "size" not in d:
        return
    for a, n, lo, up in zip("xyz", d["size"], d["lower"], d["upper"]):
        args["L" + a] = n
        args["d" + a] = (up - lo) / n
        args[a + "0"] = lo
    args["origo"] = None


def main(args):
    if args["json"] is not None and os.path.exists(args["json"]):
        args.update(json.load(open(args["json"])))
    apply_domain_block(args)
    files = []
    nfiles = int((args["t1"] - args["t0"])/args["saveat"]) + 1
    nwarnings = 0
    for i in range(nfiles):
        filename = args["results"] % i
        if os.path.exists(filename):
            files.append(filename)

    expected_size = 8 * args["Lx"] * args["Ly"] * args["Lz"]
    if os.path.getsize(files[0]) != expected_size:
        print("File size mismatch, check lx, ly, lz!")
        return
    args["ntimesteps"] = len(files)
    args["timesteps"] = " ".join("%0.3f" % (i*args["saveat"]) for i in range(len(files)))

    if args["origo"] is None:
        pass  # x0/y0/z0 came from the domain block
    elif args["origo"] == "center":
        args["x0"] = -0.5 * args["Lx"] * args["dx"]
        args["y0"] = -0.5 * args["Ly"] * args["dy"]
        args["z0"] = -0.5 * args["Lz"] * args["dz"]
    else:
        args["x0"] = 0.0
        args["y0"] = 0.0
        args["z0"] = 0.0

    print(header.format(**args))
    for f in files:
        print(content.format(**args, grid_name=os.path.splitext(os.path.basename(f))[0], f=os.path.relpath(f)))
    print(footer.format(**args))


def _xml_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_manifest_xdmf(manifest_path, output=None, dx=None, dy=None, dz=None,
                        dvx=None, dvy=None):
    """Write a temporal XDMF collection next to a showcase manifest.

    * Uniform ``nx, ny, nz`` fields share one ``.xdmf`` with one Attribute
      per field.
    * Weibel reduced dumps (``nx``/``nvx``/``nvy``) get one ``.xdmf`` per
      field because the meshes differ.
    """
    manifest_path = os.path.abspath(manifest_path)
    base = os.path.dirname(manifest_path)
    with open(manifest_path) as fh:
        man = json.load(fh)
    fields = list(man.get("fields") or [])
    if not fields:
        raise SystemExit("%s: no fields" % manifest_path)
    pattern = man.get("pattern") or (
        "%s_{field}_{index:04d}.bin" % man.get("run_id", "run")
    )
    times = man.get("times")
    steps = man.get("steps")
    if times is None and steps is None:
        raise SystemExit("%s: need times or steps" % manifest_path)
    n = len(times) if times is not None else len(steps)
    if times is None:
        times = [float(s) for s in steps]

    def brick(field, index):
        patterns = man.get("patterns")
        if isinstance(patterns, list):
            try:
                slot = fields.index(field)
            except ValueError as exc:
                raise SystemExit(
                    "%s: unknown field %s" % (manifest_path, field)
                ) from exc
            if slot >= len(patterns):
                raise SystemExit("%s: patterns shorter than fields" % manifest_path)
            rendered = str(patterns[slot])
            rendered = rendered % index if "%" in rendered else rendered
        else:
            rendered = pattern.format(field=field, index=index)
        path = rendered if os.path.isabs(rendered) else os.path.join(base, rendered)
        if not os.path.isfile(path):
            raise SystemExit("missing %s" % path)
        return os.path.relpath(path, base), os.path.getsize(path)

    # Weibel reduced: different arrays on different meshes.
    if "nvx" in man and "nvy" in man and "nx" in man and "ny" not in man:
        nx = int(man["nx"])
        nvx = int(man["nvx"])
        nvy = int(man["nvy"])
        dx_x = float(dx if dx is not None else man.get("dx", 1.0))
        dvx = float(dvx if dvx is not None else man.get("dvx", 1.0))
        dvy = float(dvy if dvy is not None else man.get("dvy", 1.0))
        written = []
        for field in fields:
            if field == "f_xvx":
                shape = (nx, nvx, 1)
                sp = (dx_x, dvx, 1.0)
                org = (0.0, -0.5 * nvx * dvx, 0.0)
            elif field == "f_xvy":
                shape = (nx, nvy, 1)
                sp = (dx_x, dvy, 1.0)
                org = (0.0, -0.5 * nvy * dvy, 0.0)
            else:
                shape = (nx, 1, 1)
                sp = (dx_x, 1.0, 1.0)
                org = (0.0, 0.0, 0.0)
            expected = 8 * shape[0] * shape[1] * shape[2]
            rels = []
            for i in range(n):
                rel, size = brick(field, i)
                if size != expected:
                    raise SystemExit(
                        "%s: size %d, expected %d" % (rel, size, expected)
                    )
                rels.append(rel)
            out = output
            if out is None or len(fields) > 1:
                stem = os.path.splitext(os.path.basename(manifest_path))[0]
                out = os.path.join(base, "%s_%s.xdmf" % (stem.replace("_manifest", ""), field))
            _write_xdmf(
                out, shape[0], shape[1], shape[2], org, sp, [field],
                {field: rels}, times,
            )
            written.append(out)
        return written

    nx = int(man["nx"])
    ny = int(man.get("ny", 1))
    nz = int(man.get("nz", 1))
    ddx = float(dx if dx is not None else man.get("dx", 1.0))
    ddy = float(dy if dy is not None else man.get("dy", ddx))
    ddz = float(dz if dz is not None else man.get("dz", ddx if nz > 1 else 1.0))
    if nz == 1:
        ddz = 1.0
    expected = 8 * nx * ny * nz
    rels = {field: [] for field in fields}
    for i in range(n):
        for field in fields:
            rel, size = brick(field, i)
            if size != expected:
                raise SystemExit(
                    "%s: size %d, expected %d" % (rel, size, expected)
                )
            rels[field].append(rel)
    if output is None:
        stem = os.path.splitext(os.path.basename(manifest_path))[0]
        output = os.path.join(base, stem.replace("_manifest", "") + ".xdmf")
    raw_origin = man.get("origin") or [0.0, 0.0, 0.0]
    if len(raw_origin) >= 3:
        origin = (
            float(raw_origin[0]),
            float(raw_origin[1]),
            float(raw_origin[2]),
        )
    else:
        origin = (0.0, 0.0, 0.0)
    _write_xdmf(
        output, nx, ny, nz, origin, (ddx, ddy, ddz), fields, rels, times
    )
    return [output]


def _write_xdmf(path, nx, ny, nz, origin, spacing, field_names, rels, times):
    ox, oy, oz = origin
    dx, dy, dz = spacing
    n = len(times)
    two_d = nz == 1
    lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0">',
        "  <Domain>",
    ]
    if two_d:
        lines.append(
            '    <Topology name="topo" TopologyType="2DCoRectMesh" Dimensions="%d %d"/>'
            % (ny, nx)
        )
        lines.append('    <Geometry name="geo" Type="ORIGIN_DXDY">')
        lines.append(
            '      <DataItem Format="XML" Dimensions="2">%.17g %.17g</DataItem>'
            % (oy, ox)
        )
        lines.append(
            '      <DataItem Format="XML" Dimensions="2">%.17g %.17g</DataItem>'
            % (dy, dx)
        )
        dim_attr = "%d %d" % (ny, nx)
    else:
        lines.append(
            '    <Topology name="topo" TopologyType="3DCoRectMesh" Dimensions="%d %d %d"/>'
            % (nz, ny, nx)
        )
        lines.append('    <Geometry name="geo" Type="ORIGIN_DXDYDZ">')
        lines.append(
            '      <DataItem Format="XML" Dimensions="3">%.17g %.17g %.17g</DataItem>'
            % (oz, oy, ox)
        )
        lines.append(
            '      <DataItem Format="XML" Dimensions="3">%.17g %.17g %.17g</DataItem>'
            % (dz, dy, dx)
        )
        dim_attr = "%d %d %d" % (nz, ny, nx)
    lines.extend(
        [
            "    </Geometry>",
            '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
        ]
    )
    for i, t in enumerate(times):
        lines.append('      <Grid Name="step_%04d" GridType="Uniform">' % i)
        lines.append("        <Time Value=\"%.17g\"/>" % float(t))
        lines.append('        <Topology Reference="/Xdmf/Domain/Topology[1]"/>')
        lines.append('        <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>')
        for name in field_names:
            lines.append(
                '        <Attribute Name="%s" Center="Node">' % _xml_escape(name)
            )
            lines.append(
                '          <DataItem Format="Binary" DataType="Float" Precision="8" Endian="Little" Dimensions="%s">%s</DataItem>'
                % (dim_attr, _xml_escape(rels[name][i]))
            )
            lines.append("        </Attribute>")
        lines.append("      </Grid>")
    lines.extend(["    </Grid>", "  </Domain>", "</Xdmf>", ""])
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", path, "(%d steps, fields: %s)" % (n, ", ".join(field_names)))


def cli():
    parser = argparse.ArgumentParser(
        description="Write XDMF so OpenPFC .bin dumps open in ParaView."
    )
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--json", default="input.json", help="legacy JSON settings")
    parser.add_argument("--results_dir", default="", help="legacy data directory")
    parser.add_argument("--Lx", type=int)
    parser.add_argument("--Ly", type=int)
    parser.add_argument("--Lz", type=int)
    parser.add_argument("--x0", type=float)
    parser.add_argument("--y0", type=float)
    parser.add_argument("--z0", type=float)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--dy", type=float)
    parser.add_argument("--dz", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--saveat", type=float, default=1.0)
    parser.add_argument(
        "--manifest",
        help="showcase *_manifest.json (writes .xdmf next to the bricks)",
    )
    parser.add_argument("-o", "--output", help="XDMF path (manifest mode)")
    parser.add_argument(
        "--dvx", type=float, help="Weibel vx spacing (stored on the manifest path)"
    )
    parser.add_argument("--dvy", type=float, help="Weibel vy spacing")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = cli()
    if args.get("manifest"):
        write_manifest_xdmf(
            args["manifest"],
            output=args.get("output"),
            dx=args.get("dx"),
            dy=args.get("dy"),
            dz=args.get("dz"),
            dvx=args.get("dvx"),
            dvy=args.get("dvy"),
        )
        sys.exit(0)
    if args["legacy"]:
        main_old(args)
    else:
        main(args)
