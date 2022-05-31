#!/usr/bin/env python

"""
Generate xmf file for bunch of binary files, so that simulation results can be
opened easily using Paraview.
"""

import argparse
import glob
import os
import re

header = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
    <Domain>
        <Topology name="topo" TopologyType="3DCoRectMesh" Dimensions="{lx} {ly} {lz}"></Topology>
        <Geometry name="geo" Type="ORIGIN_DXDYDZ">
            <DataItem Format="XML" Dimensions="3">{x0} {y0} {z0}</DataItem>
            <DataItem Format="XML" Dimensions="3">{dx} {dy} {dz}</DataItem>
        </Geometry>

        <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
            <Time TimeType="HyperSlab">
                <DataItem Format="XML" NumberType="Float" Dimensions="{ntimesteps}">{timesteps}</DataItem>
            </Time>
"""

footer = """
        </Grid>
    </Domain>
</Xdmf>"""


def extract_timestep(f):
    return re.findall(r"\d+", f)[-1]


def main(args):
    if args.dir != "" and not os.path.isdir(args.dir):
        print("Directory %s does not exist!" % args.dir)
        return
    files = glob.glob(os.path.join(args.dir, "*.bin"))
    if len(files) == 0:
        print("No data files found!")
        return
    files = sorted(files, key=lambda k: int(extract_timestep(k)))
    expected_size = 8 * args.lx * args.ly * args.lz
    if os.path.getsize(files[0]) != expected_size:
        print("File size mismatch, check lx, ly, lz!")
        return
    data = vars(args)
    data["ntimesteps"] = len(files)
    data["timesteps"] = " ".join(map(extract_timestep, files))
    print(header.format(**data))
    for f in files:
        item = """            <Grid GridType="Uniform"><Topology Reference="/Xdmf/Domain/Topology[1]" /><Geometry Reference="/Xdmf/Domain/Geometry[1]" /><Attribute Name="u" Center="Node"><DataItem Format="Binary" DataType="Float" Precision="8" Endian="Little" Dimensions="{lx} {ly} {lz}">{f}</DataItem></Attribute></Grid>"""
        print(item.format(**data, f=f))
    print(footer.format(**data))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="", help="data directory (default .)")
    parser.add_argument("--lx", type=int, default=256)
    parser.add_argument("--ly", type=int, default=256)
    parser.add_argument("--lz", type=int, default=256)
    parser.add_argument("--x0", default=-128.0)
    parser.add_argument("--y0", default=-128.0)
    parser.add_argument("--z0", default=-128.0)
    parser.add_argument("--dx", default=1.0)
    parser.add_argument("--dy", default=1.0)
    parser.add_argument("--dz", default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(cli())
