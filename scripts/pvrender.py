#!/usr/bin/env python

import argparse
import sys
import os


__epilog__ = """
## Usage examples:

Range is given in "Julia syntax" (https://erik-engheim.medium.com/ranges-and-slices-in-julia-and-python-bb0fd893a20c)

# load state isometric.pvsm and render frame #0
pvbatch -- $(which pvrender.py) isometric.pvsm 0

# render frames 3 ... 8
pvbatch -- $(which pvrender.py) isometric.pvsm 3:8

# render frames 1, 3, 5
pvbatch -- $(which pvrender.py) isometric.pvsm 1:2:5

# render starting from frame 500 and render to the end
pvbatch -- $(which pvrender.py) isometric.pvsm 500:end

## Usage with Slurm or other batch system

Typical batch script would be

```bash
#!/bin/bash

#SBATCH --partition=medium24
#SBATCH --job-name=r1024x512
#SBATCH --output=%x-%j.log
#SBATCH --cpus-per-task=24
#SBATCH --array=0-4

module load cuda paraview/5.10.1-egl
# module load paraview/5.10.1-osmesa
SCRIPT=$(which pvrender.py)  # path to pvrender.py
pvbatch --force-offscreen-rendering -- $SCRIPT $SLURM_ARRAY_TASK_ID:5:end
```

Here, first task is rendering frames 0, 5, 10, ..., second is rendering
frames 1, 6, 1, ..., and so on.

## Creating animations

Good ffmpeg settings:

ffmpeg -r 30 -i images/frame.%04d.png -c:v libx264 -preset veryslow -tune film anim.mp4

"""

def create_range(s, end):
    p = str.split(s, ":")
    start = 0
    step = 1

    if len(p) == 1:
        start = int(p[0])
        end = start + 1

    if len(p) == 2:
        start = int(p[0])
        if p[1] != "end":
            end = int(p[1]) + 1

    if len(p) == 3:
        start = int(p[0])
        step = int(p[1])
        if p[2] != "end":
            end = int(p[2]) + 1

    return range(start, end, step)


def main(args):
    import paraview.simple as ps
    #### disable automatic camera reset on 'Show'
    ps._DisableFirstRenderCameraReset()

    ps.LoadState(args.state)
    #SetActiveView(renderView1) # set active view
    #materialLibrary1 = GetMaterialLibrary() # get the material library
    #layout1 = GetLayout() # get layout
    #layout1.SetSize(1920, 1080) # layout/tab size in pixels
    basepath = os.path.dirname(args.format)
    if not os.path.exists(basepath):
        print("Creating directory %s" % basepath)
        os.makedirs(basepath)


    renderView1 = ps.FindViewOrCreate('RenderView1', viewtype='RenderView') # find view
    animationScene1 = ps.GetAnimationScene()
    animationScene1.UpdateAnimationUsingDataTimeSteps()
    timeKeeper1 = ps.GetTimeKeeper()
    timesteps = timeKeeper1.TimestepValues
    ntimesteps = len(timesteps)

    if args.resolution:
        xs, ys = args.resolution.split("x")
        x = int(xs)
        y = int(ys)
        print("Set resolution to %dx%d" % (x, y))
        renderView1.ViewSize = [x, y]
        renderView1.Update()

    if args.scale != 1.0:
        x, y = renderView1.ViewSize
        x2, y2 = int(args.scale*x), int(args.scale*y)
        print("Scaling images by %0.2f, %dx%d -> %dx%d" % (args.scale, x, y, x2, y2))
        renderView1.ViewSize = [x2, y2]
        renderView1.Update()

    for idx in create_range(args.range, end=ntimesteps):
        if idx >= ntimesteps:
            print("idx = %d, ntimesteps = %d, quitting" % (idx, ntimesteps))
            break
        imgfile = args.format % idx
        if os.path.exists(imgfile) and not args.overwrite:
            print("%s already exists, not rendering" % imgfile)
            continue
        print("Rendering frame %d (time = %0.3f) ... " % (idx, timesteps[idx]), end="")
        animationScene1.AnimationTime = timesteps[idx]
        print("done! Writing image to %s" % imgfile)
        ps.SaveScreenshot(imgfile, renderView1, OverrideColorPalette='WhiteBackground')

    print("\nAll done! Consider creating animation:")
    print("\n    ffmpeg -r 30 -i %s -c:v libx264 -preset veryslow -tune film anim.mp4\n" % args.format)


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Render images from ParaView state", epilog=__epilog__)
    parser.add_argument("state", type=str)
    parser.add_argument("range", type=str, default="0:end")
    parser.add_argument("--format", default="images/frame.%04d.png")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    sys.exit(main(args))
