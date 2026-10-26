#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check native vortex snapshots, or draw them with Matplotlib 3.11.2."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile


def read_run(folder):
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["case"] == "taylor_green"
    n, steps, dt, nu = (metadata[key] for key in ("N", "steps", "dt", "nu"))
    fields = []
    for step in (0, steps):
        data = (folder / f"omega_{step:04d}.bin").read_bytes()
        assert len(data) == n * n * 8
        values = struct.unpack(f"={n * n}d", data)
        expected = [
            2
            * math.sin(2 * math.pi * i / n)
            * math.sin(2 * math.pi * j / n)
            * math.exp(-2 * nu * step * dt)
            for j in range(n)
            for i in range(n)
        ]
        assert all(math.isfinite(v) for v in values)
        assert max(abs(a - b) for a, b in zip(values, expected)) < 1e-9
        fields.append((values, expected))
    with (folder / "diagnostics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert int(rows[0]["step"]) == 0 and int(rows[-1]["step"]) == steps
    # The legacy diagnostics deliberately print six significant digits.
    for row in rows:
        time = int(row["step"]) * dt
        assert math.isclose(
            float(row["ke"]),
            0.25 * math.exp(-4 * nu * time),
            rel_tol=5e-6,
            abs_tol=5e-7,
        )
        assert math.isclose(
            float(row["enstrophy"]),
            0.5 * math.exp(-4 * nu * time),
            rel_tol=5e-6,
            abs_tol=5e-7,
        )
        assert float(row["omega_linf"]) < 1e-9
    return metadata, fields


def plot(folder, output):
    import matplotlib.pyplot as plt
    import numpy as np
    from render_utilities import save_svg

    metadata, fields = read_run(folder)
    n = metadata["N"]
    x = np.arange(n) * 2 * math.pi / n
    figure, axes = plt.subplot_mosaic(
        [["initial", "final"], ["error", "error"]], figsize=(7, 6), layout="constrained"
    )
    for name, title, values in (
        ("initial", "Initial vorticity", fields[0][0]),
        ("final", f"After t = {metadata['steps'] * metadata['dt']:g}", fields[1][0]),
    ):
        image = axes[name].pcolormesh(
            x,
            x,
            np.array(values).reshape(n, n),
            shading="nearest",
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
        )
        axes[name].set(
            xlabel="x",
            ylabel="y",
            title=title,
            xlim=(x[0], x[-1]),
            ylim=(x[0], x[-1]),
            aspect="equal",
        )
    figure.colorbar(image, ax=[axes["initial"], axes["final"]], label="ω", pad=0.02)
    values, expected = fields[1]
    j = n // 4
    error = [values[j * n + i] - expected[j * n + i] for i in range(n)]
    axes["error"].plot(x, error, color="#286b83", marker="o", markersize=4)
    axes["error"].set(
        xlabel=f"x at y = {x[j]:.3g}",
        ylabel="Native − exact",
        title="Final nodal error along one grid line",
    )
    axes["error"].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axes["error"].grid(alpha=0.15)
    save_svg(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "flow/vorticity/figures/vortex.svg",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1 or (args.ranks != 1 and not args.launcher):
        parser.error(
            "positive rank count and a launcher are required for multiple ranks"
        )
    if args.execute:
        args.run.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="vortex-", dir=args.run.resolve()
        ) as temp:
            command = (
                [args.launcher, args.rank_flag, str(args.ranks)]
                if args.launcher
                else []
            )
            command += [
                str(args.execute.resolve()),
                "--verify",
                "--N",
                "16",
                "--steps",
                "8",
                "--dt",
                "0.05",
                "--nu",
                "0.1",
                "--outdir",
                temp,
            ]
            subprocess.run(command, check=True, timeout=60)
            metadata, _ = read_run(Path(temp))
            assert metadata["nproc"] == args.ranks
        print("Vortex output: complete native snapshots and analytic decay pass.")
    elif args.check:
        read_run(args.run)
        print("Vortex snapshots and diagnostics pass.")
    else:
        plot(args.run, args.output)


if __name__ == "__main__":
    main()
