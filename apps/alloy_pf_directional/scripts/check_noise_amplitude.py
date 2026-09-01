#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare interface roughness of noisy vs quiet Al-Cu ds runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_meta(path: Path) -> dict[str, float]:
    meta: dict[str, float] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            meta[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return meta


def load_phi_raw(root: Path) -> tuple[dict[str, float], np.ndarray]:
    meta = load_meta(root / "meta.txt")
    nx = int(meta["Nx"])
    ny = int(meta["Ny"])
    phi = np.fromfile(root / "phi_final.raw", dtype=np.float64).reshape(ny, nx)
    return meta, phi


def load_vti_phi(path: Path, nx: int, ny: int) -> np.ndarray:
    data = path.read_bytes()
    key = b"<AppendedData encoding=\"raw\">"
    i = data.find(key)
    if i < 0:
        raise ValueError(f"no appended data in {path}")
    i = data.find(b"_", i)
    if i < 0:
        raise ValueError(f"no raw marker in {path}")
    i += 1
    n = int(nx) * int(ny)
    nbytes = int(np.frombuffer(data[i : i + 8], dtype="<u8")[0])
    i += 8
    i += nbytes  # skip c
    nbytes_phi = int(np.frombuffer(data[i : i + 8], dtype="<u8")[0])
    i += 8
    phi = np.frombuffer(data[i : i + nbytes_phi], dtype="<f8")
    if phi.size != n:
        raise ValueError(f"{path}: expected {n} phi values, got {phi.size}")
    return phi.reshape(ny, nx)


def interface_x(phi: np.ndarray, dx: float) -> np.ndarray:
    """Rightmost φ=0 crossing in each row (solid left, liquid right)."""
    ny, nx = phi.shape
    xs = np.full(ny, np.nan)
    for j in range(ny):
        row = phi[j]
        for i in range(nx - 1):
            p0 = row[i]
            p1 = row[i + 1]
            if p0 >= 0.0 and p1 < 0.0:
                a = p0 / (p0 - p1 + 1.0e-30)
                xs[j] = (i + 0.5) * dx + a * dx
    return xs


def bulk_stats(phi: np.ndarray) -> dict[str, float]:
    solid = phi > 0.8
    liquid = phi < -0.8
    return {
        "min_phi": float(phi.min()),
        "max_phi": float(phi.max()),
        "frac_|phi|>0.9": float(np.mean(np.abs(phi) > 0.9)),
        "mean_solid": float(phi[solid].mean()) if solid.any() else float("nan"),
        "mean_liquid": float(phi[liquid].mean()) if liquid.any() else float("nan"),
        "min_liquid": float(phi[liquid].min()) if liquid.any() else float("nan"),
    }


def pick_vti(root: Path, frac: float = 0.55) -> Path | None:
    vtis = sorted(root.glob("output_c_phi_*.vti"), key=lambda p: int(p.stem.split("_")[-1]))
    vtis = [p for p in vtis if not p.stem.endswith("_0")]
    if not vtis:
        return None
    idx = min(len(vtis) - 1, max(0, int(frac * (len(vtis) - 1))))
    return vtis[idx]


def report(label: str, root: Path, quiet: Path | None = None) -> None:
    meta, phi_end = load_phi_raw(root)
    w0 = meta["W0"]
    dx = meta["dx"]
    dt = meta["dt"]
    tau0 = meta["tau0"]
    f0 = meta.get("noise_F0", 0.0)
    sigma_dphi = float(np.sqrt(2.0 * f0 * dt / tau0)) if f0 > 0 else 0.0
    print(f"\n=== {label}  {root} ===")
    print(
        f"  F0={f0:g}  seed={meta.get('noise_seed', float('nan')):g}  "
        f"W0={w0*1e9:.2f} nm  dx/W0={dx/w0:.3f}  Nx={int(meta['Nx'])} Ny={int(meta['Ny'])}"
    )
    print(
        f"  n_steps_done={int(meta['n_steps_done'])}  hit_right={int(meta.get('hit_right', 0))}  "
        f"blew_up={int(meta.get('blew_up', 0))}"
    )
    print(f"  Euler Δφ_rms at φ=0 (theory) = sqrt(2 F0 dt/τ) = {sigma_dphi:.4f}")
    bs = bulk_stats(phi_end)
    print(
        f"  final φ: min={bs['min_phi']:.5f} max={bs['max_phi']:.5f}  "
        f"|φ|>0.9 fraction={bs['frac_|phi|>0.9']:.3f}  "
        f"<φ>_solid={bs['mean_solid']:.5f}  min liquid={bs['min_liquid']:.5f}"
    )
    vti = pick_vti(root)
    if vti is None:
        print("  no VTI snapshots")
        return
    phi = load_vti_phi(vti, int(meta["Nx"]), int(meta["Ny"]))
    xs = interface_x(phi, dx)
    valid = np.isfinite(xs)
    rms = float(np.nanstd(xs))
    ptp = float(np.nanmax(xs) - np.nanmin(xs)) if valid.any() else float("nan")
    print(f"  snapshot {vti.name}: interface rms={rms/w0:.3f} W0  peak-to-peak={ptp/w0:.3f} W0")
    if quiet is not None and (quiet / "phi_final.raw").is_file():
        mq = load_meta(quiet / "meta.txt")
        vq = pick_vti(quiet)
        if vq is not None:
            phi_q = load_vti_phi(vq, int(mq["Nx"]), int(mq["Ny"]))
            xq = interface_x(phi_q, mq["dx"])
            # Compare roughness, not the mean tip (noise run may be a slightly different time).
            print(
                f"  quiet {vq.name}: interface rms={np.nanstd(xq)/mq['W0']:.3f} W0  "
                f"peak-to-peak={(np.nanmax(xq)-np.nanmin(xq))/mq['W0']:.3f} W0"
            )
            if valid.any() and np.isfinite(xq).any():
                extra = rms / max(float(np.nanstd(xq)), 1.0e-30)
                print(f"  roughness ratio noisy/quiet = {extra:.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("noisy")
    p.add_argument("quiet", nargs="?")
    p.add_argument("--label", default="")
    args = p.parse_args()
    report(args.label or Path(args.noisy).name, Path(args.noisy), Path(args.quiet) if args.quiet else None)


if __name__ == "__main__":
    main()
