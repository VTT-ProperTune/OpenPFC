#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw adaptive and coupled-field lessons from native CSVs; Matplotlib 3.11.2."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def table(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def values(rows, name):
    return np.array([float(row[name]) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "openpfc-workflows"})
    blue, red, green = "#526a82", "#a44d2d", "#358075"

    def read(case, name):
        return table(args.runs / case / "results" / name)

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.15)
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)

    data = read("adaptive", "attempts_rank_0.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout="constrained")
    for controller, color in (("memoryless", blue), ("pi", green)):
        ledger = [r for r in data if r["controller"] == controller]
        accepted = [r for r in ledger if int(r["accepted"])]
        rejected = [r for r in ledger if not int(r["accepted"])]
        axes[0, 0].semilogy(values(accepted, "time"), values(accepted, "y"), ".-", color=color, ms=3, label=controller)
        axes[0, 1].semilogy(values(accepted, "t0"), values(accepted, "dt"), ".-", color=color, ms=3, label=controller)
        axes[0, 1].plot(values(rejected, "t0"), values(rejected, "dt"), "x", color=color, ms=6)
        axes[1, 0].semilogy(values(ledger, "t0"), np.maximum(values(ledger, "metric"), 1e-16),
                            ".", color=color, ms=3, label=controller)
        axes[1, 0].plot(values(rejected, "t0"), values(rejected, "metric"), "x", color=color, ms=6)
        axes[1, 1].semilogy(values(accepted, "time"),
                            np.maximum(abs(values(accepted, "y")-values(accepted, "exact")), 1e-17),
                            ".-", color=color, ms=3, label=controller)
    exact = [r for r in data if r["controller"] == "memoryless" and int(r["accepted"])]
    axes[0, 0].semilogy(values(exact, "time"), values(exact, "exact"), "--", color=red, label="Exact")
    axes[1, 0].axhline(1, color=red, ls="--", lw=1, label="Admission threshold")
    axes[0, 0].set(title="Same decay, two policies", ylabel="Accepted y", xlabel="Time")
    axes[0, 1].set(title="Step sizes · crosses are rejected", ylabel="Attempted dt", xlabel="Start time")
    axes[1, 0].set(title="Local error metric", ylabel="Scaled difference", xlabel="Start time")
    axes[1, 1].set(title="Global error at accepted times", ylabel="Absolute error", xlabel="Time")
    for ax in axes.flat:
        ax.axvline(.15, color="#777777", alpha=.35, lw=1)
        ax.legend(frameon=False, fontsize=8)
    save(fig, "integration/adaptive-rk/figures/adaptive.svg")

    data = read("etd", "attempts_rank_0.csv")
    fields = read("etd", "fields_rank_0.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    for name, color in (("heat", blue), ("uniform", red), ("reaction", green)):
        accepted = [r for r in data if r["case"] == name and int(r["accepted"])]
        axes[0].semilogy(values(accepted, "t0"), values(accepted, "dt"), "o-", color=color, ms=3, label=name)
        line = sorted((r for r in fields if r["case"] == name and int(r["j"]) == int(r["k"]) == 0),
                      key=lambda r: int(r["i"]))
        x = values(line, "x")
        axes[1].plot(x, values(line, "u"), "o", color=color, ms=3, label=name + " · computed")
        if name != "reaction":
            exact = .4*np.exp(-.2)*np.cos(x) if name == "heat" else np.full_like(x, .4/np.sqrt(1+2*.4**2*.2))
            axes[1].plot(x, exact, "--", color=color, label=name + " · exact")
    axes[0].set(title="Accepted ETD step sizes", xlabel="Start time", ylabel="dt")
    axes[1].set(title="Final profiles · t = 0.2", xlabel="x", ylabel="u(x)")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    save(fig, "integration/adaptive-etd/figures/etd.svg")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    fine = read("oscillator", "oscillator_rank_0.csv")
    line = sorted((r for r in fine if int(r["j"]) == int(r["k"]) == 0), key=lambda r:int(r["i"]))
    x = values(line, "x")
    for field, color, exact in (("u", blue, np.cos(1)*np.cos(x)), ("v", red, -np.sin(1)*np.cos(x))):
        axes[0].plot(x, values(line, field), "o", color=color, ms=4, label=field + " · RK4")
        axes[0].plot(x, exact, "--", color=color, label=field + " · exact")
    errors=[]
    for case in ("oscillator-coarse", "oscillator"):
        samples = read(case, "oscillator_rank_0.csv")
        error = max(abs(complex(float(r["u"]),float(r["v"]))-np.exp(-1j)*np.cos(float(r["x"]))) for r in samples)
        errors.append(error)
    dt = np.array([.1,.05])
    axes[1].loglog(dt,errors,"o-",color=blue,label="Observed field error")
    axes[1].loglog(dt,errors[0]*(dt/.1)**4,"--",color=red,label="Fourth-order guide")
    axes[0].set(title="Two fields at t = 1",xlabel="x",ylabel="Field value")
    axes[1].set(title="Halve the RK4 step at fixed final time",xlabel="dt",ylabel="Maximum |z − exact|")
    axes[1].set_xticks([.05, .1], ["0.05", "0.10"])
    axes[1].tick_params(axis="x", which="minor", labelbottom=False)
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
    save(fig,"coupling/runtime-fields/figures/oscillator.svg")


if __name__ == "__main__":
    main()
