#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Legacy domain blocks become size/lower/upper without changing the grid."""

import json
import math
import os
import sys
import tomllib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import migrate_domain_block as m  # noqa: E402

DX = 1.1107207345395915


def spacing(block, axis):
    return (block["upper"][axis] - block["lower"][axis]) / block["size"][axis]


def test_center_origin_is_symmetric_and_exact():
    new = m.convert({"Lx": 256, "Ly": 128, "Lz": 1, "dx": DX, "dy": DX,
                     "dz": DX, "origin": "center"})
    assert new["size"] == [256, 128, 1]
    for axis, n in enumerate((256, 128, 1)):
        # The old parser placed index 0 at -0.5 * dx * Lx.
        assert new["lower"][axis] == -0.5 * DX * n
        assert spacing(new, axis) == DX


def test_corner_origin_starts_at_zero():
    new = m.convert({"Lx": 100, "Ly": 100, "Lz": 100, "dx": 0.1, "dy": 0.1,
                     "dz": 0.1, "origo": "corner"})
    assert new["lower"] == [0.0, 0.0, 0.0]
    assert all(spacing(new, a) == 0.1 for a in range(3))


@pytest.mark.parametrize("n", [3, 7, 96, 100, 1000, 1200])
@pytest.mark.parametrize("d", [0.1, 0.3, 1.0 / 3.0, DX, 2.5e-3])
def test_spacing_round_trips_to_one_ulp(n, d):
    # Exact in most cases; where no exact upper exists, within one ulp.
    for origin in ("center", "corner"):
        new = m.convert({"Lx": n, "dx": d, "origin": origin})
        assert abs(spacing(new, 0) - d) <= math.ulp(d)


def test_shipped_style_grids_are_exact():
    for n in (32, 64, 128, 256, 512, 768, 1024):
        new = m.convert({"Lx": n, "dx": DX, "origin": "center"})
        assert spacing(new, 0) == DX


def test_thin_film_style_block_uses_dx_everywhere():
    new = m.convert({"Lx": 64, "Ly": 32, "dx": 0.5})
    assert new["size"] == [64, 32, 1]
    assert new["lower"] == [0.0, 0.0, 0.0]
    assert new["upper"] == [32.0, 16.0, 0.5]


def test_json_rewrite_keeps_the_rest_of_the_file():
    text = ('{\n  "model": {"name": "x"},\n  "domain": {\n    "Lx": 4,\n'
            '    "Ly": 4,\n    "Lz": 4,\n    "dx": 1.0,\n    "dy": 1.0,\n'
            '    "dz": 1.0,\n    "origin": "corner"\n  },\n  "t": 1e-05\n}\n')
    out = m.migrate_json(text)
    assert '"t": 1e-05' in out
    assert json.loads(out)["domain"] == {"size": [4, 4, 4],
                                         "lower": [0.0, 0.0, 0.0],
                                         "upper": [4.0, 4.0, 4.0]}


def test_toml_rewrite_keeps_other_tables():
    text = ('[model]\nname = "x"\n\n[domain]\nLx = 4\nLy = 4\nLz = 4\n'
            'dx = 2.0\ndy = 2.0\ndz = 2.0\norigin = "center"\n\n'
            '[timestepping]\nt0 = 0.0\n')
    out = tomllib.loads(m.migrate_toml(text))
    assert out["domain"]["lower"] == [-4.0, -4.0, -4.0]
    assert out["domain"]["upper"] == [4.0, 4.0, 4.0]
    assert out["timestepping"] == {"t0": 0.0}


def test_refuses_to_guess():
    with pytest.raises(m.MigrationError):
        m.convert({"Lx": 4, "dx": 1.0, "origin": "middle"})
    with pytest.raises(m.MigrationError):
        m.convert({"size": [4, 4, 4]})
    with pytest.raises(m.MigrationError):
        m.migrate_json('{"Lx": 4, "dx": 1.0}')
