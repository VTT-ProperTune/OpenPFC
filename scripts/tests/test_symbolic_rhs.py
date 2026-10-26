# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generation checks; numerical runtime checks are separate mandatory CTest cases."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples" / "coupling" / "symbolic-rhs"


def load_generator():
    sp = pytest.importorskip("sympy", reason="generation-only SymPy not installed")
    if sp.__version__ != "1.14.0":
        pytest.skip("generation checks require pinned SymPy 1.14.0")
    spec = importlib.util.spec_from_file_location("generate_rhs", SOURCE / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sp, module


def test_cli_determinism_and_checked_artifacts(tmp_path):
    _, generator = load_generator()
    outputs = []
    for seed in ["1", "12345"]:
        destination = tmp_path / seed
        subprocess.run([sys.executable, str(SOURCE / "generate.py"), "--output-dir", str(destination)],
                       env=dict(os.environ, PYTHONHASHSEED=seed), check=True)
        outputs.append({f.name: f.read_bytes() for f in destination.iterdir()})
    assert outputs[0] == outputs[1] == generator.artifacts()
    subprocess.run([sys.executable, str(SOURCE / "generate.py"), "--check"], check=True)
    (tmp_path / "1" / "generated_rhs.hpp").write_text("corrupted")
    result = subprocess.run([sys.executable, str(SOURCE / "generate.py"), "--check",
                             "--output-dir", str(tmp_path / "1")], capture_output=True)
    assert result.returncode != 0


def test_cse_reconstruction_dependencies_and_perturbations():
    sp, generator = load_generator()
    s, expr = generator.expressions()
    replacements, reduced = sp.cse(expr, symbols=sp.numbered_symbols("cse"), order="canonical")
    for temporary, value in reversed(replacements):
        reduced = [e.subs(temporary, value) for e in reduced]
    assert all(sp.expand(a-b) == 0 for a,b in zip(reduced, expr))
    manifest = json.loads(generator.artifacts()["dependency_manifest.json"])
    assert manifest["cse_temporaries"] == 1
    assert manifest["operations_after_cse"] < manifest["operations_before_cse"]
    assert manifest["fields"] == {"u": ["xx", "xy", "yy", "yz", "zz"], "v": ["value", "xz"]}
    # Independently known sensitivities identify all mixed derivative slots.
    expected = {"u_xy": (s["a"],0), "u_yz": (0,s["d"]), "v_xz": (0,s["c"]),
                "v_value": (1,0), "u_xx": (0,s["b"]), "u_yy": (0,s["b"]), "u_zz": (0,s["b"])}
    delta = sp.Rational(1,1024)
    for name, derivative in expected.items():
        for e, exact in zip(expr, derivative):
            change=sp.expand(e.subs(s[name],s[name]+delta)-e)
            assert sp.simplify(change-exact*delta)==0
