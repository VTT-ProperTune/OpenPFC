<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Bounded symbolic pointwise RHS

This generation example implements exactly two fields:

```text
du = v.value + a*u.xy
dv = b*(u.xx+u.yy+u.zz) + c*v.xz + d*u.yz
```

An optional, explicit manufactured source adds
`source_rate*t*(x+y+z)` to both outputs. Its default is zero; nonzero
source rates describe a different forced problem. This source exercises
shared expression elimination and time/position arguments. It is not
solidification physics.

`generated_rhs.hpp` accepts model-owned `Local{u,v}`, `Parameters{a,b,c,d}`,
position with `operator[]`, and an explicit output aggregate template
argument. A model's existing `rhs(t, local)` wraps this function, supplying
position through its local evaluator/model data. No framework interface or
new application is introduced. CPU outputs opt into the existing
`as_tuple()` protocol. GPU adapters return existing `DeviceInc2`; they do
not call host tuple methods from a kernel. Storage, derivatives, halos,
integrator stages and output remain driver responsibilities.

Generate with Python 3.9 or newer:

```sh
python3 -m venv /path/outside/checkout/generation-env
/path/outside/checkout/generation-env/bin/pip install -r examples/coupling/symbolic-rhs/requirements.txt
/path/outside/checkout/generation-env/bin/python examples/coupling/symbolic-rhs/generate.py
/path/outside/checkout/generation-env/bin/python examples/coupling/symbolic-rhs/generate.py --check
```

SymPy 1.14.0 and mpmath 1.3.0 are pinned generation dependencies. C++
consumers require neither Python nor SymPy. A different SymPy version
fails generation explicitly. Artifacts contain no timestamp or absolute
path; CSE uses canonical ordering. The machine-readable dependency
manifest records requested derivatives, arithmetic counts and the header's
SHA-256. Arithmetic counts describe this expression, not runtime speed,
register usage or an optimization claim. No GPU performance improvement is
claimed.

Canonical `scripts/build.sh` tests invoke the compiled CPU model, canonical
host tuple scatter, an independent 80-digit decimal oracle (273 cases),
and, on HIP builds, actual device evaluation and existing named scatter
with output guards. HIP verification fails when hardware is absent.
Generation checks in `scripts/tests/test_symbolic_rhs.py` require the pinned
SymPy environment and report a skip when it is unavailable. Runtime
verification needs no symbolic dependency. Run the generation checks with
`python -m pytest scripts/tests/test_symbolic_rhs.py` in the environment
above after installing pytest.

This is a bounded checked-in example, not a symbolic PDE compiler. A
pointwise Hessian of a field is not a conservative divergence of a
spatially varying flux. Conservative alloy transport needs a separate
flux evaluation and divergence operator; this generator does not provide
one. Rotation/orientation and coefficient calibration also belong to the
physical model, outside this example.
