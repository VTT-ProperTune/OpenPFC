# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exercise the installed Burgers CLI, data format, and negative exit paths."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def invoke(executable: str, method: str, folder: Path, *, dt: str = "0.001"):
    return subprocess.run(
        [executable, "--method", method, "--N", "64", "--nu", "0.01",
         "--dt", dt, "--time", dt if dt == "1" else "0.02",
         "--outdir", str(folder)],
        capture_output=True, text=True, timeout=30, check=False
    )


def verify_output(executable: str, method: str, root: Path):
    folder = root / method
    result = invoke(executable, method, folder)
    assert result.returncode == 0, (method, result.stdout, result.stderr)
    assert "status=ok" in result.stdout, result.stdout
    filename = folder / "u.dat"
    assert filename.is_file(), filename
    lines = filename.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ("# x_center u" if method == "fvm" else "# x u")
    assert len(lines) == 65, (method, len(lines))
    spacing = 2.0 * math.pi / 64.0
    for i, line in enumerate(lines[1:]):
        x, u = map(float, line.split())
        center = i + (0.5 if method == "fvm" else 0.0)
        assert math.isclose(x, center * spacing, abs_tol=1e-12)
        assert math.isfinite(u), (method, i, u)


def main():
    executable = sys.argv[1]
    spectral = sys.argv[2] == "1"
    with tempfile.TemporaryDirectory(prefix="burgers-cli-") as tmp:
        root = Path(tmp)
        for method in (["fdm2", "fvm", "spectral"] if spectral else ["fdm2", "fvm"]):
            verify_output(executable, method, root)
        # The pre-step method-specific CFL guard must refuse unsafe dt.
        unstable = invoke(executable, "fvm", root / "unstable", dt="1")
        assert unstable.returncode != 0, unstable.stdout
        assert "status=ok" not in unstable.stdout, unstable.stdout

        # Attempting to use a regular file as an output directory must fail.
        blocked = root / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        failed = invoke(executable, "fdm2", blocked)
        assert failed.returncode != 0, failed.stdout
        assert "status=ok" not in failed.stdout, failed.stdout

        if not spectral:
            missing = invoke(executable, "spectral", root / "spectral")
            assert missing.returncode != 0, missing.stdout
            assert "need FFTW3" in missing.stderr, missing.stderr
    print("BURGERS_CLI_OUTPUT_OK")


if __name__ == "__main__":
    main()
