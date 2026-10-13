# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exercise the production JSON/TOML entry point and communicator rejection."""

import argparse
import ctypes
import json
import re
from pathlib import Path
import subprocess
import tempfile
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--backend", choices=("cufft", "rocfft"), required=True)
    parser.add_argument("--mpiexec")
    parser.add_argument("--numproc-flag", default="-n")
    args = parser.parse_args()
    library, probe = (("libcudart.so", "cudaGetDeviceCount")
                      if args.backend == "cufft"
                      else ("libamdhip64.so", "hipGetDeviceCount"))
    try:
        runtime = ctypes.CDLL(library)
    except OSError:
        print(f"SKIP: {library} is unavailable")
        return 77
    count = ctypes.c_int()
    if getattr(runtime, probe)(ctypes.byref(count)) != 0 or count.value == 0:
        print(f"SKIP: no usable {args.backend} device")
        return 77
    settings = {
        "fft": "local",
        "domain": {"size": [8, 8, 8], "lower": [0, 0, 0], "upper": [8, 8, 8]},
        "timestepping": {"t0": 0, "t1": 0.02, "dt": 0.01, "saveat": -1},
        "initial_conditions": [{"type": "constant", "n0": -0.4}],
    }
    with tempfile.TemporaryDirectory(prefix="local-fft-cli-", dir=".") as name:
        root = Path(name).resolve()
        for extension in ("json", "toml"):
            config = root / f"input.{extension}"
            if extension == "json":
                config.write_text(json.dumps(settings))
            else:
                config.write_text('''fft = "local"
[domain]
size = [8, 8, 8]
lower = [0, 0, 0]
upper = [8, 8, 8]
[timestepping]
t0 = 0.0
t1 = 0.02
dt = 0.01
saveat = -1.0
[[initial_conditions]]
type = "constant"
n0 = -0.4
''')
            command = [args.binary, str(config)]
            if args.mpiexec:
                command = [args.mpiexec, args.numproc_flag, "2"] + command
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            if args.mpiexec:
                assert result.returncode != 0, output
                assert "local FFT needs exactly one rank" in output, output
            else:
                assert result.returncode == 0, output
                assert (f"SPECTRAL_FFT implementation=local backend={args.backend} ranks=1"
                        in output), output
                assert "SPECTRAL_CHECKSUM field=psi" in output, output
                done = re.search(r"done t=(\S+)", output)
                assert done, output
                token = done.group(1)
                final_time = float.fromhex(token) if "0x" in token else float(token)
                assert abs(final_time - 0.02) < 1e-14, output
            print(output)


if __name__ == "__main__":
    sys.exit(main())
