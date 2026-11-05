# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Negative MPMD MPI test: one HIP rank fails while the other has valid input.

An all-ranks-invalid test cannot prove the rank-local error contract: the
valid peer must be allowed to reach an MPI collective before MPI_Abort
terminates the job. A timeout is a failure, never a skip or an expected pass.
"""
import pathlib
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: check_hip_mpi_abort.py MPIEXEC NUMPROC_FLAG EXE VALID_JSON")
    launcher, numproc_flag, binary, valid_json = sys.argv[1:]
    invalid = pathlib.Path(valid_json).with_name("__missing_hip_rank0__.json")
    if invalid.exists():
        raise SystemExit("negative fixture unexpectedly exists")
    command = [
        launcher, numproc_flag, "1", binary, str(invalid),
        ":", numproc_flag, "1", binary, valid_json,
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=25, check=False
        )
    except subprocess.TimeoutExpired as error:
        print("CH_HIP_ASYMMETRIC_ABORT_FAILED: MPI job hung", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    output = process.stdout + "\n" + process.stderr
    if process.returncode == 0:
        print("CH_HIP_ASYMMETRIC_ABORT_FAILED: unexpected success", file=sys.stderr)
        return 1
    if "CH HIP rank" not in output or "cannot open" not in output:
        print("CH_HIP_ASYMMETRIC_ABORT_FAILED: no rank-local JSON diagnostic",
              file=sys.stderr)
        print(output[-4000:], file=sys.stderr)
        return 1
    print("CH_HIP_ASYMMETRIC_ABORT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
