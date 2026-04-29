<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Debugging

## Debug Mode

In the Debug build mode, additional features are activated to aid in debugging
and validation. These features provide enhanced error checking and diagnostics,
but they may introduce some performance overhead. It is recommended to use the
Debug mode during development and testing.

### NaN Check

When compiling with `CMAKE_BUILD_TYPE=Debug`, the software includes NaN
(Not-a-Number) checks to detect and handle floating-point NaN values. This check
helps identify potential numerical issues and prevents the propagation of NaN
values in calculations.

Debug builds enable the NaN check feature automatically. To enable it explicitly
for another build type, use the OpenPFC CMake option:

```bash
cmake -DOpenPFC_ENABLE_NAN_CHECK=ON path/to/source
```

When NaN checks are enabled, the software includes a macro called
`CHECK_AND_ABORT_IF_NAN`, which can be used to check individual floating-point
values for NaN. If a NaN value is detected, the application will be aborted, and
an error message will be displayed, indicating the process rank, file name, and
line number where the NaN was detected. The rank and abort communicator follow a
process-wide default (`pfc::utils::default_nan_check_mpi_comm()`): JSON `App`
drivers set it to the application communicator in `App::main`; other entry points
should call `pfc::utils::set_default_nan_check_mpi_comm` or use the `*_MPI`
macro variants with an explicit `MPI_Comm`.

It is recommended to disable NaN checks in release builds
(`CMAKE_BUILD_TYPE=Release`) to optimize performance. NaN checks are primarily
intended for debugging and validation purposes.

Note: Enabling NaN checks may introduce some performance overhead. Therefore, it
is advisable to enable these checks only during the development and testing
stages to ensure numerical stability and correctness.
