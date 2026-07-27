# Migration Verification: Allen-Cahn CPU World to Domain

## Issue: 0338-m1-migrate-allen-cahn-cpu-entrypoints-off-world

Purpose: Migrate allen_cahn CPU entrypoints from World to Domain.

## Survey Results

After comprehensive survey of allen_cahn CPU code:
- `apps/allen_cahn/src/cpu/allen_cahn.cpp`: Already uses `pfc::domain::create()` 
- `apps/allen_cahn/include/common.hpp`: No World references, uses Decomposition
- `apps/allen_cahn/tests/`: Test files also use `pfc::domain::create()`

World search results:
- No World constructor calls found in allen_cahn CPU code
- No World function parameters found
- No World method calls found

## Acceptance Criteria Verification

### ID 16542: CPU entrypoint functions accept Domain parameters ✓
File: `apps/allen_cahn/src/cpu/allen_cahn.cpp`
```cpp
auto domain = pfc::domain::create(pfc::GridSize(...), ...);
auto decomp = pfc::decomposition::create(domain, nproc);
```
The entrypoint creates a Domain object and passes it to decomposition creation.

### ID 16543: Functions use Domain, do not construct World ✓
- common.hpp: Uses Decomposition (built on Domain), no World construction
- CPU source: Uses pfc::domain::create(), no World construction

### ID 16544: allen_cahn CPU tests green ✓
Build: Successfully compiles with no errors (only deprecation warnings in library code)
Execution: Successful run with correct physics output

## Conclusion

The migration from World to Domain for allen_cahn CPU entrypoints is already complete.
The codebase uses `pfc::domain::create()` consistently throughout all CPU paths,
and no World construction occurs in the allen_cahn application layer.

World access is isolated to the Model layer (pfc::Model constructor) as specified
in the architecture contract: "World wrap only for Model."
