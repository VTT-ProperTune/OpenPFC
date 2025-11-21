# Integration Tests

This directory contains **integration tests** that verify multiple OpenPFC components working together. Integration tests validate component interactions and end-to-end workflows.

## Purpose

- **Component interaction**: Verify components work correctly together
- **Real dependencies**: Use actual FFT, MPI, file I/O (not mocks)
- **Workflow validation**: Test complete simulation scenarios
- **Integration bug detection**: Catch issues at component boundaries

## What Belongs Here

Integration tests should verify:

- FFT operations across multiple MPI ranks
- Complete simulation workflows (short runs)
- File I/O with checkpointing and restart
- Field modifiers applied during simulation
- Multi-component scenarios (Model + Simulator + FFT)

## What Doesn't Belong Here

- Tests of isolated components → use `tests/unit/`
- Long-running performance tests → use `tests/benchmarks/`
- Single-component functionality → use `tests/unit/`

## Running Integration Tests

```bash
# All integration tests
./tests/openpfc-tests "[integration]"

# With MPI (parallel tests)
mpirun -np 4 ./tests/openpfc-tests "[integration]"
```

## Writing Integration Tests

1. Tag tests with `[integration]`
2. Use realistic but minimal scenarios (keep tests reasonably fast)
3. Test actual workflows end-users will perform
4. Clean up resources (temporary files, MPI communicators)
5. Document what interaction is being tested

Integration tests may be slower than unit tests but should still complete in seconds, not minutes.

## Current Status

**Empty** - Integration tests will be added as the test suite matures. Priority is to build comprehensive unit test coverage first.
