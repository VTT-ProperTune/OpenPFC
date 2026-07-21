<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Solver contract for linear and implicit subsystems

This document describes the semantic contract for solving implicit systems in OpenPFC. The contract enables backend-agnostic solver composition without prescribing abstract `LinearSolver` base classes, `solve(A,b,x)` interfaces, virtual methods, or matrix representations.

## Overview

The solver contract establishes a capability-based approach where:

- **Operators** provide descriptor-based field-bundle transformations
- **Callers** provide right-hand-side and target storage
- **Solvers** produce solution state plus convergence evidence
- **Intermediate work** uses isolated buffers without modifying accepted solution state

This design supports:
- Spectral diagonal solvers (element-wise division for phase-field)
- Iterative solvers with preconditioning for finite-difference stencils
- Extension to nonlinear implicit systems

## Core types

### LinearOperatorDesc

Identifies the transformation to apply without prescribing implementation:

```cpp
struct LinearOperatorDesc {
    std::string operator_identifier;           // e.g., "spectral_diagonal"
    std::optional<std::string> preconditioner_identifier;
    std::variant<std::monostate, std::vector<double>, std::string> operator_context;
};
```

The `operator_context` can hold:
- Spectral propagator arrays for diagonal operators
- Stencil coefficients for finite-difference methods
- Opaque handles for external solver libraries

### SolveOptions

Bundles stopping criteria for solver attempts:

```cpp
struct SolveOptions {
    int max_iterations = 1000;
    double tolerance = 1e-6;                    // relative tolerance
    std::optional<double> absolute_tolerance;   // disables pure relative if set
    std::optional<LinearOperatorDesc> preconditioner_desc;
};
```

### ConvergenceStatus

Indicates the outcome of a solve attempt:

```cpp
enum class ConvergenceStatus {
    converged,                 // Converged within tolerance
    max_iterations_reached,    // Stopping criteria met without convergence
    stagnation_detected,       // Residual stopped decreasing
    ill_conditioned,           // System is numerically ill-conditioned
    cancelled,                 // Collective cancellation for MPI
    unknown_failure            // Uncategorized failure
};
```

### SolveOutcome

Templates on the solution field bundle type, supporting both in-place and out-of-place patterns:

```cpp
template<typename Fields>
struct SolveOutcome {
    Fields solution;                      // Reference or view to solution
    ConvergenceStatus status;
    int iteration_count;
    double final_residual_norm;
    std::optional<std::string> failure_cause;
};
```

**Commit contract:**
- **In-place solvers:** Mutate `target_out` directly and return `solution = std::ref(target_out)`
- **Out-of-place solvers:** Use internal buffers and return `solution = internal_buffer_view`
- Callers examine `solution` and `status` to determine whether/how to commit
- On failure or cancellation, `target_out` is NOT mutated
- For MPI runs, convergence status must be consistent across ranks

### ExecutionService

Interface for driver-provided distributed operations:

```cpp
class ExecutionService {
public:
    virtual ~ExecutionService() = default;

    virtual void request_halo_exchange(const std::vector<std::string>& field_names) = 0;
    virtual void prepare_boundaries(const std::vector<std::string>& field_names) = 0;
    virtual void global_reduce(const std::vector<double>& data, MPI_Op op) = 0;
};
```

Implemented by the simulation driver (adapting `SimulationContext`) to coordinate halo exchange, boundary preparation, and global reductions during solver iterations.

### StageContext

Context passed to solver functions:

```cpp
struct StageContext {
    double evaluation_time;
    ExecutionService& execution_service;
};
```

### SolveFunction concept

Solver implementations model this concept without inheritance:

```cpp
template<typename Func, typename RHSFields, typename TargetFields>
concept SolveFunction = requires(Func solver,
                                    const LinearOperatorDesc& op_desc,
                                    const RHSFields& rhs,
                                    TargetFields& target_out,
                                    const SolveOptions& options,
                                    const StageContext& ctx) {
    requires tuple_protocol<RHSFields>;
    requires tuple_protocol<TargetFields>;

    { solver(op_desc, rhs, target_out, options, ctx) }
        -> std::same_as<SolveOutcome<TargetFields>>;
};
```

Both `RHSFields` and `TargetFields` must satisfy the tuple protocol from `field/tuple_protocol.hpp`:
- Single field: wrapped via `std::tie(scalar_field)`
- Multiple fields: `std::tuple` or struct with `as_tuple()` member

## Usage examples

### SpectralDiagonalSolver (production CPU diagonal solve)

Header: `include/openpfc/kernel/simulation/spectral_diagonal_solver.hpp`.

`SpectralDiagonalSolver` is a header-only value type that models `SolveFunction`.
It reads real diagonal coefficients from `LinearOperatorDesc::operator_context`
(`std::vector<double>`), writes element-wise @f$ s = b / d @f$ into
caller-owned result storage, and returns `SolveOutcome` residual evidence.
Direct-solve semantics: `iteration_count = 1` on a completed attempt; there is
no separate algorithm field on `SolveOutcome`.

```cpp
#include "openpfc/kernel/simulation/spectral_diagonal_solver.hpp"

using namespace pfc::sim;

SpectralDiagonalConfig config;
config.nullspace_policy = DiagonalNullspacePolicy::fail; // or project / regularize
config.singular_threshold = 1e-14;  // τ
config.null_mode_value = 0.0;      // project only
config.regularization = 0.0;       // λ; must be > 0 when policy is regularize

SpectralDiagonalSolver solver(config);

std::vector<double> diag{2.0, 4.0, 5.0};
std::vector<double> rhs{2.0, 8.0, 15.0};
std::vector<double> target(3, 0.0);

LinearOperatorDesc op_desc{"spectral_diagonal", std::nullopt, diag};
SolveOptions opts{};
opts.absolute_tolerance = 1e-12;
StageContext ctx{current_time, execution_service};

auto outcome = solver(op_desc, rhs, target, opts, ctx);
if (outcome.status == ConvergenceStatus::converged) {
    // Solution is already in target; outcome.solution references it
}
```

#### Nullspace mathematics

Let @f$ \tau @f$ = `singular_threshold`. Mode @f$ i @f$ is singular when
@f$ |d_i| < \tau @f$. Residual always uses the **original** diagonal:
@f$ r_i = d_i s_i - b_i @f$ (not @f$ d_i + \lambda @f$).

| Policy | Behavior |
|--------|----------|
| `fail` | Any singular mode → `ill_conditioned`, `target_out` unchanged |
| `project` | Singular → @f$ s_i = @f$ `null_mode_value`; else @f$ s_i = b_i / d_i @f$. For @f$ d_i=0 @f$, @f$ s_i=0 @f$, residual @f$ r_i = -b_i @f$ — keep @f$ \|b_i\| @f$ small on null modes when compatibility is required |
| `regularize` | Require @f$ \lambda = @f$ `regularization` @f$ > 0 @f$; @f$ s_i = b_i / (d_i + \lambda) @f$ for every @f$ i @f$ (explicit additive shift in units of @f$ d @f$, not a silent epsilon) |

`operator_identifier` must be empty or `"spectral_diagonal"`. Absolute
threshold is `absolute_tolerance.value_or(tolerance)`.

#### Workspace ownership (scratch vs checkpoint)

Solver-owned `residual_scratch_` holds the candidate solution during a solve
attempt. It is **recomputable transient state** and is **not** part of any
checkpoint, `save_state`, or `restore_state` API. On failure or cancellation,
`target_out` is left unmodified (contract rule).

### Iterative solver with preconditioning (out-of-place)

```cpp
auto iterative_solver = [](const LinearOperatorDesc& desc,
                           const auto& rhs,
                           auto& target,
                           const SolveOptions& opts,
                           const StageContext& ctx) {
    // Internal workspace for Krylov subspace
    std::vector<double> internal_buffer(rhs.size());
    std::vector<double> residual(rhs.size());

    // Iteration loop
    for (int iter = 0; iter < opts.max_iterations; ++iter) {
        // Request halo exchange before operator application
        ctx.execution_service.request_halo_exchange({"solution"});
        ctx.execution_service.prepare_boundaries({"solution"});

        // Apply operator: residual = rhs - A * x
        apply_operator(desc, internal_buffer, residual);

        // Compute residual norm
        double local_norm = compute_norm(residual);
        double global_norm;
        ctx.execution_service.global_reduce({local_norm}, MPI_SUM);
        global_norm = std::sqrt(global_norm);

        // Check convergence
        if (global_norm < opts.tolerance) {
            return SolveOutcome<std::vector<double>>{
                internal_buffer,
                ConvergenceStatus::converged,
                iter + 1,
                global_norm,
                std::nullopt
            };
        }

        // Apply preconditioner if specified
        if (opts.preconditioner_desc) {
            apply_preconditioner(*opts.preconditioner_desc, residual);
        }

        // Update solution (Krylov step)
        update_solution(internal_buffer, residual);
    }

    return SolveOutcome<std::vector<double>>{
        internal_buffer,
        ConvergenceStatus::max_iterations_reached,
        opts.max_iterations,
        global_norm,
        std::nullopt
    };
};
```

## Contracts

### Workspace ownership

- **Solver-owned:** Member buffers with lifetime bounded by solve attempt
  (e.g. `SpectralDiagonalSolver::residual_scratch_` — recomputable transient
  state, **excluded from checkpoint / save_state serialization**)
- **Caller-lent:** Scratch arrays passed via `SolveOptions` context
- **Execution-pooled:** Requested through a workspace service
- **Never leaks** into physics models
- **Allocation and lifetime** stay with explicit owner

### Backend neutrality

- CPU, CUDA, and HIP implementations obey identical semantic contracts
- Field bundle structure and interpretation are consistent across backends
- Memory-access patterns, parallel execution, and distribution are backend-specific
- Same convergence evidence for same inputs within defined tolerances

### Failure and cancellation

- On failure or cancellation, `target_out` is NOT mutated
- Cancellation is collective and rank-consistent for distributed solves
- All ranks report `cancelled` status for MPI runs
- Failure reasons provided via `failure_cause` when available

### Nonlinear extension

- Same `SolveOutcome` and `ExecutionService` contracts apply
- Jacobian-vector products fit within operator-application patterns
- Additional incremental update mechanisms supported
- No separate nonlinear base class required

## Preconditioning

Preconditioning is optional and exposed through:

1. **Operator-defined:** Built into linear operator application
2. **Caller-selected:** Via `SolveOptions::preconditioner_desc`
3. **Solver-managed:** Internal algorithm choice

No `Preconditioner` inheritance hierarchy is required; descriptors are solver-specific.

## Distributed preparation

During solver iterations or operator evaluations:

1. Solver requests halo exchange via `ExecutionService::request_halo_exchange`
2. Solver requests boundary preparation via `ExecutionService::prepare_boundaries`
3. Solver performs global reductions via `ExecutionService::global_reduce`

The driver implements `ExecutionService` and coordinates requests without understanding solver iteration structure. Requests occur per-iteration or per-operator-evaluation as needed.

## Field bundle protocol

Field bundles use the tuple protocol from `field/tuple_protocol.hpp`:

- **Single field:** `std::tie(scalar_field)` creates a one-element bundle
- **Multiple fields:** `std::tuple` or struct with `as_tuple()` member
- **Iteration:** Use `for_each_interior` and related utilities for point-wise operations

This protocol enables multi-field systems (e.g., coupled phase-field equations) without requiring specific field types.

## Integration with time integration

The solver contract is the solve vocabulary for IMEX (implicit-explicit) stage
composition. The kernel seam lives in
[`imex_stage_composition.hpp`](../../include/openpfc/kernel/simulation/steppers/imex_stage_composition.hpp)
(`pfc::sim::steppers::ImexEulerComposer`):

1. **Explicit RHS:** Evaluated into stage storage (`ExplicitOperatorEval`)
2. **Implicit solve:** Invoked via `SolveFunction` with `LinearOperatorDesc` + RHS
3. **Isolated candidate:** Written to method-owned buffers; accepted state is
   unchanged until the driver calls `apply_candidate`
4. **Driver:** Decides whether to commit based on `ImexStepAttemptResult`
   / `ConvergenceStatus`

Use `pfc::sim::StageContext` from this contract header (not
`pfc::integrator::StageContext`). First-order IMEX Euler
(`ImexEulerStepper`) is available separately; this contract does not prescribe
tableau coefficients or spectral/Krylov solver implementations.
