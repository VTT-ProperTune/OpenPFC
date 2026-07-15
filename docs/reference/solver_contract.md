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

### Spectral diagonal solver (in-place)

```cpp
auto spectral_diagonal_solver = [](const LinearOperatorDesc& desc,
                                    const auto& rhs,
                                    auto& target,
                                    const SolveOptions& opts,
                                    const StageContext& ctx) {
    // Extract spectral propagator from operator_context
    const auto& propagator = std::get<std::vector<double>>(desc.operator_context);
    
    // Element-wise division: target = rhs / propagator
    // (using field bundle iteration utilities)
    for_each_interior(rhs, target, [&](auto rhs_val, auto target_val) {
        target_val = rhs_val / propagator[/* index */];
    });
    
    return SolveOutcome<decltype(target)>{
        target,
        ConvergenceStatus::converged,
        1,
        0.0,
        std::nullopt
    };
};

// Usage
double rhs_field = 1.0;
double solution_field = 0.0;
auto rhs_bundle = std::tie(rhs_field);
auto target_bundle = std::tie(solution_field);

LinearOperatorDesc op_desc{"spectral_diagonal", std::nullopt, propagator_array};
SolveOptions opts{1000, 1e-8};
StageContext ctx{current_time, execution_service};

auto outcome = spectral_diagonal_solver(op_desc, rhs_bundle, target_bundle, opts, ctx);
if (outcome.status == ConvergenceStatus::converged) {
    // Solution is already in target_bundle
}
```

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

The solver contract is designed to work with IMEX (implicit-explicit) time integration:

1. **Explicit RHS:** Directly evaluated without solving
2. **Implicit RHS:** Solved through solver contract with operator descriptor
3. **IMEX integrator:** Combines explicit evaluation with implicit solves
4. **Driver:** Manages step acceptance based on convergence evidence

See the implicit and IMEX time integration framework documentation for full integration patterns.
