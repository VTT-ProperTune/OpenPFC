// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file solver_contract.hpp
 * @brief Capability contract for linear and implicit subsystem solvers
 *
 * @details
 * Defines the semantic contract that solver implementations must honor without
 * prescribing abstract LinearSolver, solve(A,b,x) interfaces, virtual methods,
 * or matrix representations. Operators provide descriptor-based field-bundle
 * transformations, callers provide right-hand-side and target storage, solvers
 * produce solution state plus convergence evidence, and all intermediate work
 * uses isolated buffers without modifying accepted solution state.
 *
 * The contract supports:
 * - Spectral diagonal solvers (element-wise division for phase-field)
 * - Iterative solvers with preconditioning for finite-difference stencils
 * - Extension to nonlinear implicit systems
 *
 * ## Workspace ownership contract
 *
 * - Solver implementations own intermediate work arrays and temporary field
 *   bundles (e.g., Krylov subspace vectors, residual accumulators)
 * - Workspace is either solver-owned (member buffers with lifetime bounded by
 *   solve attempt), caller-lent (scratch arrays passed via SolveOptions
 *   context), or execution-pooled (requested through a workspace service)
 * - Workspace never leaks into physics models; models only provide operator
 *   descriptors and field bundle access
 * - Allocation and lifetime stay with the explicit owner (solver instance,
 *   execution pool, or caller)
 *
 * ## Backend-neutrality contract
 *
 * - CPU, CUDA, and HIP implementations obey identical semantic contracts for
 *   operator-application, solve outcome, and convergence evidence
 * - Field bundle structure and interpretation are consistent across backends
 * - Memory-access patterns, parallel execution, and distribution are
 *   backend-specific
 *
 * ## Failure and cancellation contract
 *
 * - On failure or cancellation, target_out is NOT mutated (solution state
 *   remains isolated until caller commits)
 * - Cancellation is collective and rank-consistent for distributed solves
 *   (all ranks report cancelled status)
 *
 * ## Nonlinear extension contract
 *
 * - Same SolveOutcome and ExecutionService contracts apply for nonlinear
 *   solvers
 * - Additional Jacobian-vector products or incremental update mechanisms fit
 *   within operator-application patterns
 * - No separate nonlinear base class is required
 *
 * @see docs/reference/solver_contract.md for detailed contract documentation
 * and usage examples
 */

#ifndef OPENPFC_KERNEL_SIMULATION_SOLVER_CONTRACT_HPP
#define OPENPFC_KERNEL_SIMULATION_SOLVER_CONTRACT_HPP

#include <complex>
#include <functional>
#include <mpi.h>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "openpfc/kernel/field/tuple_protocol.hpp"

namespace pfc::sim {

// Forward declaration for tuple_protocol concept
// This will be defined below using utilities from tuple_protocol.hpp

/**
 * @brief Concept for types that satisfy the tuple protocol
 *
 * A type satisfies tuple_protocol if it can be normalized into a tuple-like
 * view for field-bundle operations:
 * - Has a member `as_tuple()` that returns a std::tuple of references
 * - Is already a std::tuple specialization
 * - Is a scalar type (treated as single-field bundle)
 */
template<typename T>
concept tuple_protocol = requires(T& t) {
    { pfc::field::detail::to_tuple(t) } -> std::same_as<decltype(pfc::field::detail::to_tuple(t))>;
};

/**
 * @brief Linear operator descriptor
 *
 * Identifies the transformation to apply without prescribing implementation
 * representation (no matrix type required). The operator_context can hold
 * operator-specific data such as spectral propagator arrays, stencil
 * coefficients, or opaque handles.
 */
struct LinearOperatorDesc {
    /// Operator identifier (e.g., "spectral_diagonal", "finite_difference_stencil")
    std::string operator_identifier;

    /// Optional preconditioner identifier
    std::optional<std::string> preconditioner_identifier;

    /// Operator-specific context (spectral propagator — real or complex —
    /// stencil coeffs, or opaque handle)
    std::variant<std::monostate, std::vector<double>,
                 std::vector<std::complex<double>>, std::string>
        operator_context;
};

/**
 * @brief Stopping criteria for solver iterations
 *
 * Bundles convergence control parameters for solver attempts. Tolerance is
 * relative to the initial residual norm unless absolute_tolerance is set.
 */
struct SolveOptions {
    /// Maximum number of solver iterations
    int max_iterations = 1000;

    /// Relative tolerance on residual norm (default 1e-6)
    double tolerance = 1e-6;

    /// Optional absolute tolerance (disables pure relative tolerance if set)
    std::optional<double> absolute_tolerance;

    /// Optional caller-selected preconditioning descriptor
    std::optional<LinearOperatorDesc> preconditioner_desc;
};

/**
 * @brief Convergence status for the solve attempt
 *
 * Indicates the outcome of a solver attempt, distinguishing between successful
 * convergence and various failure modes.
 */
enum class ConvergenceStatus {
    converged,                 ///< Solver converged within tolerance
    max_iterations_reached,    ///< Stopping criteria met without convergence
    stagnation_detected,       ///< Residual stopped decreasing
    ill_conditioned,           ///< System is numerically ill-conditioned
    cancelled,                 ///< Solve was cancelled (collective for MPI)
    unknown_failure            ///< Uncategorized failure occurred
};

/**
 * @brief Outcome returned by solver implementations
 *
 * Templates on the solution field bundle type (Fields), supporting both
 * in-place and out-of-place solver patterns:
 *
 * - In-place solvers: mutate target_out directly and return solution =
 *   std::ref(target_out)
 * - Out-of-place solvers: use internal buffers and return solution =
 *   internal_buffer_view
 *
 * Callers examine solution and status to determine whether/how to commit to
 * target_out. On failure or cancellation, target_out is NOT mutated and the
 * solution reference is meaningless. For MPI runs, convergence status must be
 * consistent across ranks (collective).
 *
 * @tparam Fields Field bundle type satisfying tuple_protocol
 */
template<typename Fields>
struct SolveOutcome {
    /// Solution state: reference to target_out or view into internal buffer
    Fields solution;

    /// Convergence status of the solve attempt
    ConvergenceStatus status;

    /// Number of iterations performed
    int iteration_count;

    /// Final residual norm
    double final_residual_norm;

    /// Optional failure cause description
    std::optional<std::string> failure_cause;
};

/**
 * @brief Type trait to detect if a type is a SolveOutcome specialization
 */
template<typename T>
struct is_solve_outcome : std::false_type {};

template<typename Fields>
struct is_solve_outcome<SolveOutcome<Fields>> : std::true_type {};

template<typename T>
concept solve_outcome_type = is_solve_outcome<T>::value;

/**
 * @brief Interface for driver-provided distributed operations
 *
 * Implemented by simulation driver (e.g., adapts SimulationContext) to provide
 * halo exchange, boundary preparation, and global reduction capabilities for
 * solver iterations and operator evaluations.
 */
class ExecutionService {
public:
    virtual ~ExecutionService() = default;

    /**
     * @brief Request halo exchange for specified fields before operator evaluation
     *
     * @param field_names Names of fields requiring halo exchange
     */
    virtual void request_halo_exchange(const std::vector<std::string>& field_names) = 0;

    /**
     * @brief Apply boundary conditions to specified fields
     *
     * @param field_names Names of fields requiring boundary preparation
     */
    virtual void prepare_boundaries(const std::vector<std::string>& field_names) = 0;

    /**
     * @brief Perform global reduction of scalar values across all ranks
     *
     * @param data Data values to reduce
     * @param op MPI reduction operation (e.g., MPI_SUM, MPI_MAX)
     */
    virtual void global_reduce(const std::vector<double>& data, MPI_Op op) = 0;
};

/**
 * @brief Context passed to solver functions
 *
 * Contains evaluation time and ExecutionService reference for solver-distributed
 * coordination during iterations and operator evaluations.
 */
struct StageContext {
    /// Evaluation time for the current stage
    double evaluation_time;

    /// Reference to execution service for distributed operations
    ExecutionService& execution_service;
};

/**
 * @brief Concept for solver functions
 *
 * Solver implementations model this concept without inheritance. Both RHS and
 * Target field bundles must satisfy tuple_protocol from field/tuple_protocol.hpp.
 *
 * The concept is backend-agnostic; implementations specify memory-access patterns
 * and distribution details internally. The return type must be a SolveOutcome
 * specialization, but the solution field type may differ from TargetFields to
 * support out-of-place solvers that return internal buffers.
 *
 * @tparam Func Solver function type
 * @tparam RHSFields Right-hand-side field bundle type
 * @tparam TargetFields Target output field bundle type
 */
template<typename Func, typename RHSFields, typename TargetFields>
concept SolveFunction = requires(Func solver,
                                    const LinearOperatorDesc& op_desc,
                                    const RHSFields& rhs,
                                    TargetFields& target_out,
                                    const SolveOptions& options,
                                    const StageContext& ctx) {
    // Both RHS and Target must satisfy tuple_protocol
    requires tuple_protocol<RHSFields>;
    requires tuple_protocol<TargetFields>;

    // Solver must be callable with specified signature
    // and return a SolveOutcome (solution field type may differ from TargetFields)
    { solver(op_desc, rhs, target_out, options, ctx) } -> solve_outcome_type;
};

} // namespace pfc::sim

#endif // OPENPFC_KERNEL_SIMULATION_SOLVER_CONTRACT_HPP
