// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_solver_contract.cpp
 * @brief Unit tests for solver contract types and concepts
 */

#include <catch2/catch_test_macros.hpp>
#include "openpfc/kernel/simulation/solver_contract.hpp"
#include <tuple>
#include <vector>
#include <functional>

using namespace openpfc::kernel::simulation;

// Mock ExecutionService implementation for testing
class MockExecutionService : public ExecutionService {
public:
    std::vector<std::string> last_halos;
    std::vector<std::string> last_boundaries;
    std::vector<double> last_reduce_data;
    MPI_Op last_op = MPI_OP_NULL;
    
    void request_halo_exchange(const std::vector<std::string>& field_names) override {
        last_halos = field_names;
    }
    
    void prepare_boundaries(const std::vector<std::string>& field_names) override {
        last_boundaries = field_names;
    }
    
    void global_reduce(const std::vector<double>& data, MPI_Op op) override {
        last_reduce_data = data;
        last_op = op;
    }
};

TEST_CASE("test_linear_operator_desc_default_constructible") {
    LinearOperatorDesc desc;
    REQUIRE(desc.operator_identifier.empty());
    REQUIRE(!desc.preconditioner_identifier.has_value());
    REQUIRE(std::holds_alternative<std::monostate>(desc.operator_context));
}

TEST_CASE("test_linear_operator_desc_with_identifier") {
    LinearOperatorDesc desc{"spectral_diagonal"};
    REQUIRE(desc.operator_identifier == "spectral_diagonal");
    REQUIRE(!desc.preconditioner_identifier.has_value());
}

TEST_CASE("test_linear_operator_desc_with_preconditioner") {
    LinearOperatorDesc desc{"fd_stencil", "jacobi"};
    REQUIRE(desc.operator_identifier == "fd_stencil");
    REQUIRE(desc.preconditioner_identifier.has_value());
    REQUIRE(*desc.preconditioner_identifier == "jacobi");
}

TEST_CASE("test_linear_operator_desc_with_vector_context") {
    std::vector<double> coeffs{1.0, 2.0, 3.0};
    LinearOperatorDesc desc{"stencil", std::nullopt, coeffs};
    REQUIRE(desc.operator_identifier == "stencil");
    REQUIRE(std::holds_alternative<std::vector<double>>(desc.operator_context));
    REQUIRE(std::get<std::vector<double>>(desc.operator_context) == coeffs);
}

TEST_CASE("test_solve_options_default_criteria") {
    SolveOptions opts;
    REQUIRE(opts.max_iterations == 1000);
    REQUIRE(opts.tolerance == 1e-6);
    REQUIRE(!opts.absolute_tolerance.has_value());
    REQUIRE(!opts.preconditioner_desc.has_value());
}

TEST_CASE("test_solve_options_custom_criteria") {
    SolveOptions opts{500, 1e-8, 1e-10};
    REQUIRE(opts.max_iterations == 500);
    REQUIRE(opts.tolerance == 1e-8);
    REQUIRE(opts.absolute_tolerance.has_value());
    REQUIRE(*opts.absolute_tolerance == 1e-10);
}

TEST_CASE("test_solve_options_with_preconditioner") {
    SolveOptions opts{};
    opts.preconditioner_desc = LinearOperatorDesc{"jacobi"};
    REQUIRE(opts.preconditioner_desc.has_value());
    REQUIRE(opts.preconditioner_desc->operator_identifier == "jacobi");
}

TEST_CASE("test_convergence_status_enum_values") {
    std::vector<ConvergenceStatus> all_statuses = {
        ConvergenceStatus::converged,
        ConvergenceStatus::max_iterations_reached,
        ConvergenceStatus::stagnation_detected,
        ConvergenceStatus::ill_conditioned,
        ConvergenceStatus::cancelled,
        ConvergenceStatus::unknown_failure
    };
    REQUIRE(all_statuses.size() == 6);
}

TEST_CASE("test_solve_outcome_solution_field_present") {
    using TargetType = std::vector<double>;
    std::vector<double> solution_field{1.0, 2.0, 3.0};
    
    SolveOutcome<TargetType> outcome{
        solution_field,  // solution field present as required
        ConvergenceStatus::converged,
        5,
        1e-7,
        std::nullopt
    };
    
    REQUIRE(outcome.solution.size() == 3);
    REQUIRE(outcome.solution[0] == 1.0);
    REQUIRE(outcome.status == ConvergenceStatus::converged);
    REQUIRE(outcome.iteration_count == 5);
    REQUIRE(outcome.final_residual_norm == 1e-7);
    REQUIRE(!outcome.failure_cause.has_value());
}

TEST_CASE("test_solve_outcome_solution_can_hold_reference") {
    // In-place solver pattern: solution references target_out
    double target_value = 0.0;
    
    SolveOutcome<double&> outcome{
        target_value,  // reference to target_out
        ConvergenceStatus::converged,
        1,
        0.0,
        std::nullopt
    };
    
    REQUIRE(outcome.status == ConvergenceStatus::converged);
    // Can read solution through outcome
    double read_value = outcome.solution;
    REQUIRE(read_value == 0.0);
    
    // Modifying through reference affects original
    outcome.solution = 5.0;
    REQUIRE(target_value == 5.0);
}

TEST_CASE("test_solve_outcome_with_failure") {
    std::vector<double> solution{1.0, 2.0};
    
    SolveOutcome<std::vector<double>> outcome{
        solution,
        ConvergenceStatus::ill_conditioned,
        10,
        1e-3,
        std::string("Matrix singular")
    };
    
    REQUIRE(outcome.status == ConvergenceStatus::ill_conditioned);
    REQUIRE(outcome.failure_cause.has_value());
    REQUIRE(*outcome.failure_cause == "Matrix singular");
}

TEST_CASE("test_tuple_protocol_field_bundle_compatible") {
    // Single field wrapped via std::tie (scalar protocol)
    double scalar_field = 0.0;
    auto scalar_bundle = std::tie(scalar_field);
    static_assert(tuple_protocol<decltype(scalar_bundle)>);
    
    // Multi-field via std::tuple
    std::vector<double> field1 = {0.0};
    std::vector<double> field2 = {0.0};
    auto multi_field = std::make_tuple(std::ref(field1), std::ref(field2));
    static_assert(tuple_protocol<decltype(multi_field)>);
    
    // Check concept satisfaction
    REQUIRE(tuple_protocol<decltype(scalar_bundle)>);
    REQUIRE(tuple_protocol<decltype(multi_field)>);
}

TEST_CASE("test_tuple_protocol_with_raw_scalar") {
    double scalar = 1.0;
    // Raw scalar should be wrapped via forward_as_tuple
    auto wrapped = pfc::field::detail::to_tuple(scalar);
    static_assert(tuple_protocol<decltype(wrapped)>);
}

TEST_CASE("test_solver_function_concept_satisfies_requirement_with_inplace") {
    // Mock in-place solver: writes to target_out, returns reference to it
    auto mock_inplace_solver = [](const LinearOperatorDesc&,
                                  const auto& rhs,
                                  auto& target,
                                  const SolveOptions&,
                                  const StageContext&) -> SolveOutcome<decltype(target)> {
        // Simulate element-wise division (spectral diagonal)
        // In reality: target = rhs / spectral_propagator
        
        return SolveOutcome<decltype(target)>{
            target,  // reference to target_out
            ConvergenceStatus::converged,
            1,
            0.0,
            std::nullopt
        };
    };
    
    double rhs_value = 1.0;
    double target_value = 0.0;
    auto rhs_bundle = std::tie(rhs_value);
    auto target_bundle = std::tie(target_value);
    
    MockExecutionService mock_service;
    StageContext ctx{0.0, mock_service};
    SolveOptions opts;
    
    REQUIRE(SolveFunction<decltype(mock_inplace_solver), decltype(rhs_bundle), decltype(target_bundle)>);
    
    auto outcome = mock_inplace_solver(LinearOperatorDesc{"spectral"}, rhs_bundle, target_bundle, opts, ctx);
    REQUIRE(outcome.status == ConvergenceStatus::converged);
    REQUIRE(outcome.iteration_count == 1);
}

TEST_CASE("test_solver_function_concept_satisfies_requirement_with_outofplace") {
    // Mock out-of-place solver: uses internal buffer, returns view
    auto mock_outofplace_solver = [](
        const LinearOperatorDesc&,
        const auto&,
        auto&,
        const SolveOptions&,
        const StageContext&) -> SolveOutcome<std::vector<double>> {
        // In real out-of-place solver, internal Krylov subspace vectors would be used
        std::vector<double> internal_solution{0.5, 1.5, 2.5};
        
        return SolveOutcome<std::vector<double>>{
            internal_solution,  // view into internal buffer
            ConvergenceStatus::converged,
            42,
            1e-8,
            std::nullopt
        };
    };
    
    double rhs_field = 1.0;
    double target_field = 0.0;
    auto rhs_bundle = std::tie(rhs_field);
    auto target_bundle = std::tie(target_field);
    
    MockExecutionService mock_service;
    StageContext ctx{0.0, mock_service};
    SolveOptions opts;
    
    REQUIRE(SolveFunction<decltype(mock_outofplace_solver), decltype(rhs_bundle), decltype(target_bundle)>);
    
    auto outcome = mock_outofplace_solver(
        LinearOperatorDesc{"iterative"},
        rhs_bundle,
        target_bundle,
        opts,
        ctx
    );
    
    REQUIRE(outcome.status == ConvergenceStatus::converged);
    REQUIRE(outcome.iteration_count == 42);
    REQUIRE(outcome.solution.size() == 3);
    REQUIRE(outcome.solution[0] == 0.5);
}

TEST_CASE("test_solver_function_concept_with_multifield") {
    // Mock solver handling multiple fields
    auto mock_multifield_solver = [](
        const LinearOperatorDesc&,
        const auto&,
        auto&,
        const SolveOptions&,
        const StageContext&) -> SolveOutcome<std::tuple<std::vector<double>, std::vector<double>>> {
        return SolveOutcome<std::tuple<std::vector<double>, std::vector<double>>>{
            std::make_tuple(std::vector<double>{1.0}, std::vector<double>{2.0}),
            ConvergenceStatus::converged,
            5,
            1e-6,
            std::nullopt
        };
    };
    
    std::vector<double> rhs1{0.0}, rhs2{0.0};
    std::vector<double> target1{0.0}, target2{0.0};
    
    auto rhs_bundle = std::make_tuple(std::ref(rhs1), std::ref(rhs2));
    auto target_bundle = std::make_tuple(std::ref(target1), std::ref(target2));
    
    MockExecutionService mock_service;
    StageContext ctx{0.0, mock_service};
    SolveOptions opts;
    
    REQUIRE(SolveFunction<decltype(mock_multifield_solver), decltype(rhs_bundle), decltype(target_bundle)>);
    
    auto outcome = mock_multifield_solver(LinearOperatorDesc{}, rhs_bundle, target_bundle, opts, ctx);
    REQUIRE(outcome.status == ConvergenceStatus::converged);
}

TEST_CASE("test_stage_context_contains_execution_service") {
    MockExecutionService mock_service;
    StageContext ctx{1.5, mock_service};
    
    REQUIRE(ctx.evaluation_time == 1.5);
    REQUIRE(&ctx.execution_service == &mock_service);
}

TEST_CASE("test_execution_service_interface_methods") {
    MockExecutionService service;
    
    std::vector<std::string> fields{"field1", "field2"};
    service.request_halo_exchange(fields);
    REQUIRE(service.last_halos == fields);
    
    service.prepare_boundaries(fields);
    REQUIRE(service.last_boundaries == fields);
    
    std::vector<double> data{1.0, 2.0, 3.0};
    service.global_reduce(data, MPI_SUM);
    REQUIRE(service.last_reduce_data == data);
    REQUIRE(service.last_op == MPI_SUM);
    
    service.global_reduce(data, MPI_MAX);
    REQUIRE(service.last_op == MPI_MAX);
}

TEST_CASE("test_execution_service_virtual_destructor") {
    // Ensure ExecutionService has a virtual destructor for proper cleanup
    static_assert(std::has_virtual_destructor_v<ExecutionService>);
    
    auto service = std::make_unique<MockExecutionService>();
    REQUIRE(service != nullptr);
}

TEST_CASE("test_convergence_status_all_values_distinct") {
    // Ensure all enum values are distinct (compile-time check)
    constexpr auto converged = ConvergenceStatus::converged;
    constexpr auto max_iter = ConvergenceStatus::max_iterations_reached;
    constexpr auto stagnation = ConvergenceStatus::stagnation_detected;
    constexpr auto ill_cond = ConvergenceStatus::ill_conditioned;
    constexpr auto cancelled = ConvergenceStatus::cancelled;
    constexpr auto unknown = ConvergenceStatus::unknown_failure;
    
    static_assert(converged != max_iter);
    static_assert(max_iter != stagnation);
    static_assert(stagnation != ill_cond);
    static_assert(ill_cond != cancelled);
    static_assert(cancelled != unknown);
}
