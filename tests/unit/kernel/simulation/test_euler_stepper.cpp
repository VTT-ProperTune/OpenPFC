// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <tuple>

using namespace pfc;
using namespace pfc::sim::steppers;
using Catch::Matchers::WithinRel;

// Helper RHS callable for constant du/dt = c
struct ConstantRHS {
    double c;
    
    void operator()(double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) const {
        for (size_t i = 0; i < du.size(); ++i) {
            du[i] = c;
        }
    }
};

// Helper linear ODE RHS: du/dt = -lambda * u
struct LinearDecayRHS {
    double lambda;
    
    void operator()(double /*t*/, std::vector<double>& u, std::vector<double>& du) const {
        for (size_t i = 0; i < u.size(); ++i) {
            du[i] = -lambda * u[i];
        }
    }
};

// Analytical solution for exponential decay: u(t) = u0 * exp(-lambda * t)
double analytical_solution(double u0, double lambda, double t) {
    return u0 * std::exp(-lambda * t);
}

// Helper for L2 error computation
double compute_l2_error(const std::vector<double>& numerical, const std::vector<double>& analytical) {
    double error_sq = 0.0;
    for (size_t i = 0; i < numerical.size(); ++i) {
        double diff = numerical[i] - analytical[i];
        error_sq += diff * diff;
    }
    return std::sqrt(error_sq / numerical.size());
}

// Helper to initialize vector with given value
void init_vector(std::vector<double>& vec, double value) {
    for (auto& v : vec) v = value;
}

// Local problem size for tests
constexpr size_t LOCAL_SIZE = 1000;

TEST_CASE("forward_euler_constant_rhs") {
    double c = 2.5;
    ConstantRHS rhs{c};
    
    std::vector<double> dt_values = {0.001, 0.01, 0.1};
    for (double dt : dt_values) {
        std::vector<double> u(LOCAL_SIZE);
        init_vector(u, 1.0);
        std::vector<double> u_initial = u;
        
        EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
        double t = 0.0;
        t = stepper.step(t, u);
        
        for (size_t i = 0; i < u.size(); ++i) {
            double expected = u_initial[i] + dt * c;
            REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
        }
    }
    
    SECTION("multiple steps") {
        double dt = 0.01;
        int steps = 10;
        ConstantRHS rhs{c};
        
        std::vector<double> u(LOCAL_SIZE);
        init_vector(u, 1.0);
        std::vector<double> u_initial = u;
        
        EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
        double t = 0.0;
        for (int i = 0; i < steps; ++i) {
            t = stepper.step(t, u);
        }
        
        for (size_t i = 0; i < u.size(); ++i) {
            double expected = u_initial[i] + steps * dt * c;
            REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
        }
    }
}

TEST_CASE("multieuler_tuple_protocol") {
    // Composite RHS that assigns different constants to each field using tuple protocol
    struct CompositeRHS {
        double c_field1;
        double c_field2;
        
        void operator()(double /*t*/, 
                        std::tuple<std::vector<double>&, std::vector<double>&> u_pack,
                        std::tuple<std::vector<double>&, std::vector<double>&> du_pack) const {
            auto& u1 = std::get<0>(u_pack);
            auto& u2 = std::get<1>(u_pack);
            auto& du1 = std::get<0>(du_pack);
            auto& du2 = std::get<1>(du_pack);
            
            for (size_t i = 0; i < u1.size(); ++i) {
                du1[i] = c_field1;
                du2[i] = c_field2;
            }
        }
    };
    
    constexpr int N_FIELDS = 2;
    double c1 = 2.0;
    double c2 = 3.0;
    
    std::array<std::size_t, N_FIELDS> local_sizes = {LOCAL_SIZE, LOCAL_SIZE};
    
    double dt = 0.01;
    
    std::vector<double> u1(LOCAL_SIZE);
    std::vector<double> u2(LOCAL_SIZE);
    init_vector(u1, 1.0);
    init_vector(u2, 1.0);
    std::vector<double> u1_initial = u1;
    std::vector<double> u2_initial = u2;
    
    CompositeRHS rhs{c1, c2};
    
    // Use correct template order: Rhs first, N second
    MultiEulerStepper<CompositeRHS, N_FIELDS> multi_stepper(dt, local_sizes, rhs);
    
    double t = 0.0;
    t = multi_stepper.step(t, u1, u2);
    
    // Verify tuple protocol: both fields stepped correctly with their RHS values
    for (size_t i = 0; i < u1.size(); ++i) {
        double expected1 = u1_initial[i] + dt * c1;
        double expected2 = u2_initial[i] + dt * c2;
        REQUIRE_THAT(u1[i], WithinRel(expected1, 1e-10));
        REQUIRE_THAT(u2[i], WithinRel(expected2, 1e-10));
    }
}

TEST_CASE("linear_ode_exponential_decay") {
    double lambda = 1.0;
    LinearDecayRHS rhs{lambda};
    
    double dt_coarse = 0.01;
    double dt_fine = 0.005;
    double dt_finer = 0.001;
    
    double t_final = 0.1;
    
    // Coarse grid solution
    std::vector<double> u_coarse(LOCAL_SIZE);
    init_vector(u_coarse, 1.0);
    EulerStepper<LinearDecayRHS> stepper_coarse(dt_coarse, LOCAL_SIZE, rhs);
    double t_coarse = 0.0;
    int steps_coarse = static_cast<int>(t_final / dt_coarse);
    for (int i = 0; i < steps_coarse; ++i) {
        t_coarse = stepper_coarse.step(t_coarse, u_coarse);
    }
    
    // Fine grid solution
    std::vector<double> u_fine(LOCAL_SIZE);
    init_vector(u_fine, 1.0);
    EulerStepper<LinearDecayRHS> stepper_fine(dt_fine, LOCAL_SIZE, rhs);
    double t_fine = 0.0;
    int steps_fine = static_cast<int>(t_final / dt_fine);
    for (int i = 0; i < steps_fine; ++i) {
        t_fine = stepper_fine.step(t_fine, u_fine);
    }
    
    // Finer grid solution
    std::vector<double> u_finer(LOCAL_SIZE);
    init_vector(u_finer, 1.0);
    EulerStepper<LinearDecayRHS> stepper_finer(dt_finer, LOCAL_SIZE, rhs);
    double t_finer = 0.0;
    int steps_finer = static_cast<int>(t_final / dt_finer);
    for (int i = 0; i < steps_finer; ++i) {
        t_finer = stepper_finer.step(t_finer, u_finer);
    }
    
    // Analytical solution (uniform)
    std::vector<double> u_analytical(LOCAL_SIZE);
    double u0 = 1.0;
    double exact = analytical_solution(u0, lambda, t_final);
    for (auto& val : u_analytical) val = exact;
    
    // Compute L2 errors
    double error_coarse = compute_l2_error(u_coarse, u_analytical);
    double error_fine = compute_l2_error(u_fine, u_analytical);
    double error_finer = compute_l2_error(u_finer, u_analytical);
    
    REQUIRE(error_fine <= error_coarse * 0.6 + 1e-12);
    REQUIRE(error_finer <= error_fine * 0.6 + 1e-12);
}

TEST_CASE("rhs_read_only_contract") {
    double c = 1.5;
    
    // RHS that strictly reads from u and writes only to du
    struct ReadOnlyRHS {
        double c;
        
        void operator()(double /*t*/, std::vector<double>& /*u*/, std::vector<double>& du) const {
            for (size_t i = 0; i < du.size(); ++i) {
                du[i] = c;
            }
        }
    };
    
    ReadOnlyRHS rhs{c};
    
    double dt = 0.01;
    int steps = 5;
    
    std::vector<double> u(LOCAL_SIZE);
    init_vector(u, 1.0);
    std::vector<double> u_initial = u;
    
    EulerStepper<ReadOnlyRHS> stepper(dt, LOCAL_SIZE, rhs);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
        t = stepper.step(t, u);
    }
    
    for (size_t i = 0; i < u.size(); ++i) {
        double expected = u_initial[i] + steps * dt * c;
        REQUIRE_THAT(u[i], WithinRel(expected, 1e-10));
    }
}

TEST_CASE("multistep_consistency") {
    double c = 2.0;
    ConstantRHS rhs{c};
    
    double dt = 0.01;
    int N = 10;
    
    // Multi-step evolution
    std::vector<double> u_multi(LOCAL_SIZE);
    init_vector(u_multi, 1.0);
    EulerStepper<ConstantRHS> stepper(dt, LOCAL_SIZE, rhs);
    double t_multi = 0.0;
    for (int i = 0; i < N; ++i) {
        t_multi = stepper.step(t_multi, u_multi);
    }
    
    // Single equivalent step
    std::vector<double> u_single(LOCAL_SIZE);
    init_vector(u_single, 1.0);
    EulerStepper<ConstantRHS> stepper_single(dt * N, LOCAL_SIZE, rhs);
    double t_single = 0.0;
    t_single = stepper_single.step(t_single, u_single);
    
    REQUIRE_THAT(t_multi, WithinRel(t_single, 1e-10));
    for (size_t i = 0; i < u_multi.size(); ++i) {
        REQUIRE_THAT(u_multi[i], WithinRel(u_single[i], 1e-10));
    }
}

TEST_CASE("dt_zero_edge_case") {
    double c = 1.0;
    ConstantRHS rhs{c};
    
    std::vector<double> u(LOCAL_SIZE);
    init_vector(u, 1.0);
    std::vector<double> u_initial = u;
    
    double t = 1.0;
    double dt_zero = 0.0;
    
    EulerStepper<ConstantRHS> stepper(dt_zero, LOCAL_SIZE, rhs);
    double t_out = stepper.step(t, u);
    
    REQUIRE(t_out == t);
    for (size_t i = 0; i < u.size(); ++i) {
        REQUIRE(u[i] == u_initial[i]);
    }
}
