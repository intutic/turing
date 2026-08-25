#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace turing {

/**
 * Multi-Modal Non-Convex Benchmark Objective Landscapes for Swarm Optimizer Verification.
 * Implements Ackley, Rastrigin, Griewank, and Rosenbrock fitness functions.
 */
inline double evaluate_ackley(const double* x, int d) {
    if (d <= 0) return 0.0;
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    const double M_2PI = 2.0 * M_PI;

    for (int i = 0; i < d; ++i) {
        sum_sq += x[i] * x[i];
        sum_cos += std::cos(M_2PI * x[i]);
    }

    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / static_cast<double>(d)));
    double term2 = -std::exp(sum_cos / static_cast<double>(d));
    return term1 + term2 + 20.0 + M_E;
}

inline double evaluate_rastrigin(const double* x, int d) {
    if (d <= 0) return 0.0;
    double val = 10.0 * static_cast<double>(d);
    const double M_2PI = 2.0 * M_PI;

    for (int i = 0; i < d; ++i) {
        val += (x[i] * x[i]) - (10.0 * std::cos(M_2PI * x[i]));
    }
    return val;
}

inline double evaluate_griewank(const double* x, int d) {
    if (d <= 0) return 0.0;
    double sum_sq = 0.0;
    double prod_cos = 1.0;

    for (int i = 0; i < d; ++i) {
        sum_sq += (x[i] * x[i]) / 4000.0;
        prod_cos *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return sum_sq - prod_cos + 1.0;
}

inline double evaluate_rosenbrock(const double* x, int d) {
    if (d <= 1) return 0.0;
    double val = 0.0;
    for (int i = 0; i < d - 1; ++i) {
        double t1 = x[i + 1] - (x[i] * x[i]);
        double t2 = 1.0 - x[i];
        val += (100.0 * t1 * t1) + (t2 * t2);
    }
    return val;
}

inline double evaluate_pso_objective_cpp(const std::string& name, const double* x, int d) {
    if (name == "ackley") return evaluate_ackley(x, d);
    if (name == "rastrigin") return evaluate_rastrigin(x, d);
    if (name == "griewank") return evaluate_griewank(x, d);
    if (name == "rosenbrock") return evaluate_rosenbrock(x, d);
    return 0.0;
}

} // namespace turing
