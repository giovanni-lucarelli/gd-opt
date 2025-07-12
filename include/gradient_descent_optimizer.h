#pragma once
#include "optimizer.h"

class GradientDescentOptimizer : public Optimizer {
public:

    using Optimizer::Optimizer;           // inherit constructor

    std::vector<double> minimize(
        std::function<double(const std::vector<double>&)> f,
        std::function<std::vector<double>(const std::vector<double>&)> grad_f,
        std::vector<double> initial_x,
        std::size_t max_iterations = 1000,
        double tolerance          = 1e-6
    ) override;
};
