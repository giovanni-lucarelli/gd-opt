#include "gradient_descent_optimizer.h"
#include <cmath>
#include <stdexcept>

std::vector<double> GradientDescentOptimizer::minimize(
    std::function<double(const std::vector<double>&)> f,
    std::function<std::vector<double>(const std::vector<double>&)> grad_f,
    std::vector<double> x,
    std::size_t max_iterations,
    double tolerance)
{
    if (x.empty()) throw std::invalid_argument("Initial point is empty");
    history.clear();
    history.reserve(max_iterations + 1);

    for (std::size_t k = 0; k < max_iterations; ++k) {
        double fval = f(x);
        std::vector<double> grad = grad_f(x);
        if (grad.size() != x.size())
            throw std::runtime_error("Gradient dimensionality mismatch");

        // compute the squared norm of grad
        double norm2 = 0.0;
        for (double g : grad) norm2 += g * g;
        norm2 = std::sqrt(norm2);

        history.push_back({k, x, fval});
        if (norm2 < tolerance) break;

        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] -= learning_rate * grad[i];
    }
    return x;
}
