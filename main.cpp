#include <iostream>
#include <memory>
#include "gradient_descent_optimizer.h"

int main() {

    {
        // testing on the quadratic function f(x,y) = x^2 + y^2
        auto f = [](const std::vector<double>& v) {
            return v[0]*v[0] + v[1]*v[1];
        };
        
        // grad of f = (2x, 2y)
        auto grad_f = [](const std::vector<double>& v) {
            return std::vector<double>{ 2.0 * v[0], 2.0 * v[1] };
        };

        // defining a smart ptr to the base class to test polymorphism
        // learning rate = 0.1
        std::unique_ptr<Optimizer> opt = std::make_unique<GradientDescentOptimizer>(0.1);   
        
        // setting initial condition
        std::vector<double> x0{ 1.0, -1.0 };

        auto sol = opt->minimize(f, grad_f, x0, 500, 1e-8);

        std::cout << "Solution: (" << sol[0] << ", " << sol[1] << ")\n";
        std::cout << "Iterations: " << opt->get_history().size() - 1 << '\n';
        std::cout << "Final f(x): " << f(sol) << '\n';
    }

    // testing on the Rosenbrock function
    {
        auto f = [](const std::vector<double>& v) {
            const double x = v[0], y = v[1];
            return (1 - x) * (1 - x) + 100.0 * (y - x * x) * (y - x * x);
        };

        auto grad_f = [](const std::vector<double>& v) {
            const double x = v[0], y = v[1];
            return std::vector<double>{
                -2.0 * (1 - x) - 400.0 * x * (y - x * x),  // df/dx
                 200.0 * (y - x * x)                       // df/dy
            };
        };

        std::unique_ptr<Optimizer> opt_rosenbrock = std::make_unique<GradientDescentOptimizer>(0.001);

        std::vector<double> x0{ 1.0, -2.0 };

        auto sol = opt_rosenbrock->minimize(f, grad_f, x0, 10000, 1e-8);
        // Obs: for lr = 0.1 or 0.01 the algorithm doesn't converge, with a smaller lr we need more steps 

        std::cout << "Solution: (" << sol[0] << ", " << sol[1] << ")\n";
        std::cout << "Iterations: " << opt_rosenbrock->get_history().size() - 1 << '\n';
        std::cout << "Final f(x): " << f(sol) << '\n';
    }


}
