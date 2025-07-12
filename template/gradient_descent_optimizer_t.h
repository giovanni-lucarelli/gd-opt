#pragma once

#include <cmath>
#include "optimizer_t.h"

template<typename T, std::size_t N = 0>
class GradientDescentOptimizerT : public OptimizerT<T,N>
{
    using Base = OptimizerT<T,N>;
public:
    using Vec = typename Base::Vec;
    using Base::Base;            // inherit constructors (learning rate)

    Vec minimize(
        std::function<T  (const Vec&)> f,
        std::function<Vec(const Vec&)> grad_f,
        Vec             x,
        std::size_t     max_iterations = 1000,
        T               tolerance      = static_cast<T>(1e-6)) override
    {
        if constexpr (N == 0) {               // dynamic size check
            if (x.empty())
                throw std::invalid_argument("Initial point is empty");
        }

        Base::history.clear();
        Base::history.reserve(max_iterations + 1);

        const std::size_t dim = (N == 0 ? 0 : N);  // if 0 use runtime g.size()

        for (std::size_t k = 0; k < max_iterations; ++k) {
            T   fval = f(x);
            Vec g    = grad_f(x);

            // dynamic‑size case
            if constexpr (N == 0) {
                if (g.size() != x.size())
                    throw std::runtime_error("Gradient dimensionality mismatch");
            }

            // Compute gradient norm
            T norm2{};
            const std::size_t d = (dim ? dim : g.size());
            for (std::size_t i = 0; i < d; ++i)
                norm2 += g[i] * g[i];
            norm2 = static_cast<T>(std::sqrt(norm2));

            // store history & convergence check
            Base::history.push_back({k, x, fval});
            if (norm2 < tolerance) break;

            // GD step
            for (std::size_t i = 0; i < d; ++i)
                x[i] -= Base::learning_rate_ * g[i];
        }
        return x;
    }
};