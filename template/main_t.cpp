#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

#include "gradient_descent_optimizer.h"      // original run‑time (vector<double>)
#include "gradient_descent_optimizer_t.h"    // new templated versions

using Clock = std::chrono::steady_clock;

template<typename V>
inline auto rosenbrock_f(const V& v) -> decltype(+v[0])
{
    auto x = v[0];
    auto y = v[1];
    return (1 - x)*(1 - x) + 100*(y - x*x)*(y - x*x);
}

template<typename V>
inline V rosenbrock_grad(const V& v)
{
    using T = typename std::remove_reference<decltype(v[0])>::type;
    T x = v[0], y = v[1];
    return V{ static_cast<T>(-2*(1 - x) - 400*x*(y - x*x)),
              static_cast<T>(200*(y - x*x)) };
}

// helpers


template<typename V>
void print_vec(const V& v)
{
    std::cout << '(' << v[0] << ", " << v[1] << ')';
}

auto ms(auto start, auto end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main()
{
    std::cout << std::fixed << std::setprecision(5);

    // Original version vector<double>
    {
        std::cout << "\n Original GradientDescentOptimizer (std::vector<double>)\n";
        std::unique_ptr<Optimizer> opt = std::make_unique<GradientDescentOptimizer>(0.001);
        std::vector<double> x0{ -1.2, 1.0 };

        auto t0 = Clock::now();
        auto sol = opt->minimize(rosenbrock_f<std::vector<double>>, rosenbrock_grad<std::vector<double>>, x0, 20'000, 1e-8);
        auto t1 = Clock::now();

        std::cout << "  sol     = "; print_vec(sol); std::cout << '\n';
        std::cout << "  iters   = " << opt->get_history().size() - 1 << '\n';
        std::cout << "  time    = " << ms(t0, t1) << " ms\n";
    }

    // Template, dynamic size, double (std::vector<double>)
    {
        std::cout << "\nGradientDescentOptimizerT<double,0> (std::vector<double>)\n";
        GradientDescentOptimizerT<double,0> opt(0.001);
        std::vector<double> x0{ -1.2, 1.0 };

        auto t0 = Clock::now();
        auto sol = opt.minimize(rosenbrock_f<std::vector<double>>, rosenbrock_grad<std::vector<double>>, x0, 20'000, 1e-8);
        auto t1 = Clock::now();

        std::cout << "  sol     = "; print_vec(sol); std::cout << '\n';
        std::cout << "  iters   = " << opt.get_history().size() - 1 << '\n';
        std::cout << "  time    = " << ms(t0, t1) << " ms\n";
    }

    // Template, fixed size, float (std::array<float,2>)
    {
        std::cout << "\nGradientDescentOptimizerT<float,2> (std::array<float,2>)\n";
        GradientDescentOptimizerT<float,2> opt(0.001f);
        std::array<float,2> x0{ -1.2f, 1.0f };

        auto t0 = Clock::now();
        auto sol = opt.minimize(rosenbrock_f<std::array<float,2>>, rosenbrock_grad<std::array<float,2>>, x0, 20'000, 1e-5f);
        auto t1 = Clock::now();

        std::cout << "  sol     = "; print_vec(sol); std::cout << '\n';
        std::cout << "  iters   = " << opt.get_history().size() - 1 << '\n';
        std::cout << "  time    = " << ms(t0, t1) << " ms\n";
    }

    return 0;
}
