#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>

//  N == 0  use std::vector<T>
//  N > 0   use std::array<T,N>

template<typename T, std::size_t N>
struct VecTraits                     // N > 0
{
    using Vec = std::array<T, N>;
    static constexpr std::size_t size(const Vec&) noexcept { return N; }
};

template<typename T>
struct VecTraits<T, 0>               // partial specialisation: dynamic size
{
    using Vec = std::vector<T>;
    static std::size_t size(const Vec& v) noexcept { return v.size(); }
};

// dimension generic abstract class

template<typename T, std::size_t N = 0>
class OptimizerT {
public:
    using Vec = typename VecTraits<T,N>::Vec;

    struct Record {
        std::size_t iteration;
        Vec         x;
        T           fval;
    };

    explicit OptimizerT(T lr = static_cast<T>(0.01)) : learning_rate_(lr)
    {
        if (lr <= T{}) throw std::invalid_argument("Learning rate must be > 0");
    }
    virtual ~OptimizerT() = default;

    void   set_learning_rate(T lr)
    {
        if (lr <= T{}) throw std::invalid_argument("Learning rate must be > 0");
        learning_rate = lr;
    }
    T      get_learning_rate() const               { return learning_rate; }
    const std::vector<Record>& get_history() const { return history; }

    virtual Vec minimize(
        std::function<T  (const Vec&)> f,
        std::function<Vec(const Vec&)> grad_f,
        Vec             initial_x,
        std::size_t     max_iterations = 1000,
        T               tolerance      = static_cast<T>(1e-6)) = 0;

protected:
    T learning_rate;
    std::vector<Record> history;
};
