#pragma once
#include <vector>
#include <functional>
#include <cstddef>
#include <stdexcept>

// Abstract base class for optimizers (with double)

class Optimizer {
public:

// Pure-virtual interface
virtual std::vector<double> minimize(
    std::function<double(const std::vector<double>&)>  f,
    std::function<std::vector<double>(const std::vector<double>&)> grad_f,
    std::vector<double> initial_x,
    std::size_t max_iterations = 1000,
    double tolerance          = 1e-6
    ) = 0;

    Optimizer(double lr = 0.01) : learning_rate(lr){
        if (lr <= 0.0) throw std::invalid_argument("Learning rate must be > 0");
    }

    virtual ~Optimizer() = default;

    void set_learning_rate(double lr){
        if (lr <= 0.0) throw std::invalid_argument("Learning rate must be > 0");
        learning_rate = lr;
    }

    double get_learning_rate() const { return learning_rate; }
        
    // group all the relevant quantities for the history in a struct
    struct Record {
        std::size_t iteration;
        std::vector<double> x;
        double fval;
    };

    const std::vector<Record>& get_history() const { return history; }


protected:
    double learning_rate;
    std::vector<Record> history;
};
