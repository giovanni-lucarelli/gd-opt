# Gradient Descent Optimization

This project implements a C++ gradient descent optimization library with support for multivariate function optimization. The design emphasizes object-oriented programming principles including inheritance, polymorphism, and templates to provide flexible optimization strategies and support for different floating-point types.

## 🎯 Project Goals

- **Multivariate Optimization**: Implement gradient descent for functions with multiple variables
- **Object-Oriented Design**: Utilize inheritance and polymorphism for extensible optimization strategies
- **Template Programming**: Support different floating-point types and container types (std::vector/std::array)
- **Python Bindings**: Provide Python interface using pybind11 for easy integration
- **Performance**: Efficient C++ implementation with optimization tracking and history

## 📁 Project Structure

```
gd-opt/
├── include/                          
│   ├── optimizer.h                   
│   └── gradient_descent_optimizer.h  
├── src/
│   └── gradient_descent_optimizer.cpp 
├── template/                         
│   ├── optimizer_t.h
│   ├── gradient_descent_optimizer_t.h
│   └── main_t.cpp
├── binding/
│   └── pyoptimizer.cpp
├── main.cpp
├── demo.ipynb
├── CMakeLists.txt
└── README.md
```

### Core Components

- **`optimizer.h`**: Abstract base class defining the optimization interface
- **`gradient_descent_optimizer.h/cpp`**: Concrete implementation of gradient descent algorithm
- **Template versions**: Generic implementations supporting different data types and containers. (TO BE COMPLETED!)
- **Python bindings**: pybind11-based Python interface for the C++ library
- **Tests and demos**: Example usage and validation programs

## 🚀 How to Build

### Prerequisites
- C++17 compatible compiler
- CMake 3.16 or higher
- Python 3.6+ (for Python bindings)
- pybind11 (for Python bindings)

### Building the C++ Library

From the project root directory:

```bash
mkdir build
cd build
cmake ..
make
```

This will create:
- `liboptimizer.a`: Static library
- `optimizer_tests`: Test executable
- `pyoptimizer.cpython-*.so`: Python module (if pybind11 is available)

### Running Tests

```bash
# Run the C++ test program
./optimizer_tests

# Run Python demos (if Jupyter is installed)
jupyter notebook ../demo.ipynb
```

### Build Targets

- **`optimizer`**: Core C++ library
- **`optimizer_tests`**: Test executable demonstrating usage
- **`pyoptimizer`**: Python module with pybind11 bindings

## 💡 Usage Example

### C++ Usage

```cpp
#include "gradient_descent_optimizer.h"

// Define your function and gradient
auto f = [](const std::vector<double>& x) {
    return x[0]*x[0] + x[1]*x[1];  // f(x,y) = x² + y²
};

auto grad_f = [](const std::vector<double>& x) {
    return std::vector<double>{2.0*x[0], 2.0*x[1]};  // ∇f = (2x, 2y)
};

// Create optimizer
auto optimizer = std::make_unique<GradientDescentOptimizer>(0.1);  // learning_rate = 0.1

// Optimize
std::vector<double> initial_point = {1.0, -1.0};
auto solution = optimizer->minimize(f, grad_f, initial_point);
```

### Python Usage

```python
import pyoptimizer

# Define optimizer
opt = pyoptimizer.GradientDescentOptimizer(0.1)

# Define function and gradient
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return [2*x[0], 2*x[1]]

# Optimize
solution = opt.minimize(f, grad_f, [1.0, -1.0])
```

## 🧪 Algorithm Details

The gradient descent implementation follows the iterative update rule:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

Where:
- $x_k$ is the current point
- $\alpha$ is the learning rate
- $\nabla f(x_k)$ is the gradient at the current point

The algorithm terminates when:
- Maximum iterations reached
- Convergence tolerance met ($||\nabla f(x)|| < \text{tolerance}$)

## TODO

- [ ] write tests
- [ ] fix C++ templates