#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "optimizer.h"
#include "gradient_descent_optimizer.h"

namespace py   = pybind11;
using    vec_d = std::vector<double>;

PYBIND11_MODULE(pyoptimizer, m)
{
    // record struct
    py::class_<Optimizer::Record>(m, "Record")
        .def_readonly("iteration", &Optimizer::Record::iteration)
        .def_readonly("x",         &Optimizer::Record::x)
        .def_readonly("fval",      &Optimizer::Record::fval)
        .def("__repr__", [](const Optimizer::Record& r){
            return "Record iter=" + std::to_string(r.iteration) +
                   " f=" + std::to_string(r.fval) + ">";
        });

    // Abstract base
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("set_learning_rate", &Optimizer::set_learning_rate)
        .def("get_learning_rate", &Optimizer::get_learning_rate)
        .def("get_history",       [](Optimizer& o) {            // copy into Python list
            return o.get_history();
        }, py::return_value_policy::copy);

    py::class_<GradientDescentOptimizer, Optimizer,
               std::shared_ptr<GradientDescentOptimizer>>(m, "GradientDescentOptimizer")
        .def(py::init<double>(),         py::arg("learning_rate") = 0.01,
             "Create a gradient-descent optimiser with given learning-rate")
        .def("minimize",
             &GradientDescentOptimizer::minimize,
             py::arg("f"),
             py::arg("grad_f"),
             py::arg("initial_x"),
             py::arg("max_iterations") = 1000,
             py::arg("tolerance")      = 1e-6,
             R"doc(
Minimise f starting at initial x.
`f` and `grad_f` are ordinary Python callables that take a list or np.array
and return the objective value and gradient respectively.
Returns the final position.)doc");
}
