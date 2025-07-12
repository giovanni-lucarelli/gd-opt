cmake_minimum_required(VERSION 3.16)
project(OptimizerProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Core library
add_library(optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gradient_descent_optimizer.cpp
)

target_include_directories(optimizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Test executable
add_executable(optimizer_tests main.cpp)
target_link_libraries(optimizer_tests PRIVATE optimizer)

# pybind11 setup
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${pybind11_dir})
find_package(pybind11 REQUIRED)

# Python module binding
pybind11_add_module(pyoptimizer binding/pyoptimizer.cpp)
target_link_libraries(pyoptimizer PRIVATE optimizer)
target_include_directories(pyoptimizer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
