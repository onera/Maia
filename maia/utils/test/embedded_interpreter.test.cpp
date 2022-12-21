#include "pybind11/embed.h"

// This file is not a regular test file, it contains no test
// Its name is *.test.cpp because it belong to the test executable
// It starts a Python interpreter to execute Python code in otherwise pure C++ tests
namespace {
  const auto _ = pybind11::scoped_interpreter{};
}
