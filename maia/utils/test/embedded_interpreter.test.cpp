#include "pybind11/embed.h"

// needed to run some tests in pure C++ (where no python interpreter has been started yet)
namespace {
  const pybind11::scoped_interpreter python_interpreter = {};
}
