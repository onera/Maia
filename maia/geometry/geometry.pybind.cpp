#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(geometry, m) {
  m.doc() = "pybind11 utils for geomery plugin"; // optional module docstring

  // m.def("compute_idx_from_color", &compute_idx_from_color,
  //       py::arg("color"));

}
