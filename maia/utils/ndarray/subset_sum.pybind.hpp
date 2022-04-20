#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

auto search_subset_match(pybind11::array_t<int> sorted_np, int target, int max_it) -> pybind11::list;
