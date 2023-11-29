#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pdm.h"

auto search_subset_match(pybind11::array_t<PDM_g_num_t> sorted_np, int target, int max_it) -> pybind11::list;
