/*
 * TMP wrapping of PDM_distrib_weight -- To remove when Cython method is avalaible
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pdm.h>

auto compute_weighted_distribution(pybind11::list, pybind11::list weights, pybind11::handle) -> pybind11::array_t<PDM_g_num_t>;
