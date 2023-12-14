#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "maia/utils/pybind_utils.hpp"

void find_duplicate_elt (int                                               n_elt,
                         int                                               elt_size,
                         pybind11::array_t<int, pybind11::array::f_style>& np_elt_vtx,
                         pybind11::array_t<int, pybind11::array::f_style>& np_elt_mask);

void find_duplicate_elt3(int                                               n_elt,
                         int                                               elt_size,
                         pybind11::array_t<int, pybind11::array::f_style>& np_elt_vtx,
                         pybind11::array_t<int, pybind11::array::f_style>& np_elt_mask);