#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "maia/utils/pybind_utils.hpp"
#include "std_e/logging/time_logger.hpp"

namespace py = pybind11;

void find_duplicate_elt (            int                       n_elt,
                                     int                       elt_size,
                         py::array_t<int, py::array::f_style>& np_elt_vtx,
                         py::array_t<int, py::array::f_style>& np_elt_mask);

void find_duplicate_elt2(            int                       n_elt,
                                     int                       elt_size,
                         py::array_t<int, py::array::f_style>& np_elt_vtx,
                         py::array_t<int, py::array::f_style>& np_elt_mask);

void find_duplicate_elt3(            int                       n_elt,
                                     int                       elt_size,
                         py::array_t<int, py::array::f_style>& np_elt_vtx,
                         py::array_t<int, py::array::f_style>& np_elt_mask);