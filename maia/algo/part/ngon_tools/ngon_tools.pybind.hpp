#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// --------------------------------------------------------------------
std::tuple<pybind11::array_t<int, pybind11::array::f_style>, 
           pybind11::array_t<int, pybind11::array::f_style>>
local_pe_to_local_cellface(pybind11::array_t<int, pybind11::array::f_style>& pe);
