#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

pybind11::array_t<bool>
is_unique_cst_stride_hash(int                     n_elt,
                          int                     stride,
                          pybind11::array_t<int>& np_array);

pybind11::array_t<bool>
is_unique_cst_stride_sort(int                     n_elt,
                          int                     stride,
                          pybind11::array_t<int>& np_array);