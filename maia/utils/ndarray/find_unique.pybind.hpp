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

std::tuple<pybind11::array_t<int32_t>, pybind11::array_t<int32_t>>
make_unique_by_stride_int32(pybind11::array_t<int32_t>& np_stride,
                            pybind11::array_t<int32_t>& np_array);
std::tuple<pybind11::array_t<int32_t>, pybind11::array_t<int64_t>>
make_unique_by_stride_int64(pybind11::array_t<int32_t>& np_stride,
                            pybind11::array_t<int64_t>& np_array);

void
sort_by_stride(const pybind11::array np_stride,
                     pybind11::array np_array);
