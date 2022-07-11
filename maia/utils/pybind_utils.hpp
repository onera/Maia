#pragma once

#include <pybind11/pybind11.h>

template<typename T> auto
make_raw_view(pybind11::array_t<T, pybind11::array::f_style>& x){
  pybind11::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

