#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pdm.h>

namespace py = pybind11;

template<typename T> auto
make_raw_view(py::array_t<T>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

template<typename T>
py::array_t<T, py::array::f_style>
extract_from_indices(py::array_t<T> np_array, py::array_t<int> np_indices, int stride, int shift){

  int size         = np_indices.size();
  int extract_size = size * stride;

  auto indices = make_raw_view(np_indices);
  auto array   = make_raw_view(np_array);

  auto np_extract_array = py::array_t<T>(extract_size);
  auto extract_array   = make_raw_view(np_extract_array);

  for(int i = 0; i < size; ++i) {
    int idx = indices[i]-shift;
    for(int s = 0; s < stride; ++s) {
      extract_array[stride*i + s] = array[stride*idx + s];
    }
  }

  return np_extract_array;
}


PYBIND11_MODULE(extract_from_indices, m) {
  m.doc() = "pybind11 extract_from_indices plugin"; // optional module docstring

  m.def("extract_from_indices", &extract_from_indices<double>,
        py::arg("array"  ).noconvert(),
        py::arg("indices").noconvert(),
        py::arg("stride").noconvert(),
        py::arg("shift").noconvert());

  m.def("extract_from_indices", &extract_from_indices<int>,
        py::arg("array"  ).noconvert(),
        py::arg("indices").noconvert(),
        py::arg("stride").noconvert(),
        py::arg("shift").noconvert());

  m.def("extract_from_indices", &extract_from_indices<PDM_g_num_t>,
        py::arg("array"  ).noconvert(),
        py::arg("indices").noconvert(),
        py::arg("stride").noconvert(),
        py::arg("shift").noconvert());

}
