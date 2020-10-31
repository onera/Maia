#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<int>
compute_idx_from_color(py::array_t<int, py::array::f_style>& np_color){
  int n_entitiy = np_color.size();

  py::buffer_info buf = np_color.request();
  int* color  = static_cast<int *>(buf.ptr);
  auto [min, max] = std::minmax_element(color, color+n_entitiy);

  int min_color = *min;
  int max_color = *max;

  assert(min_color == 0);
  max_color++;

  // std::cout << "min : " << *min << " | max " << max_color << std::endl;
  py::array_t<int> np_color_idx(max_color+1);

  py::buffer_info buf2 = np_color_idx.request();
  int* color_idx  = static_cast<int *>(buf2.ptr);

  for(int i = 0; i < max_color+1; ++i) {
    color_idx[i] = 0;
  }

  for(int i = 0; i < n_entitiy; ++i) {
    // printf(" color [%i] = %i \n", i, color[i]);
    color_idx[color[i]+1]++;
  }

  for (int i = 0; i < max_color; i++){
    color_idx[i+1] += color_idx[i];
    // printf("color_idx[%i] = %i\n", i+1, color_idx[i+1]);
  }

  // std::cout << " n_entitiy : " << n_entitiy << std::endl;
  // std::cout << " color_idx[max_color] : " << color_idx[max_color] << std::endl;
  assert(color_idx[max_color] == n_entitiy);

  return np_color_idx;
}


PYBIND11_MODULE(utils, m) {
  m.doc() = "pybind11 utils fpr partitioning plugin"; // optional module docstring

  m.def("compute_idx_from_color", &compute_idx_from_color,
        py::arg("color"));

}
