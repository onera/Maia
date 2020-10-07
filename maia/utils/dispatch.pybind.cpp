#include <iostream>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

template<typename g_num>
void auto_dispatch(py::array_t<g_num>& face_vtx, py::array_t<int>& face_vtx_idx){
  std::cout << "face_vtx.shape()   " << face_vtx.shape()    << std::endl;
  std::cout << "face_vtx.itemsize()" << face_vtx.itemsize() << std::endl;
  std::cout << "face_vtx.size()    " << face_vtx.size()     << std::endl;
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}


PYBIND11_MODULE(dispatch, m) {
  m.doc() = "pybind11 dispatch plugin"; // optional module docstring

  m.def("auto_dispatch", &auto_dispatch<int> , "A test with numpy array");
  m.def("auto_dispatch", &auto_dispatch<long> , "A test with numpy array");
  m.def("auto_dispatch", &auto_dispatch<double> , "A test with numpy array");

}
