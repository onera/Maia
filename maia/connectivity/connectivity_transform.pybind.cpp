#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename g_num>
void pe_cgns_to_pdm_face_cell(py::array_t<g_num>& pe, py::array_t<int>& face_cell){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  std::cout << "face_vtx.shape()   " << pe.shape()    << std::endl;
  std::cout << "face_vtx.itemsize()" << pe.itemsize() << std::endl;
  std::cout << "face_vtx.size()    " << pe.size()     << std::endl;
}


PYBIND11_MODULE(connectivity_transform, m) {
  m.doc() = "pybind11 connectivity_transform plugin"; // optional module docstring
  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int> , py::arg("pe").noconvert(), py::arg("face_cell").noconvert());
  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<long>, py::arg("pe").noconvert(), py::arg("face_cell").noconvert());

}
