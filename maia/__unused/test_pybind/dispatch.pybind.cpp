#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "std_e/utils/enum.hpp"

namespace py = pybind11;


namespace CGNS {
//Enum section
//############

//The strings defined below are type names used for node labels
//#############################################################

// Types as strings
// -----------------
namespace Label {
STD_E_ENUM(kind,
  CGNSTree_t,
  CGNSBase_t,
  Zone_t);
  }
}

template<typename g_num>
void auto_dispatch(py::array_t<g_num>& face_vtx, py::array_t<int>& /*face_vtx_idx*/){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  std::cout << "face_vtx.shape()   " << face_vtx.shape()    << std::endl;
  std::cout << "face_vtx.itemsize()" << face_vtx.itemsize() << std::endl;
  std::cout << "face_vtx.size()    " << face_vtx.size()     << std::endl;
}


PYBIND11_MODULE(dispatch, m) {
  m.doc() = "pybind11 dispatch plugin"; // optional module docstring
  m.def("auto_dispatch", &auto_dispatch<int> , py::arg("face_vtx").noconvert(), py::arg("face_vtx_idx").noconvert());
  m.def("auto_dispatch", &auto_dispatch<long>, py::arg("face_vtx").noconvert(), py::arg("face_vtx_idx").noconvert());

  py::enum_<CGNS::Label::kind>(m, "kind", py::arithmetic(), "A first enum")
  .value("CGNSTree_t", CGNS::Label::CGNSTree_t, " ooo ")
  .value("CGNSBase_t", CGNS::Label::CGNSBase_t, " aaa ")
  .value("Zone_t"    , CGNS::Label::Zone_t, " aaa ");
  // .export_values();

  m.def("test_enum", [](CGNS::Label::kind k){
    std::cout << " ----------------" << std::endl;
    std::cout << "test_enum :: " << k << std::endl;
    std::cout << " ----------------" << std::endl;
  });
}
