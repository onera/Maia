#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include "maia/cgns_registry/cgns_keywords.hpp"

namespace py = pybind11;

// void pybind_auto_enum(T& t, E /*e*/){

template <typename Type>
void pybind_auto_enum(py::enum_<Type>& t){
  int nb_enum = std_e::enum_size<Type>;
  for(int i = 0; i < nb_enum; ++i){
    CGNS::Label::kind e_id = static_cast<CGNS::Label::kind>(i);
    auto name = to_string(e_id);
    t.value(name.c_str(), e_id);
  }
}

PYBIND11_MODULE(cgns_keywords, m) {
  m.doc() = "pybind11 utils for cgns_keywords plugin"; // optional module docstring

  auto enum_cgns_kind = py::enum_<CGNS::Label::kind>(m, "kind", py::arithmetic(), "A first enum");
  pybind_auto_enum(enum_cgns_kind);
  enum_cgns_kind.export_values();
  int nb_cgns_labels = std_e::enum_size<CGNS::Label::kind>;
  m.attr("nb_cgns_labels") = nb_cgns_labels;

}
