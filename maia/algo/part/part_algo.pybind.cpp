#if __cplusplus > 201703L
#include "maia/algo/part/gcs_ghosts/gcs_only_for_ghosts.hpp"
#include "maia/__old/transform/remove_ghost_info.hpp"
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#else //C++==17
#endif //C++>17
#include "maia/algo/part/geometry/geometry.pybind.hpp"
#include "maia/algo/part/ngon_tools/ngon_tools.pybind.hpp"
#include "maia/algo/part/cgns_registry/cgns_registry.pybind.hpp"
#include "maia/algo/part/part_algo.pybind.hpp"
#include "maia/utils/pybind_utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void
enforce_pe_left_parent(
    py::array_t<int32_t, py::array::f_style>& np_face_vtx_idx,
    py::array_t<int32_t, py::array::f_style>& np_face_vtx,
    py::array_t<int32_t, py::array::f_style>& np_pe,
    std::optional<py::array_t<int32_t, py::array::f_style>> &np_cell_face_idx,
    std::optional<py::array_t<int32_t, py::array::f_style>> &np_cell_face
)
{
  int n_face = np_pe.size()/2;
  auto pe_ptr        = np_pe.template mutable_unchecked<2>();
  auto face_vtx_idx  = make_raw_view(np_face_vtx_idx);
  auto face_vtx      = make_raw_view(np_face_vtx);
  int *cell_face_idx = nullptr;
  int *cell_face     = nullptr;
  if (np_cell_face_idx.has_value()) { //Take value if not None
    cell_face_idx = make_raw_view(np_cell_face_idx.value());
    cell_face     = make_raw_view(np_cell_face.value());
  }

  int f_shift(0); // To go back to local cell numbering if ngons are before nface
  int c_shift(0); // To go back to local face numbering if nface are before ngons
  if (n_face > 0) {
    if (std::max(pe_ptr(0,0), pe_ptr(0,1)) > n_face ) { //NGons are first
      f_shift = n_face;
    }
    else if (cell_face != nullptr) { //NFace are first
      c_shift = np_cell_face_idx.value().size() - 1; // This is n_cell
    }
  }

  for(int i_face = 0; i_face < n_face; ++i_face) {
    if (pe_ptr(i_face, 0) == 0) {
      // Swap pe
      pe_ptr(i_face, 0) = pe_ptr(i_face, 1);
      pe_ptr(i_face, 1) = 0;

      // Change sign in cell_face
      if (cell_face_idx != nullptr) {
        int i_cell  = pe_ptr(i_face, 0) - f_shift - 1;
        int face_id = i_face + c_shift + 1;
        int* begin = &cell_face[cell_face_idx[i_cell]];
        int* end   = &cell_face[cell_face_idx[i_cell+1]];
        int *find = std::find_if(begin, end, [face_id](int x) {return abs(x) == face_id;});
        if (find != end) {
          *find *= - 1;
        }
      }

      //Swap vertices
      std::reverse(&face_vtx[face_vtx_idx[i_face]+1], &face_vtx[face_vtx_idx[i_face+1]]);
    }
  }
}

#if __cplusplus > 201703L
template<class F> auto
apply_cpp_cgns_function_to_py_base(F&& f) {
  return [&f](py::list py_base) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    f(base);
    update_py_tree(std::move(base),py_base);
  };
}
template<class F> auto
apply_cpp_cgns_par_function_to_py_base(F&& f) {
  return [&f](py::list py_base, py::object mpi4py_comm) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);
    f(base,comm);
    update_py_tree(std::move(base),py_base);
  };
}
const auto gcs_only_for_ghosts          = apply_cpp_cgns_function_to_py_base(cgns::gcs_only_for_ghosts);
const auto remove_ghost_info             = apply_cpp_cgns_par_function_to_py_base(maia::remove_ghost_info);
#else //C++==17
#endif //C++>17


void register_part_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("part_algo");

  m.doc() = "pybind11 part_algo module"; // optional module docstring

  m.def("enforce_pe_left_parent", &enforce_pe_left_parent,
        py::arg("ngon_eso").noconvert(),
        py::arg("ngon_ec").noconvert(),
        py::arg("ngon_pe").noconvert(),
        py::arg("nface_eso").noconvert() = py::none(),
        py::arg("nface_ec").noconvert()  = py::none());

  m.def("local_pe_to_local_cellface", &local_pe_to_local_cellface,
        py::arg("local_pe").noconvert());
  m.def("local_cellface_to_local_pe", &local_cellface_to_local_pe,
        py::arg("np_cell_face_idx").noconvert(),
        py::arg("np_cell_face").noconvert());

  m.def("adapt_match_information", &adapt_match_information,
        py::arg("np_neighbor_idx"    ).noconvert(),
        py::arg("np_neighbor_desc"   ).noconvert(),
        py::arg("np_recv_entity_stri").noconvert(),
        py::arg("np_point_list"      ).noconvert(),
        py::arg("np_point_list_donor").noconvert());

  m.def("compute_face_center_and_characteristic_length", &compute_face_center_and_characteristic_length,
        py::arg("np_point_list").noconvert(),
        py::arg("np_cx").noconvert(),
        py::arg("np_cy").noconvert(),
        py::arg("np_cz").noconvert(),
        py::arg("np_face_vtx").noconvert(),
        py::arg("np_face_vtx_idx").noconvert());

  m.def("compute_center_cell_u", &compute_center_cell_u,
        py::arg("n_cell").noconvert(),
        py::arg("np_cx").noconvert(),
        py::arg("np_cy").noconvert(),
        py::arg("np_cz").noconvert(),
        py::arg("np_face_vtx").noconvert(),
        py::arg("np_face_vtx_idx").noconvert(),
        py::arg("np_parent_elemnts").noconvert());

  m.def("compute_center_cell_s", &compute_center_cell_s,
        py::arg("nx").noconvert(),
        py::arg("ny").noconvert(),
        py::arg("nz").noconvert(),
        py::arg("np_cx").noconvert(),
        py::arg("np_cy").noconvert(),
        py::arg("np_cz").noconvert());

  #if __cplusplus > 201703L
  m.def("gcs_only_for_ghosts"                     , gcs_only_for_ghosts                     , "For GridConnectivities, keep only in the PointList the ones that are ghosts");
  m.def("remove_ghost_info"                       , remove_ghost_info                       , "Remove ghost nodes and ghost elements of base");
  #else //C++==17
  #endif //C++>17

  register_cgns_registry_module(m);
  
}
