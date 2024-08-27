#include "maia/algo/dist/dist_algo.pybind.hpp"

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

template<typename T> T find_unshared(const T* reference_first, const T* reference_last,
                                     const T* candidate_first, const T* candidate_last) {
  // Return the first element of candidate list which does not appears in reference list
  auto unshared_ptr = std::find_if(candidate_first, candidate_last,
    [&](T e){return std::find(reference_first, reference_last, e) == reference_last;});
  assert (unshared_ptr != candidate_last);
  return *unshared_ptr;
}



template<typename I> I
node_above(I vtx, std::vector<std::array<I,4>>& quads) {

  auto contains_vtx = [vtx](const auto& quad){return std::find(begin(quad),end(quad),vtx) != end(quad);};
  auto quad0 = std::find_if(begin(quads),end(quads),contains_vtx);
  assert(quad0!=end(quads));
  auto quad1 = std::find_if(quad0+1,end(quads),contains_vtx);
  assert(quad1!=end(quads));

  std::array<I,4> q0, q1;
  std::copy((*quad0).begin(),(*quad0).end(),begin(q0));
  std::copy((*quad1).begin(),(*quad1).end(),begin(q1));
  std::sort(begin(q0),end(q0));
  std::sort(begin(q1),end(q1));
  std::array<I,2> q0_inter_q1;
  std::set_intersection(begin(q0),end(q0),begin(q1),end(q1),begin(q0_inter_q1));
  if (q0_inter_q1[0]!=vtx)
    return q0_inter_q1[0];
  else { 
    assert (q0_inter_q1[1]!=vtx); 
    return q0_inter_q1[1];
  }
}


template<typename T>
void combine_to_tetra(py::array_t<int>& np_face_vtx_n,
                      py::array_t<T>& np_face_vtx,
                      py::array_t<T>& np_cell_face,
                      py::array_t<T>& np_cell_vtx){
  
  int n_elt = np_face_vtx_n.size() / 4;
  auto face_vtx  = np_face_vtx .data();
  auto cell_face = np_cell_face.data();
  auto cell_vtx  = np_cell_vtx .mutable_data();

  for (int i=0; i < n_elt; ++i) {
    auto first_face  = face_vtx + 12*i;
    auto second_face = first_face + 3;
    if (cell_face[4*i] > 0) // Outward normal
      std::reverse_copy(first_face, first_face+3, cell_vtx+4*i);
    else  // Inward normal
      std::copy(first_face, first_face+3, cell_vtx+4*i);
    
    cell_vtx[4*i+3] = find_unshared(first_face, first_face+3, second_face, second_face+3);
  }
}

template<typename T>
void combine_to_pyra(py::array_t<int>& np_face_vtx_n,
                     py::array_t<T>& np_face_vtx,
                     py::array_t<T>& np_cell_face,
                     py::array_t<T>& np_cell_vtx){
  
  int n_elt = np_face_vtx_n.size() / 5;
  auto face_vtx_n = np_face_vtx_n.data();
  auto face_vtx   = np_face_vtx  .data();
  auto cell_face  = np_cell_face .data();
  auto cell_vtx   = np_cell_vtx  .mutable_data();

  for (int i=0; i < n_elt; ++i) {
    int  quad_idx = 0;
    auto quad_face = face_vtx + 16*i;
    while (face_vtx_n[5*i + quad_idx] != 4) { // Search the quad face
      quad_face += 3;
      quad_idx++;
    }
    int  tri_idx = (quad_idx + 1) % 5;
    auto tri_face = face_vtx + 16*i + 3*tri_idx + int(tri_idx > quad_idx); 
    if (cell_face[5*i + quad_idx] > 0) // Outward normal
      std::reverse_copy(quad_face, quad_face+4, cell_vtx+5*i);
    else  // Inward normal
      std::copy(quad_face, quad_face+4, cell_vtx+5*i);

    cell_vtx[5*i+4] = find_unshared(quad_face, quad_face+4, tri_face, tri_face+3);
  }
}


template<typename T>
void combine_to_penta(py::array_t<int>& np_face_vtx_n,
                      py::array_t<T>& np_face_vtx,
                      py::array_t<T>& np_cell_face,
                      py::array_t<T>& np_cell_vtx){
  
  int n_elt = np_face_vtx_n.size() / 5;
  auto face_vtx_n = np_face_vtx_n.data();
  auto face_vtx   = np_face_vtx  .data();
  auto cell_face  = np_cell_face .data();
  auto cell_vtx   = np_cell_vtx  .mutable_data();

  std::vector<std::array<T, 4>> quads(3, {0,0,0,0});
  const T* cur_face = face_vtx;

  for (int i=0; i < n_elt; ++i) {
    bool tri_found = false;
    int quad_cnt = 0;

    for (int j=0; j < 5; ++j) {
      if (face_vtx_n[5*i+j] == 3 && !tri_found) {
        tri_found = true;
        if (cell_face[5*i+j] > 0) // Outward normal
          std::reverse_copy(cur_face, cur_face+3, cell_vtx+6*i);
        else  // Inward normal
          std::copy(cur_face, cur_face+3, cell_vtx+6*i);
      }
      else if (face_vtx_n[5*i+j] == 4) {
        std::copy(cur_face, cur_face+4, quads[quad_cnt].begin());
        quad_cnt++;
      }
      cur_face += face_vtx_n[5*i+j];
    }
    cell_vtx[6*i+3] = node_above(cell_vtx[6*i+0], quads);
    cell_vtx[6*i+4] = node_above(cell_vtx[6*i+1], quads);
    cell_vtx[6*i+5] = node_above(cell_vtx[6*i+2], quads);
  }
}

template<typename T>
void combine_to_hexa(py::array_t<int>& np_face_vtx_n,
                     py::array_t<T>& np_face_vtx,
                     py::array_t<T>& np_cell_face,
                     py::array_t<T>& np_cell_vtx){
  
  int n_elt = np_face_vtx_n.size() / 6;
  auto face_vtx_n = np_face_vtx_n.data();
  auto face_vtx   = np_face_vtx  .data();
  auto cell_face  = np_cell_face .data();
  auto cell_vtx   = np_cell_vtx  .mutable_data();

  std::vector<std::array<T, 4>> quads(4, {0,0,0,0});
  const T* first_face = face_vtx;

  for (int i=0; i < n_elt; ++i) {

    int quad_cnt = 0;

    if (cell_face[6*i] > 0) // Outward normal
      std::reverse_copy(first_face, first_face+4, cell_vtx+8*i);
    else  // Inward normal
      std::copy(first_face, first_face+4, cell_vtx+8*i);

    for (int j=1; j < 6; ++j) {
        
      auto cur_face = first_face + j*4;
      auto shared_ptr = std::find_if(cur_face, cur_face+4,
        [&first_face](T e){return std::find(first_face, first_face+4, e) != first_face+4;});
      if (shared_ptr != cur_face+4) {
        std::copy(cur_face, cur_face+4, quads[quad_cnt].begin());
        quad_cnt++;
      }
    }
    first_face += 24;
    cell_vtx[8*i+4] = node_above(cell_vtx[8*i+0], quads);
    cell_vtx[8*i+5] = node_above(cell_vtx[8*i+1], quads);
    cell_vtx[8*i+6] = node_above(cell_vtx[8*i+2], quads);
    cell_vtx[8*i+7] = node_above(cell_vtx[8*i+3], quads);
  }
}



void register_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("algo");

  m.def("combine_to_tetra", &combine_to_tetra<int32_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());
  m.def("combine_to_tetra", &combine_to_tetra<int64_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());

  m.def("combine_to_pyra", &combine_to_pyra<int32_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());
  m.def("combine_to_pyra", &combine_to_pyra<int64_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());

  m.def("combine_to_penta", &combine_to_penta<int32_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());
  m.def("combine_to_penta", &combine_to_penta<int64_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());

  m.def("combine_to_hexa", &combine_to_hexa<int32_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());
  m.def("combine_to_hexa", &combine_to_hexa<int64_t>,
        py::arg("face_vtx_n").noconvert(),
        py::arg("face_vtx").noconvert(),
        py::arg("cell_face").noconvert(),
        py::arg("cell_vtx_out").noconvert());
}
