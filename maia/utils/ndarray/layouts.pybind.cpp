#include "pdm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T> auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

template<typename T>
py::array_t<T, py::array::f_style>
extract_from_indices(py::array_t<T, py::array::f_style>& np_array, 
                     py::array_t<int, py::array::f_style>& np_indices,
                     int stride, int shift){

  int size         = np_indices.size();
  int extract_size = size * stride;

  auto indices = make_raw_view(np_indices);
  auto array   = make_raw_view(np_array);

  auto np_extract_array = py::array_t<T, py::array::f_style>(extract_size);
  auto extract_array   = make_raw_view(np_extract_array);

  for(int i = 0; i < size; ++i) {
    int idx = indices[i]-shift;
    for(int s = 0; s < stride; ++s) {
      extract_array[stride*i + s] = array[stride*idx + s];
    }
  }
  return np_extract_array;
}
template<typename g_num>
void pe_cgns_to_pdm_face_cell(py::array_t<g_num, py::array::f_style>& pe,
                              py::array_t<g_num, py::array::f_style>& face_cell){
  assert(pe.ndim()        == 2        );
  assert(face_cell.ndim() == 1        );
  assert(face_cell.size() == pe.size());

  int n_face = pe.shape()[0];

  auto pe_ptr        = pe       .template mutable_unchecked<2>();
  auto face_cell_ptr = face_cell.template mutable_unchecked<1>();

  for(int i_face = 0; i_face < n_face; ++i_face){
    face_cell_ptr(2*i_face  ) = pe_ptr(i_face,0);
    face_cell_ptr(2*i_face+1) = pe_ptr(i_face,1);
  }
}

template<typename g_num>
void pdm_face_cell_to_pe_cgns(py::array_t<g_num, py::array::f_style>& face_cell,
                              py::array_t<g_num, py::array::f_style>& pe){
  assert(pe.ndim()        == 2        );
  assert(face_cell.ndim() == 1        );
  assert(face_cell.size() == pe.size());

  int n_face = pe.shape()[0];

  auto pe_ptr        = pe       .template mutable_unchecked<2>();
  auto face_cell_ptr = face_cell.template mutable_unchecked<1>();

  for(int i_face = 0; i_face < n_face; ++i_face){
    pe_ptr(i_face,0) = face_cell_ptr(2*i_face  );
    pe_ptr(i_face,1) = face_cell_ptr(2*i_face+1);
  }
}

template<typename fld_type>
std::tuple<py::array_t<fld_type, py::array::f_style>, py::array_t<fld_type, py::array::f_style>, py::array_t<fld_type, py::array::f_style>>
interlaced_to_tuple_coords(py::array_t<fld_type, py::array::f_style>& np_xyz){

  int size = np_xyz.size()/3;
  py::array_t<fld_type, py::array::f_style> np_coord_x(size);
  py::array_t<fld_type, py::array::f_style> np_coord_y(size);
  py::array_t<fld_type, py::array::f_style> np_coord_z(size);

  auto coord_xyz = make_raw_view(np_xyz);
  auto coord_x   = make_raw_view(np_coord_x);
  auto coord_y   = make_raw_view(np_coord_y);
  auto coord_z   = make_raw_view(np_coord_z);

  for(int i = 0; i < size; ++i) {
    int offset = 3*i;
    coord_x[i] = coord_xyz[offset  ];
    coord_y[i] = coord_xyz[offset+1];
    coord_z[i] = coord_xyz[offset+2];
  }

  return std::make_tuple(np_coord_x, np_coord_y, np_coord_z);
}

template<typename T>
std::tuple<py::array_t<int, py::array::f_style>, py::array_t<T, py::array::f_style>>
jagged_merge(py::array_t<int, py::array::f_style>& np_idx1,
             py::array_t<T, py::array::f_style>&   np_array1,
             py::array_t<int, py::array::f_style>& np_idx2,
             py::array_t<T, py::array::f_style>&   np_array2) {

  assert(np_idx1.size() == np_idx2.size());

  int r_n_elt = np_idx1.size() - 1;
  int r_size  = np_array1.size() + np_array2.size();
  py::array_t<int, py::array::f_style> np_idx(r_n_elt + 1);
  py::array_t<T,   py::array::f_style> np_array(r_size);

  auto idx1   = np_idx1.unchecked<1>();
  auto array1 = np_array1.template unchecked<1>();
  auto idx2   = np_idx2.unchecked<1>();
  auto array2 = np_array2.template unchecked<1>();

  auto idx   = np_idx.mutable_unchecked<1>();
  auto array = np_array.template mutable_unchecked<1>();
  
  idx[0] = 0;
  int w_idx(0);
  for (int i = 0; i < r_n_elt; ++i) {
    for (int j = idx1[i]; j < idx1[i+1]; ++j) {
      array[w_idx++] = array1[j];
    }
    for (int j = idx2[i]; j < idx2[i+1]; ++j) {
      array[w_idx++] = array2[j];
    }
    idx[i+1] = idx[i] + (idx1[i+1]-idx1[i]) + (idx2[i+1]-idx2[i]);
  }
  assert (w_idx == r_size);

  return std::make_tuple(np_idx, np_array);
}





void register_layouts_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("layouts");

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

  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int32_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());
  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int64_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());

  m.def("pdm_face_cell_to_pe_cgns", &pdm_face_cell_to_pe_cgns<int32_t>,
        py::arg("face_cell").noconvert(),
        py::arg("pe"       ).noconvert());
  m.def("pdm_face_cell_to_pe_cgns", &pdm_face_cell_to_pe_cgns<int64_t>,
        py::arg("face_cell").noconvert(),
        py::arg("pe"       ).noconvert());

  m.def("interlaced_to_tuple_coords", &interlaced_to_tuple_coords<double>,
        py::arg("np_xyz").noconvert());

  m.def("jagged_merge", &jagged_merge<int32_t>,
        py::arg("idx1"  ).noconvert(),
        py::arg("array1").noconvert(),
        py::arg("idx2"  ).noconvert(),
        py::arg("array2").noconvert());
  m.def("jagged_merge", &jagged_merge<int64_t>,
        py::arg("idx1"  ).noconvert(),
        py::arg("array1").noconvert(),
        py::arg("idx2"  ).noconvert(),
        py::arg("array2").noconvert());
  m.def("jagged_merge", &jagged_merge<double>,
        py::arg("idx1"  ).noconvert(),
        py::arg("array1").noconvert(),
        py::arg("idx2"  ).noconvert(),
        py::arg("array2").noconvert());
}
