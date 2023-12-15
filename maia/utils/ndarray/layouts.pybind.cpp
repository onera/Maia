#include "pdm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "maia/utils/pybind_utils.hpp"

namespace py = pybind11;

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

template<typename g_num>
void strided_connectivity_to_pe(py::array_t<int  , py::array::f_style>& connect_idx,
                                py::array_t<g_num, py::array::f_style>& connect,
                                py::array_t<g_num, py::array::f_style>& pe){
  int n_elts = connect_idx.size() - 1;

  assert(pe.ndim()        == 2        );
  assert(pe.size()        == 2*n_elts );

  auto _pe          = pe         .template mutable_unchecked<2>();
  auto _connect_idx = connect_idx.template mutable_unchecked<1>();
  auto _connect     = connect    .template mutable_unchecked<1>();
  assert(connect.size() == _connect_idx[n_elts]);

  for (int ielt = 0; ielt < n_elts; ++ielt) {
    int size = _connect_idx(ielt+1) - _connect_idx(ielt);
    assert (0 < size && size <= 2);
    if (size == 1) {
      g_num first = _connect(_connect_idx(ielt));
      if (first > 0) {
        _pe(ielt,0) = first;
        _pe(ielt,1) = 0;
      }
      else {
        _pe(ielt,0) = 0;
        _pe(ielt,1) = -1*first;
      }
    }
    else {
      g_num first  = _connect(_connect_idx(ielt));
      g_num second = _connect(_connect_idx(ielt)+1);
      if (first > 0) {
        assert (second < 0);
        _pe(ielt,0) = first;
        _pe(ielt,1) = -1*second;
      }
      else {
        assert (second > 0);
        _pe(ielt,0) = second;
        _pe(ielt,1) = -1*first;
      }
    }
  }
}

template<typename T>
py::array_t<T, py::array::f_style>
indexed_to_interleaved_connectivity(py::array_t<T, py::array::f_style>& np_idx,
            py::array_t<T,   py::array::f_style>& np_data) {


  auto idx  = make_raw_view(np_idx);
  auto data = make_raw_view(np_data);

  auto np_interleaved = py::array_t<T, py::array::f_style>(np_idx.size()-1+np_data.size());
  auto interleaved    = make_raw_view(np_interleaved);

  int idx_write(0);
  for (int i = 0; i < np_idx.size()-1; ++i) {
    interleaved[idx_write++] = idx[i+1] - idx[i];
    for (int j=idx[i]; j < idx[i+1]; ++j) {
      interleaved[idx_write++] = data[j];
    }
  }
  return np_interleaved;
}

template<typename T>
std::tuple<py::array_t<T, py::array::f_style>, py::array_t<T, py::array::f_style>>
interleaved_to_indexed_connectivity(int n_elem, py::array_t<T, py::array::f_style>& np_interleaved)
{
  auto interleaved = make_raw_view(np_interleaved);

  py::array_t<T, py::array::f_style> np_offset(n_elem+1);
  py::array_t<T, py::array::f_style> np_values(np_interleaved.size() - n_elem);

  auto offset = make_raw_view(np_offset);
  auto values = make_raw_view(np_values);

  offset[0] = 0;
  int i_elem = 0;
  int i = 0;
  while (i < np_interleaved.size()) {
    offset[i_elem+1] = offset[i_elem] + interleaved[i];
    for (int j = 0; j < offset[i_elem+1] - offset[i_elem]; ++j) {
      values[offset[i_elem] + j] = interleaved[i+1+j];
    }
    i += interleaved[i] + 1;
    i_elem++;
  }
  return std::make_tuple(np_offset, np_values);
}

template<typename T>
void 
create_mixed_elts_eso(py::array_t<T, py::array::f_style>& np_connec, py::array_t<T, py::array::f_style>& np_eso)
{
  int n_cell = np_eso.size() - 1;

  static int n_vtx_per_type[] = {
    -1, -1, 1, 2, 3, 3, 6, 4, 8, 9,
    4, 10, 5, 14, 6, 15, 18, 8, 20, 27,
    -1, 13, -1, -1, 4, 9, 10, 12, 16, 16,
    20, 21, 29, 30, 24, 38, 40, 32, 56, 64, 
    5, 12, 15, 16, 25, 22, 34, 35, 29, 50, 
    55, 33, 66, 75, 44, 98, 125
  };

  auto connec = make_raw_view(np_connec);
  auto eso    = make_raw_view(np_eso);

  eso[0] = 0;
  int pos = 0;
  for (int i = 0; i < n_cell; ++i) {
    int nv = n_vtx_per_type[connec[eso[i]]];
    pos += (nv + 1);
    eso[i+1] = pos;
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

  m.def("indexed_to_interleaved_connectivity", &indexed_to_interleaved_connectivity<int32_t>, 
        py::arg("indices").noconvert(),
        py::arg("array"  ).noconvert());
  m.def("indexed_to_interleaved_connectivity", &indexed_to_interleaved_connectivity<int64_t>, 
        py::arg("indices").noconvert(),
        py::arg("array"  ).noconvert());
  m.def("interleaved_to_indexed_connectivity", &interleaved_to_indexed_connectivity<int32_t>, 
        py::arg("n_elem"  ).noconvert(),
        py::arg("array"  ).noconvert());
  m.def("interleaved_to_indexed_connectivity", &interleaved_to_indexed_connectivity<int64_t>, 
        py::arg("n_elem"  ).noconvert(),
        py::arg("array"  ).noconvert());
  m.def("create_mixed_elts_eso", &create_mixed_elts_eso<int32_t>, 
        py::arg("connectivity").noconvert(),
        py::arg("eso").noconvert());
  m.def("create_mixed_elts_eso", &create_mixed_elts_eso<int64_t>, 
        py::arg("connectivity").noconvert(),
        py::arg("eso").noconvert());

  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int32_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());
  m.def("pe_cgns_to_pdm_face_cell", &pe_cgns_to_pdm_face_cell<int64_t>,
        py::arg("pe"       ).noconvert(),
        py::arg("face_cell").noconvert());

  m.def("strided_connectivity_to_pe", &strided_connectivity_to_pe<int32_t>,
        py::arg("connect_idx").noconvert(),
        py::arg("connect"    ).noconvert(),
        py::arg("pe"         ).noconvert());
  m.def("strided_connectivity_to_pe", &strided_connectivity_to_pe<int64_t>,
        py::arg("connect_idx").noconvert(),
        py::arg("connect"    ).noconvert(),
        py::arg("pe"         ).noconvert());

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
