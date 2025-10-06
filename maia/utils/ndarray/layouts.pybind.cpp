#include "pdm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::tuple<py::array_t<int64_t>, py::array_t<int64_t>>
counting_sort(py::array_t<int64_t>& np_array, int n_bins) {
  size_t size = np_array.size();

  auto np_counts = py::array_t<int>(n_bins);
  auto counts = np_counts.mutable_data();
  for(int i=0; i < n_bins; ++i) {
    counts[i] = 0;
  }

  std::vector<int> displs(n_bins+1, 0);

  auto array = np_array.data();
  for(size_t i=0; i < size; ++i) {
    counts[array[i]]++;
  }

  for(int i=0; i < n_bins; ++i) {
    displs[i+1] = displs[i] + counts[i];
    counts[i] = 0;
  }

  auto np_out = py::array_t<int>(size);
  auto out = np_out.mutable_data();
  for(size_t i=0; i < size; ++i) {
    out[i] = displs[array[i]] + counts[array[i]];
    counts[array[i]]++;
  }

  return std::make_tuple(np_out, np_counts);
}

std::tuple<py::list, py::array_t<int64_t>>
counting_sort_mult(py::list array_list, int n_bins) {

  auto np_counts = py::array_t<int64_t>(n_bins);
  auto counts = np_counts.mutable_data();
  for(int i=0; i < n_bins; ++i) {
    counts[i] = 0;
  }


  for (auto item: array_list) {
    auto np_array = py::cast<py::array_t<int>>(item);
    size_t size = np_array.size();
    auto array = np_array.data();
    for(size_t i=0; i < size; ++i) {
      counts[array[i]]++;
    }
  }

  std::vector<int> displs(n_bins+1, 0);
  for(int i=0; i < n_bins; ++i) {
    displs[i+1] = displs[i] + counts[i];
    counts[i] = 0;
  }

  py::list out_list;
  
  for (auto item: array_list) {
    auto np_array = py::cast<py::array_t<int>>(item);
    auto array = np_array.data();
    size_t size = np_array.size();
    auto np_out = py::array_t<int64_t>(size);
    auto out = np_out.mutable_data();
    for(size_t i=0; i < size; ++i) {
      out[i] = displs[array[i]] + counts[array[i]];
      counts[array[i]]++;
    }
    out_list.append(np_out);
  }

  return std::make_tuple(out_list, np_counts);
}

template<typename T>
py::array_t<T>
extract_from_indices(py::array_t<T>& np_array, 
                     py::array_t<int>& np_indices,
                     int stride, int shift){

  int size         = np_indices.size();
  int extract_size = size * stride;

  auto indices = np_indices.data();
  auto array   = np_array.data();

  auto np_extract_array = py::array_t<T>(extract_size);
  auto extract_array    = np_extract_array.mutable_data();

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
                              py::array_t<g_num                    >& face_cell){
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
void pdm_face_cell_to_pe_cgns(py::array_t<g_num                    >& face_cell,
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
void strided_connectivity_to_pe(py::array_t<int>&   connect_idx,
                                py::array_t<g_num>& connect,
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
py::array_t<T>
indexed_to_interleaved_connectivity(py::array_t<T>& np_idx, py::array_t<T>& np_data) {


  auto idx  = np_idx.data();
  auto data = np_data.data();

  auto np_interleaved = py::array_t<T>(np_idx.size()-1+np_data.size());
  auto interleaved    = np_interleaved.mutable_data();

  size_t idx_write(0);
  for (int i = 0; i < np_idx.size()-1; ++i) {
    interleaved[idx_write++] = idx[i+1] - idx[i];
    for (int j=idx[i]; j < idx[i+1]; ++j) {
      interleaved[idx_write++] = data[j];
    }
  }
  return np_interleaved;
}

template<typename T>
std::tuple<py::array_t<T>, py::array_t<T>>
interleaved_to_indexed_connectivity(int n_elem, py::array_t<T>& np_interleaved)
{
  auto interleaved = np_interleaved.data();

  py::array_t<T> np_offset(n_elem+1);
  py::array_t<T> np_values(np_interleaved.size() - n_elem);

  auto offset = np_offset.mutable_data();
  auto values = np_values.mutable_data();

  offset[0] = 0;
  py::ssize_t i_elem = 0;
  py::ssize_t i = 0;
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
void create_mixed_elts_eso(py::array_t<T>& np_connec, py::array_t<T>& np_eso)
{
  auto n_cell = np_eso.size() - 1;

  static int n_vtx_per_type[] = {
    -1, -1, 1, 2, 3, 3, 6, 4, 8, 9,
    4, 10, 5, 14, 6, 15, 18, 8, 20, 27,
    -1, 13, -1, -1, 4, 9, 10, 12, 16, 16,
    20, 21, 29, 30, 24, 38, 40, 32, 56, 64, 
    5, 12, 15, 16, 25, 22, 34, 35, 29, 50, 
    55, 33, 66, 75, 44, 98, 125
  };

  auto connec = np_connec.data();
  auto eso    = np_eso.mutable_data();

  eso[0] = 0;
  size_t pos = 0;
  for (ssize_t i = 0; i < n_cell; ++i) {
    int nv = n_vtx_per_type[connec[eso[i]]];
    pos += (nv + 1);
    eso[i+1] = pos;
  }
}

template<typename fld_type>
std::tuple<py::array_t<fld_type>, py::array_t<fld_type>, py::array_t<fld_type>>
interlaced_to_tuple_coords(py::array_t<fld_type>& np_xyz){

  int size = np_xyz.size()/3;
  py::array_t<fld_type> np_coord_x(size);
  py::array_t<fld_type> np_coord_y(size);
  py::array_t<fld_type> np_coord_z(size);

  auto coord_xyz = np_xyz.data();
  auto coord_x   = np_coord_x.mutable_data();
  auto coord_y   = np_coord_y.mutable_data();
  auto coord_z   = np_coord_z.mutable_data();

  for(int i = 0; i < size; ++i) {
    int offset = 3*i;
    coord_x[i] = coord_xyz[offset  ];
    coord_y[i] = coord_xyz[offset+1];
    coord_z[i] = coord_xyz[offset+2];
  }

  return std::make_tuple(np_coord_x, np_coord_y, np_coord_z);
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

  m.def("counting_sort", &counting_sort,
        py::arg("array").noconvert(),
        py::arg("n_bins").noconvert());
  m.def("counting_sort_mult", &counting_sort_mult,
        py::arg("arrays").noconvert(),
        py::arg("n_bins").noconvert());
  
}
