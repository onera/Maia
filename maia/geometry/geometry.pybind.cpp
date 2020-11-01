#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T> constexpr auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

auto
compute_face_center_and_characteristic_length(py::array_t<int   , py::array::f_style>& np_point_list,
                                              py::array_t<double, py::array::f_style>& np_cx,
                                              py::array_t<double, py::array::f_style>& np_cy,
                                              py::array_t<double, py::array::f_style>& np_cz,
                                              py::array_t<int   , py::array::f_style>& np_face_vtx,
                                              py::array_t<int   , py::array::f_style>& np_face_vtx_idx)
{
  int bnd_size = np_point_list.size();

  auto point_list   = make_raw_view(np_point_list);
  auto cx           = make_raw_view(np_cx);
  auto cy           = make_raw_view(np_cy);
  auto cz           = make_raw_view(np_cz);
  auto face_vtx     = make_raw_view(np_face_vtx);
  auto face_vtx_idx = make_raw_view(np_face_vtx_idx);

  py::array_t<double, py::array::f_style> np_bnd_coord(3*bnd_size);
  py::array_t<double, py::array::f_style> np_characteristic_lenght(bnd_size);

  auto bnd_coord             = make_raw_view(np_bnd_coord);
  auto characteristic_lenght = make_raw_view(np_characteristic_lenght);

  for(int idx = 0; idx < bnd_size; ++idx) {
    int i_face = point_list[idx]-1;
    bnd_coord[3*idx  ] = 0.;
    bnd_coord[3*idx+1] = 0.;
    bnd_coord[3*idx+2] = 0.;

    characteristic_lenght[idx] = std::numeric_limits<double>::max();

    int beg = face_vtx_idx[i_face];
    int n_vtx_on_face = face_vtx_idx[i_face+1]-beg;

    for(int idx_vtx = beg; idx_vtx < face_vtx_idx[i_face+1]; ++idx_vtx){

      int pos1 =   idx_vtx - beg;
      int pos2 = ( idx_vtx - beg + 1 ) % n_vtx_on_face;

      int ivtx1 = face_vtx[beg+pos1] - 1;
      int ivtx2 = face_vtx[beg+pos2] - 1;

      // std::cout << "pos1 = " << pos1 << " | pos2 = " << pos2 << std::endl;
      // std::cout << "ivtx1 = " << ivtx1 << " | ivtx2 = " << ivtx2 << std::endl;

      bnd_coord[3*idx  ] += cx[ivtx1];
      bnd_coord[3*idx+1] += cy[ivtx1];
      bnd_coord[3*idx+2] += cz[ivtx1];

      double dx = cx[ivtx1] - cx[ivtx2];
      double dy = cy[ivtx1] - cy[ivtx2];
      double dz = cz[ivtx1] - cz[ivtx2];
      double le = std::sqrt(dx*dx + dy*dy + dz*dz);

      characteristic_lenght[idx] = std::min(characteristic_lenght[idx], le);

    }

    // Finish
    double inv = 1./n_vtx_on_face;
    bnd_coord[3*idx  ] = bnd_coord[3*idx  ] * inv;
    bnd_coord[3*idx+1] = bnd_coord[3*idx+1] * inv;
    bnd_coord[3*idx+2] = bnd_coord[3*idx+2] * inv;

  }

  return std::make_tuple(np_bnd_coord, np_characteristic_lenght);
}



PYBIND11_MODULE(geometry, m) {
  m.doc() = "pybind11 utils for geomery plugin"; // optional module docstring

  m.def("compute_face_center_and_characteristic_length", &compute_face_center_and_characteristic_length,
        py::arg("np_point_list").noconvert(),
        py::arg("np_cx").noconvert(),
        py::arg("np_cy").noconvert(),
        py::arg("np_cz").noconvert(),
        py::arg("np_face_vtx").noconvert(),
        py::arg("np_face_vtx_idx").noconvert());

}
