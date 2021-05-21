#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "std_e/algorithm/permutation.hpp"
#include "maia/cgns_registry/cgns_registry.hpp"

namespace py = pybind11;

template<typename T> auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

auto adr = [](const auto& i, const auto& j, const auto& k, const auto& im, const auto& jm) {
  return (i-1)
       + (j-1)*im
       + (k-1)*im*jm;
};

// --------------------------------------------------------------------
auto
prepare_extract_bc_u(py::array_t<int, py::array::f_style>& np_point_list,
                     py::array_t<int, py::array::f_style>& np_face_vtx,
                     py::array_t<int, py::array::f_style>& np_face_vtx_idx,
                     py::array_t<int, py::array::f_style>& np_work_vtx)
{
  int n_face_bnd = np_point_list.size();

  auto point_list   = make_raw_view(np_point_list);
  auto face_vtx     = make_raw_view(np_face_vtx);
  auto face_vtx_idx = make_raw_view(np_face_vtx_idx);
  auto work_vtx     = make_raw_view(np_work_vtx);

  int n_face_vtx_bnd = 0;
  int n_vtx_bnd      = 0;
  for(int idx = 0; idx < n_face_bnd; ++idx) {
    int iface = point_list[idx]-1;

    int beg = face_vtx_idx[iface];
    int n_vtx_on_face = face_vtx_idx[iface+1]-beg;

    n_face_vtx_bnd += n_vtx_on_face;

    for(int idx_vtx = beg; idx_vtx < face_vtx_idx[iface+1]; ++idx_vtx) {
      int ivtx = face_vtx[idx_vtx] - 1;

      if (work_vtx[ivtx] == -1) {
        work_vtx[ivtx] = n_vtx_bnd;
        n_vtx_bnd += 1;
      }
    }
  }

  return std::make_tuple(n_face_vtx_bnd, n_vtx_bnd);
}

auto
prepare_extract_bc_s(py::array_t<int, py::array::f_style>& np_vtx_size,
                     py::array_t<int, py::array::f_style>& np_point_range,
                     py::array_t<int, py::array::f_style>& np_work_vtx)
{
  // std::cout << "np_vtx_size.size() = " << np_vtx_size.size() << std::endl;
  // std::cout << "np_vtx_size.shape()[0] = " << np_vtx_size.shape()[0] << std::endl;
  auto vtx_size    = make_raw_view(np_vtx_size);
  auto point_range = make_raw_view(np_point_range);
  auto work_vtx    = make_raw_view(np_work_vtx);

  int im = vtx_size[0];
  int jm = vtx_size[1];
  // int km = vtx_size[2];
  // std::cout << "im = " << im << ", jm = " << jm << ", km = " << km << std::endl;

  int imin = point_range[0]; int imax = point_range[3];
  int jmin = point_range[1]; int jmax = point_range[4];
  int kmin = point_range[2]; int kmax = point_range[5];
  // std::cout << "imin = " << imin << ", jmin = " << jmin << ", kmin = " << kmin << std::endl;
  // std::cout << "imax = " << imax << ", jmax = " << jmax << ", kmax = " << kmax << std::endl;

  auto window_size = std::vector<int>(3);
  window_size[0] = std::max(1, imax-imin);
  window_size[1] = std::max(1, jmax-jmin);
  window_size[2] = std::max(1, kmax-kmin);
  // std::cout << "window_size[" << window_size[0] << ", " << window_size[1] << ", " << window_size[2] << "]" << std::endl;

  int n_face_vtx_bnd = 0;
  int n_vtx_bnd      = 0;

  auto apply_prepare = [&work_vtx, &n_vtx_bnd](const auto& ind) {
    if (work_vtx[ind] == -1) {
      work_vtx[ind] = n_vtx_bnd;
      n_vtx_bnd += 1;
    }
  };

  if (imin == imax) {
    int ic = imin;
    for(int kc = kmin; kc < kmin+window_size[2]; ++kc) {
      for(int jc = jmin; jc < jmin+window_size[1]; ++jc) {
        int ind1 = adr(ic  , jc  , kc  , im, jm);
        int ind2 = adr(ic  , jc+1, kc  , im, jm);
        int ind3 = adr(ic  , jc+1, kc+1, im, jm);
        int ind4 = adr(ic  , jc  , kc+1, im, jm);

        n_face_vtx_bnd += 4;

        apply_prepare(ind1);
        apply_prepare(ind2);
        apply_prepare(ind3);
        apply_prepare(ind4);
      }
    }
  }
  else if (jmin == jmax) {
    int jc = jmin;
    for(int kc = kmin; kc < kmin+window_size[2]; ++kc) {
      for(int ic = imin; ic < imin+window_size[0]; ++ic) {
        int ind1 = adr(ic  , jc  , kc  , im, jm);
        int ind2 = adr(ic+1, jc  , kc  , im, jm);
        int ind3 = adr(ic+1, jc  , kc+1, im, jm);
        int ind4 = adr(ic  , jc  , kc+1, im, jm);

        n_face_vtx_bnd += 4;

        apply_prepare(ind1);
        apply_prepare(ind2);
        apply_prepare(ind3);
        apply_prepare(ind4);
      }
    }
  }
  else if (kmin == kmax) {
    int kc = kmin;
    for(int jc = jmin; jc < jmin+window_size[1]; ++jc) {
      for(int ic = imin; ic < imin+window_size[0]; ++ic) {
        int ind1 = adr(ic  , jc  , kc, im, jm);
        int ind2 = adr(ic+1, jc  , kc, im, jm);
        int ind3 = adr(ic+1, jc+1, kc, im, jm);
        int ind4 = adr(ic  , jc+1, kc, im, jm);

        n_face_vtx_bnd += 4;

        apply_prepare(ind1);
        apply_prepare(ind2);
        apply_prepare(ind3);
        apply_prepare(ind4);
      }
    }
  }
  else {
    throw std::runtime_error("Unable to determine the direction of the BC window.");
  }

  return std::make_tuple(n_face_vtx_bnd, n_vtx_bnd);
}

// --------------------------------------------------------------------
auto
compute_extract_bc_u(const int&                               ibeg_face_vtx_idx,
                     py::array_t<int, py::array::f_style>&    np_point_list,
                     py::array_t<int, py::array::f_style>&    np_face_vtx,
                     py::array_t<int, py::array::f_style>&    np_face_vtx_idx,
                     py::array_t<int, py::array::f_style>&    np_work_vtx,
                     py::array_t<double, py::array::f_style>& np_x,
                     py::array_t<double, py::array::f_style>& np_y,
                     py::array_t<double, py::array::f_style>& np_z,
                     int&                                     i_vtx_bnd,
                     py::array_t<int, py::array::f_style>&    np_face_vtx_bnd,
                     py::array_t<int, py::array::f_style>&    np_face_vtx_bnd_idx,
                     py::array_t<double, py::array::f_style>& np_vtx_bnd)
{
  int n_face_bnd = np_point_list.size();

  auto point_list       = make_raw_view(np_point_list);
  auto face_vtx         = make_raw_view(np_face_vtx);
  auto face_vtx_idx     = make_raw_view(np_face_vtx_idx);
  auto work_vtx         = make_raw_view(np_work_vtx);
  auto x                = make_raw_view(np_x);
  auto y                = make_raw_view(np_y);
  auto z                = make_raw_view(np_z);
  auto face_vtx_bnd     = make_raw_view(np_face_vtx_bnd);
  auto face_vtx_bnd_idx = make_raw_view(np_face_vtx_bnd_idx);
  auto vtx_bnd          = make_raw_view(np_vtx_bnd);

  int idx = face_vtx_bnd_idx[ibeg_face_vtx_idx];

  for(int idx_face = 0; idx_face < n_face_bnd; ++idx_face) {
    int iface = point_list[idx_face]-1;

    int beg = face_vtx_idx[iface];
    int n_vtx_on_face = face_vtx_idx[iface+1]-beg;

    int global_idx_face = ibeg_face_vtx_idx+idx_face;
    face_vtx_bnd_idx[global_idx_face+1] = face_vtx_bnd_idx[global_idx_face] + n_vtx_on_face;

    for(int idx_vtx = beg; idx_vtx < face_vtx_idx[iface+1]; ++idx_vtx) {
      int ivtx = face_vtx[idx_vtx] - 1;

      if (work_vtx[ivtx] == -1) {
        work_vtx[ivtx] = i_vtx_bnd;
        vtx_bnd[3*i_vtx_bnd  ] = x[ivtx];
        vtx_bnd[3*i_vtx_bnd+1] = y[ivtx];
        vtx_bnd[3*i_vtx_bnd+2] = z[ivtx];
        i_vtx_bnd += 1;
      }

      face_vtx_bnd[idx] = work_vtx[ivtx]+1;  /// because numerotation starts to 1
      idx += 1;
    }
  }
  return i_vtx_bnd;
}

auto
compute_extract_bc_s(const int&                               ibeg_face_vtx_idx,
                     py::array_t<int, py::array::f_style>&    np_vtx_size,
                     py::array_t<int, py::array::f_style>&    np_point_range,
                     py::array_t<int, py::array::f_style>&    np_work_vtx,
                     py::array_t<double, py::array::f_style>& np_x,
                     py::array_t<double, py::array::f_style>& np_y,
                     py::array_t<double, py::array::f_style>& np_z,
                     int&                                     i_vtx_bnd,
                     py::array_t<int, py::array::f_style>&    np_face_vtx_bnd,
                     py::array_t<int, py::array::f_style>&    np_face_vtx_bnd_idx,
                     py::array_t<double, py::array::f_style>& np_vtx_bnd)
{
  // std::cout << "np_vtx_size.size() = " << np_vtx_size.size() << std::endl;
  // std::cout << "np_vtx_size.shape()[0] = " << np_vtx_size.shape()[0] << std::endl;
  auto vtx_size         = make_raw_view(np_vtx_size);
  auto point_range      = make_raw_view(np_point_range);
  auto work_vtx         = make_raw_view(np_work_vtx);
  auto x                = make_raw_view(np_x);
  auto y                = make_raw_view(np_y);
  auto z                = make_raw_view(np_z);
  auto face_vtx_bnd     = make_raw_view(np_face_vtx_bnd);
  auto face_vtx_bnd_idx = make_raw_view(np_face_vtx_bnd_idx);
  auto vtx_bnd          = make_raw_view(np_vtx_bnd);

  int im = vtx_size[0];
  int jm = vtx_size[1];
  // int km = vtx_size[2];
  // std::cout << "im = " << im << ", jm = " << jm << ", km = " << km << std::endl;

  int imin = point_range[0]; int imax = point_range[3];
  int jmin = point_range[1]; int jmax = point_range[4];
  int kmin = point_range[2]; int kmax = point_range[5];
  // std::cout << "imin = " << imin << ", jmin = " << jmin << ", kmin = " << kmin << std::endl;
  // std::cout << "imax = " << imax << ", jmax = " << jmax << ", kmax = " << kmax << std::endl;

  auto window_size = std::vector<int>(3);
  window_size[0] = std::max(1, imax-imin);
  window_size[1] = std::max(1, jmax-jmin);
  window_size[2] = std::max(1, kmax-kmin);
  // std::cout << "window_size[" << window_size[0] << ", " << window_size[1] << ", " << window_size[2] << "]" << std::endl;

  int idx = face_vtx_bnd_idx[ibeg_face_vtx_idx];

  auto apply_compute = [&idx, &work_vtx, &x, &y, &z, &i_vtx_bnd, &face_vtx_bnd, &vtx_bnd](const auto& ind) {
    if (work_vtx[ind] == -1) {
      work_vtx[ind] = i_vtx_bnd;
      vtx_bnd[3*i_vtx_bnd  ] = x[ind];
      vtx_bnd[3*i_vtx_bnd+1] = y[ind];
      vtx_bnd[3*i_vtx_bnd+2] = z[ind];
      i_vtx_bnd += 1;
    }

    face_vtx_bnd[idx] = work_vtx[ind]+1;  /// because numerotation starts to 1
    idx += 1;
  };

  int idx_face = 0;
  if (imin == imax) {
    int ic = imin;
    for(int kc = kmin; kc < kmin+window_size[2]; ++kc) {
      for(int jc = jmin; jc < jmin+window_size[1]; ++jc) {
        int ind1 = adr(ic  , jc  , kc  , im, jm);
        int ind2 = adr(ic  , jc+1, kc  , im, jm);
        int ind3 = adr(ic  , jc+1, kc+1, im, jm);
        int ind4 = adr(ic  , jc  , kc+1, im, jm);

        apply_compute(ind1);
        apply_compute(ind2);
        apply_compute(ind3);
        apply_compute(ind4);

        int global_idx_face = ibeg_face_vtx_idx+idx_face;
        face_vtx_bnd_idx[global_idx_face+1] = face_vtx_bnd_idx[global_idx_face] + 4;
        idx_face += 1;
      }
    }
  }
  else if (jmin == jmax) {
    int jc = jmin;
    for(int kc = kmin; kc < kmin+window_size[2]; ++kc) {
      for(int ic = imin; ic < imin+window_size[0]; ++ic) {
        int ind1 = adr(ic  , jc  , kc  , im, jm);
        int ind2 = adr(ic+1, jc  , kc  , im, jm);
        int ind3 = adr(ic+1, jc  , kc+1, im, jm);
        int ind4 = adr(ic  , jc  , kc+1, im, jm);

        apply_compute(ind1);
        apply_compute(ind2);
        apply_compute(ind3);
        apply_compute(ind4);

        int global_idx_face = ibeg_face_vtx_idx+idx_face;
        face_vtx_bnd_idx[global_idx_face+1] = face_vtx_bnd_idx[global_idx_face] + 4;
        idx_face += 1;
      }
    }
  }
  else if (kmin == kmax) {
    int kc = kmin;
    for(int jc = jmin; jc < jmin+window_size[1]; ++jc) {
      for(int ic = imin; ic < imin+window_size[0]; ++ic) {

        int ind1 = adr(ic  , jc  , kc, im, jm);
        int ind2 = adr(ic+1, jc  , kc, im, jm);
        int ind3 = adr(ic+1, jc+1, kc, im, jm);
        int ind4 = adr(ic  , jc+1, kc, im, jm);

        apply_compute(ind1);
        apply_compute(ind2);
        apply_compute(ind3);
        apply_compute(ind4);

        int global_idx_face = ibeg_face_vtx_idx+idx_face;
        face_vtx_bnd_idx[global_idx_face+1] = face_vtx_bnd_idx[global_idx_face] + 4;
        idx_face += 1;
      }
    }
  }
  else {
    throw std::runtime_error("Unable to determine the direction of the BC window.");
  }

  return i_vtx_bnd;
}

// --------------------------------------------------------------------
PYBIND11_MODULE(wall_distance, m) {
  m.doc() = "pybind11 utils for wall_distance plugin";

  m.def("prepare_extract_bc_u", &prepare_extract_bc_u,
        py::arg("np_point_list").noconvert(),
        py::arg("np_face_vtx").noconvert(),
        py::arg("np_face_vtx_idx").noconvert(),
        py::arg("np_work_vtx").noconvert());

  m.def("prepare_extract_bc_s", &prepare_extract_bc_s,
        py::arg("np_vtx_size").noconvert(),
        py::arg("np_point_range").noconvert(),
        py::arg("np_work_vtx").noconvert());

  m.def("compute_extract_bc_u", &compute_extract_bc_u);
        // py::arg("ibeg_face_vtx_idx").noconvert(),
        // py::arg("np_point_list").noconvert(),
        // py::arg("np_face_vtx").noconvert(),
        // py::arg("np_face_vtx_idx").noconvert(),
        // py::arg("np_work_vtx").noconvert(),
        // py::arg("np_x").noconvert(),
        // py::arg("np_y").noconvert(),
        // py::arg("np_z").noconvert(),
        // py::arg("i_vtx_bnd").noconvert(),
        // py::arg("np_face_vtx_bnd").noconvert(),
        // py::arg("np_face_vtx_bnd_idx").noconvert(),
        // py::arg("np_vtx_bnd").noconvert());

  m.def("compute_extract_bc_s", &compute_extract_bc_s);
}
