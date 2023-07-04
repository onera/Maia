#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "std_e/algorithm/permutation.hpp"
#include "maia/utils/pybind_utils.hpp"

namespace py = pybind11;

// --------------------------------------------------------------------
py::array_t<double, py::array::f_style>
compute_center_cell_u(int n_cell,
                      py::array_t<double, py::array::f_style>& np_cx,
                      py::array_t<double, py::array::f_style>& np_cy,
                      py::array_t<double, py::array::f_style>& np_cz,
                      py::array_t<int,    py::array::f_style>& np_face_vtx,
                      py::array_t<int,    py::array::f_style>& np_face_vtx_idx,
                      py::array_t<int,    py::array::f_style>& np_parent_elements)
{
  int n_face = np_parent_elements.shape()[0];
  // int n_vtx = np_cx.size();
  // std::cout << "compute_center_cell_u: n_cell = " << n_cell << std::endl;
  // std::cout << "compute_center_cell_u: n_face = " << n_face << std::endl;
  // std::cout << "compute_center_cell_u: np_face_vtx.size() = " << np_face_vtx.size() << std::endl;
  // std::cout << "compute_center_cell_u: np_face_vtx_idx.size() = " << np_face_vtx_idx.size() << std::endl;
  // std::cout << "compute_center_cell_u: n_vtx = " << n_vtx << std::endl;
  py::array_t<double, py::array::f_style> np_center_cell(3*n_cell);
  std::vector<int> countc(n_cell, 0);
  // std::cout << "compute_center_cell_u: countc.size() = " << countc.size() << std::endl;

  auto cx              = make_raw_view(np_cx);
  auto cy              = make_raw_view(np_cy);
  auto cz              = make_raw_view(np_cz);
  auto face_vtx        = make_raw_view(np_face_vtx);
  auto face_vtx_idx    = make_raw_view(np_face_vtx_idx);
  auto parent_elements = make_raw_view(np_parent_elements);
  auto center_cell     = make_raw_view(np_center_cell);

  // Init volume to ZERO
  // ---------------
  for (int icell = 0; icell < n_cell; ++icell) {
    center_cell[3*icell  ] = 0.;
    center_cell[3*icell+1] = 0.;
    center_cell[3*icell+2] = 0.;
    countc[icell] = 0;
  }

  // Loop over faces
  // ---------------
  int f_shift(0); // To go back to local cell numbering if ngons are before nface
  if (n_face > 0) {
    if (std::max(parent_elements[0], parent_elements[0+n_face]) > n_face ) {
      f_shift = n_face;
    }
  }
  for (int iface = 0; iface < n_face; ++iface) {
    // -> Face -> Cell connectivity
    int il = parent_elements[iface       ]-1-(f_shift*(parent_elements[iface] > 0));
    int ir = parent_elements[iface+n_face]-1-(f_shift*(parent_elements[iface+n_face] > 0));
    // std::cout << "compute_center_cell_u: iface = " << iface << std::endl;
    // std::cout << "compute_center_cell_u: il = " << il << ", ir = " << ir << std::endl;
    assert(((il >= -1) && (il < n_cell)));
    assert(((ir >= -1) && (ir < n_cell)));

    // Compute the indices of vtx on faces
    int begin_vtx = face_vtx_idx[iface  ];
    int end_vtx   = face_vtx_idx[iface+1];
    // std::cout << "compute_center_cell_u: begin_vtx = " << begin_vtx << ", end_vtx = " << end_vtx << std::endl;

    // Loop over  vertex of each face
    for (int indvtx = begin_vtx; indvtx < end_vtx; ++indvtx) {
      // assert(((indvtx >= 0) && (indvtx < np_face_vtx.size())));
      int ivtx = face_vtx[indvtx]-1;
      // assert(((ivtx >= 0) && (ivtx < n_vtx)));
      // std::cout << "compute_center_cell_u: ivtx = " << ivtx << ", np_face_vtx.size() = " << np_face_vtx.size() << std::endl;

       if (il >= 0) {
         center_cell[3*il  ] += cx[ivtx];
         center_cell[3*il+1] += cy[ivtx];
         center_cell[3*il+2] += cz[ivtx];
         countc[il] += 1;
       }

       if (ir >= 0) {
         center_cell[3*ir  ] += cx[ivtx];
         center_cell[3*ir+1] += cy[ivtx];
         center_cell[3*ir+2] += cz[ivtx];
         countc[ir] += 1;
       }
    }
  }

  // Finalize center cell computation
  // --------------------------------
  for(int icell = 0; icell < n_cell; ++icell) {
    assert(countc[icell] > 0);
    center_cell[3*icell  ] /= countc[icell];
    center_cell[3*icell+1] /= countc[icell];
    center_cell[3*icell+2] /= countc[icell];
    // std::cout << "compute_center_cell_u: center_cell[3*icell  ] = " << center_cell[3*icell  ]
    //                                << ", center_cell[3*icell+1] = " << center_cell[3*icell+1]
    //                                << ", center_cell[3*icell+2] = " << center_cell[3*icell+2] << std::endl;
  }

  return np_center_cell;
}

// --------------------------------------------------------------------
py::array_t<double, py::array::f_style>
compute_center_cell_s(int nx, int ny, int nz,
                      py::array_t<double, py::array::f_style>& np_cx,
                      py::array_t<double, py::array::f_style>& np_cy,
                      py::array_t<double, py::array::f_style>& np_cz)
{

  auto cx = np_cx.template mutable_unchecked<3>();
  auto cy = np_cy.template mutable_unchecked<3>();
  auto cz = np_cz.template mutable_unchecked<3>();

  py::array_t<double, py::array::f_style> np_center(3*nx*ny*nz);
  auto center = make_raw_view(np_center);

  int idx = 0;
  for(int k = 0; k < nz; ++k) {
    for(int j = 0; j < ny; ++j) {
      for(int i = 0; i < nx; ++i) {
        center[3*idx+0] = 0.125*(cx(i  ,j,k) + cx(i  ,j+1,k) + cx(i  ,j+1,k+1) + cx(i  ,j,k+1) 
                               + cx(i+1,j,k) + cx(i+1,j+1,k) + cx(i+1,j+1,k+1) + cx(i+1,j,k+1));
        center[3*idx+1] = 0.125*(cy(i  ,j,k) + cy(i  ,j+1,k) + cy(i  ,j+1,k+1) + cy(i  ,j,k+1)
                               + cy(i+1,j,k) + cy(i+1,j+1,k) + cy(i+1,j+1,k+1) + cy(i+1,j,k+1));
        center[3*idx+2] = 0.125*(cz(i  ,j,k) + cz(i  ,j+1,k) + cz(i  ,j+1,k+1) + cz(i  ,j,k+1)
                               + cz(i+1,j,k) + cz(i+1,j+1,k) + cz(i+1,j+1,k+1) + cz(i+1,j,k+1));
        idx++;
      }
    }
  }

  return np_center;
}
// --------------------------------------------------------------------
py::array_t<double, py::array::f_style>
compute_center_face_s(int nx, int ny, int nz,
                      py::array_t<double, py::array::f_style>& np_cx,
                      py::array_t<double, py::array::f_style>& np_cy,
                      py::array_t<double, py::array::f_style>& np_cz)
{

  auto cx = np_cx.template mutable_unchecked<3>();
  auto cy = np_cy.template mutable_unchecked<3>();
  auto cz = np_cz.template mutable_unchecked<3>();

  int n_face_tot = (nz - 1)*(nx - 1)*ny + (nz - 1)*(ny - 1)*nx + nz*(ny - 1)*(nx - 1); // A CHANGER

  py::array_t<double, py::array::f_style> np_center(3*n_face_tot);
  auto center = make_raw_view(np_center);

  int idx = 0;
  for(int k = 0; k < nz-1; ++k) {
    for(int j = 0; j < ny-1; ++j) {
      for(int i = 0; i < nx; ++i) {
        center[3*idx+0] = 0.250 * (cx(i, j, k) + cx(i, j+1, k) + cx(i, j, k+1) + cx(i, j+1, k+1));
        center[3*idx+1] = 0.250 * (cy(i, j, k) + cy(i, j+1, k) + cy(i, j, k+1) + cy(i, j+1, k+1));
        center[3*idx+2] = 0.250 * (cz(i, j, k) + cz(i, j+1, k) + cz(i, j, k+1) + cz(i, j+1, k+1));
        idx++;
      }
    }
  };
  for(int k = 0; k < nz-1; ++k) {
    for(int j = 0; j < ny; ++j) {
      for(int i = 0; i < nx-1; ++i) {
        center[3*idx+0] = 0.250 * (cx(i, j, k) + cx(i+1, j, k) + cx(i, j, k+1) + cx(i+1, j, k+1));
        center[3*idx+1] = 0.250 * (cy(i, j, k) + cy(i+1, j, k) + cy(i, j, k+1) + cy(i+1, j, k+1));
        center[3*idx+2] = 0.250 * (cz(i, j, k) + cz(i+1, j, k) + cz(i, j, k+1) + cz(i+1, j, k+1));
        idx++;
      }
    }
  };
  for(int k = 0; k < nz; ++k) {
    for(int j = 0; j < ny-1; ++j) {
      for(int i = 0; i < nx-1; ++i) {
        center[3*idx+0] = 0.250 * (cx(i, j, k) + cx(i+1, j, k) + cx(i, j+1, k) + cx(i+1, j+1, k));
        center[3*idx+1] = 0.250 * (cy(i, j, k) + cy(i+1, j, k) + cy(i, j+1, k) + cy(i+1, j+1, k));
        center[3*idx+2] = 0.250 * (cz(i, j, k) + cz(i+1, j, k) + cz(i, j+1, k) + cz(i+1, j+1, k));
        idx++;
      }
    }
  }
  return np_center;
}

// --------------------------------------------------------------------
py::array_t<double, py::array::f_style>
compute_face_normal_u(py::array_t<int   , py::array::f_style>& np_face_vtx_idx,
                      py::array_t<double, py::array::f_style>& np_cx,
                      py::array_t<double, py::array::f_style>& np_cy,
                      py::array_t<double, py::array::f_style>& np_cz)
{
  // Compute face normal ponderated by face area, assuming that coords cx,cy,cz are
  // the coordinates of face vertices (with repetitions)
  // Eg if we have tri face [3,2,4,  5,4,6] np_cx is [X_3, X_2, X_4,  X5,X4,X6]

  auto cx              = make_raw_view(np_cx);
  auto cy              = make_raw_view(np_cy);
  auto cz              = make_raw_view(np_cz);
  auto face_vtx_idx    = make_raw_view(np_face_vtx_idx);

  int n_face = np_face_vtx_idx.shape()[0] - 1;

  py::array_t<double, py::array::f_style> np_face_normal(3*n_face);
  auto face_normal = make_raw_view(np_face_normal);

  for (int i = 0; i < n_face; ++i) {
    int start = face_vtx_idx[i];
    int end   = face_vtx_idx[i+1];
    int n_vtx = end - start;

    double mean_center_x = std::accumulate(&cx[start], &cx[end], 0) / n_vtx;
    double mean_center_y = std::accumulate(&cy[start], &cy[end], 0) / n_vtx;
    double mean_center_z = std::accumulate(&cz[start], &cz[end], 0) / n_vtx;

    double face_normal_x = 0;
    double face_normal_y = 0;
    double face_normal_z = 0;
    // Compute cross product (uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)
    // between OVI, OVI+1 where O is the mean center, and VI / VI+1 are the vertices
    for (int j = start; j < end-1; ++j) {
      face_normal_x += ((cy[j] - mean_center_y)*(cz[j+1]-mean_center_z) - (cz[j] - mean_center_z)*(cy[j+1]-mean_center_y));
      face_normal_y += ((cz[j] - mean_center_z)*(cx[j+1]-mean_center_x) - (cx[j] - mean_center_x)*(cz[j+1]-mean_center_z));
      face_normal_z += ((cx[j] - mean_center_x)*(cy[j+1]-mean_center_y) - (cy[j] - mean_center_y)*(cx[j+1]-mean_center_x));
    }
    face_normal_x += ((cy[end-1] - mean_center_y)*(cz[start]-mean_center_z) - (cz[end-1] - mean_center_z)*(cy[start]-mean_center_y));
    face_normal_y += ((cz[end-1] - mean_center_z)*(cx[start]-mean_center_x) - (cx[end-1] - mean_center_x)*(cz[start]-mean_center_z));
    face_normal_z += ((cx[end-1] - mean_center_x)*(cy[start]-mean_center_y) - (cy[end-1] - mean_center_y)*(cx[start]-mean_center_x));

    face_normal[3*i+0] = 0.5*face_normal_x;
    face_normal[3*i+1] = 0.5*face_normal_y;
    face_normal[3*i+2] = 0.5*face_normal_z;
  }

  return np_face_normal;
}
