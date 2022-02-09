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

// --------------------------------------------------------------------
auto
adapt_match_information(py::array_t<int, py::array::f_style>& np_neighbor_idx,
                        py::array_t<int, py::array::f_style>& np_neighbor_desc,
                        py::array_t<int, py::array::f_style>& np_recv_entity_stri,
                        py::array_t<int, py::array::f_style>& np_point_list,
                        py::array_t<int, py::array::f_style>& np_point_list_donor)
{
  int join_size = np_point_list.size();

  auto neighbor_idx     = make_raw_view(np_neighbor_idx);
  auto neighbor_desc    = make_raw_view(np_neighbor_desc);     /* i_proc, i_cloud, i_entity */
  auto entity_stri      = make_raw_view(np_recv_entity_stri);  /* stride of pointlist       */
  auto point_list       = make_raw_view(np_point_list);        /* point_list                */
  auto point_list_donor = make_raw_view(np_point_list_donor);  /* point_list_donor          */

  // Sanity check - For now only faces so 1to1 - En vertex on aura le pb
  for(int i = 0; i < join_size; ++i){
    assert(neighbor_idx[i+1] - neighbor_idx[i] == 1);
    assert(entity_stri[i] == 1);
  }

  /* We first need to sort the neighbor_desc lexicographicly */
  std::vector<int> order(join_size);
  std::iota(begin(order), end(order), 0);
  std::sort(begin(order), end(order), [&](auto& i, auto& j){
    return std::tie(neighbor_desc[3*i], neighbor_desc[3*i+1], neighbor_desc[3*i+2])
         < std::tie(neighbor_desc[3*j], neighbor_desc[3*j+1], neighbor_desc[3*j+2]);
  });

  std_e::permute(point_list      , order);
  std_e::permute(point_list_donor, order);

  /* Now we have section with all coherent partition */
  // Count how many new section is here
  std::vector<int> section_idx(join_size, 0);
  int n_entity  = 0;
  int n_section = 0;
  int cur_proc = neighbor_desc[3*order[0]  ];
  int cur_part = neighbor_desc[3*order[0]+1];
  for(int i = 0; i < join_size; ++i){
    int p = order[i];
    int next_proc = neighbor_desc[3*p  ];
    int next_part = neighbor_desc[3*p+1];
    if( next_part != cur_part || next_proc != cur_proc ){
      cur_proc = next_proc;
      cur_part = next_part;
      section_idx[n_section] = n_entity;
      n_section++;
      n_entity = 0;
    } else {
      n_entity++;
    }
  }
  n_section++;
  section_idx[n_section] = n_entity;
  printf(" n_section = %i\n", n_section);
  section_idx.resize(n_section+1);

  int n_entity_max = -1;
  for (int i = 0; i < n_section; i++){
    section_idx[i+1] += section_idx[i];
    n_entity_max = std::max(n_entity_max, section_idx[i+1]-section_idx[i]);
    // printf("section_idx[%i] = %i \n", i+1, section_idx[i+1]);
  }
  // printf("n_entity_max = %i \n", n_entity_max);
  // Now we order the pointlist

  for(int i_section = 0; i_section < n_section; ++i_section) {

    int begs = section_idx[i_section];
    int n_entity_per_join = section_idx[i_section+1] - begs;

    std::vector<int> point_list_min(n_entity_per_join);
    std::vector<int> point_list_max(n_entity_per_join);
    std::vector<int> order_pl(n_entity_per_join);

    // printf("begs = %i | n_entity_per_join = %i \n", begs, n_entity_per_join);

    for(int i = 0; i < n_entity_per_join; ++i){
      point_list_min[i] = std::min(point_list[begs+i], point_list_donor[begs+i]);
      point_list_max[i] = std::max(point_list[begs+i], point_list_donor[begs+i]);
    }

    // Pour chaque raccord il faut trier
    std::iota(begin(order_pl), begin(order_pl)+n_entity_per_join, 0);

    std::sort(begin(order_pl), begin(order_pl)+n_entity_per_join, [&](auto& i, auto& j){
      return std::tie(point_list_min[i], point_list_max[i])
           < std::tie(point_list_min[j], point_list_max[j]);
    });

    // Well permutation now !
    std_e::permute(point_list      +begs, order_pl);
    std_e::permute(point_list_donor+begs, order_pl);

  }


  // Panic verbose
  if(0 == 1){
    for(int i = 0; i < join_size; ++i){
      std::cout << "Info :: pl = " << point_list[i] << " | pld = " << point_list_donor[i];
      std::cout << " | order = " << order[i];
      std::cout << " | " << neighbor_desc[3*i  ] << "/";
      std::cout <<          neighbor_desc[3*i+1] << "/";
      std::cout <<          neighbor_desc[3*i+2] << std::endl;
    }
  }

  // py::array_t<int, py::array::f_style> np_section_idx(n_section+1);
  py::array_t<int, py::array::f_style> np_section_idx(section_idx.size());
  auto np_section_idx_ptr = make_raw_view(np_section_idx);
  for(int i = 0; i < n_section + 1; ++i){
    np_section_idx_ptr[i] = section_idx[i];
  }

  return np_section_idx;
}

// --------------------------------------------------------------------
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
  auto characteristic_length = make_raw_view(np_characteristic_lenght);

  for(int idx = 0; idx < bnd_size; ++idx) {
    int i_face = point_list[idx]-1;
    bnd_coord[3*idx  ] = 0.;
    bnd_coord[3*idx+1] = 0.;
    bnd_coord[3*idx+2] = 0.;

    characteristic_length[idx] = std::numeric_limits<double>::max();

    int beg = face_vtx_idx[i_face];
    int n_vtx_on_face = face_vtx_idx[i_face+1]-beg;

    for(int idx_vtx = beg; idx_vtx < face_vtx_idx[i_face+1]; ++idx_vtx) {

      int pos1 =   idx_vtx - beg;
      int pos2 = ( pos1 + 1 ) % n_vtx_on_face;

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
      double length = std::sqrt(dx*dx + dy*dy + dz*dz);

      characteristic_length[idx] = std::min(characteristic_length[idx], length);
    }

    // Finish
    double inv = 1./n_vtx_on_face;
    bnd_coord[3*idx  ] = bnd_coord[3*idx  ] * inv;
    bnd_coord[3*idx+1] = bnd_coord[3*idx+1] * inv;
    bnd_coord[3*idx+2] = bnd_coord[3*idx+2] * inv;
  }

  return std::make_tuple(np_bnd_coord, np_characteristic_lenght);
}

// --------------------------------------------------------------------
auto
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
  for (int iface = 0; iface < n_face; ++iface) {
    // -> Face -> Cell connectivity
    int il = parent_elements[iface       ]-1;
    int ir = parent_elements[iface+n_face]-1;
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
auto
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
PYBIND11_MODULE(geometry, m) {
  m.doc() = "pybind11 utils for geomery plugin"; // optional module docstring

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
}
