#pragma once


#include "std_e/multi_index/multi_index.hpp"


template<class Multi_index> auto
block_cell_dims(const Multi_index& vtx_dims) -> Multi_index {
  int rank = vtx_dims.size();
  auto nb_cells = std_e::make_array_of_size<Multi_index>(rank);
  for (int i=0; i<rank; ++i) {
    nb_cells[i] = vtx_dims[i]-1;
  }
  return nb_cells;
}


template<class Multi_index> auto
block_face_dims_normal_to(const Multi_index& vtx_dims, int d) -> Multi_index {
  // Precondition: 0 <= d < rank
  int rank = vtx_dims.size();
  Multi_index d_face_dims;
  for (int dir=0; dir<rank; ++dir) {
    if (dir==d) {
      d_face_dims[dir] = vtx_dims[dir];
    } else {
      d_face_dims[dir] = vtx_dims[dir]-1;
    }
  }
  return d_face_dims;
}

template<class Multi_index, class I = typename Multi_index::value_type> auto
dims_of_sheet_normal_to(const Multi_index& vtx_dims, int d) -> std_e::multi_index<I,2> {
  STD_E_ASSERT(0<=d && d<3);
  if (d==0) return {vtx_dims[1]-1,vtx_dims[2]-1};
  if (d==1) return {vtx_dims[2]-1,vtx_dims[0]-1};
  if (d==2) return {vtx_dims[0]-1,vtx_dims[1]-1};
  throw;
}
