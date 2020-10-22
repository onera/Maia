#pragma once


#include "std_e/multi_index/multi_index.hpp"


template<class Multi_index> inline auto
structured_block_cell_dims(const Multi_index& vertex_dims) -> Multi_index {
  int rank = vertex_dims.size();
  auto nb_cells = std_e::make_array_of_size<Multi_index>(rank);
  for (int i=0; i<rank; ++i) {
    nb_cells[i] = vertex_dims[i]-1;
  }
  return nb_cells;
}


template<class Multi_index> inline auto
structured_block_face_dims_in_direction(const Multi_index& vertex_dims, int d) -> Multi_index {
  // Precondition: 0 <= d < rank
  int rank = vertex_dims.size();
  Multi_index d_face_dims;
  for (int dir=0; dir<rank; ++dir) {
    if (dir==d) {
      d_face_dims[dir] = vertex_dims[dir];
    } else {
      d_face_dims[dir] = vertex_dims[dir]-1;
    }
  }
  return d_face_dims;
}
