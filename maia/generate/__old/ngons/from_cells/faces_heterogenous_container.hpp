#pragma once

#include "std_e/data_structure/heterogenous_vector.hpp"
#include "maia/generate/__old/ngons/from_cells/face_with_parents.hpp"

// TODO clean
template<class I>
struct faces_heterogenous_container {
  using faces_heterogenous_container_by_parent_type = std_e::hvector<
    tri_3_with_sorted_connectivity<I>,
    quad_4_with_sorted_connectivity<I>
  >;

  faces_heterogenous_container_by_parent_type from_vol;
  faces_heterogenous_container_by_parent_type from_face;
};


template<class I>
struct faces_container {
  using interior_faces_container = std_e::hvector<
    interior_tri_3<I>,
    interior_quad_4<I>
  >;
  using boundary_faces_container = std_e::hvector<
    boundary_tri_3<I>,
    boundary_quad_4<I>
  >;

  boundary_faces_container boundary;
  interior_faces_container interior;
};
