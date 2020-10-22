#pragma once


#include <vector>
#include "std_e/future/algorithm.hpp"
#include "std_e/multi_index/cartesian_product_size.hpp"
#include "std_e/multi_index/fortran_order.hpp"
#include "std_e/multi_index/multi_index_range.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"
#include "maia/generate/__old/structured_grid_utils.hpp"
#include "range/v3/view/transform.hpp"
#include "range/v3/view/concat.hpp"


template<class Multi_index, class I = typename Multi_index::value_type> inline auto
generate_hex_8(const Multi_index& vertex_dims, const Multi_index& is) -> cgns::hex_8<I> {
  cgns::hex_8<I> hex;
  hex[0] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
  hex[1] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
  hex[2] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]  } );
  hex[3] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
  hex[4] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
  hex[5] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]+1} );
  hex[6] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]+1} );
  hex[7] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]+1} );
  return hex;
}

template<class Multi_index> constexpr auto
generate_cells(const Multi_index& vertex_dims) {
  auto gen_hex_8 = [&vertex_dims](const Multi_index& is){ return generate_hex_8(vertex_dims,is); };
 
  Multi_index cell_dims = structured_block_cell_dims(vertex_dims);
  return std_e::fortran_multi_index_range(cell_dims) | ranges::views::transform(gen_hex_8);
}





template<class Multi_index, class I = typename Multi_index::value_type> inline auto
generate_quad_4(const Multi_index& vertex_dims, const Multi_index& is, const Multi_index& dirs) -> cgns::quad_4<I> {
  cgns::quad_4<I> quad;
//
  std::array<I,3> vertex_0_ids;
  vertex_0_ids[dirs[2]] = is[2]  ;
  vertex_0_ids[dirs[0]] = is[0]  ;
  vertex_0_ids[dirs[1]] = is[1]  ;

  std::array<I,3> vertex_1_ids;
  vertex_1_ids[dirs[2]] = is[2]  ;
  vertex_1_ids[dirs[0]] = is[0]+1;
  vertex_1_ids[dirs[1]] = is[1]  ;

  std::array<I,3> vertex_2_ids;
  vertex_2_ids[dirs[2]] = is[2]  ;
  vertex_2_ids[dirs[0]] = is[0]+1;
  vertex_2_ids[dirs[1]] = is[1]+1;

  std::array<I,3> vertex_3_ids;
  vertex_3_ids[dirs[2]] = is[2]  ;
  vertex_3_ids[dirs[0]] = is[0]  ;
  vertex_3_ids[dirs[1]] = is[1]+1;

//
  //std::array<I,3> vertex_0_ids;
  //vertex_0_ids[0] = is[0]  ;
  //vertex_0_ids[1] = is[1]  ;
  //vertex_0_ids[2] = is[2]  ;

  //std::array<I,3> vertex_1_ids;
  //vertex_1_ids[0] = is[0]  ;
  //vertex_1_ids[1] = is[1]+1;
  //vertex_1_ids[2] = is[2]  ;

  //std::array<I,3> vertex_2_ids;
  //vertex_2_ids[0] = is[0]  ;
  //vertex_2_ids[1] = is[1]+1;
  //vertex_2_ids[2] = is[2]+1;

  //std::array<I,3> vertex_3_ids;
  //vertex_3_ids[0] = is[0]  ;
  //vertex_3_ids[1] = is[1]  ;
  //vertex_3_ids[2] = is[2]+1;

//
  //std::array<I,3> vertex_0_ids;
  //vertex_0_ids[2] = is[2]  ;
  //vertex_0_ids[0] = is[0]  ;
  //vertex_0_ids[1] = is[1]  ;

  //std::array<I,3> vertex_1_ids;
  //vertex_1_ids[2] = is[2]  ;
  //vertex_1_ids[0] = is[0]+1;
  //vertex_1_ids[1] = is[1]  ;

  //std::array<I,3> vertex_2_ids;
  //vertex_2_ids[2] = is[2]  ;
  //vertex_2_ids[0] = is[0]+1;
  //vertex_2_ids[1] = is[1]+1;

  //std::array<I,3> vertex_3_ids;
  //vertex_3_ids[2] = is[2]  ;
  //vertex_3_ids[0] = is[0]  ;
  //vertex_3_ids[1] = is[1]+1;

//
  //std::array<I,3> vertex_0_ids;
  //vertex_0_ids[dirs[0]] = is[0]  ;
  //vertex_0_ids[dirs[1]] = is[1]  ;
  //vertex_0_ids[dirs[2]] = is[2]  ;

  //std::array<I,3> vertex_1_ids;
  //vertex_1_ids[dirs[0]] = is[0]  ;
  //vertex_1_ids[dirs[1]] = is[1]+1;
  //vertex_1_ids[dirs[2]] = is[2]  ;

  //std::array<I,3> vertex_2_ids;
  //vertex_2_ids[dirs[0]] = is[0]  ;
  //vertex_2_ids[dirs[1]] = is[1]+1;
  //vertex_2_ids[dirs[2]] = is[2]+1;

  //std::array<I,3> vertex_3_ids;
  //vertex_3_ids[dirs[0]] = is[0]  ;
  //vertex_3_ids[dirs[1]] = is[1]  ;
  //vertex_3_ids[dirs[2]] = is[2]+1;

  quad[0] = std_e::fortran_order_from_dimensions( vertex_dims , vertex_0_ids );
  quad[1] = std_e::fortran_order_from_dimensions( vertex_dims , vertex_1_ids );
  quad[2] = std_e::fortran_order_from_dimensions( vertex_dims , vertex_2_ids );
  quad[3] = std_e::fortran_order_from_dimensions( vertex_dims , vertex_3_ids );
  return quad;
}


template<class Multi_index, class I = typename Multi_index::value_type> inline auto
generate_faces(const Multi_index& vertex_dims, int d) {
  // Generates connectivities along direction "d"
  // Vertices on which faces are built are ordered along i,j,k following fortran order
  // Generated faces are given by sheets normal to d, by fortran order along (d+1)%3,(d+2)%3

  // Precondition: 0 <= d < 3
  auto d_face_dims = structured_block_face_dims_in_direction(vertex_dims,d);

  Multi_index dirs = {(d+1)%3,(d+2)%3,d}; // fill sheet by sheet, hence "d" is the last varying index

  Multi_index dim2 = {d_face_dims[dirs[0]],d_face_dims[dirs[1]],d_face_dims[dirs[2]]};
  auto d_faces_indices_range = std_e::fortran_multi_index_range(dim2);

  // TODO
  //auto d_faces_indices_range = std_e::multi_index_range_with_order(d_face_dims,dirs);

  auto gen_quad_4 = [&vertex_dims,dirs](const Multi_index& is){ return generate_quad_4(vertex_dims,is,dirs); };
  return d_faces_indices_range | ranges::views::transform(gen_quad_4);
}

template<class Multi_index> inline auto
generate_faces(const Multi_index& vertex_dims) -> auto {
  return ranges::views::concat(
    generate_faces(vertex_dims,0),
    generate_faces(vertex_dims,1),
    generate_faces(vertex_dims,2)
  );
}




template<class Multi_index, class I = typename Multi_index::value_type> inline auto
generate_parent_cell_id(const Multi_index& cell_dims, const Multi_index& is, const Multi_index& dirs) -> I {
  std::array<I,3> parent_cell_id;
  parent_cell_id[dirs[2]] = is[2];
  parent_cell_id[dirs[0]] = is[0];
  parent_cell_id[dirs[1]] = is[1];

  return std_e::fortran_order_from_dimensions( cell_dims, parent_cell_id );
}

constexpr int no_parent_element = -1;
// TODO factor with generate_faces (same iterator)
template<class Multi_index> inline auto
generate_faces_right_parent_cell_ids(const Multi_index& vertex_dims, int d) {
  using I = std_e::index_type_of<Multi_index>;
  // Gives parent cell of each face generated by generate_faces
  
  // Precondition: 0 <= d < 3
  auto d_face_dims = structured_block_face_dims_in_direction(vertex_dims,d);
  auto cell_dims = structured_block_cell_dims(vertex_dims);

  Multi_index dirs = {(d+1)%3,(d+2)%3,d}; // fill sheet by sheet, hence "d" is the last varying index

  Multi_index             dim2 = {d_face_dims[dirs[0]],d_face_dims[dirs[1]],d_face_dims[dirs[2]]-1}; // no right parents for last sheet
  std_e::multi_index<I,2> dim1 = {d_face_dims[dirs[0]],d_face_dims[dirs[1]]};

  auto d_faces_indices_range = std_e::fortran_multi_index_range(dim2);
  auto d_faces_indices_range_bnd = std_e::fortran_multi_index_range(dim1);

  auto gen_parent_cell_id = [cell_dims,dirs](const Multi_index& is){ return generate_parent_cell_id(cell_dims,is,dirs); };
  auto gen_no_parent_cell_id = [](const auto&){ return no_parent_element; };

  return ranges::views::concat(
    d_faces_indices_range     | ranges::views::transform(gen_parent_cell_id),
    d_faces_indices_range_bnd | ranges::views::transform(gen_no_parent_cell_id) // no right parents for last sheet
  );
}
// TODO factor with generate_faces (same iterator)
template<class Multi_index> inline auto
generate_faces_left_parent_cell_ids(const Multi_index& vertex_dims, int d) {
  using I = std_e::index_type_of<Multi_index>;
  // Gives parent cell of each face generated by generate_faces
  
  // Precondition: 0 <= d < 3
  auto d_face_dims = structured_block_face_dims_in_direction(vertex_dims,d);
  auto cell_dims = structured_block_cell_dims(vertex_dims);

  Multi_index dirs = {(d+1)%3,(d+2)%3,d}; // fill sheet by sheet, hence "d" is the last varying index

  Multi_index             dim2 = {d_face_dims[dirs[0]],d_face_dims[dirs[1]],d_face_dims[dirs[2]]-1}; // no left parents for first sheet
  std_e::multi_index<I,2> dim1 = {d_face_dims[dirs[0]],d_face_dims[dirs[1]]};

  auto d_faces_indices_range = std_e::fortran_multi_index_range(dim2);
  auto d_faces_indices_range_bnd = std_e::fortran_multi_index_range(dim1);

  auto gen_parent_cell_id = [cell_dims,dirs](const Multi_index& is){ return generate_parent_cell_id(cell_dims,is,dirs); };
  auto gen_no_parent_cell_id = [](const auto&){ return no_parent_element; };

  return ranges::views::concat(
    d_faces_indices_range_bnd | ranges::views::transform(gen_no_parent_cell_id), // no left parents for first sheet
    d_faces_indices_range     | ranges::views::transform(gen_parent_cell_id)
  );
}

template<class Multi_index> inline auto
generate_faces_left_parent_cell_ids(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_faces_left_parent_cell_ids(vertex_dims,0),
    generate_faces_left_parent_cell_ids(vertex_dims,1),
    generate_faces_left_parent_cell_ids(vertex_dims,2)
  );
}
template<class Multi_index> inline auto
generate_faces_right_parent_cell_ids(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_faces_right_parent_cell_ids(vertex_dims,0),
    generate_faces_right_parent_cell_ids(vertex_dims,1),
    generate_faces_right_parent_cell_ids(vertex_dims,2)
  );
}

template<class Multi_index> inline auto
generate_faces_parent_cell_ids(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_faces_left_parent_cell_ids (vertex_dims),
    generate_faces_right_parent_cell_ids(vertex_dims)
  );
}

