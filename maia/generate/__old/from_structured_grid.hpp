#pragma once


#include <vector>
#include "std_e/future/algorithm.hpp"
#include "std_e/multi_index/cartesian_product_size.hpp"
#include "std_e/multi_index/multi_index_range.hpp"
#include "maia/generate/__old/structured_grid_utils.hpp"
#include "std_e/utils/concatenate.hpp"
#include <algorithm>
#include <ranges>
#include <span>
#include "std_e/future/ranges.hpp"
#include "std_e/future/ranges/repeat.hpp"
#include "maia/generate/connectivity/from_structured_grid.hpp"
#include "range/v3/view/concat.hpp"


namespace maia {


// generate_cells {
template<class Multi_index> constexpr auto
generate_cells(const Multi_index& vertex_dims) {
  auto gen_hex_8 = [&vertex_dims](const Multi_index& is){ return generate_hex_8(vertex_dims,is); };

  Multi_index cell_dims = block_cell_dims(vertex_dims);
  return
      std_e::fortran_multi_index_range(cell_dims)
    | std::views::transform(gen_hex_8);
}
// generate_cells }

// generate_face {
// Generates connectivities along direction "d"
// Vertices on which faces are built are ordered along i,j,k following fortran order
// Generated faces are given by sheets normal to d, by fortran order
template<int d, class Multi_index, class I = typename Multi_index::value_type> auto
generate_faces_normal_to(const Multi_index& vertex_dims) {
  STD_E_ASSERT(0<=d && d<3);
  auto gen_quad_4 = [&vertex_dims](const Multi_index& is){ return generate_quad_4_normal_to<d>(vertex_dims,is); };

  Multi_index iter_order = {(d+1)%3,(d+2)%3,d}; // fill sheet by sheet, hence "d" is the last varying index
  Multi_index face_sheets_dims = block_face_dims_normal_to(vertex_dims,d);

  return
      std_e::multi_index_range_with_order(face_sheets_dims,iter_order)
    | std::views::transform(gen_quad_4);
}

template<class Multi_index> auto
generate_faces(const Multi_index& vertex_dims) {
  return
      ranges::views::concat(
        generate_faces_normal_to<0>(vertex_dims),
        generate_faces_normal_to<1>(vertex_dims),
        generate_faces_normal_to<2>(vertex_dims)
      );
}
// generate_face }


// generate_parents {
template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_parent_cell_id(const Multi_index& cell_dims, const Multi_index& is) -> I {
  return std_e::fortran_order_from_dimensions( cell_dims, is );
}


template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_cell_parents(const Multi_index& vertex_dims, int d) {
  STD_E_ASSERT(0<=d && d<3);

  Multi_index iter_order = {(d+1)%3,(d+2)%3,d}; // fill sheet by sheet, hence "d" is the last varying index
  auto cell_dims = block_cell_dims(vertex_dims);

  auto gen_parent_cell_id = [cell_dims](const Multi_index& is){ return generate_parent_cell_id(cell_dims,is); };

  return
      std_e::multi_index_range_with_order(cell_dims,iter_order)
    | std::views::transform(gen_parent_cell_id);
}

template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_l_parents(const Multi_index& vertex_dims, int d) {
  auto sheet_dims = dims_of_sheet_normal_to(vertex_dims,d);
  auto sheet_sz = std_e::cartesian_product_size(sheet_dims);

  return
    ranges::views::concat(
      std_e::ranges::repeat(-1,sheet_sz), // no left parents for first sheet
      generate_cell_parents(vertex_dims,d)
    );
}

template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_r_parents(const Multi_index& vertex_dims, int d) {
  auto sheet_dims = dims_of_sheet_normal_to(vertex_dims,d);
  auto sheet_sz = std_e::cartesian_product_size(sheet_dims);

  return
    ranges::views::concat(
      generate_cell_parents(vertex_dims,d),
      std_e::ranges::repeat(-1,sheet_sz) // no right parents for last sheet
    );
}

template<class Multi_index> auto
generate_l_parents(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_l_parents(vertex_dims,0),
    generate_l_parents(vertex_dims,1),
    generate_l_parents(vertex_dims,2)
  );
}
template<class Multi_index> auto
generate_r_parents(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_r_parents(vertex_dims,0),
    generate_r_parents(vertex_dims,1),
    generate_r_parents(vertex_dims,2)
  );
}

template<class Multi_index> auto
generate_faces_parent_cell_ids(const Multi_index& vertex_dims) {
  return ranges::views::concat(
    generate_l_parents (vertex_dims),
    generate_r_parents(vertex_dims)
  );
}
// generate_parents }


} // maia
