#pragma once


#include "std_e/multi_index/fortran_order.hpp"
#include <array>


namespace maia {


template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_hex_8(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,8> {
  std::array<I,8> hex;
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

template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_quad_4_normal_to_i(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,4> {
  std::array<I,4> quad;
  quad[0] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
  quad[1] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
  quad[2] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]+1} );
  quad[3] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
  return quad;
}
template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_quad_4_normal_to_j(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,4> {
  std::array<I,4> quad;
  quad[0] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
  quad[1] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
  quad[2] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]+1} );
  quad[3] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
  return quad;
}
template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_quad_4_normal_to_k(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,4> {
  std::array<I,4> quad;
  quad[0] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
  quad[1] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
  quad[2] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]  } );
  quad[3] = std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
  return quad;
}

template<int d, class Multi_index, class I = typename Multi_index::value_type> auto
generate_quad_4_normal_to(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,4> {
  static_assert(0<=d && d<3);
       if constexpr (d==0) { return generate_quad_4_normal_to_i(vertex_dims,is); }
  else if constexpr (d==1) { return generate_quad_4_normal_to_j(vertex_dims,is); }
  else if constexpr (d==2) { return generate_quad_4_normal_to_k(vertex_dims,is); }
  else throw;
}


} // maia
