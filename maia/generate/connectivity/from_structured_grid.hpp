#pragma once


#include "std_e/multi_index/fortran_order.hpp"
#include <array>


namespace maia {


template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_hex_8(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,8> {
  std::array<I,8> hex;
  hex[0] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
  hex[1] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
  hex[2] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]  } );
  hex[3] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
  hex[4] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
  hex[5] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]+1} );
  hex[6] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]+1} );
  hex[7] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]+1} );
  return hex;
}

template<int d, class Multi_index, class I = typename Multi_index::value_type> auto
generate_quad_4_normal_to(const Multi_index& vertex_dims, const Multi_index& is) -> std::array<I,4> {
  std::array<I,4> quad;
  static_assert(0<=d && d<3);
  if constexpr (d==0) {
    quad[0] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
    quad[1] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
    quad[2] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]+1} );
    quad[3] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
  } else if constexpr (d==1) {
    quad[0] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
    quad[1] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]+1} );
    quad[2] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]+1} );
    quad[3] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
  } else if constexpr (d==2) {
    quad[0] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]  ,is[2]  } );
    quad[1] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]  ,is[2]  } );
    quad[2] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]+1,is[1]+1,is[2]  } );
    quad[3] = 1 + std_e::fortran_order_from_dimensions( vertex_dims , Multi_index{is[0]  ,is[1]+1,is[2]  } );
  }
  return quad;
}


template<class Multi_index, class I = typename Multi_index::value_type> auto
generate_parent_cell_id(const Multi_index& cell_dims, const Multi_index& is) -> I {
  return 1 + std_e::fortran_order_from_dimensions( cell_dims, is );
}


} // maia
