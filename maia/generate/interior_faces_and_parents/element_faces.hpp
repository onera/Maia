#pragma once


#include "cpp_cgns/sids/cgnslib.h"
#include "std_e/base/not_implemented_exception.hpp"


namespace maia {


template<
  cgns::ElementType_t elt_type,
  class connectivity_type,
  class tri_iterator, class quad_iterator
> auto
generate_faces(const connectivity_type& e, tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  using namespace cgns;
  if constexpr (elt_type==TRI_3) {
    *tri_it++ = e;
  }
  else if constexpr (elt_type==QUAD_4) {
    *quad_it++ = e;
  }
  else if constexpr (elt_type==TETRA_4) {
    *tri_it++ = {e[0],e[2],e[1]};
    *tri_it++ = {e[0],e[1],e[3]};
    *tri_it++ = {e[1],e[2],e[3]};
    *tri_it++ = {e[2],e[0],e[3]};
  }
  else if constexpr (elt_type==PYRA_5) {
    *quad_it++ = {e[0],e[3],e[2],e[1]};
    * tri_it++ = {e[0],e[1],e[4]};
    * tri_it++ = {e[1],e[2],e[4]};
    * tri_it++ = {e[2],e[3],e[4]};
    * tri_it++ = {e[3],e[0],e[4]};
  }
  else if constexpr (elt_type==PENTA_6) {
    *quad_it++ = {e[0],e[1],e[4],e[3]};
    *quad_it++ = {e[1],e[2],e[5],e[4]};
    *quad_it++ = {e[2],e[0],e[3],e[5]};
    * tri_it++ = {e[0],e[2],e[1]};
    * tri_it++ = {e[3],e[4],e[5]};
  }
  else if constexpr (elt_type==HEXA_8) {
    *quad_it++ = {e[0],e[3],e[2],e[1]};
    *quad_it++ = {e[0],e[1],e[5],e[4]};
    *quad_it++ = {e[1],e[2],e[6],e[5]};
    *quad_it++ = {e[2],e[3],e[7],e[6]};
    *quad_it++ = {e[0],e[4],e[7],e[3]};
    *quad_it++ = {e[4],e[5],e[6],e[7]};
  }
  else {
    throw std_e::not_implemented_exception("unsupported ElementType_t "+to_string(elt_type));
  }
}


template<
  cgns::ElementType_t elt_type,
  class tri_iterator, class quad_iterator
> auto
generate_parent_positions(tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  using namespace cgns;
  if constexpr (elt_type==TRI_3) {
    *tri_it++ = 1;
  }
  else if constexpr (elt_type==QUAD_4) {
    *quad_it++ = 1;
  }
  else if constexpr (elt_type==TETRA_4) {
    *tri_it++ = 1;
    *tri_it++ = 2;
    *tri_it++ = 3;
    *tri_it++ = 4;
  }
  else if constexpr (elt_type==PYRA_5) {
    *quad_it++ = 1;
    * tri_it++ = 2;
    * tri_it++ = 3;
    * tri_it++ = 4;
    * tri_it++ = 5;
  }
  else if constexpr (elt_type==PENTA_6) {
    *quad_it++ = 1;
    *quad_it++ = 2;
    *quad_it++ = 3;
    * tri_it++ = 4;
    * tri_it++ = 5;
  }
  else if constexpr (elt_type==HEXA_8) {
    *quad_it++ = 1;
    *quad_it++ = 2;
    *quad_it++ = 3;
    *quad_it++ = 4;
    *quad_it++ = 5;
    *quad_it++ = 6;
  }
  else {
    throw std_e::not_implemented_exception("unsupported ElementType_t "+to_string(elt_type));
  }
}


} // maia
