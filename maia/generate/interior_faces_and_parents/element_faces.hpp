#pragma once

#include <tuple>
#include <array>
#include "maia/connectivity/iter/concepts.hpp"
#include "maia/connectivity/iter/connectivity.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"


namespace cgns {


template<
  ElementType_t elt_type,
  class face_connectivity_type,
  class tri_iterator, class quad_iterator
> auto
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  if constexpr (elt_type==TRI_3) {
    *tri_it++ = e;
  }
  else if constexpr (elt_type==QUAD_4) {
    *quad_it++ = e;
  }
  else if constexpr (elt_type==TETRA_4) {
    *tri_it++ = {e[0],e[2],e[1]};
    *tri_it++ = {e[0],e[1],e[3]};
    *tri_it++ = {e[0],e[3],e[2]};
    *tri_it++ = {e[1],e[2],e[3]};
  }
  else if constexpr (elt_type==PYRA_5) {
    *quad_it++ = {e[0],e[3],e[2],e[1]};
    * tri_it++ = {e[0],e[1],e[4]};
    * tri_it++ = {e[1],e[2],e[4]};
    * tri_it++ = {e[2],e[3],e[4]};
    * tri_it++ = {e[3],e[0],e[4]};
  }
  else if constexpr (elt_type==PENTA_6) {
    * tri_it++ = {e[0],e[2],e[1]};
    * tri_it++ = {e[3],e[4],e[5]};
    *quad_it++ = {e[0],e[1],e[4],e[3]};
    *quad_it++ = {e[0],e[3],e[5],e[2]};
    *quad_it++ = {e[1],e[2],e[5],e[4]};
  }
  else if constexpr (elt_type==HEXA_8) {
    *quad_it++ = {e[0],e[3],e[2],e[1]};
    *quad_it++ = {e[4],e[5],e[6],e[7]};
    *quad_it++ = {e[0],e[1],e[5],e[4]};
    *quad_it++ = {e[0],e[4],e[7],e[3]};
    *quad_it++ = {e[1],e[2],e[6],e[5]};
    *quad_it++ = {e[2],e[3],e[7],e[6]};
  }
  else {
    throw;
    //static_assert(0,"unsupported ElementType_t");
  }
}


} // cgns
