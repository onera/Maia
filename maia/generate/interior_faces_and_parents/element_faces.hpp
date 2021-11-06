#pragma once

#include <tuple>
#include <array>
#include "maia/connectivity/iter/concepts.hpp"
#include "maia/connectivity/iter/connectivity.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"


namespace cgns {


template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==TRI_3,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator&) -> void {
  *tri_it++ = e;
}
template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==QUAD_4,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator&, quad_iterator& quad_it) -> void {
  *quad_it++ = e;
}


template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==TETRA_4,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator&) -> void {
  *tri_it++ = {e[0],e[2],e[1]};
  *tri_it++ = {e[0],e[1],e[3]};
  *tri_it++ = {e[0],e[3],e[2]};
  *tri_it++ = {e[1],e[2],e[3]};
}

template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==PYRA_5,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  *quad_it++ = {e[0],e[3],e[2],e[1]};
  * tri_it++ = {e[0],e[1],e[4]};
  * tri_it++ = {e[1],e[2],e[4]};
  * tri_it++ = {e[2],e[3],e[4]};
  * tri_it++ = {e[3],e[0],e[4]};
}

template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==PENTA_6,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  * tri_it++ = {e[0],e[2],e[1]};
  * tri_it++ = {e[3],e[4],e[5]};
  *quad_it++ = {e[0],e[1],e[4],e[3]};
  *quad_it++ = {e[0],e[3],e[5],e[2]};
  *quad_it++ = {e[1],e[2],e[5],e[4]};
}

template<
  class face_connectivity_type,
  class tri_iterator, class quad_iterator,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==HEXA_8,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e, tri_iterator& tri_it, quad_iterator& quad_it) -> void {
  *quad_it++ = {e[0],e[3],e[2],e[1]};
  *quad_it++ = {e[4],e[5],e[6],e[7]};
  *quad_it++ = {e[0],e[1],e[5],e[4]};
  *quad_it++ = {e[0],e[4],e[7],e[3]};
  *quad_it++ = {e[1],e[2],e[6],e[5]};
  *quad_it++ = {e[2],e[3],e[7],e[6]};
}


} // cgns
