#pragma once

#include <tuple>
#include <array>
#include "maia/connectivity/iter/concepts.hpp"
#include "maia/connectivity/iter/connectivity.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"


namespace cgns {


template<
  class face_connectivity_type,
  class I = typename face_connectivity_type::index_type,
  int N = face_connectivity_type::nb_nodes,
  int connec_type = face_connectivity_type::elt_t,
  std::enable_if_t<connec_type==TRI_3 || connec_type==QUAD_4,int> =0
> auto
// requires face_connectivity_type is Element
generate_faces(const face_connectivity_type& e) {
  using kind = typename face_connectivity_type::kind;
  connectivity<I,kind> c;
  // TODO no copy
  for (int i=0; i<N; ++i) {
      c[i] = e[i];
  }
  return std::array<connectivity<I,kind>,1>{c};
}


template<
  class cell_connectivity_type,
  class I = typename cell_connectivity_type::index_type,
  std::enable_if_t<cell_connectivity_type::elt_t==TETRA_4,int> =0
> auto
// requires cell_connectivity_type is Element
generate_faces(cell_connectivity_type const& e)
    -> std::array<tri_3<I>,4>
{
  tri_3<I>  face0 = {e[0],e[2],e[1]};
  tri_3<I>  face1 = {e[0],e[1],e[3]};
  tri_3<I>  face2 = {e[0],e[3],e[2]};
  tri_3<I>  face3 = {e[1],e[2],e[3]};

  return {face0,face1,face2,face3};
}


template<
  class cell_connectivity_type,
  class I = typename cell_connectivity_type::index_type,
  std::enable_if_t<cell_connectivity_type::elt_t==PYRA_5,int> =0
> auto
// requires cell_connectivity_type is Element
generate_faces(cell_connectivity_type const& e) {
  quad_4<I> face0 = {e[0],e[3],e[2],e[1]};
  tri_3<I>  face1 = {e[0],e[1],e[4]}; 
  tri_3<I>  face2 = {e[1],e[2],e[4]}; 
  tri_3<I>  face3 = {e[2],e[3],e[4]}; 
  tri_3<I>  face4 = {e[3],e[0],e[4]}; 

  return std::make_tuple(face0,face1,face2,face3,face4);
}


template<
  class cell_connectivity_type,
  class I = typename cell_connectivity_type::index_type,
  std::enable_if_t<cell_connectivity_type::elt_t==PENTA_6,int> =0
> auto
// requires cell_connectivity_type is Element
generate_faces(cell_connectivity_type const& e) {
  tri_3<I>  face0 = {e[0],e[2],e[1]}; 
  tri_3<I>  face1 = {e[3],e[4],e[5]}; 
  quad_4<I> face2 = {e[0],e[1],e[4],e[3]};
  quad_4<I> face3 = {e[0],e[3],e[5],e[2]};
  quad_4<I> face4 = {e[1],e[2],e[5],e[4]};

  return std::make_tuple(face0,face1,face2,face3,face4);
}


template<
  class cell_connectivity_type,
  class I = typename cell_connectivity_type::index_type,
  std::enable_if_t<cell_connectivity_type::elt_t==HEXA_8,int> =0
> auto
// requires cell_connectivity_type is Element
generate_faces(cell_connectivity_type const& e)
    -> std::array<quad_4<I>,6>
{
  quad_4<I> face0 = {e[0],e[3],e[2],e[1]};
  quad_4<I> face1 = {e[4],e[5],e[6],e[7]};
  quad_4<I> face2 = {e[0],e[1],e[5],e[4]};
  quad_4<I> face3 = {e[0],e[4],e[7],e[3]};
  quad_4<I> face4 = {e[1],e[2],e[6],e[5]};
  quad_4<I> face5 = {e[2],e[3],e[7],e[6]};

  return {face0,face1,face2,face3,face4,face5};
}


} // cgns 
