#pragma once


#include "maia/generate/__old/ngons/from_cells/faces_heterogenous_container.hpp"
#include "maia/generate/__old/ngons/from_cells/generate_faces.hpp"
#include "std_e/utils/tuple.hpp"
#include "std_e/algorithm/sorting_networks.hpp"


namespace cgns {


// TODO test
template<class I, class CK> auto
create_face_with_sorted_connectivity(const connectivity<I,CK>& c, I parent_id) {
  constexpr int N = CK::nb_nodes;
  std::array<I,N> sorted_nodes;
  std::copy(c.begin(),c.end(),sorted_nodes.begin());
  std_e::sorting_network<N>::sort(sorted_nodes.begin());
  return face_with_sorted_connectivity<I,CK>{c,sorted_nodes,parent_id};
}


// TODO test
template<
  class Elt,
  class I
> auto
// requires Elt is Element
append_faces_with_parent_id(faces_heterogenous_container<I>& faces, Elt const& e, I e_id) {
  auto e_faces = generate_faces(e);

  if constexpr (std::tuple_size_v<decltype(e_faces)> == 1) {
    auto append_face_of_face = [&faces,e_id](auto face){ 
      faces.from_face.push_back(create_face_with_sorted_connectivity(face,e_id));
    };
    append_face_of_face(e_faces[0]);
  } else {
    auto append_face_of_vol = [&faces,e_id](auto face){ 
      faces.from_vol.push_back(create_face_with_sorted_connectivity(face,e_id));
    };
    std_e::for_each(e_faces,append_face_of_vol);
  }
}


} // cgns
