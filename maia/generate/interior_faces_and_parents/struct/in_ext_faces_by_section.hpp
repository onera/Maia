#pragma once


#include <vector>


namespace maia {


// Note: we don't bother encapsulating data members
//       because this type is only used as a temporary computation result


template<class I>
struct ext_faces_with_parents {
  // Class invariant:
  //   all sizes are equal
  std::vector<I> bnd_face_parent_elements;
  std::vector<I> cell_parent_elements;
  std::vector<I> cell_parent_positions;
  auto size() const -> I { return bnd_face_parent_elements.size(); }
};
template<class I>
struct in_faces_with_parents {
  // Class invariant:
  //   all sizes are equal
  std::vector<I> connec;
  std::vector<I> l_parent_elements;
  std::vector<I> r_parent_elements;
  std::vector<I> l_parent_positions;
  std::vector<I> r_parent_positions;
  auto size() const -> I { return l_parent_elements.size(); }
};


template<class I>
struct in_ext_faces {
  cgns::ElementType_t face_type;
  ext_faces_with_parents<I> ext;
  in_faces_with_parents <I> in;
};


template<class I>
using in_ext_faces_by_section = std::vector<in_ext_faces<I>>;


} // maia
