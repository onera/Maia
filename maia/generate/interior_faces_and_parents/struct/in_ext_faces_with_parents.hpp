#pragma once


#include <vector>


namespace maia {


// faces_and_parents_by_section {
// TODO class because invariant: size is same for all (with connec by block)
// TODO add first_id
template<class I>
struct in_faces_with_parents {
  std::vector<I> connec;
  std::vector<I> l_parents;
  std::vector<I> r_parents;
  std::vector<I> l_parent_positions;
  std::vector<I> r_parent_positions;
  auto size() const -> I { return l_parents.size(); }
};
// TODO class because invariant: size is same for all (with connec by block)
// TODO add first_id
template<class I>
struct ext_faces_with_parents {
  std::vector<I> boundary_parents;
  std::vector<I> vol_parents;
  std::vector<I> vol_parent_positions;
  auto size() const -> I { return boundary_parents.size(); }
};

template<class I>
struct in_ext_faces_with_parents {
  in_faces_with_parents <I> in;
  ext_faces_with_parents<I> ext;
};


} // maia
