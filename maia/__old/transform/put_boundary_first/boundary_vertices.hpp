#pragma once


#include "cpp_cgns/cgns.hpp"


namespace maia {


template<class Rng, class PE_array> auto
find_boundary_vertices(const Rng& cs, const PE_array& pe) {
  STD_E_ASSERT(size_t(cs.size()) == size_t(pe.extent(0)));
  STD_E_ASSERT(pe.extent(1) == 2);
  using I = typename PE_array::value_type;
  std::vector<I> bnd_vertices;

  I n = cs.size();
  for (I i=0; i<n; ++i) {
    if (cgns::is_boundary(pe,i)) {
      for (I vtx : cs[i]) {
        bnd_vertices.push_back(vtx);
      }
    }
  }

  return bnd_vertices;
}


template<class I> auto get_ordered_boundary_vertex_ids(const cgns::tree_range& element_sections) -> std::vector<I>;


} // maia
