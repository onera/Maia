#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"
#include "std_e/parallel/mpi/base.hpp"
#include "std_e/parallel/all_to_all.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"
#include "std_e/algorithm/partition_sort.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/algorithm/sorting_networks.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "maia/sids/element_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/log.hpp" // TODO
#include "std_e/logging/time_logger.hpp" // TODO


using namespace cgns;


namespace maia {


constexpr auto
number_of_TRI_3_faces(ElementType_t elt_type) {
  auto& ts = face_types(elt_type);
  auto pos = std_e::find_if(begin(ts),end(ts),[](const auto& t){ return t==cgns::TRI_3; });
  auto idx = pos-begin(ts);
  return number_of_faces_by_type(elt_type)[idx];
}
constexpr auto
number_of_QUAD_4_faces(ElementType_t elt_type) {
  auto& ts = face_types(elt_type);
  auto pos = std_e::find_if(begin(ts),end(ts),[](const auto& t){ return t==cgns::QUAD_4; });
  auto idx = pos-begin(ts);
  return number_of_faces_by_type(elt_type)[idx];
}


auto
number_of_faces(const tree_range& elt_sections) {
  I8 n_tri = 0;
  I8 n_quad = 0;
  for (const tree& e : elt_sections) {
    auto elt_type = element_type(e);
    I8 n_elt = distribution_local_size(ElementDistribution(e));
    n_tri  += n_elt * number_of_TRI_3_faces (elt_type);
    n_quad += n_elt * number_of_QUAD_4_faces(elt_type);
  }
  return std::make_pair(n_tri,n_quad);
}

template<ElementType_t elt_type> auto
gen_faces(const tree& elt_node, auto& tri_it, auto& quad_it, auto& tri_parent_it, auto& quad_parent_it) {
  using I = typename std::remove_cvref_t<decltype(tri_it)>::index_type;
  I elt_start = ElementRange<I>(elt_node)[0];
  I index_dist_start = ElementDistribution(elt_node)[0];
  auto elt_connec = ElementConnectivity<I>(elt_node);

  constexpr int n_vtx = number_of_vertices(elt_type);
  auto connec_range = make_block_range<n_vtx>(elt_connec);

  I elt_id = elt_start + index_dist_start;
  for (const auto& elt : connec_range) {
    generate_faces<elt_type>(elt,tri_it,quad_it);
    tri_parent_it  = std::fill_n( tri_parent_it,number_of_TRI_3_faces (elt_type),elt_id);
    quad_parent_it = std::fill_n(quad_parent_it,number_of_QUAD_4_faces(elt_type),elt_id);
    ++elt_id;
  }
}


template<class I> auto
generate_element_faces_and_parents(const tree_range& elt_sections) -> faces_and_parents_by_section<I> {
  //auto _ = std_e::stdout_time_logger("generate_element_faces_and_parents");
  auto [n_tri,n_quad] = number_of_faces(elt_sections);
  faces_and_parents_by_section<I> faces_and_parents(n_tri,n_quad);

  auto tri_it         = faces_and_parents.tris .connectivities().begin();
  auto quad_it        = faces_and_parents.quads.connectivities().begin();
  auto tri_parent_it  = faces_and_parents.tris .parents()       .begin();
  auto quad_parent_it = faces_and_parents.quads.parents()       .begin();
  for (const auto& elt_section : elt_sections) {
    auto elt_type = element_type(elt_section);
    switch(elt_type){
      case TRI_3: {
        gen_faces<TRI_3>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case QUAD_4: {
        gen_faces<QUAD_4>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case TETRA_4: {
        gen_faces<TETRA_4>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case PENTA_6: {
        gen_faces<PENTA_6>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case PYRA_5: {
        gen_faces<PYRA_5>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case HEXA_8: {
        gen_faces<HEXA_8>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      default: {
        throw cgns_exception("Function generate_element_faces_and_parents: not implemented for element type \""+to_string(elt_type)+"\"");
      }
    }
  }
  return faces_and_parents;
}


// These template functions are declared in the .hpp, but defined in the .cpp, so they need to be explicitly instanciated
template auto generate_element_faces_and_parents<I4>(const tree_range& elt_sections) -> faces_and_parents_by_section<I4>;
template auto generate_element_faces_and_parents<I8>(const tree_range& elt_sections) -> faces_and_parents_by_section<I8>;

} // maia
