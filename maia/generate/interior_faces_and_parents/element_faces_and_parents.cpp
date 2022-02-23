#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces.hpp"
#include "maia/generate/interior_faces_and_parents/struct/faces_and_parents_by_section.hpp"
//#include "std_e/logging/time_logger.hpp"


using namespace cgns;


namespace maia {


auto
number_of_faces(const tree_range& elt_sections) -> std::array<I8,n_face_type> {
  std::array<I8,n_face_type> n_faces_by_type;
  std::ranges::fill(n_faces_by_type,0);

  for (const tree& e : elt_sections) {
    auto elt_type = element_type(e);
    I8 n_elt = distribution_local_size(ElementDistribution(e));
    for (int i=0; i<n_face_type; ++i) {
      auto face_type = all_face_types[i];
      n_faces_by_type[i] += n_elt * number_of_faces(elt_type,face_type);
    }
  }
  return n_faces_by_type;
}

template<ElementType_t elt_type> auto
gen_faces(
  const tree& elt_node,
  auto& tri_it, auto& quad_it,
  auto& tri_parent_it, auto& quad_parent_it,
  auto& tri_ppos_it, auto& quad_ppos_it
)
{
  using I = typename std::remove_cvref_t<decltype(tri_it)>::index_type;

  constexpr int n_vtx = number_of_vertices(elt_type);
  auto elt_connec = ElementConnectivity<I>(elt_node);
  auto connec_range = view_as_block_range<n_vtx>(elt_connec);

  I elt_start = ElementRange<I>(elt_node)[0];
  I index_dist_start = ElementDistribution(elt_node)[0];
  I elt_id = elt_start + index_dist_start;

  for (const auto& elt : connec_range) {
    generate_faces<elt_type>(elt,tri_it,quad_it);
    generate_parent_positions<elt_type>(tri_ppos_it,quad_ppos_it);
    tri_parent_it  = std::fill_n( tri_parent_it,number_of_faces(elt_type,TRI_3 ),elt_id);
    quad_parent_it = std::fill_n(quad_parent_it,number_of_faces(elt_type,QUAD_4),elt_id);
    ++elt_id;
  }
}


template<class I> auto
generate_element_faces_and_parents(const tree_range& elt_sections) -> faces_and_parents_by_section<I> {
  //auto _ = std_e::stdout_time_logger("generate_element_faces_and_parents");

  auto n_faces_by_type = number_of_faces(elt_sections);
  connectivities_with_parents<I> tris (TRI_3 ,cgns::at_face_type(n_faces_by_type,TRI_3 ));
  connectivities_with_parents<I> quads(QUAD_4,cgns::at_face_type(n_faces_by_type,QUAD_4));

  auto tri_it         = connectivities<3>(tris ).begin();
  auto quad_it        = connectivities<4>(quads).begin();
  auto tri_parent_it  = parent_elements  (tris ).begin();
  auto quad_parent_it = parent_elements  (quads).begin();
  auto tri_ppos_it    = parent_positions (tris ).begin();
  auto quad_ppos_it   = parent_positions (quads).begin();
  for (const auto& elt_section : elt_sections) {
    auto elt_type = element_type(elt_section);
    switch(elt_type){
      case TRI_3: {
        gen_faces<TRI_3  >(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      case QUAD_4: {
        gen_faces<QUAD_4 >(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      case TETRA_4: {
        gen_faces<TETRA_4>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      case PENTA_6: {
        gen_faces<PENTA_6>(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      case PYRA_5: {
        gen_faces<PYRA_5 >(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      case HEXA_8: {
        gen_faces<HEXA_8 >(elt_section,tri_it,quad_it,tri_parent_it,quad_parent_it,tri_ppos_it,quad_ppos_it);
        break;
      }
      default: {
        throw cgns_exception(std::string("Function \"")+__func__+"\": not implemented for element type \""+to_string(elt_type)+"\"");
      }
    }
  }
  return {std::move(tris),std::move(quads)};
}


// Explicit instanciations of functions defined in this .cpp file
template auto generate_element_faces_and_parents<I4>(const tree_range& elt_sections) -> faces_and_parents_by_section<I4>;
template auto generate_element_faces_and_parents<I8>(const tree_range& elt_sections) -> faces_and_parents_by_section<I8>;

} // maia
