#include "maia/generate/__old/ngons/from_cells/generate_ngons_from_elts.hpp"

#include "maia/utils/log/log.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/creation.hpp"


namespace cgns {


auto
generate_ngons_from_elts(tree& b, factory& F) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  for (tree& z : get_children_by_label(b,"Zone_t")) {
    if (is_unstructured_zone(z)) {
      auto _ = maia_time_log("Generation of ngons of zone");
      auto ngon = generate_ngons_from_elts(get_children_by_label(z,"Elements_t"),F);
      emplace_child(z,std::move(ngon));
    }
  }
}

auto
generate_ngons_from_elts(const tree_range& elt_pools, factory& F) -> tree {
  using I = int32_t; // TODO std::int32_t -> template I
  faces_heterogenous_container<I> all_faces;
  
  // TODO correct bug of all_faces_unique being passed by value if "cgns::visit( LIFT(append_faces) )" 
  for (const tree& elt_pool : elt_pools) {
    auto connectivities = ElementConnectivity<I>(elt_pool);
    I first_elt_id = ElementRange<I>(elt_pool)[0];
    ElementType_t elt_type = static_cast<ElementType_t>(ElementType<I>(elt_pool));
    append_faces(elt_type,all_faces,connectivities,first_elt_id);

  }
  I first_ngon = 1; // TODO

  faces_container<I> all_faces_unique;
  append_boundary_and_interior_faces<I>(all_faces_unique,all_faces);
  return create_ngon(all_faces_unique,first_ngon,F);
}


} // cgns
