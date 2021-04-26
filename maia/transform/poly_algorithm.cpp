#include "maia/transform/poly_algorithm.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include <algorithm>
#include <vector>
#include "cpp_cgns/sids/utils.hpp"
#include "maia/connectivity/iter_cgns/range.hpp"
#include "maia/partitioning/gc_name_convention.hpp"
#include "maia/sids/element_sections.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/buffer/buffer_vector.hpp"
#include "std_e/data_structure/multi_range.hpp"
#include "std_e/utils/concatenate.hpp"

using namespace cgns;

namespace maia {

auto
ngon_new_to_old(tree& b) -> void {
  for (tree& elt : get_nodes_by_matching(b,"Zone_t/Elements_t")) {
    if (element_type(elt)==NGON_n || element_type(elt)==NFACE_n) {
      auto eso = ElementStartOffset<I4>(elt); // TODO I4/I8
      auto connectivity = ElementConnectivity<I4>(elt);
      auto new_ngon_range = polygon_range(eso,connectivity);

      I4 old_connectivity_sz = eso.size()-1 + eso.back();
      std_e::buffer_vector<I4> old_connectivity(old_connectivity_sz);
      auto old_ngon_range = cgns::interleaved_ngon_range(old_connectivity);

      std::copy(new_ngon_range.begin(),new_ngon_range.end(),old_ngon_range.begin());
      auto sp = std_e::make_span(old_connectivity.data(),30);

      rm_child_by_name(elt,"ElementConnectivity");
      rm_child_by_name(elt,"ElementStartOffset");

      emplace_child(elt,new_DataArray("ElementConnectivity",std::move(old_connectivity)));
    }
  }
}

auto
sids_conforming_ngon_nface(tree& b) -> void {
  // Del bnd elts, in ngon PE replace bnd by 0
  // Ngon ElementRange at 1, shift PL
  // Make PE begin at n_ngon
  // NFace: del faces, ElementRange
  // TODO manage GC with apply_base_renumbering
  for (tree& z : get_children_by_label(b,"Zone_t")) {
    auto elt_sections = element_sections_ordered_by_range(z);
    int n_section = elt_sections.size();
    STD_E_ASSERT(n_section>=2); // at least poly2d and poly3d

    tree& poly2d = elt_sections[n_section-2];
    tree& poly3d = elt_sections[n_section-1];
    STD_E_ASSERT(element_type(poly2d)==NGON_n);
    STD_E_ASSERT(element_type(poly3d)==NFACE_n);

    auto std_elt_sections = std_e::make_span(elt_sections.data(),n_section-2);
    auto is_2D_section = [](const tree& elt_section){ return element_dimension(element_type(elt_section))==2; };
    STD_E_ASSERT(std::is_partitioned(begin(std_elt_sections),end(std_elt_sections),is_2D_section));

    auto first_3D_section = std::partition_point(begin(std_elt_sections),end(std_elt_sections),is_2D_section);
    STD_E_ASSERT(first_3D_section != end(std_elt_sections)); // at least one 3D element section
    auto id_first_3D_elt = ElementRange<I4>(*first_3D_section)[0];
    auto n_face = id_first_3D_elt-1;

    // 0. shift poly2d to begining
    // 0.0 shift
    auto poly2d_range = ElementRange<I8>(poly2d);
    auto offset = -poly2d_range[0]+1;
    auto n_poly2d = poly2d_range[1] - poly2d_range[0] + 1;
    poly2d_range[0] = 1;
    poly2d_range[1] = n_poly2d;
    // 0.1 report to all FaceCenter PL
    auto indexed_nodes = get_nodes_by_labels(z,{"BC_t","BCDataSet_t","ZoneGridConnectivity_t","ZoneSubRegion_t"});
    for (tree& node : indexed_nodes) {
      if (has_child_of_name(node,"GridLocation") && GridLocation(node)=="FaceCenter" && has_child_of_name(node,"PointList")) {
        auto pl = PointList<I4>(node);
        for (I4& id : pl) {
          id += offset;
        }
      }
    }

    // 1. Correct PE
    auto parent_elts = ParentElements<I4>(poly2d);
    for (I4& pe : parent_elts) {
      // 1.0. replace boundary parent elements by 0
      if (pe < id_first_3D_elt) {
        pe = 0;
      } else { // 1.1. make PE refer to NFace ids
        pe += n_poly2d; // n_poly2d is the shift
      }
    }

    // 2. poly3d
    // 2.1 range
    auto poly3d_range = ElementRange<I8>(poly3d);
    auto n_poly3d = poly3d_range[1] - poly3d_range[0] + 1 - n_face;
    poly3d_range[0] = n_poly2d+1;
    poly3d_range[1] = n_poly2d+n_poly3d;

    // 2.0 remove faces at the beginning
    auto poly3d_eso_old = ElementStartOffset<I4>(poly3d);
    std_e::buffer_vector<I4> poly3d_eso_new(poly3d_eso_old.begin()+n_face,poly3d_eso_old.end());
    std::for_each(begin(poly3d_eso_new),end(poly3d_eso_new),std_e::offset(-poly3d_eso_new[0]));

    auto poly3d_eco_old = ElementConnectivity<I4>(poly3d);
    std_e::buffer_vector<I4> poly3d_eco_new(poly3d_eco_old.begin()+n_face,poly3d_eco_old.end());

    rm_child_by_name(poly3d,"ElementStartOffset");
    rm_child_by_name(poly3d,"ElementConnectivity");
    emplace_child(poly3d,new_DataArray("ElementStartOffset",std::move(poly3d_eso_new)));
    emplace_child(poly3d,new_DataArray("ElementConnectivity",std::move(poly3d_eco_new)));

    // 3. delete std elts
    rm_children(z,std_elt_sections);
  }
}

} // maia
