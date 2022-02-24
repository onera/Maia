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
#include "std_e/data_structure/multi_range.hpp"
#include "std_e/utils/concatenate.hpp"
#include "maia/sids/element_sections.hpp"
#include "maia/transform/renumber/shift.hpp"

using namespace cgns;

namespace maia {

auto
ngon_new_to_old(tree& b) -> void {
  for (tree& z : get_nodes_by_label(b,"Zone_t")) {
    auto elt_sections = element_sections(z);
    auto face_sections = surface_element_sections(z);

    // 1. transform ngon/nface indexed->interleaved
    for (tree& elt : elt_sections) {
      if (element_type(elt)==NGON_n || element_type(elt)==NFACE_n) {
        auto eso = ElementStartOffset<I4>(elt); // TODO I4/I8
        auto connectivity = ElementConnectivity<I4>(elt);
        auto new_ngon_range = polygon_range(eso,connectivity);

        I4 old_connectivity_sz = eso.size()-1 + eso.back();
        std::vector<I4> old_connectivity(old_connectivity_sz);
        auto old_ngon_range = cgns::interleaved_ngon_range(old_connectivity);

        std::copy(new_ngon_range.begin(),new_ngon_range.end(),old_ngon_range.begin());

        rm_child_by_name(elt,"ElementConnectivity");
        rm_child_by_name(elt,"ElementStartOffset");

        emplace_child(elt,new_DataArray("ElementConnectivity",std::move(old_connectivity)));
      }
    }

    // 2. shift parent element to start at 1
    I4 n_faces = length(elements_interval(face_sections));
    shift_parent_elements(face_sections,-n_faces);
  }
}

} // maia
