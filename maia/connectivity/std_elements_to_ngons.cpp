#include "maia/connectivity/std_elements_to_ngons.hpp"

#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "std_e/parallel/struct/distributed_array.hpp"
#include "std_e/algorithm/distribution/weighted.hpp"
#include "maia/sids/element_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/parallel/algorithm/concatenate/concatenate.hpp"
#include "std_e/future/ranges.hpp"
#include "maia/utils/log/log.hpp"
#include "maia/sids/maia_cgns.hpp"
#include "cpp_cgns/sids.hpp"
#include "maia/utils/parallel/utils.hpp"


using namespace cgns;


namespace maia {

auto
element_conversion_traits(ElementType_t from_type, ElementType_t to_type) -> std::pair<std::string,int> {
  if (to_type==cgns::NGON_n ) return {"ElementConnectivity",number_of_vertices(from_type)};
  if (to_type==cgns::NFACE_n) return {"CellFace"           ,number_of_faces   (from_type)};
  STD_E_ASSERT(0); throw std::logic_error("expects NGON_n or NFACE_n");
}

template<class I> auto
concatenate_into_poly_section(const cgns::tree_range& section_nodes, ElementType_t poly_section_type, MPI_Comm comm) {
  int n_section = section_nodes.size();
  std::vector<I> n_elts(n_section);
  std::vector<I> w_elts(n_section);
  std::vector<std_e::interval_vector<I>> connec_distri_by_section(n_section);
  std::vector<std_e::span<const I>> connec_by_section(n_section);
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = section_nodes[i];

    ElementType_t elt_type = element_type(elt_node);
    auto [connec_name,connec_weight] = element_conversion_traits(elt_type,poly_section_type);
    n_elts[i] = length(element_range(elt_node));
    w_elts[i] = connec_weight;

    connec_by_section[i] = get_node_value_by_matching<I>(elt_node,connec_name);

    auto elt_distri = ElementDistribution<I>(elt_node);
    connec_distri_by_section[i] = partial_to_full_distribution(elt_distri,comm);
    std_e::scale(connec_distri_by_section[i],w_elts[i]);
  }

  // 0. create a balanced distribution
  auto [distri_cat,connec_distri_cat] = std_e::balanced_distribution(std_e::n_rank(comm),n_elts,w_elts);

  // 1. concatenate connectivities
  std::vector<I> connec_array = std_e::concatenate_arrays(connec_by_section,connec_distri_by_section,connec_distri_cat,comm);

  // 2. compute connectivity indices (ElementStartOffset)
  int rk = std_e::rank(comm);
  auto ns = std_e::n_elements_in_interval(distri_cat.interval(rk),n_elts);
  std::vector<I> elt_start_offset(distri_cat.length(rk) + 1);
  elt_start_offset[0] = connec_distri_cat[rk];

  auto f = elt_start_offset.begin()+1;
  for (int i=0; i<n_section; ++i) {
    f = std_e::inclusive_iota_n(f,ns[i],*(f-1),w_elts[i]);
  }

  // 3. poly section node
  // connec
  I n_elt_tot = distri_cat.back();
  I first_id = elements_interval(section_nodes).first();
  tree ngon_section_node = new_Elements(to_string(poly_section_type),poly_section_type,std::move(connec_array),first_id,first_id+n_elt_tot-1);

  // elt_start_offset
  tree eso_node = new_DataArray("ElementStartOffset",std::move(elt_start_offset));
  emplace_child(ngon_section_node,std::move(eso_node));

  // distribution
  tree distri_node        = new_DataArray("Element"            ,full_to_partial_distribution(distri_cat       ,comm));
  tree distri_connec_node = new_DataArray("ElementConnectivity",full_to_partial_distribution(connec_distri_cat,comm));

  auto distri_section_node = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(distri_section_node,std::move(distri_node));
  emplace_child(distri_section_node,std::move(distri_connec_node));
  emplace_child(ngon_section_node,std::move(distri_section_node));

  return std::make_pair(std::move(distri_cat),std::move(ngon_section_node));
}

template<class I> auto
ngon_from_faces(const tree_range& face_sections, MPI_Comm comm) -> tree {
  // 0. concatenate connectivities
  auto [distri_cat,ngon_section_node] = concatenate_into_poly_section<I>(face_sections,cgns::NGON_n,comm);

  // 1. concatenate parents
  /// 1.0. query tree
  int n_section = face_sections.size();
  std::vector<std_e::span<const I>> l_pe_by_section(n_section);
  std::vector<std_e::span<const I>> r_pe_by_section(n_section);
  std::vector<std_e::span<const I>> l_pp_by_section(n_section);
  std::vector<std_e::span<const I>> r_pp_by_section(n_section);
  std::vector<std_e::interval_vector<I>> distri_by_section(n_section);
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = face_sections[i];
    auto pe = ParentElements<I>(elt_node);
    auto pp = ParentElementsPosition<I>(elt_node);
    l_pe_by_section[i] = column(pe,0);
    r_pe_by_section[i] = column(pe,1);
    l_pp_by_section[i] = column(pp,0);
    r_pp_by_section[i] = column(pp,1);

    auto elt_distri = ElementDistribution<I>(elt_node);
    distri_by_section[i] = partial_to_full_distribution(elt_distri,comm);
  }
  /// 1.2. concatenate
  std::vector<I> l_pe_cat = std_e::concatenate_arrays(l_pe_by_section,distri_by_section,distri_cat,comm);
  std::vector<I> r_pe_cat = std_e::concatenate_arrays(r_pe_by_section,distri_by_section,distri_cat,comm);
  std::vector<I> l_pp_cat = std_e::concatenate_arrays(l_pp_by_section,distri_by_section,distri_cat,comm);
  std::vector<I> r_pp_cat = std_e::concatenate_arrays(r_pp_by_section,distri_by_section,distri_cat,comm);

  I8 parent_cat_sz = l_pe_cat.size();
  /// 1.3. parent elements
  cgns::md_array<I,2> pe_array(parent_cat_sz,2);
  auto [_0,mid_pe_array] = std::ranges::copy(l_pe_cat,begin(pe_array));
                           std::ranges::copy(r_pe_cat,mid_pe_array);
  tree pe_node = cgns::new_DataArray("ParentElements",std::move(pe_array));
  emplace_child(ngon_section_node,std::move(pe_node));

  /// 1.4. parent positions
  cgns::md_array<I,2> pp_array(parent_cat_sz,2);
  auto [_1,mid_pp_array] = std::ranges::copy(l_pp_cat,begin(pp_array));
                           std::ranges::copy(r_pp_cat,mid_pp_array);
  tree pp_node = cgns::new_DataArray("ParentElementsPosition",std::move(pp_array));
  emplace_child(ngon_section_node,std::move(pp_node));

  #if defined REAL_GCC && __GNUC__ >= 11
    return ngon_section_node;
  #else // It seems like GCC 10 uses the copy-ctor instead of mandatory RVO
    return std::move(ngon_section_node);
  #endif
}

template<class I> auto
nface_from_cells(const tree_range& cell_sections, MPI_Comm comm) -> tree {
  auto [_,nfaces_section_node] = concatenate_into_poly_section<I>(cell_sections,cgns::NFACE_n,comm);
  #if defined REAL_GCC && __GNUC__ >= 11
    return nfaces_section_node;
  #else // It seems like GCC 10 uses the copy-ctor instead of mandatory RVO
    return std::move(nfaces_section_node);
  #endif
}

template<class I> auto
_turn_into_ngon_nface(tree& z, MPI_Comm comm) -> void {
  auto elt_sections = element_sections(z);
  auto elt_2D_3D_names = elt_sections
                       | std::views::filter([](const tree& x){ return is_section_of_dimension(x,2) || is_section_of_dimension(x,3); })
                       | std::views::transform([](const tree& x){ return name(x); })
                       | std_e::to_vector();
  emplace_child(z, ngon_from_faces <I>(surface_element_sections(z),comm));
  emplace_child(z, nface_from_cells<I>(volume_element_sections (z),comm));
  cgns::rm_children_by_names(z,elt_2D_3D_names);
}

auto
turn_into_ngon_nface(tree& z, MPI_Comm comm) -> void {
  if (value(z).data_type()=="I4") return _turn_into_ngon_nface<I4>(z,comm);
  if (value(z).data_type()=="I8") return _turn_into_ngon_nface<I8>(z,comm);
  throw cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}

auto
std_elements_to_ngons(tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_compliant_zone(z));
  auto _ = maia_perf_log_lvl_0("std_elements_to_ngons");

  generate_interior_faces_and_parents(z,comm);
  turn_into_ngon_nface(z,comm);
}


} // maia
