#pragma once


#include "std_e/algorithm/algorithm.hpp"
#include <algorithm>
#include <tuple>
#include <numeric>
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "maia/utils/log/log.hpp"
#include "maia/transform/renumber/permute.hpp"
#include "maia/transform/__old/renumber_point_lists.hpp" // TODO move in renumber/
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "std_e/algorithm/mismatch_points.hpp"
#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "std_e/data_structure/block_range/vblock_permutation.hpp"
#include "maia/connectivity/utils/connectivity_range.hpp"


using cgns::tree;
using cgns::ParentElements;


namespace maia {


// DOC the returned boundary/interior partition is *stable*
template<class I> auto
boundary_interior_permutation(const cgns::md_array_view<I,2>& parent_elts) -> std::pair<std::vector<I>,I> {
  I n_connec = parent_elts.extent(0);
  STD_E_ASSERT(parent_elts.extent(1)==2);

  // init
  auto perm = std_e::iota_vector(n_connec);

  // permute
  auto connectivity_is_on_boundary = [&parent_elts](I i){ return cgns::is_boundary(parent_elts,i); };
  auto partition_sub_rng = std::ranges::stable_partition(perm,connectivity_is_on_boundary);
  I partition_index = partition_sub_rng.begin() - perm.begin();

  return {perm,partition_index};
}


template<class I> auto
permute_boundary_ngons_at_beginning(tree& ngons, tree& nfaces, const std::vector<std_e::span<I>>& pls) -> I {
  STD_E_ASSERT(label(ngons)=="Elements_t");
  STD_E_ASSERT(element_type(ngons)==cgns::NGON_n);

  if (cgns::is_boundary_partitioned_element_section<I>(ngons)) return ElementSizeBoundary(ngons);

  auto parent_elts = cgns::ParentElements<I>(ngons);
  auto face_vtx = make_connectivity_range<I>(ngons);

  // compute permutation
  auto [perm,partition_index] = boundary_interior_permutation(parent_elts);

  // apply permutation
  std_e::permute_vblock_range(face_vtx,perm);
  permute_parent_elements(parent_elts,perm);
  I offset = cgns::ElementRange<I>(ngons)[0];
  auto cell_face_cs = cgns::ElementConnectivity<I>(nfaces);
  inv_permute_connectivity(cell_face_cs,perm,offset);

  // TODO bundle with inv_permute_connectivity(cell_face_cs,perm);
  auto inv_p = std_e::inverse_permutation(perm);
  std_e::offset_permutation elts_perm(offset,inv_p);
  renumber_point_lists(pls,elts_perm);

  // record number of bnd elements
  ElementSizeBoundary(ngons) = partition_index;

  return partition_index;
}


// =================
// NGons
template<class Rng, class I = typename Rng::value_type> auto
indirect_partition_by_number_of_vertices(const Rng& face_vtx_offsets) -> std::pair<std::vector<I>,I> {
  // Precondition: only tet/pyra/prism/hexa
  auto faces_n_vtx = std_e::interval_lengths(face_vtx_offsets);
  STD_E_ASSERT(std::ranges::all_of(faces_n_vtx,[](I n_vtx){ return n_vtx==3 || n_vtx==4; }));

  // partition permutation
  auto [perm,partition_indices] = std_e::indirect_partition_by_block_size(face_vtx_offsets);

  // get tri/quad split TODO: retrieve them from partition_indices
  auto last_tri = std::ranges::find_if_not(perm,[&faces_n_vtx](I i){ return faces_n_vtx[i]==3; });

  return std::make_pair(std::move(perm),last_tri-perm.begin());
}

template<class I> auto
create_permuted_ngon_connectivities(std_e::span<I> old_connectivities, std_e::span<I> old_eso, const std::vector<I>& permutation)
  -> std::vector<I>
{
  auto _ = maia_time_log("create_permuted_ngon_connectivities");

  // prepare accessors
  auto old_ngon_accessor = std_e::view_as_vblock_range(old_connectivities,old_eso);

  std::vector<I> new_connectivities(old_ngon_accessor.size());
  std::vector<I> new_eso(old_eso.size());
  auto new_ngon_accessor = std_e::view_as_vblock_range(new_connectivities,new_eso);

  // permute
  std_e::permute_copy_n(old_ngon_accessor.begin(),new_ngon_accessor.begin(),permutation.begin(),permutation.size());

  return new_connectivities;
}

template<class I> auto
apply_permutation_to_ngon(std_e::span<I> old_ngon_cs, std_e::span<I> old_eso, const std::vector<I>& permutation) -> void {
  auto new_connectivities = create_permuted_ngon_connectivities(old_ngon_cs,old_eso,permutation);
  std::ranges::copy(new_connectivities,old_ngon_cs.begin());
}

template<class I> auto
partition_bnd_faces_by_number_of_vertices(tree& ngons, tree& nfaces, const std::vector<std_e::span<I>>& pls) -> I {
  // Preconditions:
  STD_E_ASSERT(label(ngons)=="Elements_t");
  STD_E_ASSERT(element_type(ngons)==cgns::NGON_n);
  I n_bnd_faces = ElementSizeBoundary(ngons);
  auto parent_elts = ParentElements<I>(ngons);
  STD_E_ASSERT(n_bnd_faces != 0);

  auto bnd_face_vtx = make_connectivity_subrange(ngons,I(0),n_bnd_faces);

  // compute permutation
  auto [perm,last_tri_index] = indirect_partition_by_number_of_vertices(bnd_face_vtx.offsets());

  // apply permutation
  std_e::permute_vblock_range(bnd_face_vtx,perm);
  permute_parent_elements(parent_elts,perm); // TODO supposed to work with sub-arrays?

  I offset = cgns::ElementRange<I>(ngons)[0];
  auto cell_face_cs = cgns::ElementConnectivity<I>(nfaces);
  inv_permute_connectivity_sub(cell_face_cs,perm,offset); // TODO either _sub not needed, nor renumber_point_lists (below) also needs that

  // TODO bundle with inv_permute_connectivity(cell_face_cs,perm);
  auto inv_p = std_e::inverse_permutation(perm);
  std_e::offset_permutation elts_perm(offset,inv_p);
  renumber_point_lists(pls,elts_perm);

  return last_tri_index;
}

// =================
// NFaces
template<class Rng, class I = typename Rng::value_type> auto
indirect_partition_by_number_of_faces(const Rng& cell_face_offsets) -> std::tuple<std::vector<I>,I,I> {
  // Precondition: only tet/pyra/prism/hexa
  auto cell_n_faces = std_e::interval_lengths(cell_face_offsets);
  STD_E_ASSERT(std::ranges::all_of(cell_n_faces,[](I n_face){ return 4<=n_face && n_face<= 6; }));

  // partition permutation
  auto [perm,partition_indices] = std_e::indirect_partition_by_block_size(cell_face_offsets);

  // get tet/pyra-prism and pyra-prism/hex splits TODO: retrieve them from partition_indices
  auto last_tet        = std::ranges::find_if_not(perm              ,[&cell_n_faces](I i){ return cell_n_faces[i]==4; });
  auto last_pyra_prism = std::ranges::find_if_not(last_tet,end(perm),[&cell_n_faces](I i){ return cell_n_faces[i]==5; });

  return std::make_tuple( std::move(perm) , last_tet-begin(perm) , last_pyra_prism-begin(perm) );
}

template<class I> auto
partition_cells_by_number_of_faces(auto& cell_face, auto& face_cell, I first_cell_id) -> std::pair<I,I> {
  auto [perm,last_tet,last_pyra_prism] = indirect_partition_by_number_of_faces(cell_face.offsets());
  std_e::permute_vblock_range(cell_face,perm);
  inv_permute_parent_elements(face_cell,perm,first_cell_id);
  return {last_tet,last_pyra_prism};
}


template<class I> auto
is_pyra_among_pyra_prism(const auto& cell_face_connec, const auto& face_vtx, I first_face_id){
  STD_E_ASSERT(cell_face_connec.size()==5);
  int n_quad = 0;
  for (I face_id : cell_face_connec) {
    I face_idx = face_id-first_face_id;
    auto&& face = face_vtx[face_idx];
    if (face.size()==4) {
      ++n_quad;
    } else {
      STD_E_ASSERT(face.size()==3);
    }
  }
  return n_quad==1;
};

template<class I> auto
pyra_prism_permutation(auto& cell_face, const auto& face_vtx, I first_face_id) -> std::pair<std::vector<I>,I> {
  auto is_pyra = [&cell_face,&face_vtx,first_face_id](I i){ return is_pyra_among_pyra_prism(cell_face[i],face_vtx,first_face_id); };
  auto n_connec = cell_face.size();
  auto perm = std_e::iota_vector(n_connec);
  auto [partition_point,_] = std::ranges::stable_partition(perm,is_pyra);
  I partition_index = partition_point - perm.begin();

  return {perm,partition_index};
}
template<class I> auto
partition_cells_pyra_prism(auto& cell_face, auto& face_cell, I first_pyra_id, const auto& face_vtx, I first_face_id, I inf, I sup) -> I {
  auto [perm,first_prism_index] = pyra_prism_permutation(cell_face,face_vtx,first_face_id);
  std_e::permute_vblock_range(cell_face,perm);
  inv_permute_parent_elements_sub(face_cell,perm,first_pyra_id,inf,sup);
  return first_prism_index;
}

template<class I> auto
partition_cells_into_simple_types(tree& ngons, tree& nfaces) -> std::vector<I> {
  // Precondition: the cells described by nfaces are of type Tet4, Pyra5, Prism6 or Hex8

  // 0. queries
  I first_face_id = cgns::ElementRange<I>(ngons)[0];
  I first_cell_id = cgns::ElementRange<I>(nfaces)[0];

  auto face_vtx = make_connectivity_range<I>(ngons);
  auto cell_face = make_connectivity_range<I>(nfaces);
  auto face_cell = ParentElements<I>(ngons);

  // 1. apply partition // TODO also apply to PointList (see e.g. partition_bnd_faces_by_number_of_vertices)
  /// 1.0. first partition by number of faces
  auto [last_tet_index,last_pyra_prism_index] = partition_cells_by_number_of_faces(cell_face,face_cell,first_cell_id);
  /// 1.1. still need to distinguish pyra/prism
  auto pyra_prism_cell_face = make_connectivity_subrange(nfaces,last_tet_index,last_pyra_prism_index);
  auto first_pyra_id = first_cell_id+last_tet_index;

  I first_prism_among_pyra_prism_index = partition_cells_pyra_prism(pyra_prism_cell_face,face_cell,first_pyra_id,face_vtx,first_face_id,last_tet_index,last_pyra_prism_index);
  I first_prism_index = last_tet_index + first_prism_among_pyra_prism_index;
  return { 0,  last_tet_index,  first_prism_index,  last_pyra_prism_index,  cell_face.size() };
}


} // cgns
