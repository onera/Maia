#pragma once


#include "std_e/algorithm/algorithm.hpp"
#include <algorithm>
#include <tuple>
#include <numeric>
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "maia/utils/log/log.hpp"
#include "maia/connectivity/iter_cgns/range.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "std_e/algorithm/mismatch_points.hpp"


namespace cgns {


// DOC the returned boundary/interior partition is *stable*
template<class I> auto
boundary_interior_permutation(const md_array_view<I,2>& parent_elts) -> std::pair<std::vector<I>,I> {
  I nb_connec = parent_elts.extent(0);
  STD_E_ASSERT(parent_elts.extent(1)==2);

  // init
  std::vector<I> permutation(nb_connec);
  std::iota(begin(permutation),end(permutation),0);

  // permute
  auto connectivity_is_on_boundary = [&parent_elts](I i){ return is_boundary(parent_elts,i); };
  auto partition_ptr = std::stable_partition(permutation.begin(),permutation.end(),connectivity_is_on_boundary);
  I partition_index = partition_ptr - permutation.begin();

  return {permutation,partition_index};
}




template<class I> auto
create_partitionned_ngon_connectivities(std_e::span<I> old_connectivities, const std::vector<I>& permutation, I partition_index)
  -> std::pair<std::vector<I>,I> 
{
  auto _ = maia_time_log("create_partitionned_ngon_connectivities");

  // prepare accessors
  auto old_ngon_accessor = cgns::interleaved_ngon_random_access_range(old_connectivities);

  std::vector<I> new_connectivities(old_ngon_accessor.memory_length());
  auto new_ngon_accessor = cgns::interleaved_ngon_range(new_connectivities);

  // prepare partition segments
  auto permutation_first = permutation.begin();
  auto permutation_partition_point = permutation.begin() + partition_index;
  auto first_part_size = partition_index;
  auto second_part_size = permutation.size() - partition_index;

  // apply permutation copy at beginning
  auto new_ngon_partition_point_it =
  std_e::permute_copy_n(old_ngon_accessor.begin(),new_ngon_accessor.begin(),permutation_first          ,first_part_size);

  // record partition point in ngon
  I ngon_partition_index = new_ngon_partition_point_it.data() - new_ngon_accessor.begin().data();

  // finish permutation copy
  std_e::permute_copy_n(old_ngon_accessor.begin(),new_ngon_partition_point_it,permutation_partition_point,second_part_size);

  return {new_connectivities,ngon_partition_index};
}

template<class I> auto
apply_partition_to_ngons(std_e::span<I> old_ngon_cs, const std::vector<I>& permutation, I partition_index) -> I {
  auto [new_connectivities,ngon_partition_index]  = create_partitionned_ngon_connectivities(old_ngon_cs,permutation,partition_index);
  std::copy(new_connectivities.begin(),new_connectivities.end(),old_ngon_cs.begin());

  return ngon_partition_index;
}




template<class I> auto
apply_ngon_permutation_to_parent_elts(md_array_view<I,2>& parent_elts, const std::vector<I>& permutation) -> void {
  std_e::permute(column(parent_elts,0).begin(),permutation);
  std_e::permute(column(parent_elts,1).begin(),permutation);
}
template<class I> auto
apply_nface_permutation_to_parent_elts(md_array_view<I,2>& parent_elts, const std::vector<I>& permutation) -> void {
  auto p = std_e::inverse_permutation(permutation);
  for (I4& cell_id : parent_elts) {
    cell_id = p[cell_id-1] + 1; // TODO 1 -> start
  }
}




inline auto
mark_as_boundary_partitionned(tree& ngons, I8 partition_index, I8 ngon_partition_index, factory F) -> void {
  ElementSizeBoundary<I4>(ngons) = partition_index;

  node_value part_idx_val = create_node_value_1d({ngon_partition_index},F.alloc());
  tree pt_node = F.newUserDefinedData(".#PartitionIndex",part_idx_val);

  emplace_child(ngons,std::move(pt_node));
}

template<class T> inline auto
mark_polygon_groups(const std::string& bnd_or_interior, T& ngon_accessor, const I4* global_start, factory F) -> std::vector<tree> {
  auto polygon_types = make_cgns_vector<I4>(F.alloc());
  auto polygon_type_starts = make_cgns_vector<I4>(F.alloc());

  auto equal_nb_vertices = [](const auto& conn0, const auto& conn1){ return conn0.nb_nodes()==conn1.nb_nodes(); };
  auto record_type_and_start = [&polygon_types,&polygon_type_starts,global_start](const auto& conn_it){
    polygon_types.push_back(conn_it->nb_nodes());
    polygon_type_starts.push_back(conn_it.data()-global_start);
  };
  std_e::for_each_mismatch(ngon_accessor.begin(),ngon_accessor.end(),equal_nb_vertices,record_type_and_start);
  polygon_type_starts.push_back(ngon_accessor.data()+ngon_accessor.memory_length()-global_start);

  node_value polygon_type_val = view_as_node_value(polygon_types);
  node_value polygon_type_start_val = view_as_node_value(polygon_type_starts);
  return {
    F.newUserDefinedData(".#PolygonType"+bnd_or_interior,polygon_type_val),
    F.newUserDefinedData(".#PolygonTypeStart"+bnd_or_interior,polygon_type_start_val)
  };
}
inline auto
mark_polygon_groups(tree& ngons, factory F) -> void {
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto ngon_start = ngon_connectivity.data();
  auto ngon_size = ngon_connectivity.size();
  I4 bnd_finish_idx = view_as_span<I8>(get_child_by_name(ngons,".#PartitionIndex").value)[0];

  auto ngon_bnd_connectivity = std_e::make_span(ngon_start,bnd_finish_idx);
  auto ngon_iterior_connectivity = std_e::make_span(ngon_start+bnd_finish_idx,ngon_start+ngon_size);

  auto ngon_bnd_accessor = cgns::interleaved_ngon_range(ngon_bnd_connectivity);
  auto ngon_interior_accessor = cgns::interleaved_ngon_range(ngon_iterior_connectivity);

  emplace_children(ngons,mark_polygon_groups("Boundary",ngon_bnd_accessor,ngon_start,F));
  emplace_children(ngons,mark_polygon_groups("Interior",ngon_interior_accessor,ngon_start,F));
}
//auto
//mark_polyhedron_groups(tree& nfaces, factory F) -> void {
  //auto ngon_connectivity = ElementConnectivity<I4>(nfaces);
//  auto conn_start = ngon_connectivity.data();
//  auto nface_accessor = cgns::interleaved_nface_range(ngon_connectivity);
//
//  auto polyhedron_type = make_cgns_vector<I4>(F.alloc());
//  auto polyhedron_type_start = make_cgns_vector<I4>(F.alloc());
//
//  auto equal_nb_faces = [](const auto& conn0, const auto& conn1){ return conn0.size()==conn1.size(); };
//  auto record_type_and_start = [&polyhedron_type,&polyhedron_type_start,conn_start](const auto& conn_it){
//    polyhedron_type.push_back(conn_it->nb_nodes());
//    polyhedron_type_start.push_back(conn_it.data()-conn_start);
//  };
//  std_e::for_each_mismatch(ngon_accessor.begin(),ngon_accessor.end(),equal_nb_vertices,record_type_and_start);
//
//  node_value polygon_type_val = view_as_node_value(polygon_type);
//  node_value polygon_type_start_val = view_as_node_value(polygon_type_start);
//  tree pt_node = F.newUserDefinedData(".#PolygonType",polygon_type_val);
//  tree pts_node = F.newUserDefinedData(".#PolygonTypeStart",polygon_type_start_val);
//  emplace_child(nfaces,std::move(pt_node));
//  emplace_child(nfaces,std::move(pts_node));
//}
inline auto
mark_simple_polyhedron_groups(tree& nfaces, const tree& ngons, I4 penta_start, factory F) -> void {
  // Precondition: nfaces is sorted with tet,pyra,penta,hex; with no other polyhedron type
  auto nface_connectivity = ElementConnectivity<I4>(nfaces);
  auto nface_accessor = cgns::interleaved_nface_random_access_range(nface_connectivity);

  auto last_tet = std::find_if_not(nface_accessor.begin(),nface_accessor.end(),[](const auto& c){ return c.size()==4; });
  auto last_penta = std::find_if_not(last_tet,nface_accessor.end(),[](const auto& c){ return c.size()==5; });
  auto polyhedron_type_starts = make_cgns_vector<I4>(5,F.alloc()); // tet, pyra, penta, hex, end
  polyhedron_type_starts[0] = 0; // tets start at 0
  polyhedron_type_starts[1] = last_tet.data()-nface_accessor.data();
  polyhedron_type_starts[2] = penta_start; // penta start
  polyhedron_type_starts[3] = last_penta.data()-nface_accessor.data();
  polyhedron_type_starts[4] = nface_connectivity.size();

  //// CHECK Hex
  //auto first_ngon_id = ElementRange<I4>(ngons)[0];
  //auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  //auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);
  //auto hexa_range = std_e::make_span(nface_connectivity.data()+polyhedron_type_starts[3],nface_connectivity.data()+polyhedron_type_starts[4]);
  //auto hexa_accessor = cgns::interleaved_nface_random_access_range(hexa_range);
  //for (const auto& hexa : hexa_accessor) {
  //  bool hex = true;
  //  for (int i=0; i<6; ++i) {
  //    I4 face_idx = hexa[i]-first_ngon_id;
  //    auto face = ngon_accessor[face_idx];
  //    if (face.size()!=4) {
  //      std::cout << "face.size() = " << face.size() << "\n";
  //      hex = false;
  //    }
  //    //STD_E_ASSERT(face.size()==4);
  //  }
  //  STD_E_ASSERT(hex);
  //}
  //// end CHECK

  node_value polyhedron_type_start_val = view_as_node_value(polyhedron_type_starts);
  tree pts_node = F.newUserDefinedData(".#PolygonSimpleTypeStart",polyhedron_type_start_val);
  tree desc = F.newDescriptor("Node info","The .#PolygonSimpleTypeStart node is present for an Elements_t of NFACE_n type if the polyhedrons are only simple, linear ones and sorted with Tets firsts, then Pyras, then Pentas then Hexes. The .#PolygonSimpleTypeStart values are the starting indices for each of these elements, in respective order (the last number is the size of the NFACE connectivity)");
  emplace_child(pts_node,std::move(desc));
  emplace_child(nfaces,std::move(pts_node));
}

inline auto
permute_boundary_ngons_at_beginning(tree& ngons, factory F) -> std::vector<I4> { // TODO find a way to do it for I4 and I8
  // Precondition: ngons.type = "Elements_t" and elements of type NGON_n
  auto parent_elts = ParentElements<I4>(ngons);
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);

  // compute permutation
  auto [permutation,partition_index] = boundary_interior_permutation(parent_elts);

  // apply permutation
  auto ngon_partition_index = apply_partition_to_ngons(ngon_connectivity,permutation,partition_index);
  apply_ngon_permutation_to_parent_elts(parent_elts,permutation);

  mark_as_boundary_partitionned(ngons,partition_index,ngon_partition_index,F);

  return permutation;
}


// =================
// TODO factor with above, test
template<class I> auto
sorting_by_nb_vertices_permutation(std_e::span<I> connectivities) -> std::vector<I> {
  auto _ = maia_time_log("sorting_by_nb_vertices_permutation");
  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(connectivities);

  // init
  auto nb_connec = ngon_accessor.size();
  std::vector<I> permutation(nb_connec);
  std::iota(begin(permutation),end(permutation),0);

  // permute
  auto comp_by_nb_vertices = [&ngon_accessor](I i, I j){ return ngon_accessor[i].size()<ngon_accessor[j].size(); };
  std::stable_sort(permutation.begin(),permutation.end(),comp_by_nb_vertices);

  return permutation;
}

template<class I> auto
create_permuted_ngon_connectivities(std_e::span<I> old_connectivities, const std::vector<I>& permutation)
  -> std::vector<I> 
{
  auto _ = maia_time_log("create_permuted_ngon_connectivities");

  // prepare accessors
  auto old_ngon_accessor = cgns::interleaved_ngon_random_access_range(old_connectivities);

  std::vector<I> new_connectivities(old_ngon_accessor.memory_length());
  auto new_ngon_accessor = cgns::interleaved_ngon_range(new_connectivities);

  // permute
  std_e::permute_copy_n(old_ngon_accessor.begin(),new_ngon_accessor.begin(),permutation.begin(),permutation.size());

  return new_connectivities;
}

template<class I> auto
apply_permutation_to_ngon(std_e::span<I> old_ngon_cs, const std::vector<I>& permutation) -> void {
  auto new_connectivities  = create_permuted_ngon_connectivities(old_ngon_cs,permutation);
  std::copy(new_connectivities.begin(),new_connectivities.end(),old_ngon_cs.begin());
}

inline auto
sort_ngons_by_nb_vertices(tree& ngons) -> std::vector<I4> {
  // Precondition: ngons.type = "Elements_t" and elements of type NGON_n
  auto parent_elts = ParentElements<I4>(ngons);
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);

  // compute permutation
  auto permutation = sorting_by_nb_vertices_permutation(ngon_connectivity);

  // apply permutation
  apply_permutation_to_ngon(ngon_connectivity,permutation);
  apply_ngon_permutation_to_parent_elts(parent_elts,permutation);

  return permutation;
}

// =================
// TODO factor with above, test
template<class I> auto
sorting_by_nb_faces_permutation(std_e::span<I> connectivities) -> std::vector<I> {
  auto _ = maia_time_log("sorting_by_nb_faces_permutation");
  auto nface_accessor = cgns::interleaved_nface_random_access_range(connectivities);

  // init
  auto nb_connec = nface_accessor.size();
  std::vector<I> permutation(nb_connec);
  std::iota(begin(permutation),end(permutation),0);

  // permute
  auto comp_by_nb_faces = [&nface_accessor](I i, I j){ return nface_accessor[i].size()<nface_accessor[j].size(); };
  std::stable_sort(permutation.begin(),permutation.end(),comp_by_nb_faces);

  return permutation;
}

template<class I> auto
create_permuted_nface_connectivities(std_e::span<I> old_connectivities, const std::vector<I>& permutation)
  -> std::vector<I> 
{
  auto _ = maia_time_log("create_permuted_nface_connectivities");

  // prepare accessors
  auto old_nface_accessor = cgns::interleaved_nface_random_access_range(old_connectivities);

  std::vector<I> new_connectivities(old_nface_accessor.memory_length());
  auto new_nface_accessor = cgns::interleaved_nface_range(new_connectivities);

  // permute
  std_e::permute_copy_n(old_nface_accessor.begin(),new_nface_accessor.begin(),permutation.begin(),permutation.size());

  return new_connectivities;
}

template<class I> auto
apply_permutation_to_nface(std_e::span<I> old_nface_cs, const std::vector<I>& permutation) -> void {
  auto new_connectivities  = create_permuted_nface_connectivities(old_nface_cs,permutation);
  std::copy(new_connectivities.begin(),new_connectivities.end(),old_nface_cs.begin());
}

inline auto
sort_nfaces_by_nb_faces(std_e::span<I4> nface_connectivity, tree& ngons) -> std::vector<I4> {
  auto permutation = sorting_by_nb_faces_permutation(nface_connectivity);
  apply_permutation_to_nface(nface_connectivity,permutation);
  auto parent_elts = ParentElements<I4>(ngons);
  apply_nface_permutation_to_parent_elts(parent_elts,permutation);
  return permutation;
}

inline auto
pyra_penta_permutation(std_e::span<I4> nface_connectivity, const tree& ngons) -> std::pair<std::vector<I4>,I4> {
  I4 ngon_start_id = ElementRange<I4>(ngons)[0];
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);

  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);

  auto nface_accessor = cgns::interleaved_nface_random_access_range(nface_connectivity);

  auto first_5_faces = std_e::find_if(nface_accessor.begin(),nface_accessor.end(),[](const auto& c){ return c.size()==5; });
  auto first_6_faces = std_e::find_if(nface_accessor.begin(),nface_accessor.end(),[](const auto& c){ return c.size()==6; });
  I4 first_5_index = first_5_faces-nface_accessor.begin();
  I4 last_5_index = first_6_faces-nface_accessor.begin();

  auto is_pyra = [ngon_start_id,&ngon_accessor,&nface_accessor](I4 i){ 
    auto polyhedron = nface_accessor[i];
    STD_E_ASSERT(polyhedron.size()==5);
    int nb_quad = 0;
    for (I4 polygon_id : polyhedron) {
      I4 polygon_idx = polygon_id-ngon_start_id;
      auto polygon = ngon_accessor[polygon_idx];
      if (polygon.size()==4) {
        ++nb_quad;
      } else {
        STD_E_ASSERT(polygon.size()==3);
      }
    }
    return nb_quad==1;
  };

  auto nb_connec = nface_accessor.size();
  std::vector<I4> permutation(nb_connec);
  std::iota(begin(permutation),end(permutation),0);
  auto partition_ptr = std::stable_partition(permutation.begin()+first_5_index,permutation.begin()+last_5_index,is_pyra);
  I4 partition_index = partition_ptr - permutation.begin();
  I4 partition_penta_start = (nface_accessor.begin()+partition_index).data() - nface_accessor.data();

  return {permutation,partition_penta_start};
}
inline auto
partition_pyra_penta(std_e::span<I4> nface_connectivity, tree& ngons) -> I4 {
  auto [permutation,partition_penta_start] = pyra_penta_permutation(nface_connectivity,ngons);
  apply_permutation_to_nface(nface_connectivity,permutation);
  auto parent_elts = ParentElements<I4>(ngons);
  apply_nface_permutation_to_parent_elts(parent_elts,permutation);
  // TODO: replace global begin,end by nb_faces=5 begin/end
  return partition_penta_start;
}

inline auto
sort_nfaces_by_simple_polyhedron_type(tree& nfaces, tree& ngons) -> I4 {
  auto nface_connectivity = ElementConnectivity<I4>(nfaces);
  sort_nfaces_by_nb_faces(nface_connectivity,ngons);
  I4 partition_penta_start = partition_pyra_penta(nface_connectivity,ngons);
  return partition_penta_start;
}

} // cgns
