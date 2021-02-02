#include "maia/transform/__old/convert_to_simple_connectivities.hpp"


#include "maia/transform/__old/partition_with_boundary_first/boundary_ngons_at_beginning.hpp" // TODO rename file
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "std_e/future/span.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "maia/generate/__old/ngons/from_cells/cast_heterogenous_to_homogenous.hpp"


// TODO factor, nface/ngon iterator/ test
namespace cgns {


auto
sort_zone_nface_into_simple_connectivities(tree& z, factory F) -> void {
  tree& ngons = element_pool<I4>(z,NGON_n);
  tree& nfaces = element_pool<I4>(z,NFACE_n);
  I4 partition_penta_start = sort_nfaces_by_simple_polyhedron_type(nfaces,ngons);
  mark_simple_polyhedron_groups(nfaces,ngons,partition_penta_start,F);
}

auto
only_contains_tris_and_quads(std_e::span<const I4> polygon_types) -> bool {
  return 
      polygon_types == std::vector{3}
   || polygon_types == std::vector{4}
   || polygon_types == std::vector{3,4};
}
auto
convert_to_simple_boundary_connectivities(const tree& ngons, factory F) -> std::vector<tree> {
  auto ngon_range = ElementRange<I4>(ngons);
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto polygon_types = view_as_span<I4>(get_child_by_name(ngons,".#PolygonTypeBoundary").value);
  auto polygon_type_starts = view_as_span<I4>(get_child_by_name(ngons,".#PolygonTypeStartBoundary").value);
 
  STD_E_ASSERT(only_contains_tris_and_quads(polygon_types));
  STD_E_ASSERT(polygon_type_starts.size()==polygon_types.size()+1);

  int nb_polygon_types = polygon_types.size();
  std::vector<tree> elt_pools;
  I4 elt_pool_start = ngon_range[0];
  for (int i=0; i<nb_polygon_types; ++i) {
    I4 polygon_type = polygon_types[i];
    const I4* poly_start = ngon_connectivity.data()+polygon_type_starts[i];
    const I4* poly_finish = ngon_connectivity.data()+polygon_type_starts[i+1];
    auto homogenous_range = std_e::make_span(poly_start,poly_finish);
    auto ngon_accessor = cgns::interleaved_ngon_random_access_range(homogenous_range);
    I4 nb_connec = ngon_accessor.size();
    I4 nb_vertices = nb_connec*polygon_type;
    auto homogenous_connectivities = make_cgns_vector<I4>(nb_vertices,F.alloc());
    I4 cgns_type = 0;
    if (polygon_type==3) { // TODO remove the if once make_connectivity_range is not templated anymore
      cgns_type = cgns::TRI_3;
      using Tri_kind = cgns::connectivity_kind<cgns::TRI_3>;
      auto tris = make_connectivity_range<Tri_kind>(homogenous_connectivities);
      std::transform(ngon_accessor.begin(),ngon_accessor.end(),tris.begin(),[](const auto& het_c){ return cgns::cast_as<cgns::TRI_3>(het_c); });
    }
    if (polygon_type==4) { // TODO remove the if once make_connectivity_range is not templated anymore
      cgns_type = cgns::QUAD_4;
      using Quad_kind = cgns::connectivity_kind<cgns::QUAD_4>;
      auto quads = make_connectivity_range<Quad_kind>(homogenous_connectivities);
      std::transform(ngon_accessor.begin(),ngon_accessor.end(),quads.begin(),[](const auto& het_c){ return cgns::cast_as<cgns::QUAD_4>(het_c); });
    }

    elt_pools.push_back(
      F.newElements(
        "Poly_"+std::to_string(polygon_type),
        cgns_type,
        std_e::make_span(homogenous_connectivities),
        elt_pool_start,elt_pool_start+nb_connec-1
      )
    );
    elt_pool_start += nb_connec;
  }
  return elt_pools;
}

template<class T> I4
find_vertex_not_in_first(const T& connec_0, const T& connec_1) {
  for (I4 vertex : connec_1) {
    if (std::find(begin(connec_0),end(connec_0),vertex)==end(connec_0)) return vertex;
  }
  STD_E_ASSERT(false); return 0;
}


template<class T> auto
convert_to_tetra(const T& tetra_accessor, const tree& ngons, I4 elt_pool_start, I4 elt_pool_start2, factory F) -> tree {
  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto parent_elts = ParentElements<I4>(ngons);

  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);

  I4 nb_tets = tetra_accessor.size();
  I4 nb_vertices = nb_tets*4;
  auto homogenous_connectivities = make_cgns_vector<I4>(nb_vertices,F.alloc());
  auto d_first = homogenous_connectivities.data();

  for (int k=0; k<tetra_accessor.size(); ++k) {
    const auto& tet = tetra_accessor[k];
    I4 tetra_id = elt_pool_start2+k;
    I4 tri_0_idx = tet[0]-first_ngon_id;
    I4 tri_1_idx = tet[1]-first_ngon_id;
    auto tri_0 = ngon_accessor[tri_0_idx];
    auto tri_1 = ngon_accessor[tri_1_idx];
    if (parent_elts(tri_0_idx,0)==tetra_id) { // outward normal
      d_first = std::reverse_copy(begin(tri_0),end(tri_0),d_first);
    } else {
      STD_E_ASSERT(parent_elts(tri_0_idx,1)==tetra_id); // inward normal
      d_first = std::copy(begin(tri_0),end(tri_0),d_first);
    }
    I4 other_vertex = find_vertex_not_in_first(tri_0,tri_1);
    *d_first++ = other_vertex;
  }

  return F.newElements(
    "TETRA_4",
    cgns::TETRA_4,
    std_e::make_span(homogenous_connectivities),
    elt_pool_start,elt_pool_start+nb_tets-1
  );
}
template<class T> auto
convert_to_pyra(const T& pyra_accessor, const tree& ngons, I4 elt_pool_start, I4 elt_pool_start2, factory F) -> tree {
  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto parent_elts = ParentElements<I4>(ngons);

  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);

  I4 nb_pyras = pyra_accessor.size();
  I4 nb_vertices = nb_pyras*5;
  auto homogenous_connectivities = make_cgns_vector<I4>(nb_vertices,F.alloc());
  auto d_first = homogenous_connectivities.data();

  for (int k=0; k<pyra_accessor.size(); ++k) {
    const auto& pyra = pyra_accessor[k];
    I4 pyra_id = elt_pool_start2+k;
    for (int i=0; i<pyra.size(); ++i) {
      I4 face_idx = pyra[i]-first_ngon_id;
      auto face = ngon_accessor[face_idx];
      if (face.size()==4) {
        if (parent_elts(face_idx,0)==pyra_id) { // outward normal
          d_first = std::reverse_copy(begin(face),end(face),d_first);
        } else {
          STD_E_ASSERT(parent_elts(face_idx,1)==pyra_id); // inward normal
          d_first = std::copy(begin(face),end(face),d_first);
        }
        I4 tri_idx = pyra[(i+1)%5]-first_ngon_id; // since we just found the quad, face (i+1)%5 has to be a tri
        auto tri = ngon_accessor[tri_idx];
        I4 other_vertex = find_vertex_not_in_first(face,tri);
        *d_first++ = other_vertex;
        break;
      }
    }
  }

  return F.newElements(
    "PYRA_5",
    cgns::PYRA_5,
    std_e::make_span(homogenous_connectivities),
    elt_pool_start,elt_pool_start+nb_pyras-1
  );
}


template<class I, class Node_connectivity> auto
contains_vertex(I vtx, const Node_connectivity& c) -> bool {
  return std::find(begin(c),end(c),vtx) != end(c);
}

template<class I, class Quad> auto
other_common(I vtx, const Quad& quad0, const Quad& quad1) -> I4 {
  std::array<I,4> q0;
  std::copy(begin(quad0),end(quad0),begin(q0));
  std::sort(begin(q0),end(q0));
  std::array<I,4> q1;
  std::copy(begin(quad1),end(quad1),begin(q1));
  std::sort(begin(q1),end(q1));
  std::array<I,2> q0_inter_q1;
  std::set_intersection(begin(q0),end(q0),begin(q1),end(q1),begin(q0_inter_q1));
  if (q0_inter_q1[0]!=vtx) return q0_inter_q1[0];
  else { STD_E_ASSERT(q0_inter_q1[0]==vtx); return q0_inter_q1[1]; }
}

template<class T> auto
node_above(I4 vtx, const T& quads) -> I4 {
  auto contains_vtx = [vtx](const auto& quad){ return contains_vertex(vtx,quad); };
  auto quad0 = std::find_if(begin(quads),end(quads),contains_vtx);
  auto quad1 = std::find_if(quad0+1,end(quads),contains_vtx);
  STD_E_ASSERT(quad0!=end(quads));
  STD_E_ASSERT(quad1!=end(quads));
  return other_common(vtx,*quad0,*quad1);
}

template<class T> auto
convert_to_penta(const T& penta_accessor, const tree& ngons, I4 elt_pool_start, I4 elt_pool_start2, factory F) -> tree {
  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto parent_elts = ParentElements<I4>(ngons);

  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);

  I4 nb_pentas = penta_accessor.size();
  I4 nb_vertices = nb_pentas*6;
  auto homogenous_connectivities = make_cgns_vector<I4>(nb_vertices,F.alloc());
  auto d_first = homogenous_connectivities.data();

  for (int k=0; k<penta_accessor.size(); ++k) {
    const auto& penta = penta_accessor[k];
    I4 penta_id = elt_pool_start2+k;
    using face_type = heterogenous_connectivity_view<I4,const I4,maia::interleaved_polygon_kind>;
    const I4 not_found = -1;
    I4 tri_idx = not_found;
    std::array<I4,3> tri;
    std::array<face_type,3> quads;
    int quad_pos = 0;
    for (const auto& face_id : penta) {
      I4 face_idx = face_id-first_ngon_id;
      auto face = ngon_accessor[face_idx];
      if (face.size()==3) {
        if (tri_idx==not_found) {
          tri_idx = face_idx;
          std::copy(begin(face),end(face),begin(tri));
        }
      } else {
        quads[quad_pos] = face;
        ++quad_pos;
      }
    }
    // use the tri as nodes 1,2,3
    if (parent_elts(tri_idx,0)==penta_id) { // outward normal
      std::reverse(begin(tri),end(tri));
    } else {
      STD_E_ASSERT(parent_elts(tri_idx,1)==penta_id);
    }
    d_first = std::copy(begin(tri),end(tri),d_first);

    // nodes "above" tri
    *d_first++ = node_above(tri[0],quads);
    *d_first++ = node_above(tri[1],quads);
    *d_first++ = node_above(tri[2],quads);
  }

  return F.newElements(
    "PENTA_6",
    cgns::PENTA_6,
    std_e::make_span(homogenous_connectivities),
    elt_pool_start,elt_pool_start+nb_pentas-1
  );
}

template<class Connecivity_type_0, class Connecivity_type_1> auto
share_vertices(const Connecivity_type_0& c0, const Connecivity_type_1& c1) -> bool {
  // TODO replace by std_e::set_intersection_size if size_0*size_1 >> size*log(size)
  for (I4 x : c0) {
    for (I4 y : c1) {
      if (x==y) return true;
    }
  }
  return false;
}

template<class T> auto
convert_to_hexa(const T& hexa_accessor, const tree& ngons, I4 elt_pool_start, I4 elt_pool_start2, factory F) -> tree {
  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  auto ngon_connectivity = ElementConnectivity<I4>(ngons);
  auto parent_elts = ParentElements<I4>(ngons);

  auto ngon_accessor = cgns::interleaved_ngon_random_access_range(ngon_connectivity);

  I4 nb_hexas = hexa_accessor.size();
  I4 nb_vertices = nb_hexas*8;
  auto homogenous_connectivities = make_cgns_vector<I4>(nb_vertices,F.alloc());
  auto d_first = homogenous_connectivities.data();

  for (int k=0; k<hexa_accessor.size(); ++k) {
    const auto& hexa = hexa_accessor[k];
    I4 hexa_id = elt_pool_start2+k;

    std::array<I4,4> quad_0;
    I4 quad_0_idx = hexa[0]-first_ngon_id;
    auto quad_0_in_ngon = ngon_accessor[quad_0_idx];
    std::copy(begin(quad_0_in_ngon),end(quad_0_in_ngon),begin(quad_0));

    using face_type = heterogenous_connectivity_view<I4,const I4,maia::interleaved_polygon_kind>;
    std::array<face_type,4> side_quads;
    int side_quad_pos = 0;
    for (int i=0; i<6; ++i) {
      I4 quad_idx = hexa[i]-first_ngon_id;
      auto quad = ngon_accessor[quad_idx];
      STD_E_ASSERT(quad.size()==4);
    }
    for (int i=1; i<6; ++i) {
      I4 quad_idx = hexa[i]-first_ngon_id;
      auto quad = ngon_accessor[quad_idx];
      if (share_vertices(quad,quad_0)) {
        side_quads[side_quad_pos] = quad;
        ++side_quad_pos;
      }
    }

    // use quad_0 as nodes 1,2,3,4 
    if (parent_elts(quad_0_idx,0)==hexa_id) { // outward normal
      std::reverse(begin(quad_0),end(quad_0));
    } else {
      STD_E_ASSERT(parent_elts(quad_0_idx,1)==hexa_id);
    }
    d_first = std::copy(begin(quad_0),end(quad_0),d_first);

    // nodes "above" quad_0
    *d_first++ = node_above(quad_0[0],side_quads);
    *d_first++ = node_above(quad_0[1],side_quads);
    *d_first++ = node_above(quad_0[2],side_quads);
    *d_first++ = node_above(quad_0[3],side_quads);
  }

  return F.newElements(
    "HEXA_8",
    cgns::HEXA_8,
    std_e::make_span(homogenous_connectivities),
    elt_pool_start,elt_pool_start+nb_hexas-1
  );
}

auto
convert_to_simple_volume_connectivities(const tree& nfaces, const tree& ngons, I4 vol_elt_start_id, factory F) -> std::vector<tree> {
  auto nface_connectivity = ElementConnectivity<I4>(nfaces);
  auto nface_start = nface_connectivity.data();
  auto polyhedron_type_starts = view_as_span<I4>(get_child_by_name(nfaces,".#PolygonSimpleTypeStart").value);

  auto tetra_range = std_e::make_span(nface_start+polyhedron_type_starts[0],nface_start+polyhedron_type_starts[1]);
  auto pyra_range  = std_e::make_span(nface_start+polyhedron_type_starts[1],nface_start+polyhedron_type_starts[2]);
  auto penta_range = std_e::make_span(nface_start+polyhedron_type_starts[2],nface_start+polyhedron_type_starts[3]);
  auto hexa_range  = std_e::make_span(nface_start+polyhedron_type_starts[3],nface_start+polyhedron_type_starts[4]);

  // debug
  auto nfs = cgns::interleaved_nface_random_access_range(nface_connectivity);
  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  int elt_pool_start3 = 1;
  auto parent_elts = ParentElements<I4>(ngons);
  for (int k=0; k<nfs.size(); ++k) {
    I4 nf_id = elt_pool_start3+k;
    const auto& nf = nfs[k];
    for (int f_id : nf) {
      I4 f_idx = f_id-first_ngon_id;
      STD_E_ASSERT(parent_elts(f_idx,0)==nf_id || parent_elts(f_idx,1)==nf_id);
    }
  }
  // end debug

  std::vector<tree> elt_pools;

  I4 elt_pool_start = vol_elt_start_id;
  I4 elt_pool_start2 = 1; // TODO FIXME only here as quickfix to parent element starting at 1 and not elt_pool_start (also in apply_nface_permutation_to_parent_elts)
  auto tetra_accessor = cgns::interleaved_nface_random_access_range(tetra_range);
  if (tetra_accessor.size()>0) {
    elt_pools.push_back(convert_to_tetra(tetra_accessor,ngons,elt_pool_start,elt_pool_start2,F));
    elt_pool_start += tetra_accessor.size();
    elt_pool_start2 += tetra_accessor.size();
  }

  auto pyra_accessor = cgns::interleaved_nface_random_access_range(pyra_range);
  if (pyra_accessor.size()>0) {
    elt_pools.push_back(convert_to_pyra(pyra_accessor,ngons,elt_pool_start,elt_pool_start2,F));
    elt_pool_start += pyra_accessor.size();
    elt_pool_start2 += pyra_accessor.size();
  }

  auto penta_accessor = cgns::interleaved_nface_random_access_range(penta_range);
  if (penta_accessor.size()>0) {
    elt_pools.push_back(convert_to_penta(penta_accessor,ngons,elt_pool_start,elt_pool_start2,F));
    elt_pool_start += penta_accessor.size();
    elt_pool_start2 += penta_accessor.size();
  }

  auto hexa_accessor = cgns::interleaved_nface_random_access_range(hexa_range);
  if (hexa_accessor.size()>0) {
    elt_pools.push_back(convert_to_hexa(hexa_accessor,ngons,elt_pool_start,elt_pool_start2,F));
    elt_pool_start += hexa_accessor.size();
    elt_pool_start2 += hexa_accessor.size();
  }

  return elt_pools;
}


auto
convert_zone_to_simple_connectivities(tree& z, factory F) -> void {
  STD_E_ASSERT(z.label=="Zone_t");

  sort_zone_nface_into_simple_connectivities(z,F);

  tree& ngons = element_pool<I4>(z,NGON_n);
  tree& nfaces = element_pool<I4>(z,NFACE_n);
  auto bnd_elt_pools = convert_to_simple_boundary_connectivities(ngons,F);

  auto& last_bnd_elt_pool = bnd_elt_pools.back();
  I4 vol_elt_start_id = ElementRange<I4>(last_bnd_elt_pool)[1]+1;
  auto vol_elt_pools = convert_to_simple_volume_connectivities(nfaces,ngons,vol_elt_start_id,F);

  // TODO deallocate if necessary => when is it? for which allocator?
  // A: it is right to dealloc if OWNED by F
  //    it is right to NOT dealloc if OWNED by anybody other than F
  std::string ngons_name = ngons.name;
  std::string nfaces_name = nfaces.name;
  F.rm_child_by_name(z,ngons_name);
  F.rm_child_by_name(z,nfaces_name);

  emplace_children(z,std::move(bnd_elt_pools));
  emplace_children(z,std::move(vol_elt_pools));
}


auto
sort_nface_into_simple_connectivities(tree& b, factory F) -> void {
  auto f = [&F](tree& z){ sort_zone_nface_into_simple_connectivities(z,F); };
  for_each_unstructured_zone(b,f);
}
auto
convert_to_simple_connectivities(tree& b, factory F) -> void {
  auto f = [&F](tree& z){ convert_zone_to_simple_connectivities(z,F); };
  for_each_unstructured_zone(b,f);
}



} // cgns
