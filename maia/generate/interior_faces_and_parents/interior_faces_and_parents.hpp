#pragma once

#include "maia/generate/interior_faces_and_parents/element_faces.hpp"
#include "maia/connectivity/iter/connectivity_range.hpp"
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

using namespace cgns; // TODO


namespace maia {


struct element_faces_traits {
  ElementType_t elt_type;
  int n_tri;
  int n_quad;
};
inline constexpr std::array elements_face_traits = std_e::make_array(
  //                    name  , n_tri , n_quad
  element_faces_traits{TRI_3  ,   1   ,   0  },
  element_faces_traits{QUAD_4 ,   0   ,   1  },
  element_faces_traits{TETRA_4,   4   ,   0  },
  element_faces_traits{PYRA_5 ,   4   ,   1  },
  element_faces_traits{PENTA_6,   2   ,   3  },
  element_faces_traits{HEXA_8 ,   0   ,   6  }
);

template<class I> auto
find_face_traits_by_elt_type(I elt) {
  auto elt_type = static_cast<ElementType_t>(elt);
  auto match_cgns_elt = [elt_type](const auto& e){ return e.elt_type==elt_type; };
  auto pos = std::find_if(begin(elements_face_traits),end(elements_face_traits),match_cgns_elt);
  if (pos==end(elements_face_traits)) {
    throw cgns_exception(std::string("Unknown CGNS element type for number of nodes\"")+to_string(elt_type)+"\"");
  } else {
    return pos;
  }
}
template<class I> auto
n_tri_elt(I elt) {
  auto pos = find_face_traits_by_elt_type(elt);
  return pos->n_tri;
}
template<class I> auto
n_quad_elt(I elt) {
  auto pos = find_face_traits_by_elt_type(elt);
  return pos->n_quad;
}


using Tri_kind = cgns::connectivity_kind<cgns::TRI_3>;
using Quad_kind = cgns::connectivity_kind<cgns::QUAD_4>;


auto
number_of_faces(const tree_range& elt_sections) {
  I8 n_tri = 0;
  I8 n_quad = 0;
  for (const auto& elt : elt_sections) {
    auto partial_dist = cgns::get_node_value_by_matching<I8>(elt,":CGNS#Distribution/Element");
    I8 n_elt = partial_dist[1] - partial_dist[0]; // TODO more expressive
    I4 elt_type = ElementType<I4>(elt);
    n_tri  += n_elt * n_tri_elt (elt_type);
    n_quad += n_elt * n_quad_elt(elt_type);
  }
  ELOG(n_tri);
  ELOG(n_quad);
  return std::make_pair(n_tri,n_quad);
}

template<I4 elt_type> auto
gen_faces(const tree& elt_node, auto& tri_it, auto& quad_it, auto& tri_parent_it, auto& quad_parent_it) {
  I4 elt_start = ElementRange<I4>(elt_node)[0];
  auto elt_connec = ElementConnectivity<I4>(elt_node);

  using connec_kind = cgns::connectivity_kind<elt_type>;
  auto connec_range = make_connectivity_range<connec_kind>(elt_connec);

  I4 elt_id = elt_start;
  for (const auto& elt : connec_range) {
    generate_faces(elt,tri_it,quad_it);
    tri_parent_it  = std::fill_n( tri_parent_it,n_tri_elt (elt_type),elt_id);
    quad_parent_it = std::fill_n(quad_parent_it,n_quad_elt(elt_type),elt_id);
    ++elt_id;
  }
}


template<class I, ElementType_t Elt_type>
class connectivities_with_parents {
  public:
    using connectivity_k = cgns::connectivity_kind<Elt_type>;

    // Class invariant: connectivities().size() == parents().size()
    connectivities_with_parents(I n_connec)
      : connec(n_connec*number_of_nodes(Elt_type))
      , parens(n_connec)
    {}

    auto
    size() {
      return parens.size();
    }

    auto
    connectivities() {
      return make_connectivity_range<connectivity_k>(connec);
    }

    auto
    parents() -> std_e::span<I> {
      return std_e::make_span(parens);
    }
  private:
    std::vector<I> connec;
    std::vector<I> parens;
};


template<class I>
struct faces_and_parents_t {
  faces_and_parents_t(I n_tri, I n_quad)
    : tris(n_tri)
    , quads(n_quad)
  {}
  connectivities_with_parents<I,cgns::TRI_3 > tris ;
  connectivities_with_parents<I,cgns::QUAD_4> quads;
};


auto
gen_interior_faces_and_parents(const tree_range& elt_sections) -> faces_and_parents_t<I4> { // TODO I8 also
  auto [n_tri,n_quad] = number_of_faces(elt_sections);
  faces_and_parents_t<I4> faces_and_parents(n_tri,n_quad);

  I8 i_tri = 0;
  I8 i_quad = 0;
  auto tri_connec  = faces_and_parents.tris .connectivities();
  auto quad_connec = faces_and_parents.quads.connectivities();
  auto tri_it  = tri_connec .begin();
  auto quad_it = quad_connec.begin();

  auto tri_parent_it  = faces_and_parents.tris .parents().begin();
  auto quad_parent_it = faces_and_parents.quads.parents().begin();
  for (const auto& elt : elt_sections) {
    I4 elt_type = ElementType<I4>(elt);
    switch(elt_type){
      case TRI_3: {
        gen_faces<TRI_3>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case QUAD_4: {
        gen_faces<QUAD_4>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case TETRA_4: {
        gen_faces<TETRA_4>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case PENTA_6: {
        gen_faces<PENTA_6>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case PYRA_5: {
        gen_faces<PYRA_5>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      case HEXA_8: {
        gen_faces<HEXA_8>(elt,tri_it,quad_it,tri_parent_it,quad_parent_it);
        break;
      }
      default: {
        STD_E_ASSERT(0);
      }
    }
  }
  return faces_and_parents;
}


template<class T> auto
tri_reorder_with_smallest_vertex_first(T& tri) -> void {
  auto smallest_pos = std::min_element(tri.data(),tri.data()+3);
  int i = smallest_pos-tri.data();
  int v0 = tri[i];
  int v1 = tri[i+1]%3;
  int v2 = tri[i+2]%3;
  tri[0] = v0;
  tri[1] = v1;
  tri[2] = v2;
}
template<class T> auto
quad_reorder_with_smallest_vertex_first(T& quad) -> void {
  auto smallest_pos = std::min_element(quad.data(),quad.data()+3);
  int i = smallest_pos-quad.data();
  int v0 = quad[i];
  int v1 = quad[i+1]%4;
  int v2 = quad[i+2]%4;
  int v3 = quad[i+3]%4;
  quad[0] = v0;
  quad[1] = v1;
  quad[2] = v2;
  quad[3] = v3;
}

template<class I>
struct int_ext_faces_with_parents {
  std::vector<I> face_parents_ext;
  std::vector<I> vol_parents_ext;
  std::vector<I> connec_int;
  std::vector<I> l_parents_int;
  std::vector<I> r_parents_int;
};


auto
same_face_but_flipped(const auto& f0, const auto& f1) {
  STD_E_ASSERT(f0.size()==f1.size());
  int n = f0.size();
  STD_E_ASSERT(n>0);
  STD_E_ASSERT(f0[0]==f1[0]); // Precondition here
  for (int i=1; i<n; ++i) {
    if (f0[i] != f1[n-i]) return false;
  }
  return true;
}

template<class I, ElementType_t Elt_type> auto
merge_unique_faces(connectivities_with_parents<I,Elt_type>& cps, I n_vtx, I first_3d_elt_id, MPI_Comm comm) -> int_ext_faces_with_parents<I> {
  int n_connec = cps.size();
  auto cs = cps.connectivities();
  auto parents = cps.parents();
  constexpr int n_vtx_elt = number_of_nodes(Elt_type);
  using connectivity_k = cgns::connectivity_kind<Elt_type>;

  // for each face, reorder its vertices so that the smallest is first
  for (auto c : cs) {
    tri_reorder_with_smallest_vertex_first(c);
  }

  // partition_sort by first vertex (this is a partial sort, but good enought so we can exchange based on that)
  // TODO indirect, propagate to parents
  auto less_first_vertex = [&cs](int i, int j){ return cs[i][0] < cs[i][0]; };
  auto distri = uniform_distribution(n_vtx,std_e::n_rank(comm));
  std::vector<int> perm(n_connec);
  std::iota(begin(perm),end(perm),0);
  auto partition_indices = std_e::partition_sort_indices(perm,distri,less_first_vertex);
  std_e::permute(cs.begin(),perm);
  std_e::permute(parents.begin(),perm);

  // exchange
  std_e::jagged_span<I> parents_by_rank(parents,partition_indices);
  auto res_parents = std_e::all_to_all_v(parents_by_rank,comm);
  auto res_parents2 = res_parents.flat_view();

  auto scaled_partition_indices = partition_indices;
  std_e::scale(scaled_partition_indices,n_vtx_elt);
  std_e::jagged_span<I> cs_by_rank(cs.range(),scaled_partition_indices);
  auto res_connec = std_e::all_to_all_v(cs_by_rank,comm);
  auto res_connec2 = res_connec.flat_view();
  auto res_cs = make_connectivity_range<connectivity_k>(res_connec2);

  // finish sort (if two faces are equal, they have the same first vertex, hence are on the same proc)
  // 0. sort vertices so that we can do a lexicographical comparison
  auto res_connec_ordered = res_connec;
  auto res_cs_ordered = make_connectivity_range<connectivity_k>(res_connec_ordered);
  for (auto c : res_cs_ordered) {
    // since the first vtx is already the smallest, no need to include it in the sort
    std_e::sorting_network<n_vtx_elt-1>::sort(c.begin()+1);
  }

  // 1. do the sort based on this lexico ordering
  auto less_vertices = [&res_cs_ordered](int i, int j){ return res_cs_ordered[i]<res_cs_ordered[j]; };
  int n_res_cs = res_cs.size();
  STD_E_ASSERT(n_res_cs%2==0); // each face should be there twice (either two 3d parents if interior, or one 2d and one 3d parent if exterior)
  std::vector<int> perm2(n_res_cs);
  std::iota(begin(perm2),end(perm2),0);
  std::sort(begin(perm2),end(perm2),less_vertices);
  std_e::permute(res_cs.begin(),perm2);
  std_e::permute(res_parents2.begin(),perm2);

  // gather parents two by two
  //std::vector<I> connec_ext;
  std::vector<I> face_parents_ext;
  std::vector<I> vol_parents_ext;
  std::vector<I> connec_int;
  std::vector<I> l_parents_int;
  std::vector<I> r_parents_int;
  connec_int.reserve(res_connec.size()/2);
  l_parents_int.reserve(res_parents2.size()/2);
  r_parents_int.reserve(res_parents2.size()/2);
  auto cs_int = make_connectivity_range<connectivity_k>(connec_int);
  //auto cs_ext = make_connectivity_range<connectivity_k>(connec_ext);

  for (I4 i=0; i<n_res_cs; i+=2) {
    if (res_cs[i]==res_cs[i+1]) {
      // since they are the same,
      // it follows that they are oriented in the same direction
      // this can only be the case if one parent is 2d and the other is 3d
      //cs_ext.push_back(res_cs[i]);
      if (res_parents2[i] < first_3d_elt_id) {
        STD_E_ASSERT(res_parents2[i+1] >= first_3d_elt_id);
        face_parents_ext.push_back(res_parents2[i]);
        vol_parents_ext.push_back(res_parents2[i+1]);
      } else {
        STD_E_ASSERT(res_parents2[i+1] < first_3d_elt_id);
        face_parents_ext.push_back(res_parents2[i+1]);
        vol_parents_ext.push_back(res_parents2[i]);
      }
    } else {
      STD_E_ASSERT(same_face_but_flipped(res_cs[i],res_cs[i+1])); // two 3d parents
      cs_int.push_back(res_cs[i]);
      l_parents_int.push_back(res_parents2[i]);
      r_parents_int.push_back(res_parents2[i+1]);
    }
  }

  return {face_parents_ext,vol_parents_ext,connec_int,l_parents_int,r_parents_int};
}

template<class I> auto
gen_unique_faces_and_parents(tree& surf_elt, const std::vector<I>& face_ids, const std::vector<I>& vol_parents_ext, MPI_Comm comm) {
  auto partial_dist = cgns::get_node_value_by_matching<I8>(surf_elt,":CGNS#Distribution/Element");
  auto dist_I8 = maia::distribution_from_partial(partial_dist,comm);
  auto first_face_id = cgns::get_node_value_by_matching<I>(surf_elt,"ElementRange")[0];
  ELOG(first_face_id);
  I n_face = face_ids.size();
  std::vector<I> face_indices = face_ids;
  std_e::offset(face_indices,-first_face_id);
  std_e::dist_array<I> parents(dist_I8,comm);
  std_e::scatter(parents,dist_I8,std::move(face_indices),vol_parents_ext);

  std_e::buffer_vector<I> parents_array(n_face*2);
  std::copy(begin(parents.local()),end(parents.local()),begin(parents_array)); // only half is assign, the other is 0

  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  parent_elt_node.value.dims = {n_face,2};
  emplace_child(surf_elt,std::move(parent_elt_node));
}

auto
generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) {
  int n_vtx = cgns::VertexSize_U<I4>(z);
  I4 last_2d_elt_id = boundary_elements_interval(z).last();
  I4 first_3d_elt_id = volume_elements_interval(z).first();
  ELOG(last_2d_elt_id);
  ELOG(first_3d_elt_id);
  STD_E_ASSERT(last_2d_elt_id==first_3d_elt_id);
  auto elt_sections = element_sections(z);

  auto faces_and_parents = gen_interior_faces_and_parents(elt_sections);

  auto unique_faces_tri  = merge_unique_faces(faces_and_parents.tris ,n_vtx,first_3d_elt_id,comm);
  auto unique_faces_quad = merge_unique_faces(faces_and_parents.quads,n_vtx,first_3d_elt_id,comm);

  auto ext_tri_node  = find_if(begin(elt_sections),end(elt_sections),[](const tree& t){ return ElementType<I4>(t)==cgns::TRI_3 ; });
  auto ext_quad_node = find_if(begin(elt_sections),end(elt_sections),[](const tree& t){ return ElementType<I4>(t)==cgns::QUAD_4; });

  if (ext_tri_node == end(elt_sections)) {
    STD_E_ASSERT(unique_faces_tri.face_parents_ext.size()==0);
  } else {
    gen_unique_faces_and_parents(*ext_tri_node,unique_faces_tri.face_parents_ext,unique_faces_tri.vol_parents_ext,comm);
  }

  if (ext_quad_node == end(elt_sections)) {
    STD_E_ASSERT(unique_faces_quad.face_parents_ext.size()==0);
  } else {
    gen_unique_faces_and_parents(*ext_quad_node,unique_faces_quad.face_parents_ext,unique_faces_quad.vol_parents_ext,comm);
  }
}


} // maia
