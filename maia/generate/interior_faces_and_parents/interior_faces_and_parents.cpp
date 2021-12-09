#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"
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
#include "maia/sids/maia_cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/log.hpp" // TODO
#include "std_e/logging/time_logger.hpp" // TODO


using namespace cgns;


namespace maia {


template<class I> auto
reorder_with_smallest_vertex_first(std_e::span_ref<I,3>& tri) -> void {
  auto smallest_pos = std::min_element(tri.data(),tri.data()+3);
  int i = smallest_pos-tri.data();
  int v0 = tri[i];
  int v1 = tri[(i+1)%3];
  int v2 = tri[(i+2)%3];
  tri[0] = v0;
  tri[1] = v1;
  tri[2] = v2;
}
template<class I> auto
reorder_with_smallest_vertex_first(std_e::span_ref<I,4>& quad) -> void {
  auto smallest_pos = std::min_element(quad.data(),quad.data()+4);
  int i = smallest_pos-quad.data();
  int v0 = quad[i];
  int v1 = quad[(i+1)%4];
  int v2 = quad[(i+2)%4];
  int v3 = quad[(i+3)%4];
  quad[0] = v0;
  quad[1] = v1;
  quad[2] = v2;
  quad[3] = v3;
}

template<class I>
struct in_faces_with_parents {
  std::vector<I> connec;
  std::vector<I> l_parents;
  std::vector<I> r_parents;
};
template<class I>
struct ext_faces_with_parents {
  std::vector<I> face_parents;
  std::vector<I> vol_parents;
};
template<class I>
struct in_ext_faces_with_parents {
  in_faces_with_parents <I> in;
  ext_faces_with_parents<I> ext;
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

template<class I> auto
merge_uniq(auto& cs, std_e::span<I> parents, I first_3d_elt_id) -> in_ext_faces_with_parents<I> {
  //auto _ = std_e::stdout_time_logger("  merge_uniq");
  using connectivity_k = typename std::decay_t<decltype(cs)>::kind;
  // gather parents two by two
  std::vector<I> face_parents_ext;
  std::vector<I> vol_parents_ext;
  std::vector<I> connec_int;
  std::vector<I> l_parents_int;
  std::vector<I> r_parents_int;
  connec_int.reserve(cs.n_vtx()/2);
  l_parents_int.reserve(parents.size()/2);
  r_parents_int.reserve(parents.size()/2);
  auto cs_int = make_connectivity_range<connectivity_k>(connec_int);

  auto n_res_cs = cs.size();
  for (I4 i=0; i<n_res_cs; i+=2) {
    if (cs[i]==cs[i+1]) {
      // since they are the same,
      // it follows that they are oriented in the same direction
      // this can only be the case if one parent is 2d and the other is 3d
      if (parents[i] < first_3d_elt_id) {
        STD_E_ASSERT(parents[i+1] >= first_3d_elt_id);
        face_parents_ext.push_back(parents[i]);
        vol_parents_ext.push_back(parents[i+1]);
      } else {
        STD_E_ASSERT(parents[i+1] < first_3d_elt_id);
        face_parents_ext.push_back(parents[i+1]);
        vol_parents_ext.push_back(parents[i]);
      }
    } else {
      //if (!same_face_but_flipped(cs[i],cs[i+1])) {
      //  ELOG(cs[i]);
      //  ELOG(cs[i+1]);
      //}
      STD_E_ASSERT(same_face_but_flipped(cs[i],cs[i+1])); // two 3d parents
      cs_int.push_back(cs[i]);
      l_parents_int.push_back(parents[i]);
      r_parents_int.push_back(parents[i+1]);
    }
  }

  return {in_faces_with_parents{connec_int,l_parents_int,r_parents_int},ext_faces_with_parents{face_parents_ext,vol_parents_ext}};
}

template<class I, ElementType_t elt_type> auto
merge_unique_faces(connectivities_with_parents<I,elt_type>& cps, I n_vtx, I first_3d_elt_id, MPI_Comm comm) -> in_ext_faces_with_parents<I> {
  //auto _ = std_e::stdout_time_logger("merge_unique_faces");
  int n_connec = cps.size();
  auto cs = cps.connectivities();
  auto parents = cps.parents();
  constexpr int n_vtx_elt = number_of_vertices(elt_type);
  using connectivity_k = cgns::connectivity_kind<elt_type>;

  // for each face, reorder its vertices so that the smallest is first
  for (auto c : cs) {
    reorder_with_smallest_vertex_first(c);
  }

  // partition_sort by first vertex (this is a partial sort, but good enought so we can exchange based on that)
  //auto _0 = std_e::stdout_time_logger("  partition sort");
    auto less_first_vertex = [&cs](int i, const auto& y){ return cs[i][0] < y; }; // TODO ugly (non-symmetric)
    auto distri = uniform_distribution(std_e::n_rank(comm),n_vtx);
    std::vector<int> perm(n_connec);
    std::iota(begin(perm),end(perm),0);
    auto partition_indices = std_e::partition_sort_indices(perm,distri,less_first_vertex);
    auto partition_indices2 = std_e::make_span(partition_indices.data()+1,partition_indices.size()-1);
    std_e::permute(cs.begin(),perm);
    std_e::permute(parents.begin(),perm);
  //_0.stop();


  // exchange
  //auto _1 = std_e::stdout_time_logger("  exchange");
    std_e::jagged_span<I> parents_by_rank(parents,partition_indices2);
    auto res_parents = std_e::all_to_all_v(parents_by_rank,comm);
    auto res_parents2 = res_parents.flat_view();

    auto scaled_partition_indices = partition_indices2;
    std_e::scale(scaled_partition_indices,n_vtx_elt);
    std_e::jagged_span<I> cs_by_rank(cs.underlying_range(),scaled_partition_indices);
    auto res_connec = std_e::all_to_all_v(cs_by_rank,comm);
    auto res_connec2 = res_connec.flat_view();
    auto res_cs = make_connectivity_range<connectivity_k>(res_connec2);
  //_1.stop();

  // finish sort (if two faces are equal, they have the same first vertex, hence are on the same proc)
  // 0. sort vertices so that we can do a lexicographical comparison
  //auto _2 = std_e::stdout_time_logger("  vertex sort");
    std::vector<I> res_connec_ordered(res_connec2.begin(),res_connec2.end());
    auto res_cs_ordered = make_connectivity_range<connectivity_k>(res_connec_ordered);
    for (auto c : res_cs_ordered) {
      // since the first vtx is already the smallest, no need to include it in the sort
      std_e::sorting_network<n_vtx_elt-1>::sort(c.begin()+1);
    }
  //_2.stop();

  // 1. do the sort based on this lexico ordering
  //auto _3 = std_e::stdout_time_logger("  final sort");
    auto less_vertices = [&res_cs_ordered](int i, int j){ return res_cs_ordered[i]<res_cs_ordered[j]; };
    int n_res_cs = res_cs.size();
    STD_E_ASSERT(n_res_cs%2==0); // each face should be there twice (either two 3d parents if interior, or one 2d and one 3d parent if exterior)
    std::vector<int> perm2(n_res_cs);
    std::iota(begin(perm2),end(perm2),0);
    //auto _30 = std_e::stdout_time_logger("    std::sort");
      std::sort(begin(perm2),end(perm2),less_vertices);
    //_30.stop();

    //auto _31 = std_e::stdout_time_logger("    permute cs");
      std_e::permute(res_cs.begin(),perm2);
    //_31.stop();
    //auto _32 = std_e::stdout_time_logger("    permute parents");
      std_e::permute(res_parents2.begin(),perm2);
    //_32.stop();
  //_3.stop();

  return merge_uniq(res_cs,res_parents2,first_3d_elt_id);
}

template<class I> auto
scatter_parents_to_sections(tree& surf_elt, const std::vector<I>& face_ids, const std::vector<I>& vol_parents_ext, MPI_Comm comm) {
  //auto _ = std_e::stdout_time_logger("scatter_parents_to_sections");
  auto partial_dist = cgns::get_node_value_by_matching<I8>(surf_elt,":CGNS#Distribution/Element");
  auto dist_I8 = maia::distribution_from_partial(partial_dist,comm);
  auto first_face_id = cgns::get_node_value_by_matching<I>(surf_elt,"ElementRange")[0];
  std::vector<I> face_indices = face_ids;
  std_e::offset(face_indices,-first_face_id);
  std_e::dist_array<I> parents(dist_I8,comm);
  std_e::scatter(parents,dist_I8,std::move(face_indices),vol_parents_ext);

  I n_face_res = parents.local().size();
  md_array<I,2> parents_array(n_face_res,2);
  std::copy(begin(parents.local()),end(parents.local()),begin(parents_array)); // only half is assign, the other is 0

  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(surf_elt,std::move(parent_elt_node));
}

template<class I> auto
append_element_section(cgns::tree& z, ElementType_t elt_type, in_faces_with_parents<I>&& faces_with_p, MPI_Comm comm) -> void { // TODO do not copy faces_with_p
  //auto _ = std_e::stdout_time_logger("append_element_section");
  I n_face = faces_with_p.l_parents.size();

  I n_face_tot = std_e::all_reduce(n_face,MPI_SUM,comm);
  I n_face_acc = std_e::scan(n_face,MPI_SUM,comm) - n_face;

  I first_sec_id = max_element_id(z)+1;
  I last_sec_id = first_sec_id+n_face_tot-1;

  // connec
  tree elt_section_node = new_Elements(to_string(elt_type)+"_interior",(I)elt_type,std::move(faces_with_p.connec),first_sec_id,last_sec_id);

  // parent
  md_array<I,2> parents_array(n_face,2);
  // TODO optim with no copy
  auto mid_parent_array = std::copy(begin(faces_with_p.l_parents),end(faces_with_p.l_parents),begin(parents_array));
                          std::copy(begin(faces_with_p.r_parents),end(faces_with_p.r_parents),mid_parent_array);

  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(elt_section_node,std::move(parent_elt_node));

  // distribution
  std::vector<I8> elt_distri_mem(3);
  elt_distri_mem[0] = n_face_acc;
  elt_distri_mem[1] = n_face_acc + n_face;
  elt_distri_mem[2] = n_face_tot;
  tree elt_dist = new_DataArray("Element",std::move(elt_distri_mem));
  auto dist_node = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(dist_node,std::move(elt_dist));
  emplace_child(elt_section_node,std::move(dist_node));

  emplace_child(z,std::move(elt_section_node));
}


template<class I> auto
_generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));

  auto elt_sections = element_sections(z);
  auto faces_and_parents = generate_element_faces_and_parents<I>(elt_sections);

  I n_vtx = cgns::VertexSize_U<I>(z);
  I first_3d_elt_id = volume_elements_interval(z).first();
  auto unique_faces_tri  = merge_unique_faces(faces_and_parents.tris ,n_vtx,first_3d_elt_id,comm);
  auto unique_faces_quad = merge_unique_faces(faces_and_parents.quads,n_vtx,first_3d_elt_id,comm);

  auto ext_tri_node  = std::find_if(begin(elt_sections),end(elt_sections),[](const tree& t){ return element_type(t)==cgns::TRI_3 ; });
  auto ext_quad_node = std::find_if(begin(elt_sections),end(elt_sections),[](const tree& t){ return element_type(t)==cgns::QUAD_4; });

  if (ext_tri_node == end(elt_sections)) {
    STD_E_ASSERT(unique_faces_tri.ext.face_parents.size()==0);
  } else {
    scatter_parents_to_sections(*ext_tri_node,unique_faces_tri.ext.face_parents,unique_faces_tri.ext.vol_parents,comm);
  }

  if (ext_quad_node == end(elt_sections)) {
    STD_E_ASSERT(unique_faces_quad.ext.face_parents.size()==0);
  } else {
    scatter_parents_to_sections(*ext_quad_node,unique_faces_quad.ext.face_parents,unique_faces_quad.ext.vol_parents,comm);
  }

  append_element_section(z,cgns::TRI_3 ,std::move(unique_faces_tri .in),comm);
  append_element_section(z,cgns::QUAD_4,std::move(unique_faces_quad.in),comm);
};

auto
generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));
  auto _ = std_e::stdout_time_logger("== generate_interior_faces_and_parents ==");
  if (value(z).data_type()=="I4") return _generate_interior_faces_and_parents<I4>(z,comm);
  if (value(z).data_type()=="I8") return _generate_interior_faces_and_parents<I8>(z,comm);
  throw cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}


} // maia
