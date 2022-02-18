#include "maia/generate/interior_faces_and_parents/merge_unique_faces.hpp"


#include "maia/utils/log/log.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "std_e/algorithm/rotate/rotate.hpp"
#include "std_e/parallel/all_to_all.hpp"
#include "std_e/algorithm/partition_sort.hpp"
#include "std_e/algorithm/sorting_networks.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/log.hpp" // TODO
#include "std_e/parallel/algorithm/pivot_partition_eq.hpp"
#include "std_e/data_structure/multi_range/multi_range.hpp"
#include "std_e/future/sort/sort_ranges.hpp"


using namespace cgns;


namespace maia {


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
merge_uniq(auto& cs, std_e::span<I> parents, std_e::span<I> parent_positions, I first_3d_elt_id) -> in_ext_faces_with_parents<I> {
  //auto _ = std_e::stdout_time_logger("  merge_uniq");
  constexpr int n_vtx_elt = std::decay_t<decltype(cs)>::block_size();
  // gather parents two by two
  std::vector<I> face_parents_ext;
  std::vector<I> vol_parents_ext;
  std::vector<I> vol_parent_positions_ext;
  std::vector<I> connec_int;
  std::vector<I> l_parents_int;
  std::vector<I> r_parents_int;
  std::vector<I> l_parent_positions_int;
  std::vector<I> r_parent_positions_int;
  connec_int.reserve(cs.total_size()/2);
  l_parents_int.reserve(parents.size()/2);
  r_parents_int.reserve(parents.size()/2);
  l_parents_int.reserve(parent_positions.size()/2);
  r_parents_int.reserve(parent_positions.size()/2);
  auto cs_int = std_e::view_as_block_range<n_vtx_elt>(connec_int);

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
        vol_parent_positions_ext.push_back(parent_positions[i+1]);
      } else {
        STD_E_ASSERT(parents[i+1] < first_3d_elt_id);
        face_parents_ext.push_back(parents[i+1]);
        vol_parents_ext.push_back(parents[i]);
        vol_parent_positions_ext.push_back(parent_positions[i]);
      }
    } else {
      STD_E_ASSERT(same_face_but_flipped(cs[i],cs[i+1]));
      if (parents[i] >= first_3d_elt_id && parents[i+1] >= first_3d_elt_id) { // two 3d parents
        cs_int.push_back(cs[i]);
        l_parents_int.push_back(parents[i]);
        r_parents_int.push_back(parents[i+1]);
        l_parent_positions_int.push_back(parent_positions[i]);
        r_parent_positions_int.push_back(parent_positions[i+1]);
      } else if (parents[i] < first_3d_elt_id) { // first parent is 2d and the normal is inward (wrong convention)
        STD_E_ASSERT(parents[i+1] >= first_3d_elt_id);
        face_parents_ext.push_back(parents[i]);
        vol_parents_ext.push_back(parents[i+1]);
        vol_parent_positions_ext.push_back(parent_positions[i+1]);
      } else { // second parent is 2d and the normal is inward (wrong convention)
        STD_E_ASSERT(parents[i+1] < first_3d_elt_id);
        face_parents_ext.push_back(parents[i+1]);
        vol_parents_ext.push_back(parents[i]);
        vol_parent_positions_ext.push_back(parent_positions[i]);
      }
    }
  }

  return {
    in_faces_with_parents{connec_int,l_parents_int,r_parents_int,l_parent_positions_int,r_parent_positions_int},
    ext_faces_with_parents{face_parents_ext,vol_parents_ext,vol_parent_positions_ext}
  };
}

template<class I, ElementType_t elt_type> auto
merge_unique_faces(connectivities_with_parents<I,elt_type>& cps, I n_vtx, I first_3d_elt_id, MPI_Comm comm) -> in_ext_faces_with_parents<I> {
  auto _ = maia_perf_log_lvl_2("merge_unique_faces");
  int n_connec = cps.size();
  auto cs = cps.connectivities();
  auto parents = cps.parents();
  auto parent_positions = cps.parent_positions();
  constexpr int n_vtx_elt = number_of_vertices(elt_type);

  // for each face, reorder its vertices so that the smallest is first
  for (auto c : cs) {
    std_e::rotate_min_first(c);
  }

  // partition_sort by first vertex (this is a partial sort, but good enought so we can exchange based on that)
  auto _0 = std_e::stdout_time_logger("  partition sort");
    //auto less_first_vertex = [&cs](int i, const auto& y){ return cs[i][0] < y; }; // TODO ugly (non-symmetric)
    //auto distri = uniform_distribution(std_e::n_rank(comm),n_vtx);
    //std::vector<int> perm(n_connec);
    //std::iota(begin(perm),end(perm),0);
    //auto partition_indices = std_e::partition_sort_indices(perm,distri,less_first_vertex);
    //auto partition_indices = std_e::make_span(partition_indices.data()+1,partition_indices.size()-1);
    //std_e::permute(cs.begin(),perm);
    //std_e::permute(parents.begin(),perm);
    //std_e::permute(parent_positions.begin(),perm);

    std::vector<int> perm(n_connec);
    std::iota(begin(perm),end(perm),0);
    auto proj = [&cs](I i){ return cs[i][0]; };
    //auto _00 = std_e::stdout_time_logger("    pivot_partition_eq");
      auto partition_indices = std_e::pivot_partition_eq(perm,comm,proj);
    //_00.stop();

    //auto _01 = std_e::stdout_time_logger("    perm");
      std_e::permute(cs.begin(),perm);
      std_e::permute(parents.begin(),perm);
      std_e::permute(parent_positions.begin(),perm);
    //_01.stop();
  _0.stop();


  // exchange
  //auto _1 = std_e::stdout_time_logger("  exchange");
    std_e::jagged_span<I> parents_by_rank(parents,partition_indices);
    auto res_parents = std_e::all_to_all_v(parents_by_rank,comm);
    auto res_parents2 = res_parents.flat_view();

    std_e::jagged_span<I> parent_positions_by_rank(parent_positions,partition_indices);
    auto res_parent_positions = std_e::all_to_all_v(parent_positions_by_rank,comm);
    auto res_parent_positions2 = res_parent_positions.flat_view();

    auto scaled_partition_indices = partition_indices;
    std_e::scale(scaled_partition_indices,n_vtx_elt);
    std_e::jagged_span<I> cs_by_rank(cs.underlying_range(),scaled_partition_indices);
    auto res_connec = std_e::all_to_all_v(cs_by_rank,comm);
    auto res_connec2 = res_connec.flat_view();
    auto res_cs = std_e::view_as_block_range<n_vtx_elt>(res_connec2);
  //_1.stop();

  // finish sort (if two faces are equal, they have the same first vertex, hence are on the same proc)
  // 0. sort vertices so that we can do a lexicographical comparison
  //auto _2 = std_e::stdout_time_logger("  vertex sort");
    std::vector<I> res_connec_ordered(res_connec2.begin(),res_connec2.end());
    auto res_cs_ordered = std_e::view_as_block_range<n_vtx_elt>(res_connec_ordered);
    for (auto c : res_cs_ordered) {
      // since the first vtx is already the smallest, no need to include it in the sort
      std_e::sorting_network<n_vtx_elt-1>::sort(c.begin()+1);
    }
  //_2.stop();

  // 1. do the sort based on this lexico ordering
  auto _3 = std_e::stdout_time_logger("  final sort");
    //auto less_vertices = [&res_cs_ordered](int i, int j){ return res_cs_ordered[i]<res_cs_ordered[j]; };
    int n_res_cs = res_cs.size();
    STD_E_ASSERT(n_res_cs%2==0); // each face should be there twice (either two 3d parents if interior, or one 2d and one 3d parent if exterior)
    //std::vector<int> perm2(n_res_cs);
    //std::iota(begin(perm2),end(perm2),0);

    auto proj2 = [](const auto& x){ return get<0>(x); };
    auto mr = view_as_multi_range(res_cs_ordered,res_cs,res_parents2,res_parent_positions2);
    //auto mr = std_e::view_as_multi_range(res_cs_ordered);
    std_e::ranges::sort(mr,{},proj2);
    ////auto _30 = std_e::stdout_time_logger("    sort");
    //  std::sort(begin(perm2),end(perm2),less_vertices);
    ////_30.stop();

    ////auto _31 = std_e::stdout_time_logger("    perm");
    //  std_e::permute(res_cs.begin(),perm2);
    //  std_e::permute(res_parents2.begin(),perm2);
    //  std_e::permute(res_parent_positions2.begin(),perm2);
    ////_31.stop();
  _3.stop();

  return merge_uniq(res_cs,res_parents2,res_parent_positions2,first_3d_elt_id);
}

template<class I> auto
merge_unique_faces(faces_and_parents_by_section<I>& faces_and_parents_sections, I n_vtx, I first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I>,cgns::n_face_types>
{
  return
    transform(
      faces_and_parents_sections,
      [=](auto& x){ return merge_unique_faces(x,n_vtx,first_3d_elt_id,comm); }
    );
}

// Explicit instanciations of functions defined in this .cpp file
template auto merge_unique_faces<I4>(faces_and_parents_by_section<I4>& faces_and_parents_sections, I4 n_vtx, I4 first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I4>,cgns::n_face_types>;

template auto merge_unique_faces<I8>(faces_and_parents_by_section<I8>& faces_and_parents_sections, I8 n_vtx, I8 first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I8>,cgns::n_face_types>;

} // maia
