#include "maia/generate/interior_faces_and_parents/merge_unique_faces.hpp"


#include "maia/utils/log/log.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "std_e/algorithm/rotate/rotate.hpp"
#include "std_e/parallel/all_to_all.hpp"
#include "std_e/algorithm/partition_sort.hpp"
#include "std_e/algorithm/sorting_networks.hpp"
#include "std_e/algorithm/algorithm.hpp"
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
  constexpr int n_vtx_of_face_type = std::decay_t<decltype(cs)>::block_size();
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
  auto cs_int = std_e::view_as_block_range<n_vtx_of_face_type>(connec_int);

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

template<ElementType_t face_type, class I> auto
merge_faces_of_type(connectivities_with_parents<I>& cps, I first_3d_elt_id, MPI_Comm comm) -> in_ext_faces_with_parents<I> {
  auto _ = maia_perf_log_lvl_2("merge_unique_faces");

  constexpr int n_vtx_of_face_type = number_of_vertices(face_type);
  size_t n_connec = size(cps);
  auto cs = connectivities<n_vtx_of_face_type>(cps);
  auto pe = parent_elements(cps);
  auto pp = parent_positions(cps);

  // for each face, reorder its vertices so that the smallest is first
  for (auto c : cs) {
    std_e::rotate_min_first(c);
  }

  // partition_sort by first vertex (this is a partial sort, but good enought so we can exchange based on that)
  //auto _0 = std_e::stdout_time_logger("  partition sort");
    auto proj = [](const auto& x){ return get<0>(x)[0]; }; // sort cs, according to the first vertex only
    auto mr = view_as_multi_range(cs,pe,pp);
    auto partition_indices = std_e::pivot_partition_eq(mr,comm,proj);
  //_0.stop();


  // exchange
  //auto _1 = std_e::stdout_time_logger("  exchange");
    std_e::jagged_span<I> parents_by_rank(pe,partition_indices);
    auto res_parents = std_e::all_to_all_v(parents_by_rank,comm);
    auto res_parents2 = res_parents.flat_view();

    std_e::jagged_span<I> parent_positions_by_rank(pp,partition_indices);
    auto res_parent_positions = std_e::all_to_all_v(parent_positions_by_rank,comm);
    auto res_parent_positions2 = res_parent_positions.flat_view();

    auto scaled_partition_indices = partition_indices;
    std_e::scale(scaled_partition_indices,n_vtx_of_face_type);
    std_e::jagged_span<I> cs_by_rank(cs.underlying_range(),scaled_partition_indices);
    auto res_connec = std_e::all_to_all_v(cs_by_rank,comm);
    auto res_connec2 = res_connec.flat_view();
    auto res_cs = std_e::view_as_block_range<n_vtx_of_face_type>(res_connec2);
    STD_E_ASSERT(res_cs.size()%2==0); // each face should be there twice
                                      // (either two 3d parents if interior,
                                      //  or one 2d and one 3d parent if exterior)
  //_1.stop();

  // finish sort (if two faces are equal, they have the same first vertex, hence are on the same proc)
  // 0. sort vertices so that we can do a lexicographical comparison
  //auto _2 = std_e::stdout_time_logger("  vertex sort");
    std::vector<I> res_connec_ordered(res_connec2.begin(),res_connec2.end());
    auto res_cs_ordered = std_e::view_as_block_range<n_vtx_of_face_type>(res_connec_ordered);
    for (auto c : res_cs_ordered) {
      // since the first vtx is already the smallest, no need to include it in the sort
      std_e::sorting_network<n_vtx_of_face_type-1>::sort(c.begin()+1);
    }
  //_2.stop();

  // 1. do the sort based on this lexico ordering
  //auto _3 = std_e::stdout_time_logger("  final sort");
    auto proj2 = [](const auto& x){ return get<0>(x); };
    auto mr2 = view_as_multi_range(res_cs_ordered,res_cs,res_parents2,res_parent_positions2);
    std_e::ranges::sort(mr2,{},proj2);
  //_3.stop();

  return merge_uniq(res_cs,res_parents2,res_parent_positions2,first_3d_elt_id);
}

template<class I, size_t... Is> auto
merge_faces_by_face_types(
  faces_and_parents_by_section<I>& faces_and_parents_sections, I first_3d_elt_id, MPI_Comm comm,
  std::index_sequence<Is...>
)
 -> std::array<in_ext_faces_with_parents<I>,cgns::n_face_types>
{
  return { merge_faces_of_type<cgns::all_face_types[Is]>(faces_and_parents_sections[Is],first_3d_elt_id,comm) ... };
}

template<class I> auto
merge_unique_faces(faces_and_parents_by_section<I>& faces_and_parents_sections, I first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I>,cgns::n_face_types>
{
  auto static_loop_indices = std::make_index_sequence<cgns::n_face_types>{};
  return
    merge_faces_by_face_types(
      faces_and_parents_sections,first_3d_elt_id,comm,
      static_loop_indices
    );
}

// Explicit instanciations of functions defined in this .cpp file
template auto merge_unique_faces<I4>(faces_and_parents_by_section<I4>& faces_and_parents_sections, I4 first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I4>,cgns::n_face_types>;

template auto merge_unique_faces<I8>(faces_and_parents_by_section<I8>& faces_and_parents_sections, I8 first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I8>,cgns::n_face_types>;

} // maia
