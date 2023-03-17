#if __cplusplus > 201703L
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/algo/merge_unique_faces.hpp"


#include "maia/utils/logging/log.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/algorithm/rotate/rotate.hpp"
#include "std_e/parallel/mpi/extra_types.hpp"
#include "std_e/algorithm/partition_sort.hpp"
#include "std_e/algorithm/sorting_networks.hpp"
#include "std_e/parallel/algorithm/sort_by_rank.hpp"
#include "std_e/data_structure/multi_range/multi_range.hpp"
#include "std_e/future/sort/sort_ranges.hpp"

#if defined REAL_GCC && __GNUC__ < 11
  using std_e::get; // found by ADL with GCC 11
#endif

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

template<ElementType_t face_type, class I> auto
merge_unique(const auto& cs, const auto& pe, const auto& pp, I first_3d_elt_id) -> in_ext_faces<I> {
  // NOTE:
  //     pe means parent elements
  //     pp means parent positions

  //auto _ = std_e::stdout_time_logger("  merge_unique");

  // 0. prepare
  /// 0.0. sizes
  constexpr int n_vtx_of_face_type = number_of_vertices(face_type);
  auto n_faces = cs.size();

  /// 0.1. interior faces
  std::vector<I> connec_int; connec_int.reserve(cs.total_size()/2);
  std::vector<I> l_pe_int;   l_pe_int  .reserve(pe.size()/2);
  std::vector<I> r_pe_int;   r_pe_int  .reserve(pe.size()/2);
  std::vector<I> l_pp_int;   l_pp_int  .reserve(pp.size()/2);
  std::vector<I> r_pp_int;   r_pp_int  .reserve(pp.size()/2);
  auto cs_int = std_e::view_as_block_range<n_vtx_of_face_type>(connec_int);

  /// 0.2. exterior faces
  std::vector<I> face_pe_ext;
  std::vector<I> cell_pe_ext;
  std::vector<I> cell_pp_ext;

  // 1. keep one face and gather parents two by two
  for (I4 i=0; i<n_faces; i+=2) {
    if (cs[i]==cs[i+1]) {
      // since they are the same, they are oriented in the same direction
      // this can only be the case if one parent is 2d and the other is 3d
      if (pe[i] < first_3d_elt_id) {
        STD_E_ASSERT(pe[i+1] >= first_3d_elt_id);
        face_pe_ext.push_back(pe[i]);
        cell_pe_ext.push_back(pe[i+1]);
        cell_pp_ext.push_back(pp[i+1]);
      } else {
        STD_E_ASSERT(pe[i+1] < first_3d_elt_id);
        face_pe_ext.push_back(pe[i+1]);
        cell_pe_ext.push_back(pe[i]);
        cell_pp_ext.push_back(pp[i]);
      }
    } else {
      STD_E_ASSERT(same_face_but_flipped(cs[i],cs[i+1]));
      if (pe[i] >= first_3d_elt_id && pe[i+1] >= first_3d_elt_id) { // two 3d pe
        if (pe[i] < pe[i+1]) {
          cs_int.push_back(cs[i]);
          l_pe_int.push_back(pe[i]);
          r_pe_int.push_back(pe[i+1]);
          l_pp_int.push_back(pp[i]);
          r_pp_int.push_back(pp[i+1]);
        } else {
          cs_int.push_back(cs[i+1]);
          l_pe_int.push_back(pe[i+1]);
          r_pe_int.push_back(pe[i]);
          l_pp_int.push_back(pp[i+1]);
          r_pp_int.push_back(pp[i]);
        }
      } else if (pe[i] < first_3d_elt_id) { // first parent is 2d and the normal is inward (wrong convention)
        STD_E_ASSERT(pe[i+1] >= first_3d_elt_id);
        face_pe_ext.push_back(pe[i]);
        cell_pe_ext.push_back(pe[i+1]);
        cell_pp_ext.push_back(pp[i+1]);
      } else { // second parent is 2d and the normal is inward (wrong convention)
        STD_E_ASSERT(pe[i+1] < first_3d_elt_id);
        face_pe_ext.push_back(pe[i+1]);
        cell_pe_ext.push_back(pe[i]);
        cell_pp_ext.push_back(pp[i]);
      }
    }
  }

  // 2. gather results
  return {
    face_type,
    ext_faces_with_parents{face_pe_ext,cell_pe_ext,cell_pp_ext},
    in_faces_with_parents{connec_int,l_pe_int,r_pe_int,l_pp_int,r_pp_int}
  };
}

template<ElementType_t face_type, class I> auto
merge_faces_of_type(connectivities_with_parents<I>& cps, I first_3d_elt_id, MPI_Comm comm) -> in_ext_faces<I> {
  //auto _ = maia_perf_log_lvl_2("merge_unique_faces");

  // 0. prepare
  constexpr int n_vtx_of_face_type = number_of_vertices(face_type);
  auto cs = connectivities<n_vtx_of_face_type>(cps);
  auto pe = parent_elements(cps);
  auto pp = parent_positions(cps);

  // 1. for each face, reorder its vertices so that the smallest is first
  for (auto c : cs) {
    std_e::rotate_min_first(c);
  }

  // 2. pre-sort the arrays
  //       Note: elements are compared by their first vertex only
  //             this is good enought for load balancing
  auto mr = view_as_multi_range(cs,pe,pp);
  auto proj_on_first_vertex = [](const auto& x){ return get<0>(x)[0]; }; // returns cs[0], i.e. the first vertex
  auto rank_indices = std_e::sort_by_rank(mr,comm,proj_on_first_vertex);


  // 3. exchange
  auto [res_cs,_0] = std_e::all_to_all(cs,rank_indices,comm);
  auto [res_pe,_1] = std_e::all_to_all(pe,rank_indices,comm);
  auto [res_pp,_2] = std_e::all_to_all(pp,rank_indices,comm);

  STD_E_ASSERT(res_cs.size()%2==0); // each face should be there twice
                                    // (either two 3d parents if interior,
                                    //  or one 2d and one 3d parent if exterior)

  // 4. continue sorting locally
  /// 4.0. do a lexico ordering of each face, in an auxilliary array
  auto res_cs_ordered = deep_copy(res_cs);
  for (auto c : res_cs_ordered) {
    // since the first vtx is already the smallest, no need to include it in the sort
    std_e::sorting_network<n_vtx_of_face_type-1>::sort(c.begin()+1);
  }
  /// 4.1. do the sort based on this lexico ordering
  auto res_mr = view_as_multi_range(res_cs_ordered,res_cs,res_pe,res_pp);
  auto proj_on_ordered_cs = [](const auto& x){ return get<0>(x); }; // returns res_cs_ordered
  std_e::ranges::sort(res_mr,{},proj_on_ordered_cs);

  return merge_unique<face_type>(res_cs,res_pe,res_pp,first_3d_elt_id);
}

template<class I> auto
merge_unique_faces(faces_and_parents_by_section<I>& faces_and_parents_sections, I first_3d_elt_id, MPI_Comm comm)
 -> in_ext_faces_by_section<I>
{
  in_ext_faces_by_section<I> ie_faces;
  for (auto& fps : faces_and_parents_sections) {
    auto elt_type = element_type(fps);
    if (elt_type==TRI_3) {
      ie_faces.emplace_back( merge_faces_of_type<TRI_3>(fps,first_3d_elt_id,comm) );
    } else if (elt_type==QUAD_4) {
      ie_faces.emplace_back( merge_faces_of_type<QUAD_4>(fps,first_3d_elt_id,comm) );
    } else {
      throw cgns_exception(std::string("Function \"")+__func__+"\": not implemented for element type \""+to_string(elt_type)+"\"");
    }
  }
  return ie_faces;
}

// Explicit instanciations of functions defined in this .cpp file
template auto merge_unique_faces<I4>(faces_and_parents_by_section<I4>& faces_and_parents_sections, I4 first_3d_elt_id, MPI_Comm comm)
 -> in_ext_faces_by_section<I4>;

template auto merge_unique_faces<I8>(faces_and_parents_by_section<I8>& faces_and_parents_sections, I8 first_3d_elt_id, MPI_Comm comm)
 -> in_ext_faces_by_section<I8>;

} // maia
#endif // C++>17
