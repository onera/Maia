#include "maia/algo/dist/dist_algo.pybind.hpp"
#if __cplusplus > 201703L
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/__old/transform/convert_to_std_elements.hpp"
#include "maia/algo/dist/rearrange_element_sections/rearrange_element_sections.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "maia/algo/dist/elements_to_ngons/elements_to_ngons.hpp"
#include "maia/algo/dist/fsdm_distribution/fsdm_distribution.hpp"
#include "maia/__old/transform/put_boundary_first/put_boundary_first.hpp"
#include "maia/algo/dist/split_boundary_subzones_according_to_bcs/split_boundary_subzones_according_to_bcs.hpp"
#include "maia/algo/dist/adaptation_utils.hpp"

namespace py = pybind11;

template<class F> auto
apply_cpp_cgns_function_to_py_base(F&& f) {
  return [&f](py::list py_base) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    f(base);
    update_py_tree(std::move(base),py_base);
  };
}
template<class F> auto
apply_cpp_cgns_par_function_to_py_base(F&& f) {
  return [&f](py::list py_base, py::object mpi4py_comm) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);
    f(base,comm);
    update_py_tree(std::move(base),py_base);
  };
}

const auto generate_interior_faces_and_parents = apply_cpp_cgns_par_function_to_py_base(maia::generate_interior_faces_and_parents);
const auto elements_to_ngons               = apply_cpp_cgns_par_function_to_py_base(maia::elements_to_ngons);
const auto convert_zone_to_std_elements = apply_cpp_cgns_function_to_py_base(maia::convert_zone_to_std_elements);
const auto add_fsdm_distribution         = apply_cpp_cgns_par_function_to_py_base(maia::add_fsdm_distribution);
const auto rearrange_element_sections             = apply_cpp_cgns_par_function_to_py_base(maia::rearrange_element_sections);
const auto put_boundary_first = apply_cpp_cgns_par_function_to_py_base(maia::put_boundary_first);
const auto split_boundary_subzones_according_to_bcs = apply_cpp_cgns_par_function_to_py_base(maia::split_boundary_subzones_according_to_bcs);


void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");

  m.def("generate_interior_faces_and_parents"     , generate_interior_faces_and_parents     , "Generate TRI_3_interior and QUAD_4_interior element sections, and adds ParentElement to interior and exterior faces");
  m.def("elements_to_ngons"                       , elements_to_ngons                       , "Convert to NGon");
  m.def("convert_zone_to_std_elements"            , convert_zone_to_std_elements            , "ngon to elements");
  m.def("add_fsdm_distribution"                   , add_fsdm_distribution                   , "Add FSDM-specific distribution info");
  m.def("rearrange_element_sections"              , rearrange_element_sections              , "For a distributed base, merge Elements_t nodes of the same type and does the associated renumbering");
  m.def("put_boundary_first"                      , put_boundary_first                      , "ngon sorted with boundary faces first");
  m.def("split_boundary_subzones_according_to_bcs", split_boundary_subzones_according_to_bcs, "Split a ZoneSubRegion node with a PointRange spaning all boundary faces into multiple ZoneSubRegion with a BCRegionName");
  
}
#else //C++==17

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "maia/utils/pybind_utils.hpp"

namespace py = pybind11;

void find_tri_in_tetras(            int                       n_tri,
                                    int                       n_tetra,
                        py::array_t<int, py::array::f_style>& np_tri_vtx,
                        py::array_t<int, py::array::f_style>& np_tetra_vtx,
                        py::array_t<int, py::array::f_style>& np_tri_mask) {
  /*
    Go through all triangles verifying that they are described by at least on cell.
    n_tri*n_cell complexity, could be improve using ngon description and hash table.
    Args : 
      - n_tri        [in] : triangle number
      - n_tetra      [in] : tetrahedra number
      - np_tri_vtx   [in] : triangle connectivity
      - np_tetra_vtx [in] : tetrahedra connectivity
      - np_tri_mask  [out]: which triangle are not duplicated

    TODO: n_tri*n_tetra algo, improve with ngon reconstruction then hash table
  */

  auto tri_vtx   = make_raw_view(np_tri_vtx);
  auto tetra_vtx = make_raw_view(np_tetra_vtx);
  auto tri_mask  = make_raw_view(np_tri_mask);

  int size_tri   = 3;
  int size_tetra = 4;
  
  int n_vtx_in_tet = 0;
  int is_in_tetra  = 0;

  for (int i_elt1=0; i_elt1<n_tri; i_elt1++) {
    for (int i_elt2=0; i_elt2<n_tetra; i_elt2++) {
      is_in_tetra = 0;
      n_vtx_in_tet = 0;
      for (int i_vtx1=i_elt1*size_tri; i_vtx1<(i_elt1+1)*size_tri; i_vtx1++) {

        for (int i_vtx2=i_elt2*size_tetra; i_vtx2<(i_elt2+1)*size_tetra; i_vtx2++) {
          if (tri_vtx[i_vtx1]==tetra_vtx[i_vtx2]) {
            n_vtx_in_tet ++;
            break;
          }
        }
        // Vtx from element 1 is not in element 2
        if (n_vtx_in_tet==3) {
          is_in_tetra = 1;
          break;
        }
      }
      if (is_in_tetra==1) {
        tri_mask[i_elt1] = 1;
      }      
    }
  }
}


void find_duplicate_elt3(          int                       n_elt,
                                  int                       elt_size,
                      py::array_t<int, py::array::f_style>& np_elt_vtx,
                      py::array_t<int, py::array::f_style>& np_elt_mask) {
  /*
    Go through all elements verifying that has not already be defined.
    Args : 
      - n_elt       [in] : element number
      - elt_size    [in] : vertex number in one element
      - np_elt_vtx  [in] : element connectivity
      - np_elt_mask [out]: which element are not duplicated

    TODO:
     - delete elt_size arguement (to be retrieve in place)
     - demander a Berenger si il a ca en stock
     - Beware of gnum
     - Renvoyer la liste PL
     - Resoudre les conflit en triangulaire
  */
  using int_t = int;
  auto elt_vtx  = make_raw_view(np_elt_vtx);
  auto elt_mask = make_raw_view(np_elt_mask);

  // > Build keys array
  int *elt_key = (int *) malloc(n_elt*sizeof(int));
  for (int i_elt=0; i_elt<n_elt; i_elt++) {
    elt_key[i_elt] = 0;
    for (int i_vtx=i_elt*elt_size; i_vtx<(i_elt+1)*elt_size; i_vtx++) {
      elt_key[i_elt] = elt_key[i_elt] + elt_vtx[i_vtx];
    }
  }

  // > Sort array
  int *order = (int *)  malloc(n_elt*sizeof(int));
  std::iota(order, order+n_elt, 0);
  std::sort(order, order+n_elt, [&](int i, int j) {return elt_key[i] < elt_key[j];});

  // > Create conflict idx
  int n_elem_in_conflict = 0;
  int n_conflict = 0;
  int *conflict_idx = (int *)  malloc(n_elt*sizeof(int));
  conflict_idx[0] = 0;
  for (int i_elt=0; i_elt<n_elt-1; i_elt++) {
    if (elt_key[order[i_elt]]!=elt_key[order[i_elt+1]]) {
      n_conflict++;
      conflict_idx[n_conflict] = i_elt+1;
    }
  }
  n_conflict++;
  conflict_idx[n_conflict] = n_elt;

  // > Resolve conflict
  int n_elt_in_conflict = 0;
  for (int i_conflict=0; i_conflict<n_conflict; i_conflict++) {
    n_elt_in_conflict = conflict_idx[i_conflict+1]-conflict_idx[i_conflict];
    // printf("i_conflict = %d with %d conflicts\n", i_conflict, n_elt_in_conflict);
    if (n_elt_in_conflict>1) {
      // > Local copy of elt_vtx
      auto j=0;
      std::vector<int_t> elt_vtx_lex(n_elt_in_conflict*elt_size);
      for (int i=conflict_idx[i_conflict]; i<conflict_idx[i_conflict+1]; i++) {
        auto i_elt = order[i];
        for (int i_vtx=i_elt*elt_size; i_vtx<(i_elt+1)*elt_size; i_vtx++) {
          elt_vtx_lex[j] = elt_vtx[i_vtx];
          j++;
        }
      }


      // > Sort vtx in each element of conflict
      for (int i_elt=0; i_elt<n_elt_in_conflict; ++i_elt) {
        std::sort(elt_vtx_lex.data()+i_elt*elt_size, elt_vtx_lex.data()+(i_elt+1)*elt_size);
      }


      // > Lambda function to compare two elements
      // > Nothing in [&] because elt_vtx_lex and elt_size needed, and when 2 norm say to put nothing
      auto elt_comp = [&](int i, int j) { 
        auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
        auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
        auto elt_i_end = elt_i_beg + elt_size;
        auto elt_j_end = elt_j_beg + elt_size;
        return std::lexicographical_compare(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
      };

      std::vector<int_t> lorder(n_elt_in_conflict); 
      std::iota(lorder.begin(), lorder.end(), 0);
      std::sort(lorder.begin(), lorder.end(), elt_comp);
      
      // > Lambda function equal elements
      // auto is_same_elt = [&elt_vtx_lex](int i, int j) {
      auto is_same_elt = [&](int i, int j) { // Nothin in & because 
        auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
        auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
        auto elt_i_end = elt_i_beg + elt_size;
        auto elt_j_end = elt_j_beg + elt_size;
        return std::equal(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
      };

      int compteur=1;
      int idx_previous = 0;
      for (int i_elt=1; i_elt<n_elt_in_conflict; ++i_elt) {
        if (is_same_elt(lorder[idx_previous], lorder[i_elt])) {
          compteur++;
          if ((i_elt==n_elt_in_conflict-1)&&(compteur!=1)) {
            for (int i_elt2=idx_previous; i_elt2<i_elt+1; ++i_elt2) {
              auto id = order[conflict_idx[i_conflict]+lorder[i_elt2]];
              elt_mask[id] = 0;
            }
          }
        }
        else {
          if (compteur!=1) {
            for (int i_elt2=idx_previous; i_elt2<i_elt; ++i_elt2) {
              auto id = order[conflict_idx[i_conflict]+lorder[i_elt2]];
              elt_mask[id] = 0;
            }
          }
          idx_previous = i_elt;
          compteur = 1;
        }
      }
    }
  }
}


void find_duplicate_elt2(          int                       n_elt,
                                   int                       elt_size,
                       py::array_t<int, py::array::f_style>& np_elt_vtx,
                       py::array_t<int, py::array::f_style>& np_elt_mask) {
  /*
    Go through all elements verifying that has not already be defined.
    Args : 
      - n_elt       [in] : element number
      - elt_size    [in] : vertex number in one element
      - np_elt_vtx  [in] : element connectivity
      - np_elt_mask [out]: which element are not duplicated

    TODO:
     - delete elt_size arguement (to be retrieve in place)
     - demander a Berenger si il a ca en stock
     - Beware of gnum
     - Renvoyer la liste PL
  */
  using int_t = int;
  auto elt_vtx  = make_raw_view(np_elt_vtx);
  auto elt_mask = make_raw_view(np_elt_mask);

  // > Local copy of elt_vtx
  std::vector<int_t> elt_vtx_lex(elt_vtx, elt_vtx+n_elt*elt_size); 

  // > Sort vtx in each element 
  for (int i_elt=0; i_elt<n_elt; ++i_elt) {
    std::sort(elt_vtx_lex.data()+i_elt*elt_size, elt_vtx_lex.data()+(i_elt+1)*elt_size);
  }

  // printf("Sort vtx in each element :\n");
  // for (int i_vtx=0; i_vtx<n_elt*elt_size; ++i_vtx) {
  //   if (i_vtx%elt_size==0) {
  //     printf("  ");
  //   }
  //   printf("%d ", elt_vtx_lex[i_vtx]);
  // }
  // printf("\n\n");

  // > Lambda function to compare two elements
  // > Nothing in [&] because elt_vtx_lex and elt_size needed, and when 2 norm say to put nothing
  // auto elt_comp = [&elt_vtx_lex](int i, int j) {
  auto elt_comp = [&](int i, int j) { 
    auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
    auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
    auto elt_i_end = elt_i_beg + elt_size;
    auto elt_j_end = elt_j_beg + elt_size;
    return std::lexicographical_compare(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
  };

  std::vector<int_t> order(n_elt); 
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), elt_comp);

  // printf("Order :\n");
  // for (int i_elt=0; i_elt<n_elt; ++i_elt) {
  //   printf("%d ", order[i_elt]);
  // }
  // printf("\n\n");

  // printf("Sort elements :\n");
  // for (int i_elt=0; i_elt<n_elt; ++i_elt) {
  //   printf("  ");
  //   for (int i_vtx=0; i_vtx<elt_size; ++i_vtx) {
  //     printf("%d ", elt_vtx_lex[(order[i_elt])*elt_size+i_vtx]);
  //   }
  // }
  // printf("\n\n");

  // > Lambda function equal elements
  // auto is_same_elt = [&elt_vtx_lex](int i, int j) {
  auto is_same_elt = [&](int i, int j) { // Nothin in & because 
    auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
    auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
    auto elt_i_end = elt_i_beg + elt_size;
    auto elt_j_end = elt_j_beg + elt_size;
    return std::equal(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
  };

  int compteur=1;
  int idx_previous = 0;
  for (int i_elt=1; i_elt<n_elt; ++i_elt) {
    // printf("Compare %d with %d\n", idx_previous, i_elt);
    if (is_same_elt(order[idx_previous], order[i_elt])) {
      compteur++;
      // printf("  --> same : compteur = %d\n", compteur);
      if ((i_elt==n_elt-1)&&(compteur!=1)) {
        // printf("    --> Change from %d to %d\n", idx_previous, i_elt);
        for (int i_elt2=idx_previous; i_elt2<i_elt+1; ++i_elt2) {
          // printf("i_elt2 = %d -> order[i_elt2] = %d\n", i_elt2, order[i_elt2]);
          elt_mask[order[i_elt2]] = 0;
        }
      }
    }
    else {
      // printf("  --> different : compteur = %d\n", compteur);
      if (compteur!=1) {
        // printf("    --> Change from %d to %d\n", idx_previous, i_elt-1);
        for (int i_elt2=idx_previous; i_elt2<i_elt; ++i_elt2) {
          // printf("i_elt2 = %d -> order[i_elt2] = %d\n", i_elt2, order[i_elt2]);
          elt_mask[order[i_elt2]] = 0;
        }
      }
      idx_previous = i_elt;
      compteur = 1;
    }
  }
}


void find_duplicate_elt(          int                       n_elt,
                                  int                       elt_size,
                      py::array_t<int, py::array::f_style>& np_elt_vtx,
                      py::array_t<int, py::array::f_style>& np_elt_mask) {
  /*
    Go through all elements verifying that has not already be defined.
    Args : 
      - n_elt       [in] : element number
      - elt_size    [in] : vertex number in one element
      - np_elt_vtx  [in] : element connectivity
      - np_elt_mask [out]: which element are not duplicated

    TODO:
     - delete elt_size arguement (to be retrieve in place)
     - demander a Berenger si il a ca en stock
     - Beware of gnum
     - Renvoyer la liste PL
     - Resoudre les conflit en triangulaire
  */
  auto elt_vtx  = make_raw_view(np_elt_vtx);
  auto elt_mask = make_raw_view(np_elt_mask);

  // > Build keys array
  int *elt_key = (int *) malloc(n_elt*sizeof(int));
  for (int i_elt=0; i_elt<n_elt; i_elt++) {
    elt_key[i_elt] = 0;
    for (int i_vtx=i_elt*elt_size; i_vtx<(i_elt+1)*elt_size; i_vtx++) {
      elt_key[i_elt] = elt_key[i_elt] + elt_vtx[i_vtx];
    }
  }

  // > Sort array
  int *order = (int *)  malloc(n_elt*sizeof(int));
  std::iota(order, order+n_elt, 0);
  std::sort(order, order+n_elt, [&](int i, int j) {return elt_key[i] < elt_key[j];});

  // > Create conflict idx
  int n_elem_in_conflict = 0;
  int n_conflict = 0;
  int *conflict_idx = (int *)  malloc(n_elt*sizeof(int));
  conflict_idx[0] = 0;
  for (int i_elt=0; i_elt<n_elt-1; i_elt++) {
    if (elt_key[order[i_elt]]!=elt_key[order[i_elt+1]]) {
      n_conflict++;
      conflict_idx[n_conflict] = i_elt+1;
    }
  }
  n_conflict++;
  conflict_idx[n_conflict] = n_elt;

  // > Resolve conflict
  int i_elt1 = 0;
  int i_elt2 = 0;
  int li_vtx = 0;
  int n_similar_vtx = 0;
  int *similar_vtx = (int *) malloc(elt_size*sizeof(int)); // Array to tag vtx already taggued as similar in elt2
  int n_elt_in_conflict = 0;
  for (int i_conflict=0; i_conflict<n_conflict; i_conflict++) {
    n_elt_in_conflict = conflict_idx[i_conflict+1]-conflict_idx[i_conflict];
    if (n_elt_in_conflict>1) {
      for (int i1=conflict_idx[i_conflict]; i1<conflict_idx[i_conflict+1]; i1++) {
        i_elt1 = order[i1];
        for (int i2=i1+1; i2<conflict_idx[i_conflict+1]; i2++) {
          i_elt2 = order[i2];

          n_similar_vtx = 0;
          for (int i_vtx=0; i_vtx<elt_size; i_vtx++) {
            similar_vtx[i_vtx] = 0;
          }

          if (elt_mask[i_elt2]!=0) {
            for (int i_vtx1=i_elt1*elt_size; i_vtx1<(i_elt1+1)*elt_size; i_vtx1++) {

              li_vtx = 0;
              for (int i_vtx2=i_elt2*elt_size; i_vtx2<(i_elt2+1)*elt_size; i_vtx2++) {
                if ((similar_vtx[li_vtx]==0)&&(elt_vtx[i_vtx1]==elt_vtx[i_vtx2])) {
                  n_similar_vtx ++;
                  similar_vtx[li_vtx] = 1;
                  break;
                }
                li_vtx ++;
              }
            }
            if (n_similar_vtx==elt_size) {
              elt_mask[i_elt1] = 0;
              elt_mask[i_elt2] = 0;
            }      
          }
        }
      }
    }
  }
}


void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");
  m.def("find_tri_in_tetras", find_tri_in_tetras, "Find triangles defined in one tetrahedra at least");
  m.def("find_duplicate_elt", find_duplicate_elt, "Find elements that are duplicated");
  m.def("find_duplicate_elt2", find_duplicate_elt2, "Find elements that are duplicated");
  m.def("find_duplicate_elt3", find_duplicate_elt3, "Find elements that are duplicated");
  
}
#endif //C++>17
