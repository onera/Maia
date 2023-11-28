#include "maia/algo/dist/find_duplicate_elt.pybind.hpp"

void find_duplicate_elt (          int                       n_elt,
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
     - Beware of gnum
     - Renvoyer la liste PL
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
     - Beware of gnum
     - Renvoyer la liste PL
  */
  using int_t = int;
  auto elt_vtx  = make_raw_view(np_elt_vtx);
  auto elt_mask = make_raw_view(np_elt_mask);

  // > Local copy of elt_vtx
  std::vector<int_t> elt_vtx_lex(elt_vtx, elt_vtx+n_elt*elt_size); 

  // {std_e::time_logger tlog0("maia", "sort elt sorted by vtx");
  // > Sort vtx in each element 
  for (int i_elt=0; i_elt<n_elt; ++i_elt) {
    std::sort(elt_vtx_lex.data()+i_elt*elt_size, elt_vtx_lex.data()+(i_elt+1)*elt_size);
  }
  // }

  // > Lambda function to compare two elements
  // > Nothing in [&] because elt_vtx_lex and elt_size needed, and when 2 norm say to put nothing
  // auto elt_comp = [&elt_vtx_lex](int i, int j) {
  auto elt_comp = [&](int i, int j) { 
    auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
    auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
    auto elt_i_end = elt_i_beg + elt_size;
    auto elt_j_end = elt_j_beg + elt_size;
    return std::lexicographical_compare(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
    // return *elt_i_beg<*elt_j_beg;
  };

  std::vector<int_t> order(n_elt); 
  // {std_e::time_logger tlog0("maia", "sort order");
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), elt_comp);
  // }

  // > Lambda function equal elements
  // auto is_same_elt = [&elt_vtx_lex](int i, int j) {
  auto is_same_elt = [&](int i, int j) { // Nothin in & because 
    auto elt_i_beg = elt_vtx_lex.data()+i*elt_size;
    auto elt_j_beg = elt_vtx_lex.data()+j*elt_size;
    auto elt_i_end = elt_i_beg + elt_size;
    auto elt_j_end = elt_j_beg + elt_size;
    return std::equal(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
  };

  // {std_e::time_logger tlog0("maia", "find unique");
  int compteur=1;
  int idx_previous = 0;
  for (int i_elt=1; i_elt<n_elt; ++i_elt) {
    if (is_same_elt(order[idx_previous], order[i_elt])) {
      compteur++;
      if ((i_elt==n_elt-1)&&(compteur!=1)) {
        for (int i_elt2=idx_previous; i_elt2<i_elt+1; ++i_elt2) {
          elt_mask[order[i_elt2]] = 0;
        }
      }
    }
    else {
      if (compteur!=1) {
        for (int i_elt2=idx_previous; i_elt2<i_elt; ++i_elt2) {
          elt_mask[order[i_elt2]] = 0;
        }
      }
      idx_previous = i_elt;
      compteur = 1;
    }
  }
  // }
}


void find_duplicate_elt3(         int                       n_elt,
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
     - Beware of gnum
     - Renvoyer la liste PL
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