#include "maia/utils/ndarray/find_duplicate_elt.pybind.hpp"
namespace py = pybind11;

py::array_t<bool>
is_unique_cst_stride_hash(int               n_elt,
                          int               stride,
                          py::array_t<int>& np_array)
{
  /*
    Args : 
      - n_elt       : Number of elements
      - stride      : Value of the stride, ie number of values per elements (cst)
      - np_array    : Data array (size = n_elt*stride)
    Returns:
      - np_elt_mask : Flag indicating if elts are unique (true) or not (false)

    TODO:
     - Type generalization
  */

  auto np_elt_mask =  py::array_t<bool>(n_elt); // To be returned

  auto elt_mask = np_elt_mask.mutable_unchecked<1>();
  auto array    = np_array.unchecked<1>();

  // > Build keys array
  int *elt_key = new int[n_elt];
  for (int i_elt=0; i_elt<n_elt; i_elt++) {
    elt_key[i_elt] = 0;
    elt_mask(i_elt) = true;
    for (int j=i_elt*stride; j<(i_elt+1)*stride; ++j) {
      elt_key[i_elt] = elt_key[i_elt] + array(j);
    }
  }

  // > Sort array
  int *order = new int[n_elt];
  std::iota(order, order+n_elt, 0);
  std::sort(order, order+n_elt, [&](int i, int j) {return elt_key[i] < elt_key[j];});

  // > Create conflict idx
  int n_elem_in_conflict = 0;
  int n_conflict = 0;
  int *conflict_idx = new int[n_elt];
  conflict_idx[0] = 0;
  for (int i_elt=0; i_elt<n_elt-1; i_elt++) {
    if (elt_key[order[i_elt]]!=elt_key[order[i_elt+1]]) {
      n_conflict++;
      conflict_idx[n_conflict] = i_elt+1;
    }
  }
  n_conflict++;
  conflict_idx[n_conflict] = n_elt;
  delete[] elt_key;

  // > Resolve conflict
  int i_elt1 = 0;
  int i_elt2 = 0;
  int li = 0;
  int n_similar = 0;
  std::vector<int8_t> similar(stride); // Array to values already taggued as similar in elt2
  for (int i_conflict=0; i_conflict<n_conflict; i_conflict++) {
    for (int i1=conflict_idx[i_conflict]; i1<conflict_idx[i_conflict+1]; i1++) {
      i_elt1 = order[i1];
      for (int i2=i1+1; i2<conflict_idx[i_conflict+1]; i2++) {
        i_elt2 = order[i2];

        n_similar = 0;
        std::fill(similar.begin(), similar.end(), 0);

        if (elt_mask(i_elt2)) {
          for (int j1 = i_elt1*stride; j1 < (i_elt1+1)*stride; ++j1) {

            li = 0;
            for (int j2 = i_elt2*stride; j2 < (i_elt2+1)*stride; ++j2) {
              if (!(similar[li]) && (array(j1) == array(j2))) {
                n_similar++;
                similar[li] = 1;
                break;
              }
              li++;
            }
          }
          if (n_similar==stride) {
            elt_mask(i_elt1) = false;
            elt_mask(i_elt2) = false;
          }      
        }
      }
    }
  }
  delete[] order;
  delete[] conflict_idx;
  return np_elt_mask;
}


py::array_t<bool>
is_unique_cst_stride_sort(int               n_elt,
                          int               stride,
                          py::array_t<int>& np_array)
{
  /**
   * See is_unique_cst_stride_hash
  */
  using int_t = int;
  auto np_elt_mask =  py::array_t<bool>(n_elt); // To be returned

  auto array    = np_array.unchecked<1>();
  auto elt_mask = np_elt_mask.mutable_unchecked<1>();

  // > Build keys array
  int *elt_key = new int[n_elt];
  for (int i_elt=0; i_elt<n_elt; i_elt++) {
    elt_key[i_elt] = 0;
    elt_mask(i_elt) = true;
    for (int j = i_elt*stride; j < (i_elt+1)*stride; ++j) {
      elt_key[i_elt] = elt_key[i_elt] + array(j);
    }
  }

  // > Sort array
  int *order = new int[n_elt];
  std::iota(order, order+n_elt, 0);
  std::sort(order, order+n_elt, [&](int i, int j) {return elt_key[i] < elt_key[j];});

  // > Create conflict idx
  int n_elem_in_conflict = 0;
  int n_conflict = 0;
  int *conflict_idx = new int[n_elt];
  conflict_idx[0] = 0;
  for (int i_elt=0; i_elt<n_elt-1; i_elt++) {
    if (elt_key[order[i_elt]]!=elt_key[order[i_elt+1]]) {
      n_conflict++;
      conflict_idx[n_conflict] = i_elt+1;
    }
  }
  n_conflict++;
  conflict_idx[n_conflict] = n_elt;

  delete[] elt_key;

  // > Resolve conflict
  int n_elt_in_conflict = 0;
  for (int i_conflict=0; i_conflict<n_conflict; i_conflict++) {
    n_elt_in_conflict = conflict_idx[i_conflict+1]-conflict_idx[i_conflict];
    if (n_elt_in_conflict>1) {
      // > Local copy of array
      auto j=0;
      std::vector<int_t> elt_val_lex(n_elt_in_conflict*stride);
      for (int i=conflict_idx[i_conflict]; i<conflict_idx[i_conflict+1]; i++) {
        auto i_elt = order[i];
        for (int i_val = i_elt*stride; i_val < (i_elt+1)*stride; ++i_val) {
          elt_val_lex[j++] = array(i_val);
        }
      }

      // > Sort vtx in each element of conflict
      for (int i_elt=0; i_elt<n_elt_in_conflict; ++i_elt) {
        std::sort(elt_val_lex.data()+i_elt*stride, elt_val_lex.data()+(i_elt+1)*stride);
      }

      // > Lambda function to compare two elements
      // > Nothing in [&] because elt_val_lex and stride needed, and when 2 norm say to put nothing
      auto elt_comp = [&](int i, int j) { 
        auto elt_i_beg = elt_val_lex.data()+i*stride;
        auto elt_j_beg = elt_val_lex.data()+j*stride;
        auto elt_i_end = elt_i_beg + stride;
        auto elt_j_end = elt_j_beg + stride;
        return std::lexicographical_compare(elt_i_beg, elt_i_end, elt_j_beg, elt_j_end);
      };

      std::vector<int_t> lorder(n_elt_in_conflict); 
      std::iota(lorder.begin(), lorder.end(), 0);
      std::sort(lorder.begin(), lorder.end(), elt_comp);
      
      // > Lambda function equal elements
      auto is_same_elt = [&](int i, int j) { 
        auto elt_i_beg = elt_val_lex.data()+i*stride;
        auto elt_j_beg = elt_val_lex.data()+j*stride;
        auto elt_i_end = elt_i_beg + stride;
        auto elt_j_end = elt_j_beg + stride;
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
              elt_mask(id) = false;
            }
          }
        }
        else {
          if (compteur!=1) {
            for (int i_elt2=idx_previous; i_elt2<i_elt; ++i_elt2) {
              auto id = order[conflict_idx[i_conflict]+lorder[i_elt2]];
              elt_mask(id) = false;
            }
          }
          idx_previous = i_elt;
          compteur = 1;
        }
      }
    }
  }
  delete[] order;
  delete[] conflict_idx;
  return np_elt_mask;
}