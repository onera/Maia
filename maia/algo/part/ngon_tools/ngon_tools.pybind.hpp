#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "maia/utils/pybind_utils.hpp"

namespace py = pybind11;

// --------------------------------------------------------------------
template<typename T>
inline
std::tuple<py::array_t<T, py::array::f_style>, py::array_t<T, py::array::f_style>>
local_pe_to_local_cellface(py::array_t<T, py::array::f_style>& np_pe)
{
  int n_face = np_pe.shape()[0];
  auto pe = make_raw_view(np_pe);

  // Find number of cells == max of PE
  int n_cell = std::accumulate(pe, pe+2*n_face, T(0), [](auto x, auto y){ return std::max(x,y); });
  
  // Count the number of occurences of each cell
  std::vector<int> counts(n_cell+1, 0);
  for (int iface=0; iface < 2*n_face; ++iface) {
    counts[pe[iface]]++;
  }
  
  //Compute eso
  std::vector<T> eso(n_cell+1, 0);
  eso[0] = 0;
  std::partial_sum(counts.begin()+1, counts.end(), eso.begin()+1);

  //Allocate ec
  py::array_t<T, py::array::f_style> np_ec(eso[n_cell]);
  auto ec = make_raw_view(np_ec);


  // Now fill
  std::fill(counts.begin(), counts.end(), 0);
  for (int iface=0; iface < n_face; ++iface) {
    int icell = pe[iface];
    if (icell > 0) {
      ec[eso[icell-1]+counts[icell]++] = iface + 1;
    }
  }
  for (int iface=0; iface < n_face; ++iface) {
    int icell = pe[n_face + iface];
    if (icell > 0) {
      ec[eso[icell-1]+counts[icell]++] = -(iface + 1);
    }
  }
  return std::make_tuple(py::cast(eso), np_ec);
}

// --------------------------------------------------------------------
inline
py::array_t<int, py::array::f_style>
local_cellface_to_local_pe(py::array_t<int, py::array::f_style>& np_cellface_idx,
                           py::array_t<int, py::array::f_style>& np_cellface)
{
  int n_cell = np_cellface_idx.size() - 1;
  auto cellface_idx = np_cellface_idx.unchecked<1>();
  auto cellface     = np_cellface.unchecked<1>();
  
  // Find number of faces == max of EC
  int n_face = 0;
  for (int i=0; i < np_cellface.size(); ++i) {
    n_face = std::max(n_face, std::abs(cellface[i]));
  }

  // Declare and init PE
  py::array_t<int, py::array::f_style> np_pe({n_face,2});
  auto pe = make_raw_view(np_pe);
  for (int i=0; i < 2*n_face; i++) {
    pe[i] = 0;
  }

  //Fill PE
  for (int icell=0; icell < n_cell; ++icell) {
    for (int j=cellface_idx[icell]; j < cellface_idx[icell+1]; j++) {
      int iface = cellface[j];
      if (iface > 0) {
        pe[std::abs(iface)-1] = icell+1;
      }
      else {
        pe[std::abs(iface) -1 + n_face] = icell+1;
      }
    }
  }

  return np_pe;
}
