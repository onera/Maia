#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T> auto
make_raw_view(py::array_t<T, py::array::f_style>& x){
  py::buffer_info buf = x.request();
  return static_cast<T*>(buf.ptr);
}

// --------------------------------------------------------------------
std::tuple<py::array_t<int, py::array::f_style>, py::array_t<int, py::array::f_style>>
local_pe_to_local_cellface(py::array_t<int, py::array::f_style>& np_pe)
{
  int n_face = np_pe.shape()[0];
  auto pe = make_raw_view(np_pe);

  // Find number of cells == max of PE
  int n_cell = 0;
  for (int iface=0; iface < 2*n_face; ++iface) {
    n_cell = std::max(n_cell, pe[iface]);
  }
  
  // Count the number of occurences of each cell
  std::vector<int> counts(n_cell+1, 0);
  for (int iface=0; iface < 2*n_face; ++iface) {
    counts[pe[iface]]++;
  }
  
  //Compute eso
  std::vector<int> eso(n_cell+1, 0);
  eso[0] = 0;
  std::partial_sum(counts.begin()+1, counts.end(), eso.begin()+1);

  //Allocate ec
  py::array_t<int, py::array::f_style> np_ec(eso[n_cell]);
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
