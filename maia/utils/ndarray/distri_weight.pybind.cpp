/*
 * TMP wrapping of PDM_distrib_weight -- To remove when Cython method is avalaible
*/
#include <mpi.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "maia/utils/parallel/mpi4py.hpp"
#include "pdm_distrib.h"

namespace py = pybind11;

py::array_t<PDM_g_num_t>
compute_weighted_distribution(py::list ln_to_gn_list, py::list weights, py::handle py_comm) {
  auto MPI_Comm = maia::mpi4py_comm_to_comm(py_comm);
  int comm_size;
  MPI_Comm_size(MPI_Comm, &comm_size);

  int n_part = len(ln_to_gn_list);
  int sampling_factor = 2;
  int n_iter_max = 5;
  double tolerance = 0.5;

  std::vector<int> n_elmts;
  PDM_g_num_t** _ln_to_gn = (PDM_g_num_t**) malloc(n_part * sizeof(PDM_g_num_t*));
  double**       _weights = (double**)      malloc(n_part * sizeof(double*));

  for (int i=0; i < n_part; ++i) {
    auto lngn  = py::cast<py::array>(ln_to_gn_list[i]);
    auto weight = py::cast<py::array>(weights[i]);

    n_elmts.push_back(lngn.size());
    _ln_to_gn[i] = (PDM_g_num_t*) lngn.data();
    _weights[i]  = (double*)      weight.data();
  }

  PDM_g_num_t* distrib = NULL;
  PDM_distrib_weight(sampling_factor,
                     comm_size,
                     n_part,
                     n_elmts.data(),
(const PDM_g_num_t**)_ln_to_gn,
    (const double**) _weights,
                     n_iter_max,
                     tolerance,
                     PDM_MPI_mpi_2_pdm_mpi_comm(&MPI_Comm),
                     &distrib);
  free(_ln_to_gn);
  free(_weights);

  auto out = py::array_t<PDM_g_num_t>(comm_size+1);
  auto out_ptr = out.template mutable_unchecked<1>();
  for (int i = 0; i < comm_size+1; i++) {
    out_ptr(i) = distrib[i];
  }
  free(distrib);
  return out;
}

