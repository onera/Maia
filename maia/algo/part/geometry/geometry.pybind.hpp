#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// --------------------------------------------------------------------
pybind11::array_t<double, pybind11::array::f_style>
compute_center_cell_u(int n_cell,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cx,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cy,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cz,
                      pybind11::array_t<int,    pybind11::array::f_style>& np_face_vtx,
                      pybind11::array_t<int,    pybind11::array::f_style>& np_face_vtx_idx,
                      pybind11::array_t<int,    pybind11::array::f_style>& np_parent_elements);

// --------------------------------------------------------------------
pybind11::array_t<double, pybind11::array::f_style>
compute_center_cell_s(int nx, int ny, int nz,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cx,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cy,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cz);

// --------------------------------------------------------------------
pybind11::array_t<double, pybind11::array::f_style>
compute_center_face_s(int nx, int ny, int nz,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cx,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cy,
                      pybind11::array_t<double, pybind11::array::f_style>& np_cz);
