#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// --------------------------------------------------------------------
pybind11::array_t<int, pybind11::array::f_style>
adapt_match_information(pybind11::array_t<int, pybind11::array::f_style>& np_neighbor_idx,
                        pybind11::array_t<int, pybind11::array::f_style>& np_neighbor_desc,
                        pybind11::array_t<int, pybind11::array::f_style>& np_recv_entity_stri,
                        pybind11::array_t<int, pybind11::array::f_style>& np_point_list,
                        pybind11::array_t<int, pybind11::array::f_style>& np_point_list_donor);

// --------------------------------------------------------------------
std::tuple<pybind11::array_t<double, pybind11::array::f_style>,
           pybind11::array_t<double, pybind11::array::f_style>>
compute_face_center_and_characteristic_length(pybind11::array_t<int   , pybind11::array::f_style>& np_point_list,
                                              pybind11::array_t<double, pybind11::array::f_style>& np_cx,
                                              pybind11::array_t<double, pybind11::array::f_style>& np_cy,
                                              pybind11::array_t<double, pybind11::array::f_style>& np_cz,
                                              pybind11::array_t<int   , pybind11::array::f_style>& np_face_vtx,
                                              pybind11::array_t<int   , pybind11::array::f_style>& np_face_vtx_idx);

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
