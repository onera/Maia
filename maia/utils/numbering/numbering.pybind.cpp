#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

template<typename g_num>
inline g_num n_face_glob(py::array_t<g_num> &vtx_size) {
  const g_num *n_vtx = vtx_size.data();
  return n_vtx[0]*(n_vtx[1]-1)*(n_vtx[2]-1) +
         n_vtx[1]*(n_vtx[0]-1)*(n_vtx[2]-1) +
         n_vtx[2]*(n_vtx[0]-1)*(n_vtx[1]-1);
}

/* Generate a distributed ngon connectivity between the indicated face gnum ids for
 * a zone of a given size.
 * Faces will be generated for global id between
 *   [begin; endI[ for i-normal faces   Examples :
 *   [endI; endJ[  for j-normal faces    * [100, 200, 300, 300] -> generate ifaces 100-200 and jface 200-300
 *   [endJ; endK[  for k-normal faces    * [300, 300, 300, 400] -> generate kfaces 300-400
 * Size of dist zone must be given as the number of cells (size=3)
*/
template<typename g_num>
void ngon_dconnectivity_from_gnum(g_num begin, g_num endI, g_num endJ, g_num endK,
                                  py::array_t<g_num> &zone_size,
                                  py::array_t<g_num, py::array::f_style>& pe,
                                  py::array_t<g_num, py::array::f_style>& face_vtx) {

  //Some checks
  int n_face_loc = endK - begin;
  g_num n_face_tot = n_face_glob(zone_size);
  assert (begin <= endI && endI <= endJ && endJ <= endK);
  assert (face_vtx.ndim() == 1 && face_vtx.shape()[0] == 4*n_face_loc);
  assert (pe.ndim() == 2 && pe.shape()[0] == n_face_loc && pe.shape()[1] == 2);

  const g_num *n_vtx = zone_size.data();
  const g_num n_cell[] = {n_vtx[0]-1, n_vtx[1]-1, n_vtx[2]-1};

  auto pe_ptr       = pe      .template mutable_unchecked<2>();
  auto face_vtx_ptr = face_vtx.template mutable_unchecked<1>();

  //Manage i oriented faces
  g_num gface = begin; //Global number of iface
  for (int i = 0; i < endI - begin; ++i) {
    g_num line_nb   = (gface-1) / n_vtx[0];
    g_num plane_nb  = (gface-1) / (n_vtx[0]*n_cell[1]);
    bool is_min_bnd  = (gface%n_vtx[0] == 1);
    bool is_max_bnd  = (gface%n_vtx[0] == 0);
    bool is_internal = !is_min_bnd & !is_max_bnd;

    //Internal faces : left, right = idx-line_number-1, idx-line_number
    //Min faces      : left        = idx-line_number
    //Max faces      : left        = idx-line_number-1
    pe_ptr(i, 0) = (gface - line_nb + n_face_tot) - 1 + is_min_bnd;
    pe_ptr(i, 1) = (gface - line_nb + n_face_tot)*is_internal;

    g_num n1 = gface + plane_nb*n_vtx[0];
    face_vtx_ptr(4*i+0) = n1;
    face_vtx_ptr(4*i+2) = n1 + n_vtx[0] + n_vtx[0]*n_vtx[1];
    if (is_min_bnd) {
      face_vtx_ptr(4*i+3) = n1 + n_vtx[0];
      face_vtx_ptr(4*i+1) = n1 + n_vtx[0]*n_vtx[1];
    }
    else {
      face_vtx_ptr(4*i+1) = n1 + n_vtx[0];
      face_vtx_ptr(4*i+3) = n1 + n_vtx[0]*n_vtx[1];
    }
    gface++;
  }

  //Manage j oriented faces
  g_num nf_i = n_vtx[0]*n_cell[1]*n_cell[2];
  g_num nb_face_ij  = n_vtx[1] * n_cell[0];
  gface = endI - nf_i; //Global number of jface
  for (int i = endI - begin; i < endJ - begin; ++i) {
    g_num line_nb  = (gface-1) / n_cell[0];
    g_num plane_nb = (gface-1) / nb_face_ij;
    bool is_min_bnd  = (gface - plane_nb*nb_face_ij) < n_vtx[0];
    bool is_max_bnd  = (gface - plane_nb*nb_face_ij) > nb_face_ij - n_vtx[0] + 1;
    bool is_internal = !is_min_bnd & !is_max_bnd;
    
    //Internal faces : left, right = idx - n_cell[0]*plan_number-n_cell[0], idx - n_cell[0]*plan_number
    //Min faces      : left        = idx - n_cell[0]*plan_number
    //Max faces      : left        = idx - n_cell[0]*plan_number-n_cell[0]
    pe_ptr(i, 0) = (gface - n_cell[0]*plane_nb + n_face_tot)-  n_cell[0]*(1-is_min_bnd);
    pe_ptr(i, 1) = (gface - n_cell[0]*plane_nb + n_face_tot)*is_internal;

    g_num n1 = gface + line_nb;
    face_vtx_ptr(4*i+0) = n1;
    face_vtx_ptr(4*i+2) = n1 + n_vtx[0]*n_vtx[1] + 1;
    if (is_min_bnd) {
      face_vtx_ptr(4*i+3) = n1 + n_vtx[0]*n_vtx[1];
      face_vtx_ptr(4*i+1) = n1 + 1;
    }
    else {
      face_vtx_ptr(4*i+1) = n1 + n_vtx[0]*n_vtx[1];
      face_vtx_ptr(4*i+3) = n1 + 1;
    }
    gface++;
  }

  //Manage k oriented faces
  g_num nf_j = n_vtx[1]*n_cell[0]*n_cell[2];
  nb_face_ij =  n_cell[0] * n_cell[1];
  gface = endJ - nf_i - nf_j; //Global number of kface
  for (int i = endJ - begin ; i < endK - begin; ++i) {
    g_num line_nb = (gface - 1) / n_cell[0];
    g_num plan_nb = (gface - 1) / nb_face_ij;
    bool is_min_bnd  = gface <= nb_face_ij;
    bool is_max_bnd  = gface >  nb_face_ij*n_cell[2];
    bool is_internal = !is_min_bnd & !is_max_bnd;
    
    //Internal faces : left, right = idx - nb_face_ij, idx
    //Min faces      : left        = idx
    //Max faces      : left        = idx - nb_face_ij
    pe_ptr(i, 0) =  gface - nb_face_ij*(1-is_min_bnd) + n_face_tot;
    pe_ptr(i, 1) =  (gface + n_face_tot) * is_internal;

    g_num n1 = gface + line_nb + n_vtx[0]*plan_nb;
    face_vtx_ptr(4*i+0) = n1;
    face_vtx_ptr(4*i+2) = n1 + n_vtx[0] + 1;
    if (is_min_bnd) {
      face_vtx_ptr(4*i+3) = n1 + 1;
      face_vtx_ptr(4*i+1) = n1 + n_vtx[0];
    }
    else {
      face_vtx_ptr(4*i+1) = n1 + 1;
      face_vtx_ptr(4*i+3) = n1 + n_vtx[0];
    }
    gface++;
  }
}


void register_numbering_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("numbering");
  m.doc() = "Numbering functions for structured meshes";

  m.def("ngon_dconnectivity_from_gnum", &ngon_dconnectivity_from_gnum<int32_t>,
        "beginI"_a.noconvert(), "endI"_a.noconvert(), "endJ"_a.noconvert(), "endK"_a.noconvert(),
        "zone_size"_a.noconvert(), "face_pe"_a.noconvert(), "face_vtx"_a.noconvert(),
        "Generate NGon connectivity and parent element from global numbering bounds");
  m.def("ngon_dconnectivity_from_gnum", &ngon_dconnectivity_from_gnum<int64_t>,
        "beginI"_a.noconvert(), "endI"_a.noconvert(), "endJ"_a.noconvert(), "endK"_a.noconvert(),
        "zone_size"_a.noconvert(), "face_pe"_a.noconvert(), "face_vtx"_a.noconvert(),
        "Generate NGon connectivity and parent element from global numbering bounds");
}
