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

template<typename g_num>
inline g_num n_edge_glob(py::array_t<g_num> &vtx_size) {
  const g_num *n_vtx = vtx_size.data();
  return n_vtx[0]*(n_vtx[1]-1) + n_vtx[1]*(n_vtx[0]-1);
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

template<typename g_num>
void edge_dconnectivity_from_gnum(g_num begin, g_num endI, g_num endJ,
                                  py::array_t<g_num> &zone_size,
                                  py::array_t<g_num, py::array::f_style>& pe,
                                  py::array_t<g_num, py::array::f_style>& edge_vtx) {

  //Some checks
  int n_edge_loc = endJ - begin;
  g_num n_edge_tot = n_edge_glob(zone_size);
  assert (begin <= endI && endI <= endJ);
  assert (edge_vtx.ndim() == 1 && edge_vtx.shape()[0] == 2*n_edge_loc);
  assert (pe.ndim() == 2 && pe.shape()[0] == n_edge_loc && pe.shape()[1] == 2);

  const g_num *n_vtx = zone_size.data();
  const g_num n_cell[] = {n_vtx[0]-1, n_vtx[1]-1};

  auto pe_ptr       = pe      .template mutable_unchecked<2>();
  auto edge_vtx_ptr = edge_vtx.template mutable_unchecked<1>();

  //Manage i oriented edges
  g_num gedge = begin; //Global number of iface
  for (int i = 0; i < endI - begin; ++i) {
    g_num line_nb   = (gedge-1) / n_vtx[0];
    bool is_min_bnd  = (gedge%n_vtx[0] == 1);
    bool is_max_bnd  = (gedge%n_vtx[0] == 0);
    bool is_internal = !is_min_bnd & !is_max_bnd;

    //Internal edges : left, right = idx-line_number-1, idx-line_number
    //Min edges      : left        = idx-line_number
    //Max edges      : left        = idx-line_number-1
    pe_ptr(i, 0) = (gedge - line_nb + n_edge_tot) - 1 + is_min_bnd;
    pe_ptr(i, 1) = (gedge - line_nb + n_edge_tot)*is_internal;

    g_num n1 = gedge;
    g_num n2 = n1 + n_vtx[0];
    if (is_min_bnd) {
      edge_vtx_ptr(2*i+0) = n2;
      edge_vtx_ptr(2*i+1) = n1;
    }
    else {
      edge_vtx_ptr(2*i+0) = n1;
      edge_vtx_ptr(2*i+1) = n2;
    }
    gedge++;
  }

  //Manage j oriented edges
  g_num ne_i = n_vtx[0]*(n_vtx[1]-1);
  g_num nb_edge_j  = n_vtx[1] * n_cell[0];
  gedge = endI - ne_i; //Global number of jface
  for (int i = endI - begin; i < endJ - begin; ++i) {
    g_num line_nb  = (gedge-1) / n_cell[0];
    bool is_min_bnd  = gedge < n_vtx[0];
    bool is_max_bnd  = gedge > nb_edge_j - n_vtx[0] + 1;
    bool is_internal = !is_min_bnd & !is_max_bnd;
    
    //Internal faces : left, right = idx, idx - n_cell[0]
    //Min faces      : left        = idx
    //Max faces      : left        = idx - n_cell[0]
    pe_ptr(i, 0) = (gedge + n_edge_tot) - n_cell[0]*(is_max_bnd);
    pe_ptr(i, 1) = (gedge - n_cell[0] + n_edge_tot)*is_internal;

    g_num n1 = gedge + line_nb;
    g_num n2 = n1 + 1;
    if (is_max_bnd) {
      edge_vtx_ptr(2*i+0) = n2;
      edge_vtx_ptr(2*i+1) = n1;
    }
    else {
      edge_vtx_ptr(2*i+0) = n1;
      edge_vtx_ptr(2*i+1) = n2;
    }
    gedge++;
  }
}

template<typename g_num>
void bar2_connectivity_from_idx(py::array_t<g_num> &edge_idx,
                                py::array_t<g_num> &vtx_size,
                                py::array_t<g_num> &edge_vtx) {

auto _vtx_size = vtx_size.data();
auto _edge_idx = edge_idx.data();
auto _edge_vtx = edge_vtx.mutable_data();


g_num n_edge_i = _vtx_size[0]*(_vtx_size[1]-1);

  for (int i_edge=0; i_edge < edge_idx.size(); ++i_edge) {
    g_num vtx1, vtx2;
    g_num edge = _edge_idx[i_edge];

    // Edge is i-normal
    if (edge <= n_edge_i) {
      g_num j = (edge-1) / _vtx_size[0] + 1;
      g_num i = edge - (j-1)*_vtx_size[0];

      vtx1 = i + (j-1)*_vtx_size[0];
      vtx2 = i + (j  )*_vtx_size[0];

      if (i==1) {
        std::swap(vtx1, vtx2);
      }
    }
    // Edge is j-normal
    else {
      g_num j = (edge - 1 - n_edge_i) / (_vtx_size[0]-1) + 1;
      g_num i = edge - (j-1)*(_vtx_size[0]-1) - n_edge_i;

      vtx1 = i   + (j-1)*_vtx_size[0];
      vtx2 = i+1 + (j-1)*_vtx_size[0];

      if (j==_vtx_size[1]) {
        std::swap(vtx1, vtx2);
      }
    }
    _edge_vtx[2*i_edge+0] = vtx1;
    _edge_vtx[2*i_edge+1] = vtx2;
  }
}

template<typename g_num>
void quad4_connectivity_from_idx(py::array_t<g_num> &face_idx,
                                 py::array_t<g_num> &vtx_size,
                                 py::array_t<g_num> &face_vtx) {

auto _face_idx = face_idx.data();
auto _face_vtx = face_vtx.mutable_data();

auto n_vtx = vtx_size.data();
const g_num n_cell[] = {n_vtx[0]-1, n_vtx[1]-1, n_vtx[2]-1};

g_num n_face_i = n_vtx[0]*n_cell[1]*n_cell[2];
g_num n_face_j = n_vtx[1]*n_cell[0]*n_cell[2];

  for (int i_face=0; i_face < face_idx.size(); ++i_face) {
    g_num vtx1, vtx2, vtx3, vtx4;
    g_num face = _face_idx[i_face];

    // Face is i-normal
    if (face <= n_face_i) {
      g_num k = ((face - 1) / (n_vtx[0]*n_cell[1])) + 1;
      g_num j = (face - (k-1)*(n_vtx[0]*n_cell[1]) - 1) / n_vtx[0] + 1;
      g_num i = face - (j-1)*n_vtx[0] - (k-1)*(n_vtx[0]*n_cell[1]);

      // (i,j,k) (i,j+1,k) (i,j+1,k+1), (i,j,k+1)
      vtx1 = i + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx2 = i + (j  )*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx3 = i + (j  )*n_vtx[0] + (k  )*n_vtx[0]*n_vtx[1];
      vtx4 = i + (j-1)*n_vtx[0] + (k  )*n_vtx[0]*n_vtx[1];

      if (i==1) {
        std::swap(vtx2, vtx4);
      }
    }
    // Face is j-normal
    else if (face <= n_face_i + n_face_j) {

      g_num k = ((face - 1 - n_face_i) / (n_vtx[1]*n_cell[0])) + 1;
      g_num j = (face - (k-1)*(n_vtx[1]*n_cell[0]) - 1 - n_face_i) / n_cell[0] + 1;
      g_num i = face - (j-1)*n_cell[0] - (k-1)*(n_vtx[1]*n_cell[0]) - n_face_i;

      // (i,j,k) (i+1,j,k) (i+1,j,k+1), (i,j,k+1)
      vtx1 = i   + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx2 = i+1 + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx3 = i+1 + (j-1)*n_vtx[0] + (k  )*n_vtx[0]*n_vtx[1];
      vtx4 = i   + (j-1)*n_vtx[0] + (k  )*n_vtx[0]*n_vtx[1];

      if (j==n_vtx[1]) {
        std::swap(vtx1, vtx2);
      }
    }
    // Face is k-normal
    else {
      g_num k = ((face - 1 - n_face_i - n_face_j) / (n_cell[0]*n_cell[1])) + 1;
      g_num j = (face - (k-1)*(n_cell[0]*n_cell[1]) - 1 - n_face_i - n_face_j) / n_cell[0] + 1;
      g_num i = face - (j-1)*n_cell[0] - (k-1)*(n_cell[0]*n_cell[1]) - n_face_i - n_face_j;

      // (i,j,k) (i+1,j,k) (i+1,j+1,k), (i,j+1,k)
      vtx1 = i   + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx2 = i+1 + (j-1)*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx3 = i+1 + (j  )*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];
      vtx4 = i   + (j  )*n_vtx[0] + (k-1)*n_vtx[0]*n_vtx[1];

      if (k==1) {
        std::swap(vtx2, vtx4);
      }
    }
    _face_vtx[4*i_face+0] = vtx1;
    _face_vtx[4*i_face+1] = vtx2;
    _face_vtx[4*i_face+2] = vtx3;
    _face_vtx[4*i_face+3] = vtx4;
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

  m.def("edge_dconnectivity_from_gnum", &edge_dconnectivity_from_gnum<int32_t>,
        "beginI"_a.noconvert(), "endI"_a.noconvert(), "endJ"_a.noconvert(),
        "zone_size"_a.noconvert(), "edge_pe"_a.noconvert(), "edge_vtx"_a.noconvert(),
        "Generate 2D Edge connectivity and parent element from global numbering bounds");
  m.def("edge_dconnectivity_from_gnum", &edge_dconnectivity_from_gnum<int64_t>,
        "beginI"_a.noconvert(), "endI"_a.noconvert(), "endJ"_a.noconvert(),
        "zone_size"_a.noconvert(), "edge_pe"_a.noconvert(), "edge_vtx"_a.noconvert(),
        "Generate 2D Edge connectivity and parent element from global numbering bounds");

  m.def("bar2_connectivity_from_idx", &bar2_connectivity_from_idx<int32_t>,
        "edge_idx"_a.noconvert(), 
        "zone_size"_a.noconvert(), "edge_vtx"_a.noconvert());
  m.def("bar2_connectivity_from_idx", &bar2_connectivity_from_idx<int64_t>,
        "edge_idx"_a.noconvert(), 
        "zone_size"_a.noconvert(), "edge_vtx"_a.noconvert());

  m.def("quad4_connectivity_from_idx", &quad4_connectivity_from_idx<int32_t>,
        "face_idx"_a.noconvert(), 
        "zone_size"_a.noconvert(), "face_vtx"_a.noconvert());
  m.def("quad4_connectivity_from_idx", &quad4_connectivity_from_idx<int64_t>,
        "face_idx"_a.noconvert(), 
        "zone_size"_a.noconvert(), "face_vtx"_a.noconvert());
}
