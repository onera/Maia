#include <stdlib.h>
#include "maia/factory/dcube_gen/dcube_gen.pybind.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

py::tuple
generate_dcube(PDM_g_num_t n_vtx_seg,
               double      length,
               double      zero_x,
               double      zero_y,
               double      zero_z,
               int         i_rank,
               int         n_rank) {

  PDM_g_num_t n_vtx       = n_vtx_seg * n_vtx_seg * n_vtx_seg;
  PDM_g_num_t n_face_seg  = n_vtx_seg - 1;
  PDM_g_num_t n_face      = 3 * n_face_seg * n_face_seg * n_vtx_seg;
  PDM_g_num_t n_cell      = n_face_seg * n_face_seg * n_face_seg;
  PDM_g_num_t n_face_face = n_face_seg * n_face_seg;
  PDM_g_num_t n_vtx_face  = n_vtx_seg * n_vtx_seg;
  PDM_g_num_t n_face_lim  = 6 * n_face_face;
  double step = length / (double) n_face_seg;
  PDM_g_num_t *distrib_vtx      = (PDM_g_num_t *) malloc((n_rank + 1) * sizeof(PDM_g_num_t));
  PDM_g_num_t *distrib_face     = (PDM_g_num_t *) malloc((n_rank + 1) * sizeof(PDM_g_num_t));
  PDM_g_num_t *distrib_cell     = (PDM_g_num_t *) malloc((n_rank + 1) * sizeof(PDM_g_num_t));
  PDM_g_num_t *distrib_face_lim = (PDM_g_num_t *) malloc((n_rank + 1) * sizeof(PDM_g_num_t));


  distrib_vtx[0]      = 0;
  distrib_face[0]     = 0;
  distrib_cell[0]     = 0;
  distrib_face_lim[0] = 0;

  PDM_g_num_t step_vtx      = n_vtx / n_rank;
  PDM_g_num_t remainder_vtx = n_vtx % n_rank;

  PDM_g_num_t step_face      = n_face / n_rank;
  PDM_g_num_t remainder_face = n_face % n_rank;

  PDM_g_num_t step_cell      = n_cell / n_rank;
  PDM_g_num_t remainder_cell = n_cell % n_rank;

  PDM_g_num_t step_face_im       = n_face_lim / n_rank;
  PDM_g_num_t remainder_face_lim = n_face_lim % n_rank;

  for (int i = 1; i < n_rank + 1; i++) {
    distrib_vtx[i]     = step_vtx;
    distrib_face[i]    = step_face;
    distrib_cell[i]    = step_cell;
    distrib_face_lim[i] = step_face_im;
    const int i1 = i - 1;
    if (i1 < remainder_vtx)
      distrib_vtx[i]  += 1;
    if (i1 < remainder_face)
      distrib_face[i]  += 1;
    if (i1 < remainder_cell)
      distrib_cell[i]  += 1;
    if (i1 < remainder_face_lim)
      distrib_face_lim[i]  += 1;
  }

  for (int i = 1; i < n_rank + 1; i++) {
    distrib_vtx[i]  += distrib_vtx[i-1];
    distrib_face[i] += distrib_face[i-1];
    distrib_cell[i] += distrib_cell[i-1];
    distrib_face_lim[i] += distrib_face_lim[i-1];
  }

  int n_face_group = 6;

  PDM_g_num_t _dn_cell = distrib_cell[i_rank+1] - distrib_cell[i_rank];
  PDM_g_num_t _dn_face = distrib_face[i_rank+1]    - distrib_face[i_rank];
  PDM_g_num_t _dn_vtx  = distrib_vtx[i_rank+1]     - distrib_vtx[i_rank];
  PDM_g_num_t _dn_face_lim = distrib_face_lim[i_rank+1] - distrib_face_lim[i_rank];
  int dn_cell     = (int) _dn_cell;
  int dn_face     = (int) _dn_face;
  int dn_vtx      = (int) _dn_vtx;
  int dn_face_lim = (int) _dn_face_lim;

  auto np_dface_cell      = py::array_t<PDM_g_num_t>(2*dn_face);
  auto np_dface_vtx_idx   = py::array_t<int        >(dn_face + 1);
  auto np_dface_vtx       = py::array_t<PDM_g_num_t>(4*dn_face);
  auto np_dvtx_coord      = py::array_t<double     >(3*dn_vtx);
  auto np_dface_group_idx = py::array_t<int        >(n_face_group + 1);
  auto np_dface_group     = py::array_t<PDM_g_num_t>(dn_face_lim);

  auto dface_cell      = np_dface_cell     .mutable_data();
  auto dface_vtx_idx   = np_dface_vtx_idx  .mutable_data();
  auto dface_vtx       = np_dface_vtx      .mutable_data();
  auto dvtx_coord      = np_dvtx_coord     .mutable_data();
  auto dface_group_idx = np_dface_group_idx.mutable_data();
  auto dface_group     = np_dface_group    .mutable_data();

  dface_vtx_idx[0] = 0;
  for (int i = 1; i < dn_face + 1; i++) {
    dface_vtx_idx[i] = 4 + dface_vtx_idx[i-1];
  }
  //
  // Coordinates

  const PDM_g_num_t b_vtx_z = distrib_vtx[i_rank] / n_vtx_face;
  const PDM_g_num_t r_vtx_z = distrib_vtx[i_rank] % n_vtx_face;

  const PDM_g_num_t b_vtx_y = r_vtx_z / n_vtx_seg;
  const PDM_g_num_t b_vtx_x = r_vtx_z % n_vtx_seg;

  int i_vtx = 0;
  int cpt   = 0;

  for(PDM_g_num_t k = b_vtx_z; k < n_vtx_seg; k++) {
    PDM_g_num_t _b_vtx_y = 0;
    if (k == b_vtx_z)
      _b_vtx_y = b_vtx_y;
    for(PDM_g_num_t j = _b_vtx_y; j < n_vtx_seg; j++) {
      PDM_g_num_t _b_vtx_x = 0;
      if ((k == b_vtx_z) && (j == b_vtx_y))
        _b_vtx_x = b_vtx_x;
      for(PDM_g_num_t i = _b_vtx_x; i < n_vtx_seg; i++) {
        dvtx_coord[3 * i_vtx    ] = i * step + zero_x;
        dvtx_coord[3 * i_vtx + 1] = j * step + zero_y;
        dvtx_coord[3 * i_vtx + 2] = k * step + zero_z;
        cpt   += 1;
        i_vtx += 1;
        if (cpt == dn_vtx)
          break;
      }
      if (cpt == dn_vtx)
        break;
    }
    if (cpt == dn_vtx)
      break;
  }


  //
  // face_vtx et face_cell

  cpt = 0;

  PDM_g_num_t serie   = n_face / 3;
  PDM_g_num_t i_serie = distrib_face[i_rank] / serie;
  PDM_g_num_t r_serie = distrib_face[i_rank] % serie;

  PDM_g_num_t b1 = 0;
  PDM_g_num_t r1 = 0;

  PDM_g_num_t b2 = 0;
  PDM_g_num_t b3 = 0;

  b1 = r_serie / n_face_face;
  r1 = r_serie % n_face_face;

  b2 = r1 / n_face_seg;
  b3 = r1 % n_face_seg;


  if (i_serie == 0) {
  /* switch (i_serie) { */

  /* case 0 : */

    //
    // Faces zmin -> zmax

    for(PDM_g_num_t k = b1; k < n_vtx_seg; k++) {
      PDM_g_num_t _b2 = 0;
      if (k == b1)
        _b2 = b2;
      for(PDM_g_num_t j = _b2; j < n_face_seg; j++) {
        PDM_g_num_t _b3 = 0;
        if ((k == b1) && (j == b2))
          _b3 = b3;
        for(PDM_g_num_t i = _b3; i < n_face_seg; i++) {

          dface_vtx[cpt * 4    ] = k * n_vtx_seg * n_vtx_seg + (    j * n_vtx_seg + i + 1);
          dface_vtx[cpt * 4 + 1] = k * n_vtx_seg * n_vtx_seg + ((j+1) * n_vtx_seg + i + 1);
          dface_vtx[cpt * 4 + 2] = k * n_vtx_seg * n_vtx_seg + ((j+1) * n_vtx_seg + i + 2);
          dface_vtx[cpt * 4 + 3] = k * n_vtx_seg * n_vtx_seg + (    j * n_vtx_seg + i + 2);

          if (k == 0) {
            dface_cell[2*cpt + 0] = j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = 0;

          } else if (k == n_face_seg) {

            dface_vtx[cpt * 4    ] = k * n_vtx_seg * n_vtx_seg + (    j * n_vtx_seg + i + 2);
            dface_vtx[cpt * 4 + 1] = k * n_vtx_seg * n_vtx_seg + ((j+1) * n_vtx_seg + i + 2);
            dface_vtx[cpt * 4 + 2] = k * n_vtx_seg * n_vtx_seg + ((j+1) * n_vtx_seg + i + 1);
            dface_vtx[cpt * 4 + 3] = k * n_vtx_seg * n_vtx_seg + (    j * n_vtx_seg + i + 1);

            dface_cell[2*cpt + 0] = (k-1) * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = 0;

          } else {
            dface_cell[2*cpt + 0] =     k * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = (k-1) * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
          }
          cpt += 1;
          if (cpt == dn_face)
            break;
        }
        if (cpt == dn_face)
          break;
      }
      if (cpt == dn_face)
        break;
    }
    b1 = 0;
    b2 = 0;
    b3 = 0;
  }


  /* if (cpt == dn_face) */
  /*     break; */

  if ((i_serie == 1) || ((i_serie == 0) &&  (cpt != dn_face))) {

    //
    // Faces xmin -> xmax

    for(PDM_g_num_t i = b1; i < n_vtx_seg; i++) {
      PDM_g_num_t _b2 = 0;
      if (i == b1)
        _b2 = b2;
      for(PDM_g_num_t k = _b2; k < n_face_seg; k++) {
        PDM_g_num_t _b3 = 0;
        if ((i == b1) && (k == b2))
          _b3 = b3;
        for(PDM_g_num_t j = _b3; j < n_face_seg; j++) {

          dface_vtx[cpt * 4    ] = (k+1) * n_vtx_seg * n_vtx_seg +     j * n_vtx_seg + i + 1;
          dface_vtx[cpt * 4 + 1] = (k+1) * n_vtx_seg * n_vtx_seg + (j+1) * n_vtx_seg + i + 1;
          dface_vtx[cpt * 4 + 2] =     k * n_vtx_seg * n_vtx_seg + (j+1) * n_vtx_seg + i + 1;
          dface_vtx[cpt * 4 + 3] =     k * n_vtx_seg * n_vtx_seg +     j * n_vtx_seg + i + 1;

          if (i == 0) {
            dface_cell[2*cpt + 0] = k * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = 0;

          } else if (i == n_face_seg) {

            dface_vtx[cpt * 4    ] =     k * n_vtx_seg * n_vtx_seg +     j * n_vtx_seg + i + 1;
            dface_vtx[cpt * 4 + 1] =     k * n_vtx_seg * n_vtx_seg + (j+1) * n_vtx_seg + i + 1;
            dface_vtx[cpt * 4 + 2] = (k+1) * n_vtx_seg * n_vtx_seg + (j+1) * n_vtx_seg + i + 1;
            dface_vtx[cpt * 4 + 3] = (k+1) * n_vtx_seg * n_vtx_seg +     j * n_vtx_seg + i + 1;

            dface_cell[2*cpt + 0] = k * n_face_seg * n_face_seg + j * n_face_seg + i;
            dface_cell[2*cpt + 1] = 0;

          } else {
            dface_cell[2*cpt + 0] = k * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = k * n_face_seg * n_face_seg + j * n_face_seg + i ;
          }
          cpt += 1;
          if (cpt == dn_face)
            break;
        }
        if (cpt == dn_face)
          break;
      }
      if (cpt == dn_face)
        break;
    }
    b1 = 0;
    b2 = 0;
    b3 = 0;
  }


  /* if (cpt == dcube->dn_face) */
  /*   break; */

  if ((i_serie == 2) || ((i_serie == 1 || i_serie == 0) && (cpt != dn_face))) {
    /* case 2 : */

    //
    // Faces ymin -> ymax

    for(PDM_g_num_t j = b1; j < n_vtx_seg; j++) {
      PDM_g_num_t _b2 = 0;
      if (j == b1)
        _b2 = b2;
      for(PDM_g_num_t i = _b2; i < n_face_seg; i++) {
        PDM_g_num_t _b3 = 0;
        if ((j == b1) && (i == b2))
          _b3 = b3;
        for(PDM_g_num_t k = _b3; k < n_face_seg; k++) {
          dface_vtx[cpt * 4    ] =     k * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1    ;
          dface_vtx[cpt * 4 + 1] =     k * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1 + 1;
          dface_vtx[cpt * 4 + 2] = (k+1) * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1 + 1;
          dface_vtx[cpt * 4 + 3] = (k+1) * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1    ;

          if (j == 0) {
            dface_cell[2*cpt + 0] = k * n_face_seg * n_face_seg + j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = 0;

          } else if (j == n_face_seg) {

            dface_vtx[cpt * 4    ] = (k+1) * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1    ;
            dface_vtx[cpt * 4 + 1] = (k+1) * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1 + 1;
            dface_vtx[cpt * 4 + 2] =     k * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1 + 1;
            dface_vtx[cpt * 4 + 3] =     k * n_vtx_seg * n_vtx_seg + j * n_vtx_seg + i + 1    ;

            dface_cell[2*cpt + 0] =  k * n_face_seg * n_face_seg + (j-1) * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = 0;

          } else {
            dface_cell[2*cpt + 0] = k * n_face_seg * n_face_seg +     j * n_face_seg + i + 1;
            dface_cell[2*cpt + 1] = k * n_face_seg * n_face_seg + (j-1) * n_face_seg + i + 1;
          }
          cpt += 1;
          if (cpt == dn_face)
            break;
        }
        if (cpt == dn_face)
          break;
      }
      if (cpt == dn_face)
        break;
    }
  }

  //
  // Faces limite

  cpt = 0;
  PDM_g_num_t b_face;
  int cpt1 = 0;
  int cpt3 = 0;
  int first_group = 0;

  serie   = n_face_lim / n_face_group;
  i_serie = distrib_face_lim[i_rank] / serie;
  r_serie = distrib_face_lim[i_rank] % serie;

  for (int i = 0; i < n_face_group + 1; i++)
    dface_group_idx[i] = 0;


  //  switch (i_serie) {

  if (i_serie == 0) {

    // case 0 :

    //
    // Faces zmin

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = 0;

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

    dface_group_idx[1] = cpt - cpt1;

    /* if (cpt == dn_face_lim) */
    /*   break; */
    first_group = 0;
  }


  if ((i_serie == 1) || ((i_serie == 0) && (cpt != dn_face_lim))) {
    //  case 1 :

    //
    // Faces zmax

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = n_face_seg * n_face_seg * n_face_seg;

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

    dface_group_idx[2] = cpt - cpt1;
    first_group = 0;
  }
    /* if (cpt == dn_face_lim) */
    /*   break; */


  if ((i_serie == 2) || (((i_serie == 0) || (i_serie == 1)) && (cpt != dn_face_lim))) {
    //  case 2 :

    //
    // Faces xmin

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = n_face_seg * n_face_seg * n_vtx_seg;

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

     dface_group_idx[3] = cpt - cpt1;
     first_group = 0;
  }
    /* if (cpt == dn_face_lim) */
    /*   break; */


  if ((i_serie == 3) || (((i_serie == 0) || (i_serie == 1)  || (i_serie == 2)) && (cpt != dn_face_lim))) {
    //  case 3 :

    //
    // Faces xmax

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = n_face_seg * n_face_seg * (n_vtx_seg + n_face_seg);

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

    dface_group_idx[4] = cpt - cpt1;
    first_group = 0;
  }
    /* if (cpt == dn_face_lim) */
    /*   break; */


  if ((i_serie == 4) || (((i_serie == 0) || (i_serie == 1)  || (i_serie == 2) || (i_serie == 3)) && (cpt != dn_face_lim))) {
    //  case 4 :

    //
    // Faces ymin

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = n_face_seg * n_face_seg * (n_vtx_seg + n_vtx_seg);

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

    dface_group_idx[5] = cpt - cpt1;
    first_group = 0;
  }

    /* if (cpt == dn_face_lim) */
    /*   break; */


  if ((i_serie == 5) || (((i_serie == 0) || (i_serie == 1)  || (i_serie == 2) || (i_serie == 3) || (i_serie == 4)) && (cpt != dn_face_lim))) {
  /* case 5 : */

    //
    // Faces ymax

    if (cpt == 0)
      first_group = 1;

    cpt1 = cpt;

    b_face = n_face_seg * n_face_seg * (n_vtx_seg + n_vtx_seg + n_face_seg);

    cpt3 = 0;
    for(PDM_g_num_t j = 0; j < n_face_seg; j++) {
      for(PDM_g_num_t i = 0; i < n_face_seg; i++) {
        cpt3 += 1;
        if (!first_group || (first_group && ((cpt3 - 1)  >= r_serie))) {
          dface_group[cpt] = b_face + j * n_face_seg + i + 1;
          cpt += 1;
          if (cpt == dn_face_lim)
            break;
        }
      }
      if (cpt == dn_face_lim)
        break;
    }

  dface_group_idx[6] = cpt - cpt1;
  first_group = 0;

  }


  for (int i = 1; i < n_face_group + 1; i++)
    dface_group_idx[i] += dface_group_idx[i-1];

  free(distrib_vtx);
  free(distrib_face);
  free(distrib_cell);
  free(distrib_face_lim);


  py::dict dcube_dims("n_face_group"_a = n_face_group,
                      "dn_cell"_a = dn_cell,
                      "dn_face"_a = dn_face,
                      "dn_vtx"_a = dn_vtx,
                      "sface_vtx"_a = dface_vtx_idx[dn_face],
                      "sface_group"_a = dface_group_idx[n_face_group]);

  py::dict dcube_vals("dface_cell"_a      = np_dface_cell,
                      "dface_vtx_idx"_a   = np_dface_vtx_idx,
                      "dface_vtx"_a       = np_dface_vtx,
                      "dvtx_coord"_a      = np_dvtx_coord,
                      "dface_group_idx"_a = np_dface_group_idx,
                      "dface_group"_a     = np_dface_group);

  return py::make_tuple(dcube_dims, dcube_vals);
}

