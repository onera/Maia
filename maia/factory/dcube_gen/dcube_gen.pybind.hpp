#include "pdm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


/* This function has been cherry-picked from the ParaDiGM library, because
 * a correction concerning face orientations will occurs in next PDM
 * release (v2.6), see https://gitlab.onera.net/numerics/mesh/paradigm/-/merge_requests/77
 *
 * Since we would like maia 1.5 to be both compatible with PDM 2.5 and PDM 2.6, 
 * we copied this function to leverage the correction even with PDM 2.5.
 *
 * This function will be removed once compatibility with PDM <= 2.5 is no longer garantee
*/
pybind11::tuple
generate_dcube(PDM_g_num_t n_vtx_seg,
               double      length,
               double      zero_x,
               double      zero_y,
               double      zero_z,
               int         i_rank,
               int         n_rank);