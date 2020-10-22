#pragma once


#include <vector>
#include "pdm.h"


/*
concept Global_numbering
  Contiguous_range
  value_type==PDM_g_num_t
*/
/*
concept Global_numberings = Contiguous_range<Global_numbering>
*/
using global_numbering = std::vector<PDM_g_num_t>;
using global_numberings = std::vector<global_numbering>;
