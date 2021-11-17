import Pypdm.Pypdm as PDM

pdm_has_parmetis = PDM.pdm_has_parmetis
pdm_has_ptscotch = PDM.pdm_has_ptscotch
npy_pdm_gnum_dtype = PDM.npy_pdm_gnum_dtype

#Todo : export argument in PDM build
pdma_enabled = 'DistCellCenterSurf' in dir(PDM)

from .utils.meta import for_all_methods
