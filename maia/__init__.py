"""
Maia: Distributed algorithms and manipulations over CGNS meshes
"""
from packaging.version import Version, InvalidVersion

__version__ = '1.5.dev'

import Pypdm.Pypdm as PDM

pdm_has_parmetis = PDM.pdm_has_parmetis
pdm_has_ptscotch = PDM.pdm_has_ptscotch
npy_pdm_gnum_dtype = PDM.npy_pdm_gnum_dtype
pdma_enabled = PDM.pdm_has_pdma

_PDM_VERSION = PDM.__version__.replace('.untagged', '')
try:
  PDM_VERSION = Version(_PDM_VERSION)
except InvalidVersion:
  PDM_VERSION = Version(_PDM_VERSION[:5])

from maia import algo
from maia import factory
from maia import io
from maia import pytree
from maia import transfer
from maia import utils

# Change the default Python handling of uncaught exceptions
# By default, if one proc raises an uncaught exception, it may lead to deadlocks
# With `enable_mpi_excepthook`, if one proc raises an uncaught exception, MPI_Abort(1) is called
from maia.utils.parallel import excepthook
excepthook.enable_mpi_excepthook()
