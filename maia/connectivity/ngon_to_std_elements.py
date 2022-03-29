from cmaia.transform import transform as ctransform
from maia.transform.apply_function_to_nodes import apply_to_zones
import Converter.Internal as I
from maia import transform
from mpi4py import MPI


def ngon_to_std_elements(t,comm=MPI.COMM_WORLD):
  """
  Transform a pytree so that ngon elements are converted to standard elements
  The ngon elements are supposed to describe only standard elements
  (i.e. tris, quads, tets, pyras, prisms and hexas only)
  """
  apply_to_zones(t,ctransform.convert_zone_to_std_elements)

