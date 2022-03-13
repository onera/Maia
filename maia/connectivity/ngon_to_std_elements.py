from cmaia.transform import transform as ctransform
from maia.transform.apply_function_to_nodes import apply_to_bases
import Converter.Internal as I
from maia import transform
from mpi4py import MPI

def sort_nfaces_by_element_type(t):
  apply_to_bases(t,ctransform.sort_nfaces_by_element_type)

def sorted_nfaces_to_std_elements(t):
  apply_to_bases(t,ctransform.sorted_nfaces_to_std_elements)

def add_nfaces(t):
  apply_to_bases(t,ctransform.add_nfaces)

def ngon_to_std_elements(t,comm=MPI.COMM_WORLD):
  """
  Transform a pytree so that ngon elements are converted to standard elements
  The ngon elements are supposed to describe only standard elements
  (i.e. tris, quads, tets, pyras, prisms and hexas only)
  """
  I._adaptNFace2PE(t,remove=True) # PE = ParentElements, remove NFace (not updated by following step)
  transform.put_boundary_first(t,comm)
  I._fixNGon(t) # reconstruct NFace (TODO add_nfaces [with sign])
  sort_nfaces_by_element_type(t)
  sorted_nfaces_to_std_elements(t)
