from maia.transform import transform as cmaia
import Converter.Internal as I
from mpi4py import MPI


def apply_to_bases(t,f,*args):
  if I.getType(t)=="CGNSBase_t":
    f(t)
  elif I.getType(t)=="CGNSTree_t":
    for b in I.getBases(t):
      f(b,*args)
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+I.getType(t)+"\".")


def merge_by_elt_type(dist_tree,comm):
  apply_to_bases(dist_tree,cmaia.merge_by_elt_type,comm)


def remove_ghost_info(t):
  apply_to_bases(t,cmaia.remove_ghost_info)

def add_fsdm_distribution(t,comm):
  apply_to_bases(t,cmaia.add_fsdm_distribution,comm)


def gcs_only_for_ghosts(t):
  apply_to_bases(t,cmaia.gcs_only_for_ghosts)



def partition_with_boundary_first(t):
  apply_to_bases(t,cmaia.partition_with_boundary_first)

def sort_nface_into_simple_connectivities(t):
  apply_to_bases(t,cmaia.sort_nface_into_simple_connectivities)

def convert_to_simple_connectivities(t):
  apply_to_bases(t,cmaia.convert_to_simple_connectivities)

def add_nfaces(t):
  apply_to_bases(t,cmaia.add_nfaces)

def convert_from_ngon_to_simple_connectivities(t):
  I._adaptNFace2PE(t,remove=True) # PE = ParentElements, remove NFace (not updated by following step)
  partition_with_boundary_first(t)
  I._fixNGon(t) # reconstruct NFace (TODO add_nfaces [with sign])
  sort_nface_into_simple_connectivities(t)
  convert_to_simple_connectivities(t)

