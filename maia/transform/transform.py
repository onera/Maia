from cmaia.transform import transform as ctransform
from maia.transform.apply_to_bases import apply_to_bases
import Converter.Internal as I
from mpi4py import MPI


def apply_to_bases(t,f,*args):
  if I.getType(t)=="CGNSBase_t":
    f(t,*args)
  elif I.getType(t)=="CGNSTree_t":
    for b in I.getBases(t):
      f(b,*args)
  else:
    raise Exception("function \""+f.__name__+"\"" \
                    " can only be applied to a \"CGNSBase_t\" or on a complete \"CGNSTree_t\"," \
                    " not on a node of type \""+I.getType(t)+"\".")


def merge_by_elt_type(dist_tree,comm):
  apply_to_bases(dist_tree,ctransform.merge_by_elt_type,comm)


def remove_ghost_info(t):
  apply_to_bases(t,ctransform.remove_ghost_info)

def add_fsdm_distribution(t,comm):
  apply_to_bases(t,ctransform.add_fsdm_distribution,comm)


def gcs_only_for_ghosts(t):
  apply_to_bases(t,ctransform.gcs_only_for_ghosts)



def put_boundary_first(t,comm):
  apply_to_bases(t,ctransform.partition_with_boundary_first,comm)

def ngon_new_to_old(t):
  apply_to_bases(t,ctransform.ngon_new_to_old)
def sids_conforming_ngon_nface(t):
  apply_to_bases(t,ctransform.sids_conforming_ngon_nface)

def split_boundary_subzones_according_to_bcs(t):
  apply_to_bases(t,ctransform.split_boundary_subzones_according_to_bcs)
