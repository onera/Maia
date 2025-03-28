import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils

def add_fsdm_distribution(t, comm):
  for base in PT.get_all_CGNSBase_t(t):
    zones = PT.get_all_Zone_t(base)
    if len(zones) != 1:
      raise RuntimeError("add_fsdm_distribution (as FSDM) expects only one zone per process")
    zone = zones[0]

    n_vtx_owned = PT.get_node_from_path(zone, ':CGNS#LocalNumbering/VertexSizeOwned')[1][0]
    vtx_distri = par_utils.dn_to_distribution(n_vtx_owned, comm)
    MT.new_distribution({"Vertex" : vtx_distri}, zone)

    for elt_section in PT.get_children_from_label(zone, 'Elements_t'):
      n_owned_elt = PT.Element.Size(elt_section)
      elt_distri = par_utils.dn_to_distribution(n_owned_elt, comm)
      MT.new_distribution({"Element" : elt_distri}, elt_section)
    
