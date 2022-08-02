import maia
import maia.pytree        as PT
import Converter.PyTree   as C
import Converter.Internal as I

def save_part_tree(part_tree, filename, comm):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  zones = PT.get_all_Zone_t(part_tree)
  zone_name_list = list()
  for zone in zones:
    zone_name_list.append(zone[0])

  topfilename = filename+'_TOP.hdf'
  subfilename = filename+'_{0}_SUB.hdf'.format(i_rank)

  links      = []
  nlinks     = []
  zones_path = [f'/{path}' for path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t')]

  for zone_path in zones_path:
    links += [['',subfilename, zone_path, zone_path]]

  C.convertPyTree2File(part_tree, subfilename)

  links      = comm.gather(links, root=0)

  if(i_rank == 0):

    for l in links:
      for ii in l:
        nlinks.append(ii)

    base_name  = I.getNodeFromType1(part_tree, 'CGNSBase_t')[0]
    local_tree = I.newCGNSTree()
    local_base = I.newCGNSBase(base_name, 3, 3, parent=local_tree)
    for path_zone in nlinks:
      zone_name = path_zone[2].split("/")[2]
      I.newZone(name=zone_name, zsize=None, parent=local_base)

    C.convertPyTree2File(local_tree, topfilename, links=nlinks)
