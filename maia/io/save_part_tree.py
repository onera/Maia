import maia
import maia.pytree        as PT

from . import cgns_io_tree as IOT

def save_part_tree(part_tree, filename, comm, legacy=False):
  """
  """
  topfilename = filename + '_TOP.hdf'
  subfilename = filename + f'_{comm.Get_rank()}_SUB.hdf'

  links      = []
  for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
    links += [['',subfilename, zone_path, zone_path]]

  IOT.dump_tree(part_tree, subfilename, legacy=legacy) #Use direct API to manage name

  _links = comm.gather(links, root=0)
  links  = [l for proc_links in _links for l in proc_links] #Flatten gather result

  if(comm.Get_rank() == 0):
    top_tree = PT.new_CGNSTree()
    for link in links:
      base_name, zone_name = link[2].split("/")
      local_base = PT.update_child(top_tree, base_name, 'CGNSBase_t', value=[3,3])

    IOT.dump_tree(top_tree, topfilename, links=links, legacy=legacy)
