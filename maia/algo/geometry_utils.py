import maia.pytree as PT

# For each cell_dimension, list of output GridLocation depending of requested dim argument
DIM_TO_LOC = {3: ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter'],
              2: ['Vertex', 'EdgeCenter', 'CellCenter',  None],
              1: ['Vertex', 'CellCenter',  None,         None],}

def update_container(zone, container_name, loc, fields={}):
  """ Utility to retrieve a container from its name, or create it """
  container = PT.get_child_from_name(zone, container_name)
  if container is not None: # Container exists
    cnt_loc = PT.Subset.GridLocation(container) 
    if cnt_loc != loc:
      raise RuntimeError(f"Container {container_name} already exists in zone "
                         f"{PT.get_name(zone)} but has incompatible GridLocation "
                         f"(expected {loc}, found {cnt_loc})")
  else:  # Create container
    container = PT.new_child(zone, container_name, 'DiscreteData_t')
    PT.new_GridLocation(loc, container)

  for name, array in fields.items():
    PT.update_child(container, name, 'DataArray_t', array)

  return container
