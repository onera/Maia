import maia.pytree as PT

# For each cell_dimension, list of output GridLocation depending of requested dim argument
DIM_TO_LOC = {3: ['Vertex', 'EdgeCenter', 'FaceCenter', 'CellCenter'],
              2: ['Vertex', 'EdgeCenter', 'CellCenter',  None],
              1: ['Vertex', 'CellCenter',  None,         None],}

def get_or_create_container(zone, container_name, container_loc):
  """ Utility to retrieve a container from its name, or create it """
  container = PT.get_child_from_name(zone, container_name)
  if container is not None: # Container exists
    assert PT.Subset.GridLocation(container) == container_name, \
      f"Container {PT.get_name(container)} already exists in zone {PT.get_name(zone)} but has incompatible GridLocation"
  else:  # Create container
    container = PT.new_child(zone, container_name, 'DiscreteData_t')
    PT.new_GridLocation(container_loc, container)
  return container

def feed_container(container, datas, names):
  for data, name in zip(datas, names):
    if data is not None:
      PT.update_child(container, name, 'DataArray_t', data)
