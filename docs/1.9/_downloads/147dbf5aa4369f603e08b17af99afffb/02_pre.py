### Prelude

from mpi4py import MPI
comm = MPI.COMM_WORLD

import maia
import maia.pytree as PT

### File reading

tree = maia.io.file_to_dist_tree('rotor37_medium.cgns', comm)

### Connectivity conversion

elt_kinds = set(PT.Element.Type(e) for e in PT.iter_nodes_from_label(tree, 'Elements_t'))
if comm.Get_rank() == 0:
  print(f"Elements type was {elt_kinds}")

maia.algo.dist.convert_elements_to_ngon(tree, comm)

elt_kinds = set(PT.Element.Type(e) for e in PT.iter_nodes_from_label(tree, 'Elements_t'))
if comm.Get_rank() == 0:
  print(f"Elements type are now {elt_kinds}")


### Field initialization

zone = PT.find_node_from_label(tree, 'Zone_t') # OK because tree has only one zone
cx, cy, cz = PT.Zone.coordinates(zone)

import numpy as np
α = np.sqrt(2)/2
Θ = np.arctan(cz/cy)
vx = np.full(len(cx), α)
vy = -α * np.sin(Θ)
vz =  α * np.cos(Θ)

PT.new_FlowSolution('FlowSolution',
                    loc='Vertex',
                    fields={
                      'VelocityX' : vx,
                      'VelocityY' : vy,
                      'VelocityZ' : vz},
                    parent=zone)

### 360° duplication

#### Recovering GridConnectivity_t nodes


a = 2*np.pi / 36
maia.algo.dist.connect_1to1_families(tree,
                                     ('perleft', 'perright'), 
                                     comm,
                                     periodic={'rotation_angle' : np.array([a, 0, 0])})

if comm.Get_rank() == 0:
  PT.print_tree(tree, max_depth=4)

#### Effective duplication

zone_paths = ["Base/Rotor"]
left_jns   = ["Base/Rotor/ZoneGridConnectivity/perleft_0"]
right_jns  = ["Base/Rotor/ZoneGridConnectivity/perright_0"]

maia.algo.dist.duplicate_from_rotation_jns_to_360(tree,
                                           zone_paths,
                                           jn_paths_for_dupl = (left_jns, right_jns),
                                           comm = comm)

### Merging zones

maia.algo.dist.merge_connected_zones(tree, comm)

### Conclusion

if comm.Get_rank() == 0:
  PT.print_tree(tree)