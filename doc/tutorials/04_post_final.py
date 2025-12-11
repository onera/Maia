# Do not modify this section
# -----------------------------------------------------------------------------
def get_solver_tree():
    tree  = maia.io.file_to_dist_tree('airplane.cgns', comm)
    ptree = maia.factory.partition_dist_tree(tree, comm,
            preserve_orientation=True, data_transfer='FIELDS')
    return ptree

def plot_1d_profile(x, y, comm):
    import numpy as np
    import plotly.graph_objects as go

    x = comm.gather(x, root=0)
    y = comm.gather(y, root=0)
    
    if comm.Get_rank() != 0:
        return

    # > Concatenate arrays
    x = np.concatenate(x)
    y = np.concatenate(y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Wing',
        )
    )


    fig.update_layout(
        xaxis=dict(
            title=dict(
                text='X position',
            ),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
            ),
        yaxis=dict(
            title=dict(
                text='Pressure',
            ),
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        ),
        font=dict(
            family='Courier New, monospace',
            size=14,
        ),
        plot_bgcolor='white',
        showlegend=False,
    )

    fig.write_image('1dprofile.png')
# -----------------------------------------------------------------------------

# Warmup
from mpi4py import MPI
import numpy as np

import maia
import maia.pytree as PT

comm = MPI.COMM_WORLD

tree = get_solver_tree()

# Extraction

surf_tree = maia.algo.part.extract_part_from_family(tree, 'Walls', comm)

maia.io.part_tree_to_file(surf_tree, 'airplane_walls_part.cgns', comm)

dsurf_tree = maia.factory.recover_dist_tree(surf_tree, comm, data_transfer='FIELDS')
maia.io.dist_tree_to_file(dsurf_tree, 'airplane_walls.cgns', comm)

# Slice

slice_tree = maia.algo.part.plane_slice(tree, [0.,1.,0.,1], comm,
                                        containers_name=['FlowSolution#Centers'])

dslice_tree = maia.factory.recover_dist_tree(slice_tree, comm, data_transfer='FIELDS')
maia.io.dist_tree_to_file(dslice_tree, 'airplane_slice.cgns', comm)

# 1D profile

for zone in PT.get_all_Zone_t(tree):
  if PT.get_node_from_name_and_label(zone, 'Wing', 'BC_t') is not None:
    PT.new_ZoneSubRegion('WingFields', bc_name='Wing', parent=zone)
PT.subregion_fields_from_bcdataset(tree)

slice_tree = maia.algo.part.plane_slice(tree, [0.,1.,0.,1], comm,
                                        containers_name=['FlowSolution#Centers', 'WingFields'])

if comm.rank == 0:
  PT.print_tree(slice_tree, max_depth=3)

maia.algo.compute_elements_center(slice_tree, 1, comm)

zsr_n = PT.get_node_from_name(slice_tree, 'WingFields')
if zsr_n is not None:
    # Get X component of computed edge centers
    center_x = PT.get_value(PT.get_node_from_name(slice_tree, 'CenterX'))
    # Get edges ids belonging to Wing subset
    edge_ids = PT.get_np_value(PT.get_child_from_name(zsr_n, 'PointList'))[0]
    # Get edge offset
    edge_n = PT.get_node_from_predicate(slice_tree, PT.pred.is_element_of_type('BAR_2'))
    offset = PT.Element.Range(edge_n)[0]
    # Extract x values, and get pressure field
    x = center_x[edge_ids - offset]
    pressure = PT.get_value(PT.get_child_from_name(zsr_n, 'Pressure'))
else:
    # Create empty array for ranks not related to the Wing subset
    x = np.empty(0, dtype=np.float64)
    pressure = np.empty(0, dtype=np.float64)

plot_1d_profile(x, pressure, comm)
