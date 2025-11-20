import copy
import argparse

import nbformat
from   nbconvert.preprocessors import ExecutePreprocessor

def replace_print_tree(str_in):
    while (idx:= str_in.find('PT.print_tree')) != -1:
        p_count = 0
        first_p = -1
        end = idx
        while (p_count != 0 or first_p == -1):
            c = str_in[end]
            if c == '(':
                if first_p == -1:
                    first_p = end
                p_count += 1
            elif c == ')':
                p_count -= 1
            end += 1
        str_in = str_in[:idx] + 'print("\\n", *PT.to_string' + str_in[first_p:end] + ')' + str_in[end:]

    return str_in

def parallelize_notebook(nb, n_rank):
    nb_par = copy.deepcopy(nb)

    # Insert cluster launch
    par_header_cell = nbformat.v4.new_code_cell(f"""
    import ipyparallel as ipp
    # create a cluster
    cluster = ipp.Cluster(engines="mpi", n={n_rank})
    # start that cluster and connect to it
    rc = cluster.start_and_connect_sync()
    rc.activate()
    """)

    # Modify each cell code to make it work in //
    # Add magic command %%px to run cells in parallel
    original_source = list()
    for cell in nb_par.cells:
        tags = cell.metadata.get("tags", [])
        if cell.cell_type == 'code' and 'no-parallel' not in tags:
            original_source.append(cell.source)
            # Add magic command %%px
            cell.source = "%%px\n" + cell.source
            # Replace PT.print_tree() which doesn work (??)
            cell.source = replace_print_tree(cell.source)

    nb_par.cells.insert(0, par_header_cell)

    return nb_par


def copy_nb_output(nb, nb_mpi):
    # Remove header // cell
    cells = nb.cells
    mpi_cells = nb_mpi.cells[1:] # Ignore first cell

    is_progress_bar = lambda o: o.output_type == 'stream' and ''.join(o.text).strip().startswith('%px')
    
    for cell, mpi_cell in zip(cells, mpi_cells):
        if cell.cell_type == 'code':
            cell.outputs = [o for o in mpi_cell.outputs if not is_progress_bar(o)]







parser = argparse.ArgumentParser()
parser.add_argument('path_in', type=str)
parser.add_argument('path_out', type=str, nargs='?')
parser.add_argument('n_rank', type=int)

args = parser.parse_args()
if args.path_out is None:
    base, ext = args.path_in.rsplit('.', 1)
    args.path_out = base + '_out.' + ext

nb = nbformat.read(args.path_in, as_version=4)

nb_mpi = parallelize_notebook(nb, args.n_rank)

#nbformat.write(nb_mpi, 'tmp_parallel.ipynb')
#quit()
#nb_mpi = nbformat.read('tmp_parallel.ipynb', as_version=4)

ep = ExecutePreprocessor()
ep.preprocess(nb_mpi)

copy_nb_output(nb, nb_mpi)

nbformat.write(nb, args.path_out)
