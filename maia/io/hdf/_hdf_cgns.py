import numpy as np
import h5py
from h5py import h5, h5a, h5d, h5f, h5g, h5p, h5s, h5t, h5o

C33_t = h5t.C_S1.copy()
C33_t.set_size(33)
C3_t = h5t.C_S1.copy()
C3_t.set_size(3)

DTYPE_TO_CGNSTYPE = {'int8'    : 'B1',
                     'int32'   : 'I4',
                     'int64'   : 'I8',
                     'float32' : 'R4',
                     'float64' : 'R8',
                     'bytes8'  : 'C1'}

class AttributeRW:
  """ A singleton class usefull to read & write hdf attribute w/ allocating buffers """

  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(AttributeRW, cls).__new__(cls)
      cls.buff_S33 = np.empty(1, '|S33')
      cls.buff_S3  = np.empty(1, '|S3')
      cls.buff_flag1 = np.array([1], np.int32)
    return cls.instance

  def read_str_33(self, gid, attr_name):
    """ Read the attribute attr_name in the object gid and return it
    as a sttriped string.  """
    _name = h5a.open(gid, attr_name)
    _name.read(self.buff_S33)
    return self.buff_S33.tobytes().decode().rstrip('\x00') ###UGLY

  def read_bytes_3(self, gid, attr_name):
    """ Read the attribute attr_name in the object gid and return it
    as bytes.  """
    _name = h5a.open(gid, attr_name)
    _name.read(self.buff_S3)
    return self.buff_S3[0]

  def write_str_3(self, gid, attr_name, attr_value):
    """ Create the attribute attr_name within the object gid and
    write attr_value (of size <=2) inside.  """
    self.buff_S3[0] = attr_value
    space = h5s.create(h5s.SCALAR)
    attr_id = h5a.create(gid, attr_name, C3_t, space)
    attr_id.write(self.buff_S3)
  def write_str_33(self, gid, attr_name, attr_value):
    """ Create the attribute attr_name within the object gid and
    write attr_value (of size <=32) inside.  """
    self.buff_S33[0] = attr_value
    space = h5s.create(h5s.SCALAR)
    attr_id = h5a.create(gid, attr_name, C33_t, space)
    attr_id.write(self.buff_S33)

  def write_flag(self, gid):
    """ Create and fill the 'flags' attribute within the object gid.  """
    space = h5s.create_simple((1,))
    attr_id = h5a.create(gid, b'flags', h5t.NATIVE_INT32, space)
    attr_id.write(self.buff_flag1)

def knows_crt_order(gid):
  """ Return True if links have been added with CRT_ORDER_INDEXED prop """
  ordering_flags = gid.get_create_plist().get_link_creation_order()
  return ordering_flags & h5p.CRT_ORDER_INDEXED == h5p.CRT_ORDER_INDEXED

def is_combinated(dataspace):
  """ Check if the provided dataspace if combinated or not. """
  return not (len(dataspace) == 10)

def group_by(iterable, n):
  """ Iterate chunk by chunk (of size n) over an iterable object :
  s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ...  """
  #https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
  return zip(*[iter(iterable)] * n)

def add_root_attributes(rootid):
  """ Write the attributes of the CGNS-HDF root node.  """

  attr_writter = AttributeRW()
  attr_writter.write_str_33(rootid, b'name', 'HDF5 MotherNode')
  attr_writter.write_str_33(rootid, b'label', 'Root Node of HDF5 File')
  attr_writter.write_str_3 (rootid, b'type' , 'MT')

  if h5t.NATIVE_FLOAT == h5t.IEEE_F32BE:
    format = 'IEEE_BIG_32'
  elif h5t.NATIVE_FLOAT == h5t.IEEE_F32LE:
    format = 'IEEE_LITTLE_32'
  elif h5t.NATIVE_FLOAT == h5t.IEEE_F64BE:
    format = 'IEEE_BIG_64'
  elif h5t.NATIVE_FLOAT == h5t.IEEE_F64LE:
    format = 'IEEE_LITTLE_64'
  version = 'HDF5 Version {}.{}.{}'.format(*h5.get_libversion())

  # Those two are described at attributes in cgns doc, but the C lib
  # looks for dataset (see https://cgnsorg.atlassian.net/browse/CGNS-283)
  buffer = np.zeros(33, dtype='c')
  for i,c in enumerate(format):
    buffer[i] = c
  write_data(rootid, buffer, dataset_name=b' format')
  buffer[:] = '\0'
  for i,c in enumerate(version):
    buffer[i] = c
  write_data(rootid, buffer, dataset_name=b' hdf5version')

def open_from_path(fid, path, follow_links=True):
  """ Return the hdf node registred at the specified path in the file fid.  """
  attr_reader = AttributeRW()
  gid = h5g.open(fid, b'/')
  for name in path.split('/'):
    gid = h5g.open(gid, name.encode())
    if follow_links and attr_reader.read_bytes_3(gid, b'type') == b'LK': #Follow link
      gid = h5g.open(gid, b' link')
  return gid

def _select_file_slabs(hdf_space, filter):
  """ Performs the 'select_hyperslab' operation on a open hdf_dataset space,
  using the input filter.
  Filter must be a list of 4 elements (start, stride, count, block)
  and can be combinated.  """
  if is_combinated(filter):
    src_filter = filter[4]
    for i, src_filter_i in enumerate(group_by(src_filter, 4)): #Iterate 4 by 4 on the combinated slab
      src_start, src_stride, src_count, src_block = [tuple(src_filter_i[i][::-1]) for i in range(4)]
      op = h5s.SELECT_SET if i == 0 else h5s.SELECT_OR
      hdf_space.select_hyperslab(src_start, src_count, src_stride, src_block, op=op)
  else:
    src_start, src_stride, src_count, src_block = [tuple(filter[i][::-1]) for i in range(4,8)]
    hdf_space.select_hyperslab(src_start, src_count, src_stride, src_block)

def _create_mmry_slabs(filter):
  """ Create and return a memory dataspace from a filter object.
  Filter must be a list of 4 elements (start, stride, count, block).
  Filter can no be combinated.  """
  dst_start, dst_stride, dst_count, dst_block = [tuple(filter[i][::-1]) for i in range(0,4)]
  m_dspace = h5s.create_simple(dst_count)
  m_dspace.select_hyperslab(dst_start, dst_count, dst_stride, dst_block)
  return m_dspace

def load_data(gid):
  """ Create a numpy array from the dataset stored in the hdf node gid,
  reading all data (no hyperslab).
  HDFNode must have data (type != MT).
  Numpy array is reshaped to F order **but** kind is not converted.  """
  hdf_dataset = h5d.open(gid, b' data')

  shape = hdf_dataset.shape[::-1]

  array = np.empty(shape, hdf_dataset.dtype, order='F')
  array_view = array.T
  hdf_dataset.read(h5s.ALL, h5s.ALL, array_view)

  return array

def load_data_partial(gid, filter):
  """ Create a numpy array from the dataset stored in the hdf node gid,
  reading partial data (using global filter object).
  HDFNode must have data (type != MT).
  Numpy array is reshaped to F order **but** kind is not converted.  """
  hdf_dataset = h5d.open(gid, b' data')

  # Prepare dataspaces
  hdf_space = hdf_dataset.get_space()
  _select_file_slabs(hdf_space, filter)
  m_dspace = _create_mmry_slabs(filter)

  array = np.empty(m_dspace.shape[::-1], hdf_dataset.dtype, order='F')
  array_view = array.T
  hdf_dataset.read(m_dspace, hdf_space, array_view)

  return array


def write_data(gid, array, dataset_name=b' data'):
  """ Write a dataset on node gid from a numpy array,
  dumping all data (no hyperslab).  """
  array_view = array.T
  if array_view.dtype == 'S1':
    array_view.dtype = np.int8

  space = h5s.create_simple(array_view.shape)
  data = h5d.create(gid, dataset_name, h5t.py_create(array_view.dtype), space)
  data.write(h5s.ALL, h5s.ALL, array_view)

def write_data_partial(gid, array, filter):
  """ Write a dataset on node gid from a numpy array,
  using hyperslabls (from filter object).  """
  glob_dims = tuple(filter[-2][::-1])

  # Prepare dataspaces
  hdf_space = h5s.create_simple(glob_dims)
  _select_file_slabs(hdf_space, filter)
  m_dspace = _create_mmry_slabs(filter)

  array_view = array.T
  if array_view.dtype == 'S1':
    array_view.dtype = np.int8
  data = h5d.create(gid, b' data', h5t.py_create(array_view.dtype), hdf_space)
  xfer_plist = h5p.create(h5p.DATASET_XFER)
  xfer_plist.set_dxpl_mpio(h5py.h5fd.MPIO_INDEPENDENT)
  data.write(m_dspace, hdf_space, array_view, dxpl=xfer_plist)

def write_link(gid, node_name, target_file, target_node):
  """ Create a linked child named node_name under the open parent node gid
  Child links to the node target_node (absolute path) of file target_file. """
  node_id = h5g.create(gid, node_name.encode())

  attr_writter = AttributeRW()
  attr_writter.write_str_33(node_id, b'name',  node_name)
  attr_writter.write_str_33(node_id, b'label', '')
  attr_writter.write_str_3 (node_id, b'type',  'LK')

  write_data(node_id, np.array(tuple(target_file+'\0'), 'S1'), b' file')
  write_data(node_id, np.array(tuple(target_node+'\0'), 'S1'), b' path')

  node_id.links.create_external(" link".encode(), target_file.encode(), target_node.encode())

def _load_node_partial(gid, parent, load_if, ancestors_stack):
  """ Internal recursive implementation for load_tree_partial.  """

  attr_reader = AttributeRW()
  name  = attr_reader.read_str_33(gid, b'name')
  label = attr_reader.read_str_33(gid, b'label')
  b_kind = attr_reader.read_bytes_3(gid, b'type')
  value = None

  if b_kind == b'LK': #Follow link
    gid = h5g.open(gid, b' link')
    b_kind = attr_reader.read_bytes_3(gid, b'type')
    # Label may be empty in original node and present only in linked node
    label = attr_reader.read_str_33(gid, b'label')

  ancestors_stack[0].append(name)
  ancestors_stack[1].append(label)

  if load_if(*ancestors_stack):
    if b_kind != b'MT':
      value = load_data(gid)
      if b_kind==b'C1':
        value.dtype = 'S1'
  elif b_kind != b'MT':
    _data = h5d.open(gid, b' data')
    size_node = [name + '#Size', 
                 np.array(_data.shape[::-1]),
                 [],
                 'DataArray_t']
    parent[2].append(size_node)

  pynode = [name, value, [], label]
  parent[2].append(pynode)

  # Define the function that will be applied to the child of the current hdf node
  # thought iterate : we just start next recursion level if child is not a dataset
  iter_func = lambda n : _load_node_partial(h5g.open(gid, n), pynode, load_if, ancestors_stack) \
      if h5o.get_info(gid, n).type == h5o.TYPE_GROUP else None

  idx_type = h5.INDEX_CRT_ORDER if knows_crt_order(gid) else h5.INDEX_NAME
  gid.links.iterate(iter_func, idx_type=idx_type)
  ancestors_stack[0].pop()
  ancestors_stack[1].pop()

def _load_links(gid, links, ancestors_stack):
  """ Internal recursive implementation for load_tree_partial.  """

  attr_reader = AttributeRW()
  name  = attr_reader.read_str_33(gid, b'name')
  b_kind = attr_reader.read_bytes_3(gid, b'type')

  ancestors_stack.append(name)

  if b_kind == b'LK':
    #Target directory ; the CGNS norm is unclear about how a link should start, 
    # but other libraries are also doing that
    link = ['.']
    for ds_name in [b' file', b' path']: #Target file, then target path
      hdf_dataset = h5d.open(gid, ds_name)
      shape = hdf_dataset.shape[::-1]
      array = np.empty(shape, hdf_dataset.dtype, order='F')
      array_view = array.T
      hdf_dataset.read(h5s.ALL, h5s.ALL, array_view)
      array.dtype = 'S1'
      link.append(array.tobytes().decode().rstrip('\x00'))
    link.append('/'.join(ancestors_stack)) #Current path

    links.append(link)
    ancestors_stack.pop() #Stop exploring the node (don't explore links recursively)
    return

  # Define the function that will be applied to the child of the current hdf node
  # thought iterate : we just start next recursion level if child is not a dataset
  iter_func = lambda n : _load_links(h5g.open(gid, n), links, ancestors_stack) \
      if h5o.get_info(gid, n).type == h5o.TYPE_GROUP else None

  idx_type = h5.INDEX_CRT_ORDER if knows_crt_order(gid) else h5.INDEX_NAME
  gid.links.iterate(iter_func, idx_type=idx_type)
  ancestors_stack.pop()

def _write_node_partial(gid, node, write_if, ancestors_stack):
  """ Internal recursive implementation for write_tree_partial.  """

  cgtype = 'MT' if node[1] is None else DTYPE_TO_CGNSTYPE[node[1].dtype.name]
  ancestors_stack[0].append(node[0])
  ancestors_stack[1].append(node[3])

  gc_pl = h5p.create(h5p.GROUP_CREATE)
  gc_pl.set_link_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)

  node_id = h5g.create(gid, node[0].encode(), gcpl=gc_pl)

  # Write attributes
  attr_writter = AttributeRW()
  attr_writter.write_str_33(node_id, b'name',  node[0])
  attr_writter.write_str_33(node_id, b'label', node[3])
  attr_writter.write_str_3 (node_id, b'type',  cgtype)
  attr_writter.write_flag(node_id) 

  if write_if(*ancestors_stack) and node[1] is not None:
    write_data(node_id, node[1])

  # Write children
  for child in node[2]:
    _write_node_partial(node_id, child, write_if, ancestors_stack)
  ancestors_stack[0].pop()
  ancestors_stack[1].pop()


def load_tree_partial(filename, load_predicate):
  """
  Create a pyCGNS tree from the (partial) read of an hdf file.

  For each encountered node, the name and label of node is registered in tree,
  then the load_predicate is evaluated :
  - if load_predicate return True, the data of the node is fully loaded
    and registered in tree
  - if load_predicate return False, the data is skipped, buts its shape is registered in
    tree as the value of an additional node of name name Node#Size

  Note : if load_predicate returns always True, the tree is then fully read.
  """
  tree = ['CGNSTree', None, [], 'CGNSTree_t']

  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)
  rootid = h5g.open(fid, b'/')

  iter_func = lambda n : _load_node_partial(h5g.open(rootid, n), tree, load_predicate, ([],[])) \
      if h5o.get_info(rootid, n).type == h5o.TYPE_GROUP else None
  idx_type = h5.INDEX_CRT_ORDER if knows_crt_order(rootid) else h5.INDEX_NAME
  rootid.links.iterate(iter_func, idx_type=idx_type)

  fid.close()
  return tree

def load_tree_links(filename):
  links = list()

  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)
  rootid = h5g.open(fid, b'/')

  iter_func = lambda n : _load_links(h5g.open(rootid, n), links, []) \
      if h5o.get_info(rootid, n).type == h5o.TYPE_GROUP else None
  idx_type = h5.INDEX_CRT_ORDER if knows_crt_order(rootid) else h5.INDEX_NAME
  rootid.links.iterate(iter_func, idx_type=idx_type)

  fid.close()
  return links

def write_tree_partial(tree, filename, write_predicate):
  """
  Write a (partial) hdf file from a pyCGNS tree.

  For each encountered node, the name and label of node is registred in file,
  then the write_predicate is evaluated :
  - if write_predicate return True, the data of the CGNS is written in the file
  - if write_predicate return False, the data is no written (meaning that the ' data'
    dataset is not created; nevertheless, the attribute 'type' storing the
    datakind is written)

  Note : if write_predicate returns always False, the tree is then fully writed.
  """

  fc_pl = h5p.create(h5p.FILE_CREATE)
  fc_pl.set_link_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
  fid = h5f.create(bytes(filename, 'utf-8'), fcpl=fc_pl)

  rootid = h5g.open(fid, b'/')

  # Write some attributes of root node
  add_root_attributes(rootid)
  for node in tree[2]:
    _write_node_partial(rootid, node, write_predicate, ([],[]))

  fid.close()

