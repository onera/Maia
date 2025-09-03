import re
import sys
import os
import inspect
import h5py

from pathlib import Path

from .data import ALL_LABELS

ALLOWED_CGNS_TYPES = {'MT', 'I4', 'I8', 'U4', 'U8', 'R4', 'R8', 'X4', 'X8', 'C1', 'B1', 'LK'}

CGNS_TO_DTYPE = {
  'B1' : 'int8',
  'C1' : 'int8',
  'I4' : 'int32',
  'U4' : 'uint32',
  'I8' : 'int64',
  'U8' : 'uint64',
  'R4' : 'float32',
  'R8' : 'float64',
  'X4' : 'complex64',
  'X8' : 'complex128',
}

OK = ''

#
# Disclaimer: the reference documentation for HDF5 mapping
# (https://cgns.org/standard/hdf5.html#) seems to be incomplete
# or out of date. So somme choice are rather based on the analysis of
# cgns library itself (ADFH.c)
#
# - The attribute seems to be 'order' and not ' order' (no space)
# - In addition, we can user either 'order' OR 'flags':
#   + if ADF_NO_ORDER --> use flags with value 1
#   + Otherwise, use order with order of creation (using H5_INDEX_CRT_ORDER)
# - For root node dataset is " hdf5version" and not " version"
# - Group '/ mount' (with attrs ' file' and ' refnt') is never used
# - For links:
#   + src node is created with usual attributes but can have empty label
#   + src node have str datasets ' path' (name in file) and ' file' (filename, if external)


#
# Errors range for this file is 100 - 199
# - 101-110: File access issue
# - 111-120: Attributes related
# - 121-130: Dataset related
# - 131-140: Link related
#


# 
# The following function applies on a h5py.Group object
# 
# Note: exemples are formated following h5dump output.
# Usefull commands are:
# - h5dump -n 1  # List file contents
# - h5dump -N "flabel" # Display all attributes
# - h5dump -a "/Base/zone/type" # Display specific attribute
# - h5dump -g "/Base/zone" # Display data for a group + 
#

def invalid_name_attr(hgroup: h5py.Group) -> str:
    """E111 - Invalid 'name' attribute

    Every HDF group should have a 'name' attribute of kind S33.
    This attribute should store the name of the node, which
    is also the name of the HDF group.

    Erroneous file examples:

    GROUP "/Base/zone/GridCoordinates" {
      ATTRIBUTE "name" {
        DATATYPE  \033[91mH5T_STD_I32LE  # Datatype is wrong \033[0m
        DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
        DATA {
        (0): 1
        }
      }
    }

    GROUP "/Base/zone/\033[32mGridCoordinates\033[0m" {
      ATTRIBUTE "name" {
        DATATYPE  H5T_STRING { ... }
        DATASPACE  SCALAR
        DATA {
        (0): \033[91m"gridcoord"         # Value is wrong \033[0m
        }
      }
    }
    """
    if 'name' not in hgroup.attrs:
        return f"Attribute 'name' is missing"

    # If we want to check kind / shape direclty on h5 object, we have to use
    # raw id (otherwise h5py decode the value).
    attr_id = hgroup.attrs.get_id('name')
    type_id = attr_id.get_type()
    
    if type_id.get_class() != h5py.h5t.STRING:
        return f"Attribute 'name' is not of kind string"
    if type_id.get_size() != 33:
        return f"Attribute 'name' has invalid size (expected 33, got {type_id.get_size()})"

    group_name = hgroup.name.split('/')[-1] if hgroup.name != '/' else 'HDF5 MotherNode'
    value = hgroup.attrs['name'].decode()
    if value != group_name:
        return f"Attribute 'name' has invalid value {value}"
    
    return OK

def invalid_type_attr(hgroup: h5py.Group) -> str:
    """E112 - Invalid 'type' attribute

    Every HDF group should have a 'type' attribute of kind S3.
    This attribute should store the datakind of the node, which
    must belong to the allowed datakind set.

    Erroneous file examples:

    ATTRIBUTE "type" {
      DATATYPE  H5T_STRING {
        \033[91mSTRSIZE 33;         # Shape is wrong \033[0m
        ...
      }
      DATASPACE  SCALAR
      DATA {
      (0): "R8"
      }
    }

    ATTRIBUTE "type" {
      DATATYPE  H5T_STRING {
        STRSIZE 3;
        ...
      }
      DATASPACE  SCALAR
      DATA {
      (0): \033[91m"R1"             # Value is wrong \033[0m
      }
    }
    """
    if 'type' not in hgroup.attrs:
        return f"Attribute 'type' is missing"

    # If we want to check kind / shape direclty on h5 object, we have to use
    # raw id (otherwise h5py decode the value).
    attr_id = hgroup.attrs.get_id('type')
    type_id = attr_id.get_type()
    
    if type_id.get_class() != h5py.h5t.STRING:
        return f"Attribute 'type' is not of kind string"
    if type_id.get_size() != 3:
        return f"Attribute 'type' has invalid size (expected 3, got {type_id.get_size()})"

    value = hgroup.attrs['type'].decode()
    if value not in ALLOWED_CGNS_TYPES:
        return f"Attribute 'type' has invalid value {value}"
    
    return OK

def invalid_label_attr(hgroup: h5py.Group) -> str:
    """E113 - Invalid 'label' attribute

    Every HDF group should have a 'label' attribute of kind S33.
    This attribute should store the label of the node, which
    must belong to the allowed labels set.

    Erroneous file examples:

    ATTRIBUTE "label" {
      DATATYPE  H5T_STRING {
        \033[91mSTRSIZE 3;          # Shape is wrong \033[0m
        ...
      }
      DATASPACE  SCALAR
      DATA {
      (0): "R8"
      }
    }

    ATTRIBUTE "label" {
      DATATYPE  H5T_STRING {
        STRSIZE 33;
        ...
      }
      DATASPACE  SCALAR
      DATA {
      (0): \033[91m"WrongLabel_t"   # Value is wrong \033[0m
      }
    }
    """
    if 'label' not in hgroup.attrs:
        return f"Attribute 'label' is missing"

    # If we want to check kind / shape direclty on h5 object, we have to use
    # raw id (otherwise h5py decode the value).
    attr_id = hgroup.attrs.get_id('label')
    type_id = attr_id.get_type()
    
    if type_id.get_class() != h5py.h5t.STRING:
        return f"Attribute 'label' is not of kind string"
    if type_id.get_size() != 33:
        return f"Attribute 'label' has invalid size (expected 33, got {type_id.get_size()})"

    value = hgroup.attrs['label'].decode()
    if hgroup.name == '/':
        ok = value == 'Root Node of HDF5 File'
    elif hgroup.attrs['type'] == b'LK': # Linked nodes are allowed to have empty label
        ok = value == '' or value in ALL_LABELS
    else:
        ok = value in ALL_LABELS
    if not ok:
        return f"Attribute 'label' has invalid value {value}"
    
    return OK

def invalid_flags_attr(hgroup: h5py.Group) -> str:
    """W114 - Invalid 'flag' attribute

    Every HDF group should have a ' order' or 'flags' scalar attribute
    of integer kind.

    The role of this attribute is not clear, which is why this rule is
    only a warning.
    """
    if hgroup.name != '/':
        # Somehow this attributes is not registered for root node
        if hgroup.attrs.get(' order') is None and hgroup.attrs.get('flags') is None:
            return f"Attribute ' order' or 'flags' is missing"
    return OK

def additional_attributes(hgroup: h5py.Group) -> str:
    """W115 - Unexpected attributes
    
    The HDF5 mapping defines some required attributes for
    each HDF node: 'flags', 'type', 'label' and 'name'.

    This rule warns the user if other attributes are
    registered in the given node.

    Erroneous file example:

    FILE_CONTENTS {
    group      /
    ...
    group      /Base/zone/GridCoordinates
    attribute  /Base/zone/GridCoordinates/\033[32mflags\033[0m
    attribute  /Base/zone/GridCoordinates/\033[32mlabel\033[0m
    attribute  /Base/zone/GridCoordinates/\033[32mname\033[0m
    attribute  /Base/zone/GridCoordinates/\033[93msize  #Unexpected \033[0m
    attribute  /Base/zone/GridCoordinates/\033[32mtype\033[0m
    ...
    }
    """

    expected_attrs = {'flags', 'label', 'name', 'type', ' order'}
    attrs = set(hgroup.attrs.keys())
    diff = attrs - expected_attrs
    if len(diff) > 0:
        return f"Unexpected HDF5 attributes: {diff} "
    return OK

def root_specific_datasets(hgroup: h5py.Group) -> str:
    """E121 - Specific root datasets
    
    The root node of hdf file should have the ' format' and
    ' hdf5version' datasets.
    This rule check the existence, the shape, the datakind and
    the value of these datasets:

    +----------------+-------+-------+---------------------------+
    |                | shape | dtype |     admissible values     |
    +----------------+-------+-------+---------------------------+
    | ' format'      | (33,) | int8  | IEEE_{BIG|LITTLE}_{32|64} |
    | ' hdf5version' | (33,) | int8  | HDF5 Version {x}.{y}.{z}  |
    +----------------+-------+-------+---------------------------+

    See: https://cgns.org/standard/hdf5.html#id13
    """
    if hgroup.name != '/':
        return OK

    datasets = [k for k,v in hgroup.items() if isinstance(v, h5py.Dataset)]

    for name in [' format', ' hdf5version']:
        if name not in datasets:
            return f"Missing dataset '{name}' for root node"

        dataset = hgroup[name]
        if dataset.shape != (33,):
            return f"Invalid shape for root dataset '{name}': expected (33,), got {dataset.shape}"
        if dataset.dtype != 'i1':
            return f"Invalid datatype for root dataset '{name}': expected int8, got {dataset.dtype}"

        value = bytes(dataset).partition(b'\x00')[0].decode()
        ok = False
        if name == ' format':
            ok = value in ['H5T_IEEE_F32BE', 'IEEE_LITTLE_32', 'IEEE_BIG_64', 'IEEE_LITTLE_64']
        elif name == ' hdf5version':
            ok = value[:13] == 'HDF5 Version ' and bool(re.fullmatch(r'\d+\.\d+\.\d+', value[13:]))
        if not ok:
            return f"Invalid value for root dataset '{name}': {value}"

    return OK

def missing_data(hgroup: h5py.Group) -> str:
    """E122 - Missing dataset

    When a node is not of type 'MT', data must be stored
    in a dataset named ' data'.

    Erroneous file examples:

    HDF5 "mesh.cgns" {
      GROUP "/Base/zone/GridCoordinates/CoordinateX" {
        ATTRIBUTE "type" {
          DATATYPE  H5T_STRING { ... }
          DATASPACE  SCALAR
          DATA {
          (0): \033[32m"R8"\033[0m    # type is not "MT"
          }
        }
      }
      FILE_CONTENTS {
        group      /Base/zone/GridCoordinates
        group      /Base/zone/GridCoordinates/CoordinateX
        \033[91m           Missing dataset ' data' for CoordinateX group \033[0m
        group      /Base/zone/GridCoordinates/CoordinateY
        dataset    /Base/zone/GridCoordinates/CoordinateY/ data
      }
    }
    
    HDF5 "mesh.cgns" {
      GROUP "/Base/zone/GridCoordinates" {
        ATTRIBUTE "type" {
          DATATYPE  H5T_STRING { ... }
          DATASPACE  SCALAR
          DATA {
          (0): \033[32m"MT"\033[0m    # type is "MT"
          }
        }
      }
      FILE_CONTENTS {
        group      /Base/zone/GridCoordinates
        dataset    /Base/zone/GridCoordinates/\033[91m data #Unexpected because type is MT\033[0m
      }
    }
    """
    type = hgroup.attrs['type']
    has_dataset = ' data' in hgroup and isinstance(hgroup[' data'], h5py.Dataset)
    if type in [b'MT', b'LK']:
        if has_dataset:
            return f"Unexpected ' data' dataset for empty node"
    else:
        if not has_dataset:
            return f"Missing ' data' dataset for non empty node of type {type}"
    return OK

def incompatible_datatype(hgroup: h5py.Group) -> str:
    """E123 - Incompatible datatype

    When a node is not of type 'MT', the type of its
    dataset must be consistent with the descriptor
    registered in its 'type' attribute.

    Erroneous file example:

    GROUP "/Base/zone/GridCoordinates/CoordinateX" {
      ATTRIBUTE "type" {
        DATATYPE  H5T_STRING { ... }
        DATASPACE  SCALAR
        DATA {
        (0): \033[32m"R8"                    # Type attribute is R8 \033[0m
        }
      }
      DATASET " data" {
        DATATYPE  \033[91mH5T_STD_I32LE      # Array datatype is I4 \033[0m
        DATASPACE  SIMPLE { ( 27 ) / ( 27 ) }
        DATA { ... }
      }
    }
    """
    type = hgroup.attrs['type'].decode()
    if type not in ['MT', 'LK']:
        dtype = hgroup[' data'].dtype
        if dtype != CGNS_TO_DTYPE[type]:
            return f"Incompatible datatype between type attribute ({type}) and dataset ({dtype})"
    return OK


def additional_datasets(hgroup: h5py.Group) -> str:
    """W124 - Unexpected datasets
    
    The HDF5 mapping defines an optional dataset for
    each HDF node: ' data'.

    This rule warns the user if other datasets are
    registered in the given node.

    Erroneous file example:

    FILE_CONTENTS {
    group      /
    ...
    group      /Base/zone/GridCoordinates/CoordinateX
    dataset    /Base/zone/GridCoordinates/CoordinateX/\033[32m data\033[0m
    dataset    /Base/zone/GridCoordinates/CoordinateX/\033[93mvalues  #Unexpected \033[0m
    ...
    }
    """
    expected_datasets = {' data'}
    if hgroup.name == '/':
        expected_datasets |= {' format', ' hdf5version'}
    if hgroup.attrs['type'] == b'LK':
        expected_datasets |= {' file', ' path'}

    datasets = {k for k,v in hgroup.items() if isinstance(v, h5py.Dataset)}
    diff = datasets - expected_datasets
    if len(diff) > 0:
        return f"Unexpected HDF5 datasets: {diff} "
    return OK



def not_a_link(hgroup: h5py.Group) -> str:
    """E131 - Invalid 'LK' type

    If the type attribute of a group is 'LK', a ' link'
    child group must store an HDF ExternalLink or SoftLink.

    Erroneous file example:

    HDF5 "mesh.cgns" {
      GROUP "/Base/zone/FlowSolution/Density" {
        ATTRIBUTE "type" {
          DATATYPE  H5T_STRING { ... }
          DATASPACE  SCALAR
          DATA {
          (0): \033[32m"LK"\033[0m    # type is "LK"
          }
        }
      }
      FILE_CONTENTS {
        ...
        group      /Base/zone/FlowSolution/Density
        \033[91m           Missing dataset ' link' for Density group \033[0m
        group      /Base/zone/FlowSolution/Pressure
        dataset    /Base/zone/FlowSolution/Pressure/ data
        ...
      }
    }

    Correct file example:

    HDF5 "mesh.cgns" {
      GROUP "/Base/zone/FlowSolution/Density" {
        ATTRIBUTE "type" { ... } \033[32m# Same as above (LK) \033[0m
      FILE_CONTENTS {
        ...
        group      /Base/zone/FlowSolution/Density
        dataset    /Base/zone/FlowSolution/Density/ file
        \033[32mext link   /Base/zone/FlowSolution/Density/ link -> OUTPUT/fields.cgns Init/Density\033[0m
        dataset    /Base/zone/FlowSolution/Density/ path
        ...
      }
    }
    """
    if hgroup.attrs['type'] != b'LK':
        return OK
    
    if ' link' not in hgroup:
        return f"Attribute type is 'LK', but child ' link' is missing"

    if isinstance(hgroup.get(' link', getlink=True), h5py.HardLink):
        return f"Attribute type is 'LK', but child ' link' is a HardLink"
    
    return OK

def open_linked_file(hgroup: h5py.Group) -> str:
    """E132 - Unopenable target file

    The target (external) file of a linked node can not be open.
    Most common reasons are:
    - The target file does not exists 
    - The target file has not read permission
    - The target file is already open by an other application
    """
    link = hgroup.get(' link', getlink=True)
    if isinstance(link, h5py.ExternalLink):
        rootdir = Path(hgroup.file.filename).parent
        targetfile = rootdir / link.filename
        try:
            h5py.File(targetfile)
        except OSError as e:
            msg = str(e)
            match = re.search(r"error message\s*=\s*'([^']+)'", msg)
            if match:
                return f"Can not open target file {link.filename} of linked node ({match.group(1)})"
            else:
                return f"Can not open target file {link.filename} of linked node (unknow reason)"

    return OK

def open_linked_path(hgroup: h5py.Group) -> str:
    """E133 - Unreachable target path

    The target path of a linked node does not exists in the
    target file.
    """
    link = hgroup.get(' link', getlink=True)
    if isinstance(link, h5py.ExternalLink):
        rootdir = Path(hgroup.file.filename).parent
        targetfile = rootdir / link.filename
        try:
            with h5py.File(targetfile) as f:
                if link.path not in f:
                    return f"Can not find target node {link.path} in target file {link.filename}"
        except Exception:
            pass # File open exception are treated in an other rule
    elif isinstance(link, h5py.SoftLink):
        if link.path not in hgroup.file:
            return f"Can not find target node {link.path} in current file"
    return OK

def lk_specific_datasets(hgroup: h5py.Group) -> str:
    """E134 - Specific link datasets

    Linked nodes should have the ' path' and, if external,
    the ' file' datasets. 
    The values of these datasets must match the data registered
    in the HDF link.

    Erroneous file example:

    HDF5 "mesh.cgns" {
      GROUP "/Base/zone/FlowSolution/Density" {
        DATASET " file" {
          DATATYPE  H5T_STD_I8LE
          DATASPACE  SIMPLE { ( 19 ) / ( 19 ) }
          DATA { ... } \033[32m# Decoded value is OUTPUT/fields.cgns\033[0m
        }
      }
      FILE_CONTENTS {
        ...
        group      /Base/zone/FlowSolution/Density
        dataset    /Base/zone/FlowSolution/Density/ file
        ext link   /Base/zone/FlowSolution/Density/ link -> \033[91mOUTPUT/data.cgns\033[0m Init/Density \033[91m# Mismatch\033[0m
        \033[91m           Missing dataset ' path' for Density group \033[0m
        ...
      }
    }
    """
    if hgroup.attrs['type'] != b'LK':
        return OK

    link = hgroup.get(' link', getlink=True)

    # NB : file only for external; path for all (if not a link, do nothing)
    names = []
    if isinstance(link, h5py.ExternalLink):
        names = [' file', ' path']
    elif isinstance(link, h5py.SoftLink):
        names = [' path']

    for name in names:
        try:
            dataset = hgroup[name]
        except KeyError:
            return f"Missing '{name}' dataset for linked node"
        if not isinstance(dataset, h5py.Dataset):
            return f"Child '{name}' is not a dataset"
        if dataset.dtype != 'i1':
            return f"Invalid datatype for link dataset '{name}': expected int8, got {dataset.dtype}"
        value = bytes(dataset).decode()[:-1] # Remove trainling '\0'
        ref = {' file' : link.filename, ' path': link.path}[name] #type:ignore
        if value != ref:
            return f"Value of '{name}' dataset ({value}) is not equal to registered target {name[1:]} ({ref})"

    return OK



_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)

GROUP_RULES = {func[1].__doc__[:4] : func[1] for func in _funcs} #type:ignore
assert len(GROUP_RULES) == len(_funcs)

# 
# The following function applies on a Path object
# 
# They must be placed after GROUP_RULES creation otherwise they will be included
# in GROUP_RULES. These functions are collected manually because there is few of it.
#

def path_exists(filename:Path) -> str: 
    """F101 - File not found
    
    The provided path does not exists.
    """
    try:
        if filename.exists():
            return OK
        else:
            return f"File {filename} does not exists"
    except PermissionError:
        return f"Unable to open parent directory of {filename} (permission denied)"

def path_is_file(filename:Path) -> str: 
    """F102 - Not a file
    
    The provided path is not a file.
    """
    if not filename.is_file():
        return f"Path {filename} is not a regular file"
    return OK

def file_is_readable(filename:Path) -> str:
    """F103 - Unreadable file
    
    Access to the provided file is not granted.
    """
    if not os.access(filename, os.R_OK):
        return f"Can not read {filename} (permission denied)"
    return OK

def file_is_hdf5(filename:Path) -> str: 
    """F104 - Not a valid hdf5 file
    
    The provided file is not a hdf5 container.
    """
    if not h5py.is_hdf5(filename):
        return f"File {filename} is not a valid hdf5 file"
    return OK

FILE_RULES = {func.__doc__[:4]: func for func in \
              [path_exists, path_is_file, file_is_readable, file_is_hdf5]}
