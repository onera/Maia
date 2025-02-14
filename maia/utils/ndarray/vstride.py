import sys
import itertools
import operator
import numpy as np

from cmaia.utils import vstride as _vstride

from enum import Enum

ReduceOp = Enum('ReduceOp', 'SUM PROD MIN MAX LAND LOR BAND BOR')
Axis     = Enum('Axis',     'INNER OUTER')

ReduceOp.__doc__ = """
Enumeration storing the available operations. Members are :attr:`SUM`, :attr:`PROD`, :attr:`MIN`, :attr:`MAX`, 
:attr:`LAND`, :attr:`LOR`, :attr:`BAND` and :attr:`BOR`,
where 'L' stands for logical operations and 'B' for bitwise operations.
"""
Axis.__doc__ = """
Enumeration to indicate the axis on which a function should operate. Members are :attr:`INNER` and :attr:`OUTER`.
"""

INNER_AXIS = Axis.INNER #: A shortcut to :data:`Axis.INNER`
OUTER_AXIS = Axis.OUTER #: A shortcut to :data:`Axis.OUTER`

_UNVALID_AXIS_MSG = "Unvalid value for axis"

class VStrideArray:
  """ A class representing a variable stride array.

  Once created, a :class:`VStrideArray` instance ``arr`` has the 
  following attributes:

  +----------------+---------------------------------------------+
  | :attr:`displs` | view on displs array                        |
  +----------------+---------------------------------------------+
  | :attr:`counts` | view on counts array                        |
  +----------------+---------------------------------------------+
  | :attr:`values` | values array                                |
  +----------------+---------------------------------------------+
  | :attr:`dsize`  | size of the underlying ``values`` array     |
  +----------------+---------------------------------------------+
  | :attr:`dtype`  | datatype of the underlying ``values`` array |
  +----------------+---------------------------------------------+

  In addition, the following operations are supported:

  - Length: ``len(arr)`` returns the number of elements N.
  - Basic indexing: ``arr[i]`` returns the ith *block*, ie a view on the ``values`` array.

  - Basic assignement: ``arr[i] = val`` replace the :math:`m_i` values of the ith *block*
    using the provided input ``val``, which must be an array of relevant size :math:`m_i` or a scalar
    (it will be broadcasted).

  .. warning:: We strongly advise to **not** iterate over blocks using python loops such as
     ``func(blk) for blk in arr``. As for numpy arrays, this would lead to poor performances.
     Several functions or methods are provided to avoid such loops.

  - Unary arithmetic operations ``-``, ``+``, ``abs()`` and ``~`` applies
    element wise on the ``values`` array.
  - Binary arithmetic operations (such as ``+``, ``-``, ``*``, ``/``, ``%``, ``**``, ``&``, etc.)
    and comparison operation (such as ``==``, ``!=``, ``<``, ``>=``, etc.)
    also apply element wise on the ``values`` array. The right operand can be:

    - an other :class:`VStrideArray` object. In this case, the *strides* of the two operand must be equal:
        
        >>> a = vs.from_counts([2,2,1], [0.2, 1.4, 2.6, 0.5, 1.0])
        >>> b = vs.from_counts([2,2,1], [0.1, 2.3, 1.4, 0.6, 0.9])
        >>> a <= b
        vsarray([
          [False,  True],
          [False,  True],
          [False],
        ], dtype=bool)

    - a numpy array of size :math:`N`. In this case, each value will be repeated :math:`m_i` times::

        >>> vs.from_counts([2,2,1], np.arange(5)) + np.arange(3)
        vsarray([
          [0, 1],
          [3, 4],
          [6],
        ], dtype=int64)

    - a scalar value. In this case, it will we broadcasted to a constant array of size ``values.size``::

        >>> vs.from_counts([2,2,1], np.arange(5)) * 2
        vsarray([
          [0, 2],
          [4, 6],
          [8],
        ], dtype=int64)

    These operations return a new :class:`VStrideArray` instance. The output dtype of its values array is
    determined by numpy, to which the operation itself is delegated.

  - Inplace arithmetic operations (such as ``+=``, ``-=``, ``*=``, etc.) are also supported
    under the same conditions for the right operand.

  """


  # Constructors
  def __init__(self, displs, counts, values):
    """ Create a new VStrideArray object

    At least one of ``displs`` or ``counts`` argument must not be ``None``.
    The argument ``values`` must never be ``None``.

    Args:
      displs (integer ndarray, optional) : array of displs or None
      counts (integer ndarray, optional) : array of counts or None
      values (ndarray) : array of values

    Example: 
      >>> vs.VStrideArray(None, np.array([3,5,2]),  np.arange(10))
      vsarray([
        [0, 1, 2],
        [3, 4, 5, 6, 7],
        [8, 9],
      ], dtype=int64)
    """
    assert (displs is not None) or (counts is not None)
    assert values is not None

    self._displs = None
    self._counts = None

    if displs is not None:
      assert isinstance(displs, np.ndarray) and np.issubdtype(displs.dtype, np.integer)
      assert displs.ndim == 1 and displs.data.contiguous
      assert displs[0] == 0 and displs[-1] == values.size
      self._displs = displs

    if counts is not None:
      assert isinstance(counts, np.ndarray) and np.issubdtype(counts.dtype, np.integer)
      assert counts.ndim == 1 and counts.data.contiguous
      assert counts.sum() == values.size
      self._counts = counts

    if counts is not None and displs is not None:
      assert counts.dtype == displs.dtype

    assert isinstance(values, np.ndarray)
    assert values.ndim == 1 and values.data.contiguous
    self._values = values

  # Accessors
  @property
  def displs(self):
    if self._displs is None:
      self._displs = np.empty(self._counts.size+1, self._counts.dtype)
      self._displs[0] = 0
      np.cumsum(self._counts, out=self._displs[1:])
    view = self._displs.view()
    view.flags.writeable = False
    return view

  @property
  def counts(self):
    if self._counts is None:
      self._counts = self._displs[1:] - self._displs[:-1]

    view = self._counts.view()
    view.flags.writeable = False
    return view

  @property
  def values(self):
    return self._values

  # Other usefull properties
  @property
  def dtype(self):
    return self._values.dtype
  @property
  def dsize(self):
    return self._values.size

  #Emulate container methods
  def __len__(self):
    """ Return the number of blocks in the structure """
    return self._counts.size if self._counts is not None else self._displs.size - 1

  def __getitem__(self, key):
    """ Return a view to the requested block"""
    if isinstance(key, int):
      try:
        return self._values[self.displs[key - int(key<0)]:self.displs[key - int(key<0)+1]]
      except IndexError:
        raise IndexError(f"Tried to access index {key}, but len is {len(self)}")

    raise TypeError

  def __setitem__(self, key, val):
    """ Modify the requested block; size and dtype of val should be consistant"""
    #assert val.dtype == self.dtype
    self.__getitem__(key)[:] = val

  # BOILERPLATE part to overload classical operators
  def __numeric_iop__(self, other, op):
    if isinstance(other, VStrideArray):
      if len(self) != len(other):
        raise ValueError(f"the two VStrideArray instances does not have same len ({len(self)} vs {len(other)})")
      if not np.array_equal(self.counts, other.counts):
        diff = sum(self.counts != other.counts)
        raise ValueError(f"the two VStrideArray instances does not have same counts ({diff} indices differ)")
      op(self._values, other.values)

    elif isinstance(other, np.ndarray):
      if len(self) != len(other):
        raise ValueError(f"raw array can be broadcasted only if its size is the same than VStrideArray ({len(other)} vs {len(self)})")
      op(self._values, np.repeat(other, self.counts))

    elif np.isscalar(other):
      op(self._values, other)

    else:
      raise NotImplemented
      #raise TypeError(f"Unsupported operand type(s) for += : '{type(self).__name__}' and '{type(other).__name__}'")

    return self

  def __numeric_op__(self, other, op):

    # Trick to avoid checks at creation time
    out = super().__new__(VStrideArray)
    out._displs = self._displs
    out._counts = self._counts

    if isinstance(other, VStrideArray):
      if len(self) != len(other):
        raise ValueError(f"the two VStrideArray instances does not have same len ({len(self)} vs {len(other)})")
      if not strides_equal(self, other):
        diff = sum(self.counts != other.counts)
        raise ValueError(f"the two VStrideArray instances does not have same counts ({diff} indices differ)")
      out._values = op(self._values, other._values)

    elif isinstance(other, np.ndarray):
      if len(self) != len(other):
        raise ValueError(f"raw array can be broadcasted only if its size is the same than VStrideArray ({len(other)} vs {len(self)})")
      out._values = op(self._values, np.repeat(other, self.counts))

    elif np.isscalar(other):
      out._values = op(self._values, other)

    else:
      raise NotImplemented
      #raise TypeError(f"Unsupported operand type(s) for += : '{type(self).__name__}' and '{type(other).__name__}'")

    return out

  def __unary_op__(self, op):
    out = super().__new__(VStrideArray)
    out._counts = self._counts
    out._displs = self._displs
    out._values = op(self.values)
    return out


  def __iadd__(self, other):
    return self.__numeric_iop__(other, operator.iadd)
  def __isub__(self, other):
    return self.__numeric_iop__(other, operator.isub)
  def __imul__(self, other):
    return self.__numeric_iop__(other, operator.imul)
  def __itruediv__(self, other):
    return self.__numeric_iop__(other, operator.itruediv)
  def __ifloordiv__(self, other):
    return self.__numeric_iop__(other, operator.ifloordiv)
  def __imod__(self, other):
    return self.__numeric_iop__(other, operator.imod)
  def __ipow__(self, other):
    return self.__numeric_iop__(other, operator.ipow)
  def __ilshift__(self, other):
    return self.__numeric_iop__(other, operator.ilshift)
  def __irshift__(self, other):
    return self.__numeric_iop__(other, operator.irshift)
  def __iand__(self, other):
    return self.__numeric_iop__(other, operator.iand)
  def __ixor__(self, other):
    return self.__numeric_iop__(other, operator.ixor)
  def __ior__(self, other):
    return self.__numeric_iop__(other, operator.ior)

  def __add__(self, other):
    return self.__numeric_op__(other, operator.add)
  def __sub__(self, other):
    return self.__numeric_op__(other, operator.sub)
  def __mul__(self, other):
    return self.__numeric_op__(other, operator.mul)
  def __truediv__(self, other):
    return self.__numeric_op__(other, operator.truediv)
  def __floordiv__(self, other):
    return self.__numeric_op__(other, operator.floordiv)
  def __mod__(self, other):
    return self.__numeric_op__(other, operator.mod)
  def __pow__(self, other):
    return self.__numeric_op__(other, operator.pow)
  def __lshift__(self, other):
    return self.__numeric_op__(other, operator.lshift)
  def __rshift__(self, other):
    return self.__numeric_op__(other, operator.rshift)
  def __and__(self, other):
    return self.__numeric_op__(other, operator.and_)
  def __xor__(self, other):
    return self.__numeric_op__(other, operator.xor)
  def __or__(self, other):
    return self.__numeric_op__(other, operator.or_)

  def __lt__(self, other):
    return self.__numeric_op__(other, operator.lt)
  def __le__(self, other):
    return self.__numeric_op__(other, operator.le)
  def __eq__(self, other):
    return self.__numeric_op__(other, operator.eq)
  def __ne__(self, other):
    return self.__numeric_op__(other, operator.ne)
  def __gt__(self, other):
    return self.__numeric_op__(other, operator.gt)
  def __ge__(self, other):
    return self.__numeric_op__(other, operator.ge)

  def __neg__(self):
    return self.__unary_op__(operator.neg)
  def __pos__(self):
    return self.__unary_op__(operator.pos)
  def __abs__(self):
    return self.__unary_op__(operator.abs)
  def __invert__(self):
    return self.__unary_op__(operator.invert)

  # For compatibility

  def _inner_sort(self):
    _vstride.sort_by_stride(self.displs, self.values)
  def _inner_flip(self):
    _vstride.flip_by_stride(self.displs, self.values)

  # Methods

  def reduce(self, op:ReduceOp):
    """ Apply a reduction operation within each *block*.

    The avalaible operations are the members of the enumeration :class:`ReduceOp`.

    This function returns an array of size :math:`N` (one value per *block*).
    The datatype of the output array depends on the underlying operation, which
    is executed by numpy.

    Note:
      For empty *blocks* (*ie* for the set of ``i`` such that ``counts[i] == 0``), the corresponding
      result ``out[i]`` will be initialized with the neutral value of the corresponding operation.

      It is still possible to filter the result afterward, *eg* with ``out[self.counts > 0]``.
    
    Args:
      op (:class:`ReduceOp`): operation performed to reduce the *blocks*
    Returns:
      flat ndarray of size N : result of the reduction

    Example: 
      >>> a = vs.from_counts([3,5,2], np.arange(10))
      >>> a.reduce(vs.ReduceOp.SUM)
      array([ 3, 23, 17])
    """
    # For information : output type depending on input/op
    #
    #       add/mul   max/min land/lor  band/bor
    #  b      i8         b        b         b
    # i4      i8        i4        b        i4
    # i8      i8        i8        b        i8
    # f4      f4        f4        b         x
    # f8      f8        f8        b         x
    # 
    # Neutral 0/1                T/F      -1/0
    
    op_to_ufunc = {ReduceOp.SUM  : np.add,
                   ReduceOp.PROD : np.multiply,
                   ReduceOp.MIN  : np.minimum,
                   ReduceOp.MAX  : np.maximum,
                   ReduceOp.LAND : np.logical_and,
                   ReduceOp.LOR  : np.logical_or,
                   ReduceOp.BAND : np.bitwise_and,
                   ReduceOp.BOR  : np.bitwise_or}
    
    ufunc = op_to_ufunc[op]
    out = ufunc.reduceat(self._values, self.displs[:-1])
    
    if op == ReduceOp.MIN:
      if out.dtype.kind == 'i':
        val = np.iinfo(out.dtype).min
      elif out.dtype.kind == 'f':
        val = -np.inf
      elif out.dtype.kind == 'b':
        val = False
    elif op == ReduceOp.MAX:
      if out.dtype.kind == 'i':
        val = np.iinfo(out.dtype).max
      elif out.dtype.kind == 'f':
        val = np.inf
      elif out.dtype.kind == 'b':
        val = True
    else:
      val = ufunc.identity

    out[self.counts == 0] = val
    return out

  def restride(self, displs=None, counts=None):
    """
    Change the *strides* (``counts`` and ``displs``) of the array, without updating its ``values``.

    Eiher a new ``displs`` or a new ``counts`` array can be provided. If both arguments are ``None``,
    the object remain unchanged. Note that the new strides must be compatible with the :attr:`array.dsize`
    attribute.

    Example: 
      >>> a = vs.from_counts([1, 2, 5], [0.4, 0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
      >>> a.restride(counts=[4,4])
      >>> a
      vsarray([
        [0.4, 0.3, 0.5, 0.1],
        [0.7, 0.2, 0.6, 0.9],
      ], dtype=float64)
    """
    if displs is not None:
      displs = np.asarray(displs)
      assert np.issubdtype(displs.dtype, np.integer)
      assert displs.ndim == 1 and displs.data.contiguous
      assert displs[0] == 0 and displs[-1] == self.values.size
      self._displs = displs
      self._counts = None
    elif counts is not None:
      counts = np.asarray(counts)
      assert np.issubdtype(counts.dtype, np.integer)
      assert counts.ndim == 1 and counts.data.contiguous
      assert counts.sum() == self.values.size
      self._counts = counts
      self._displs = None


  def to_array_list(self):
    """ Return the ``values`` as a list of NumPy 1d arrays.

    The len of the output list is ``N``, and the size of each element ``i`` is ``counts[i]``.
    Data is copied into the 1d arrays.

    Returns:
      list of ndarray: values as a list of 1d arrays
    Example: 
      >>> a = vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
      >>> a.to_array_list()
      [array([], dtype=float64), array([0.3, 0.5]), array([0.1, 0.7, 0.2, 0.6, 0.9])]
    """
    return [blk.copy() for blk in self]

  def to_masked_array(self):
    """ Return the ``values`` as a NumPy `masked ndarray <https://numpy.org/doc/stable/reference/maskedarray.html>`_.

    Output is a 2d array of shape ``(N, max(counts))``, where the masked value is used to complete each row 
    ``i`` for which ``counts[i] < max(counts)``.
    Data is copied into the masked array.

    Returns:
      masked ndarray: values as a masked array
    Example: 
      >>> a = vs.from_displs([0,3,6,6,8,9], [1,3,3,4,5,6,7,8,10])
      >>> print(a.to_masked_array())
      [[1 3 3]
       [4 5 6]
       [-- -- --]
       [7 8 --]
       [10 -- --]]
    """
    shape = (len(self), self.counts.max(initial=0))
    ma = np.ma.empty(shape, self.dtype, order='F') 
    for i,blk in enumerate(self):
      ma[i,0:self.counts[i]] = blk
      ma[i,self.counts[i]:] = np.ma.masked
    return ma

  # Representation
  def __repr__(self):
    vthreshold = 50
    vedgeitems = 3

    hthreshold = 50
    hedgeitems = 3

    def split_line(line):
      # Util function that add linebreak to have at most elt_per_line items on each line
      elts = line.split(',')
      n_split = max((len(elts)-1) // elt_per_line, 0)
      for i in range(n_split):
        elts[(i+1)*elt_per_line] = '\n  ' + elts[(i+1)*elt_per_line]
      return ','.join(elts)

    # Ranges to iterate before and after vertical threshold
    if len(self) > vthreshold:
      range1, range2 = range(0, vedgeitems), range(len(self)-vedgeitems, len(self))
    else:
      range1, range2 = range(0, len(self)), iter(())

    # This mask tell us which elements will be displayed, to compute the __repr__
    # based on them (itself a vsarray)
    is_visible = VStrideArray(self.displs, self.counts, np.zeros(self.dsize, bool))
    for i in itertools.chain(range1, range2):
      is_visible[i] = True
      if is_visible[i].size > hthreshold:
        is_visible[i][hedgeitems : -hedgeitems] = False

    # Delegate repr to numpy but without any linebreak
    # Then the idea is to split this repr for each block
    with np.printoptions(threshold=sys.maxsize):
      full_repr = np.array_repr(self.values[is_visible.values], max_line_width=sys.maxsize)

    full_repr = full_repr[7:]            # Remove start of str ('array')
    full_repr = full_repr.rsplit(']')[0] # Remove end of str   ('dtype=...')
    elements = full_repr.split(',')

    if len(elements) > 0:
      elements[0] = ' ' + elements[0]    # Nupy remove first space -> add it for uniform treatment
      e_len = len(elements[0])
      elt_per_line = (np.get_printoptions()['linewidth'] - 4) // (e_len+1)

    lines = ["vsarray(["]
    read_idx = 0
    for i in itertools.chain(range1, range2):
      n_visible = is_visible[i].sum()
      subelemts = elements[read_idx : read_idx+n_visible] # Can't use [i], because elements is already filtered
      sep = [] if is_visible[i].all() else ['...'.center(e_len)]
      if n_visible > 0:
        subelemts[0] = subelemts[0][1:] # Remove first space
      subrepr = ','.join(itertools.chain(subelemts[:n_visible//2], sep, subelemts[n_visible//2:]))

      lines.append('  [' + split_line(subrepr) + '],')
      read_idx += n_visible

    if len(self) > vthreshold:
      lines.insert(1+vedgeitems, 2*' ' + '...,')

    lines.append(f'], dtype={self.dtype})')   # Finalize repr with dtype

    return '\n'.join(lines)



##### Module functions


##### Constructors

def array(data, *, dtype=None):
  """ Create a new instance from the input data.

  Input data can be either:

    - a list of objects convertible to a 1d ndarray. This include *eg* list of lists, list
      of arrays, but not list of scalars;
    - a 2d `masked ndarray <https://numpy.org/doc/stable/reference/maskedarray.html>`_, in which case
      only *visible* values are selected;
    - an other :class:`VStrideArray` instance.

  Input data is always copied to create the new array.

  Args:
    data (object): see above
    dtype (data-type, optional): expected datatype of the ``values`` array.
      If ``None``, datatype is inferred by NumPy from the input data.
  Returns:
    :class:`VStrideArray` : new array
  Examples:
    
    >>> vs.array([[1,2], [3,4,5], [], [6]]) # From nested lists
    vsarray([
      [1, 2],
      [3, 4, 5],
      [],
      [6]
    ], dtype=int64)
    >>> data = np.ma.array([[1, 2, 3], [4,5,6], [7, 8, 9]], # From masked array
    ...               mask=[[0, 0, 1], [0,0,0], [0,1,1]])
    >>> vs.array(data)
    vsarray([
      [1, 2],
      [4, 5, 6],
      [7],
    ], dtype=int64)
  """

  if isinstance(data, VStrideArray):
    new_displs = None
    new_counts = None
    new_values = np.array(data.values, dtype=dtype, copy=True)
    if data._displs is not None:
      new_displs = np.array(data._displs, dtype=int, copy=True)
    elif data._counts is not None:
      new_counts = np.array(data._counts, dtype=int, copy=True)
    return VStrideArray(new_displs, new_counts, new_values)

  elif isinstance(data, list):
    # Inner data must be 1d sequence (TODO : checks)
    #if len(data) == 0 and dtype is None:
      #raise ValueError("Can not infer dtype from empty list")
    arrays = [np.asarray(block, dtype=dtype) for block in data]
    counts = np.array([len(a) for a in arrays], int)
    if len(arrays) == 0:
      if dtype is None:
        raise ValueError("Can not concatenate empty list of arrays if dtype is not provided")
      values = np.empty(0, dtype)
    else:
      values = np.concatenate(arrays)
    return VStrideArray(None, counts, values)
    
  elif isinstance(data, np.ma.masked_array):
    assert data.ndim == 2
    selected = ~data.mask
    counts = selected.sum(axis=1)
    values = np.asarray(data.data[selected], dtype=dtype)
    return VStrideArray(None, counts, values)

  elif isinstance(data, np.ndarray):
    assert data.ndim == 2
    counts = np.full(data.shape[0], data.shape[1], int)
    values = data.flatten()
    if dtype is not None:
      values = values.astype(dtype, copy=False)
    return VStrideArray(None, counts, values)

  else:
    raise ValueError

vsarray = array # A shortcut to rebuild object from their __repr__

def from_counts(counts, values, *, dtype=None) -> VStrideArray:
  """ Create a new instance from a 1d array described by its ``counts``.

  Arguments are converted to ndarray using
  `np.asarray <https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray>`_:
  this operation is copyless if arguments are already ndarray objects and
  if ``values`` matches the requested datatype.
  
  Args:
    counts (array_like): an object convertible to a 1d integer ndarray
    values (array_like): an object convertible to a 1d ndarray
    dtype (data-type, optional): overide datatype of the ``values`` array.
      If ``None``, datatype is inferred from the input data.
  Returns:
    :class:`VStrideArray` : new array
  Example:
    >>> vs.from_counts([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2, 0.6, 0.9])
    vsarray([
      [],
      [0.3, 0.5],
      [0.1, 0.7, 0.2, 0.6, 0.9],
    ], dtype=float64)
  """
  if isinstance (counts, (int, np.integer)):
    assert len(values) % counts == 0
    counts = np.full(len(values) // counts, counts)

  if len(counts) == 0 and not isinstance(counts, np.ndarray):
    counts = np.empty(0, int)

  return VStrideArray(None, 
                      np.asarray(counts),
                      np.asarray(values, dtype=dtype))

def from_displs(displs, values, *, dtype=None) -> VStrideArray:
  """ Create a new instance from a 1d array described by its ``displs``.

  Arguments are converted to ndarray using
  `np.asarray <https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray>`_:
  this operation is copyless if arguments are already ndarray objects and
  if ``values`` matches the requested datatype.

  Args:
    displs (array_like): an object convertible to a 1d integer ndarray
    values (array_like): an object convertible to a 1d ndarray
    dtype (data-type, optional): overide datatype of the ``values`` array.
      If ``None``, datatype is inferred from the input data.
  Returns:
    :class:`VStrideArray` : new array
  Example:
    >>> vs.from_displs([0, 2, 5], [0.3, 0.5, 0.1, 0.7, 0.2], dtype='f4')
    vsarray([
      [0.3, 0.5],
      [0.1, 0.7, 0.2],
    ], dtype=float32)
    """
  return VStrideArray(np.asarray(displs),
                      None,
                      np.asarray(values, dtype=dtype))



#### Indexing

def take(array, indices):
  """ Take elements from the input array.
  
  Indices to extract directly refer to elements number,
  and must thus be included in ``[0, len(array)[``.
  A given index can be provided more than once.

  A new object is returned.

  Args:
    array (:class:`VStrideArray`): input array
    indices (array of int) : indices of the elements to extract
  Returns:
    :class:`VStrideArray` : extracted array

  Example:
    >>> a = vs.from_displs([0, 2, 4, 6, 9, 10], values=np.arange(10))
    >>> vs.take(a, [2, 1, 4, 1])
    vsarray([
      [4, 5],
      [2, 3],
      [9],
      [2, 3],
    ], dtype=int64)
  """

  dtype = int if len(indices) == 0 else None
  indices = np.asarray(indices, dtype)

  displs_in = array.displs
  values_in = array.values

  counts = displs_in[indices+1] - displs_in[indices]
  values = np.empty(counts.sum(), values_in.dtype)

  _vstride.take(displs_in, values_in, indices, values)

  return VStrideArray(None, counts, values)

def put(array:VStrideArray, indices, values:VStrideArray):
  """ Update the specified elements of input array with provided values.
  
  Indices to update directly refer to elements number,
  and must thus be included in ``[0, len(array)[``.
  A given index can be provided more than once; in this case, only the last
  corresponding value is used.

  The new ``values`` must be provided as a :class:`VStrideArray` of lenght ``len(indices)``.
  Its datatype will be converted if needed to match :attr:`array.dtype`.

  If ``indices`` is a scalar value, then a single 1d array_like object is allowed for
  ``values``.

  Note:
    Contrary to np.put, this function does not operate inplace,
    since the *strides* of the input array can be modified. A new
    object is returned.

  Args:
    array (:class:`VStrideArray`): input array
    indices (int or array of int) : indices of the elements to update
    values (array_like or :class:`VStrideArray`) : *block(s)* to write
  Returns:
    :class:`VStrideArray` : updated array

  Example:
    >>> a = vs.from_counts([2, 3, 1, 3], np.arange(9))
    >>> vals = vs.array([[-1,-2,-3,-4], [99]])
    >>> vs.put(a, [2,0], vals)
    vsarray([
      [99],
      [ 2,  3,  4],
      [-1, -2, -3, -4],
      [ 6,  7,  8],
    ], dtype=int64)
  """
  if isinstance(indices, (int, np.integer)):
    i = indices
    if not (0 <= i and i < len(array)):
      raise IndexError(f"Index {i} is out of bounds for array of size {len(array)}")
    if not np.can_cast(np.asarray(values).dtype, array.dtype):
      raise TypeError(f"Casting {np.asarray(values).dtype} into {array.dtype} is not safe")
    new_counts = array.counts.copy()
    new_counts[i] = len(values)
    displs = array.displs
    new_values = np.concatenate([array.values[:displs[i]], values, array.values[displs[i+1]:]])

    return VStrideArray(None, new_counts, new_values) 

  else:

    if len(indices) != len(values):
      msg = f"indices and values must have the same length ({len(indices)} vs {len(values)})"
      raise ValueError(msg)
    if not np.can_cast(values.dtype, array.dtype):
      raise TypeError(f"Casting {values.dtype} into {array.dtype} is not safe")
    
    # First compute the new counts arrays and allocate empty new values
    new_counts = array.counts.copy()
    np.put(new_counts, indices, values.counts)
    new_values = np.empty(new_counts.sum(), array.dtype)

    # Values to write are preexisting values + new values ; concat. indices as well
    to_write = concatenate([array, values], OUTER_AXIS)
    indices = np.concatenate([np.arange(len(array)), np.asarray(indices, dtype=int)])
    
    # Write data
    _vstride.put(new_counts, new_values, indices, to_write.counts, to_write.values)

    return VStrideArray(None, new_counts, new_values)

def delete(array, indices):
  """ Remove elements from the input array.

  Indices to delete directly refer to elements number,
  and must thus be included in ``[0, len(array)[``.

  A new object is returned.

  Args:
    array (:class:`VStrideArray`): input array
    indices (array of int) : indices of the elements to remove
  Returns:
    :class:`VStrideArray` : filtered array

  Example:
    >>> a = vs.from_displs([0, 2, 4, 6, 9, 10], np.arange(10))
    >>> vs.delete(a, [2, 1, 4])
    vsarray([
      [0, 1],
      [6, 7, 8],
    ], dtype=int64)
  """
  mask = np.ones(len(array), bool)
  indices = np.asarray(indices)
  mask[indices] = False
  extended_mask = np.repeat(mask, array.counts)

  counts = array.counts[mask]
  values = array.values[extended_mask]
  return VStrideArray(None, counts, values)

def insert(array:VStrideArray, indices, values:VStrideArray):
  """ Insert new elements in the input array.

  The indices where the new block(s) are insered must be
  included in ``[0, len(array)]``. The insered ``values``
  must be provided as a :class:`VStrideArray` of length ``len(indices)``. 
  Its datatype will be converted if needed to match :attr:`array.dtype`.

  If ``indices`` is a scalar value, then a single 1d array_like object is allowed for
  ``values``.

  A new object array is returned.

  Args:
    array (:class:`VStrideArray`): input array
    index (int of array of int) : position where the *block(s)* should be insered
    values (array_like or :class:`VStrideArray`) : *block(s)* to insert
  Returns:
    :class:`VStrideArray` : new array
  Example:
    >>> a = vs.from_counts([2, 4, 3], np.arange(9))
    >>> vs.insert(a, 1, [9,10,11])
    vsarray([
      [ 0,  1],
      [ 9, 10, 11],
      [ 2,  3,  4,  5],
      [ 6,  7,  8],
    ], dtype=int64)
  """
  if isinstance(indices, (int, np.integer)):
    if not (0 <= indices and indices <= len(array)):
      raise IndexError(f"Index {indices} is out of bounds for array of size {len(array)}")
    counts = np.insert(array.counts, indices, len(values))
    values = np.insert(array.values, array.displs[indices], values)
    return VStrideArray(None, counts, values) 

  else:
    assert isinstance(values, VStrideArray)
    if len(indices) != len(values):
      msg = f"indices and values must have the same length ({len(indices)} vs {len(values)})"
      raise ValueError(msg)
    if not np.can_cast(values.dtype, array.dtype):
      raise TypeError(f"Casting {values.dtype} into {array.dtype} is not safe")
    
    values_len     = np.array([len(v) for v in values], array.counts.dtype)
    extented_pos   = np.repeat(array.displs[indices], values_len)
    counts = np.insert(array.counts, indices, values_len)
    values = np.insert(array.values, extented_pos, values.values)
    return VStrideArray(None, counts, values)



#### INNER / OUTER algorithms


def flip(array: VStrideArray, axis:Axis):
  """ Reverse the order of values of the input array.

  Depending on the ``axis`` argument, the operation reverse:

  - the order of *blocks* if ``axis==OUTER_AXIS``, which is roughly equivalent to ::

      vs.array([blk for blk in array][::-1])

  - each *block* independently if ``axis==INNER_AXIS``, which is roughly equivalent to ::

      vs.array([blk[::-1] for blk in array)]
  
  In both cases, a copy is done and a new VStrideArray is returned.

  Args:
    array (:class:`VStrideArray`): input array
    axis (:class:`Axis`): direction used to reverse
  Returns:
    :class:`VStrideArray` : flipped array
  Example:
    >>> a = vs.from_counts([2, 4, 3], np.arange(9))
    >>> vs.flip(a, vs.OUTER_AXIS)
    vsarray([
      [6, 7, 8],
      [2, 3, 4, 5],
      [0, 1],
    ], dtype=int64)
    >>> vs.flip(a, vs.INNER_AXIS)
    vsarray([
      [1, 0],
      [5, 4, 3, 2],
      [8, 7, 6],
    ], dtype=int64)
  """

  if axis == INNER_AXIS:
    displs = array.displs
    values = array.values.copy()
    _vstride.flip_by_stride(displs, values)
    return VStrideArray(array._displs, array._counts, values)

  elif axis == OUTER_AXIS:
    indices = np.arange(len(array)-1, -1, -1)
    return take(array, indices)

  else:
    raise ValueError(_UNVALID_AXIS_MSG)

def sort(array: VStrideArray, axis:Axis):
  """ Sort the values of the input array.

  Depending on the ``axis`` argument, the operation sort:

  - the elements if ``axis==OUTER_AXIS``, which is roughly equivalent to ::

      vs.array(sorted([blk for blk in array]))
    
    In this case, lexicographic order is used to compare two *blocks*.

  - each block if ``axis==INNER_AXIS``, which is roughly equivalent to ::

      vs.array([sort(blk) for blk in array)]
  
  In both cases, a copy is done and a new VStrideArray is returned.

  Args:
    array (:class:`VStrideArray`): input array
    axis (:class:`Axis`): direction used to sort
  Returns:
    :class:`VStrideArray` : sorted array
  Example:
    >>> a = vs.from_counts([2, 4, 3], [3,2, 3,1,5,2, 9,5,8])
    >>> vs.sort(a, vs.OUTER_AXIS)
    vsarray([
      [3, 1, 5, 2],
      [3, 2],
      [9, 5, 8],
    ], dtype=int64)
    >>> vs.sort(a, vs.INNER_AXIS)
    vsarray([
      [2, 3],
      [1, 2, 3, 5],
      [5, 8, 9],
    ], dtype=int64)
  """
  if axis == INNER_AXIS:
    displs = array.displs
    values = array.values.copy()
    _vstride.sort_by_stride(displs, values)
    return VStrideArray(array._displs, array._counts, values)

  elif axis == OUTER_AXIS:
    # TODO : unoptimized version. uses lexicographic order
    return globals()['array'](sorted([blk.tolist() for blk in array]), dtype=array.dtype)

  else:
    raise ValueError(_UNVALID_AXIS_MSG)


def unique(array: VStrideArray, axis:Axis):
  """ Find the unique values of the input array.

  Depending on the ``axis`` argument, the operation applies to:

  - the elements if ``axis==OUTER_AXIS``, which is roughly equivalent to ::

      vs.array(unique([blk for blk in array]))
    
    **Not yet implemented**

  - each block if ``axis==INNER_AXIS``, which is roughly equivalent to ::

      vs.array([unique(blk) for blk in array)]

    In this case, the output array have the same number of elements :math:`N`,
    but a different ``counts`` array.

  Args:
    array (:class:`VStrideArray`): input array
    axis (:class:`Axis`): direction in which algorithm is applied
  Returns:
    :class:`VStrideArray` : unique array
  Example:
    >>> a = vs.from_counts([2, 4, 3], [2,2, 3,1,3,2, 9,5,5])
    >>> vs.unique(a, vs.INNER_AXIS)
    vsarray([
      [2],
      [3, 1, 2],
      [9, 5],
    ], dtype=int64)
  """

  if axis == INNER_AXIS:
    displs, values = _vstride.make_unique_by_stride(array.displs, array.values)
    return VStrideArray(displs, None, values)
  elif axis == OUTER_AXIS:
    raise NotImplemented

  else:
    raise ValueError(_UNVALID_AXIS_MSG)

def roll(array: VStrideArray, shift:int, axis:Axis):
  """ Roll the values of the input array.

  Positive values of ``shift`` moves the elements to the right, while negative values
  move them to the left.
  Values leaving the array are reintroduced on the opposite side. 

  Depending on the ``axis`` argument, the operation apply to:

  - the elements if ``axis==OUTER_AXIS``, which is roughly equivalent to ::

      vs.array(array[-shift:] + array[:shift+1]) # for positive shift

  - each block if ``axis==INNER_AXIS``, which is roughly equivalent to ::

      vs.array([roll(blk, shift) for blk in array)]
  
  In both cases, a copy is done and a new VStrideArray is returned.

  Args:
    array (:class:`VStrideArray`): input array
    shift (int): number of places by which the elements are shifted
    axis (:class:`Axis`): direction used to roll
  Returns:
    :class:`VStrideArray` : rolled array
  Example:
    >>> a = vs.from_counts([2, 3, 5, 4], [1,2, 3,1,1, 2,7,2,5,9, 6,4,4,2])
    >>> vs.roll(a, 2, vs.OUTER_AXIS)
    vsarray([
      [2, 7, 2, 5, 9],
      [6, 4, 4, 2],
      [1, 2],
      [3, 1, 1],
    ], dtype=int64)
    >>> vs.roll(a, -1, vs.INNER_AXIS)
    vsarray([
      [2, 1],
      [1, 1, 3],
      [7, 2, 5, 9, 2],
      [4, 4, 2, 6],
    ], dtype=int64)
  """

  if len(array) == 0: # Corner case -> avoid ZeroDivisionError
    return VStrideArray(None, np.empty(0, array.counts.dtype), np.empty(0, array.dtype))

  if axis == INNER_AXIS:
    displs = array.displs
    values = array.values.copy()
    _vstride.roll_by_stride(displs, values, shift)
    return VStrideArray(array._displs, array._counts, values)

  elif axis == OUTER_AXIS:

    if shift < 0:
      shift = -(-shift % len(array))
      vshift = -array.counts[:-shift].sum()
    else:
      shift = shift % len(array)
      vshift = array.counts[len(array) - shift:].sum()

    counts = np.roll(array.counts,  shift)
    values = np.roll(array.values, vshift)
    return VStrideArray(None, counts, values)

  else:
    raise ValueError(_UNVALID_AXIS_MSG)

def concatenate(array_l, axis:Axis):
  """ Join a sequence of arrays.

  Depending on the ``axis`` argument, the concatenation is applied to:

  - the elements if ``axis==OUTER_AXIS``, which is roughly equivalent to ::

      vs.array([blk for arr in array_l for blk in arr])

    In this case, the input arrays can have a different length; the
    number of elements of the output array is the sum of the input lenghts.

  - each block if ``axis==INNER_AXIS``, which is roughly equivalent to ::

      vs.array([concatenate(blks) for blks in zip(*array_l)])

    In this case, all the input arrays must have the same number of elements :math:`N`,
    which is the number of elements of the output array.
  
  Args:
    array_l (sequence of :class:`VStrideArray`): arrays to concatenate
    axis (:class:`Axis`): direction used to concatenate
  Returns:
    :class:`VStrideArray` : concatenated array
  Example:
    >>> a1 = vs.array([[0,1],  [2,3,4], [5,6]],  dtype=int)
    >>> a2 = vs.array([[],  [0,1,2,3], [4,6,7]], dtype=int)
    >>> vs.concatenate([a1, a2], vs.OUTER_AXIS)
    vsarray([
      [0, 1],
      [2, 3, 4],
      [5,6],
      [],
      [0, 1, 2, 3],
      [4, 5, 6],
    ], dtype=int64)
    >>> vs.concatenate([a1, a2], vs.INNER_AXIS)
    vsarray([
      [0, 1],
      [2, 3, 4, 0, 1, 2, 3],
      [5, 6, 4, 6, 7],
    ], dtype=int64)
  """
  if len(array_l) == 0:
    raise ValueError("need at least one array to concatenate") 

  if axis == INNER_AXIS:
    length_l = [len(arr) for arr in array_l]
    if len(set(length_l)) > 1:
      raise ValueError(f"lenght of all input arrays ({length_l}) must match for INNER_AXIS concatenation")
    int_dtype = np.result_type(*(arr.displs for arr in array_l))
    val_dtype = np.result_type(*(arr.values for arr in array_l))
    displs_l = [arr.displs.astype(int_dtype, copy=False) for arr in array_l]
    values_l = [arr.values.astype(val_dtype, copy=False) for arr in array_l]
    displs_out = np.empty(len(array_l[0])+1,                     int_dtype)
    values_out = np.empty(sum(array.dsize for array in array_l), val_dtype)
    _vstride.concatenate_by_stride(displs_l, values_l, displs_out, values_out)
    return VStrideArray(displs_out, None, values_out)


  elif axis == OUTER_AXIS:
    counts = np.concatenate([a.counts for a in array_l])
    values = np.concatenate([a.values for a in array_l])
    return VStrideArray(None, counts, values)

  else:
    raise ValueError(_UNVALID_AXIS_MSG)

#### Additional operators

def sign(array:VStrideArray, dtype=None):
  """ Create a new array storing the sign of :attr:`array.values`.
  The *strides* of the output array are identical to ones of the input.

  Args:
    array (:class:`VStrideArray`): input array
    dtype (data-type, optional): desired datatype for the output
  Returns:
    :class:`VStrideArray` : sign array
  Example:
    >>> vs.sign(vs.from_counts([2,3,1], [1,2,-3,4,0,-6]))
    vsarray([
      [ 1,  1],
      [-1,  1,  0],
      [-1],
    ], dtype=int64)
  """
  return VStrideArray(array._displs, array._counts, np.sign(array.values).astype(dtype=dtype, copy=False))

def strides_equal(a1:VStrideArray, a2:VStrideArray) -> bool:
  """ ``True`` if the two input arrays have the same *strides*, ``False`` otherwise.

  Args:
    a1 (:class:`VStrideArray`): first input
    a2 (:class:`VStrideArray`): second input
  Returns:
    bool  : comparison result
  Example:
    >>> vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
    ...                vs.from_counts([2,3,1], [6,5,4,3,2,1]))
    True
    >>> vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
    ...                vs.from_counts([3,2,1], [1,2,3,4,5,6]))
    False
  """
  if a1._counts is not None and a2._counts is not None:
    return np.array_equal(a1._counts, a2._counts)
  else:
    return np.array_equal(a1.displs, a2.displs)

def array_equal(a1:VStrideArray, a2:VStrideArray) -> bool:
  """ ``True`` if the two input arrays have the same *strides* and values, ``False`` otherwise.

  Args:
    a1 (:class:`VStrideArray`): first input
    a2 (:class:`VStrideArray`): second input
  Returns:
    bool  : comparison result
  Example:
    >>> vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
    ...                vs.from_counts([2,3,1], [1,2,3,4,5,6]))
    True
    >>> vs.array_equal(vs.from_counts([2,3,1], [1,2,3,4,5,6]),
    ...                vs.from_counts([2,3,1], [6,5,4,3,2,1]))
    False
  """
  return strides_equal(a1, a2) and np.array_equal(a1.values, a2.values)


def array_close(a1:VStrideArray, a2:VStrideArray, rtol=1e-5, atol=1e-8) -> bool:
  """ ``True`` if the two input arrays have the same *strides* and close values, ``False`` otherwise.

  Value comparison is performed by 
  `np.isclose <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>`_. See the
  related documentation for description of ``rtol`` and ``atol`` parameters.

  Args:
    a1 (:class:`VStrideArray`): first input
    a2 (:class:`VStrideArray`): second input
    rtol (float, optional) : relative tolerance for comparison
    atol (float, optional) : absolute tolerance for comparison
  Returns:
    bool  : comparison result
  Example:
    >>> vs.array_close(vs.from_counts([2,3,1], [1.,2,3,4,5,6]),
    ...                vs.from_counts([2,3,1], [1.,2,3,4,5,6+1e-9]))
    True
    >>> vs.array_close(vs.from_counts([2,3,1], [1.,2,3,4,5,6]),
    ...                vs.from_counts([2,3,1], [1.,2,3,4,5,6.2]))
    False
  """
  return strides_equal(a1, a2) and np.allclose(a1.values, a2.values, rtol=rtol, atol=atol)

