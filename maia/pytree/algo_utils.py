def set_intersection_difference(x, y, comp):
  """
    Returns 4 lists `(inter_x, diff_x, inter_y, diff_y)`:
      - `inter_x`, are `diff_x` are subsets of elements in `x`
      - `inter_y`, are `diff_y` are subsets of elements in `y`
    `inter_x` and `inter_y` contain elements both present in `x` and `y` according the `comp`
    `diff_x` and `diff_y` contain elements not present in both `x` and `y` according the `comp`

    `x` and `y` must be sorted according to `comp`
    The returned lists will be sorted
  """
  i = 0
  j = 0
  inter_x = []
  inter_y = []
  diff_x  = []
  diff_y  = []
  while i<len(x) and j<len(y):
    if comp(x[i],y[j]):
      diff_x.append( x[i] )
      i += 1
    elif comp(y[j],x[i]):
      diff_y.append( y[j] )
      j += 1
    else: # x[i]==y[j]:
      inter_x.append( x[i] )
      inter_y.append( y[j] )
      i += 1
      j += 1
  diff_x += x[i:]
  diff_y += y[j:]
  return inter_x, diff_x, inter_y, diff_y
