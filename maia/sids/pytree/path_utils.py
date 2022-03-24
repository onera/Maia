
def path_head(path, i=-1):
  """
  Return the start of a path until elt i
  """
  splited = path.split('/')
  return '/'.join(splited[0:i])

