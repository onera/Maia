
def path_head(path, i=-1):
  """
  Return the start of a path until elt i (excluded)
  """
  splited = path.split('/')
  return '/'.join(splited[0:i])

def path_tail(path, i=-1):
  """
  Return the end of a path from elt i (included)
  """
  splited = path.split('/')
  return '/'.join(splited[i:])

def update_path_elt(path, i, func):
  """
  Replace the ith element of the input path using the function func
  func take one argument, which is the original value of the ith element
  """
  splited = path.split('/')
  splited[i] = func(splited[i])
  return '/'.join(splited)

