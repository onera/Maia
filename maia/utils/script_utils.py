import os

def determine_output_file_path(input_path, output_path, output_default_extension):
  file_name = os.path.basename(input_path)
  file_name_base,_ = os.path.splitext(file_name)

  if output_path is None:
    return file_name_base + output_default_extension
  elif os.path.isdir(output_path):
    return os.path.join(output_path, file_name_base+output_default_extension)
  else:
    return output_path
