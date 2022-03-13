import os

def determine_output_file_path(args,output_default_extension):
  file_name = os.path.basename(args.input)
  file_name_base,_ = os.path.splitext(file_name)

  if args.output is None:
    return file_name_base + output_default_extension
  elif os.path.isdir(args.output):
    return os.path.join(args.output, file_name_base+output_default_extension)
  else:
    return args.output
