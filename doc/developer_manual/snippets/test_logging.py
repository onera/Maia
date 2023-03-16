def test_logger():
  #add_logger@start
  from maia.utils.logging import add_logger

  add_logger("my_logger")
  #add_logger@end
  #log@start
  from maia.utils.logging import log

  log('my_logger', 'my message')
  #log@end


def test_printer():
  #add_printer@start
  from maia.utils.logging import add_printer_to_logger
  add_printer_to_logger('my_logger','stdout_printer')
  #add_printer@end

  #create_printer@start
  from maia.utils.logging import add_printer_to_logger

  class my_printer:
    def log(self, msg):
      print(msg)

  add_printer_to_logger('my_logger',my_printer())
  #create_printer@end
