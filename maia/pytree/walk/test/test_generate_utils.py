import pytest

from maia.pytree.walk import generate_utils as utils

def test_camel_to_snake():
  assert utils.camel_to_snake("already_snake") == "already_snake"
  assert utils.camel_to_snake("stringInCamelCase") == "string_in_camel_case"
  assert utils.camel_to_snake("StringInCamelCase") == "string_in_camel_case"
  assert utils.camel_to_snake("stringINCamelCase", keep_upper=True) == "string_IN_camel_case"
