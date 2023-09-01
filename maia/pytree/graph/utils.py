from typing_extensions import TypeAlias
from _collections_abc import list_iterator

# Since `_collections_abc.list_iterator` is private,
# provide an alias to it so only the alias needs to be changed
# see https://stackoverflow.com/a/27046780/1583122
list_iterator_type : TypeAlias = list_iterator # or `type(iter([]))` by MyPy does not like it
