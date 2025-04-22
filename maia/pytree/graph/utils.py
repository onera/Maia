from _collections_abc import list_iterator #type:ignore[attr-defined] #(private arg)

# Since `_collections_abc.list_iterator` is private,
# provide an alias to it so only the alias needs to be changed
# see https://stackoverflow.com/a/27046780/1583122
list_iterator_type  = list_iterator # or `type(iter([]))` by MyPy does not like it