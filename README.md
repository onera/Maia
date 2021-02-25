Maia
====

`Maia` is a Python and C++ library for distributed algorithms and manipulations over CGNS meshes. It uses [ParaDiGM](https://git.onera.fr/paradigm) for parallel algorithms and [Cassiopee](http://elsa.onera.fr/Cassiopee) for CGNS tree manipulation.

## Build and install ##
1. `git clone http://gitlab-elsa-test.onecert.fr/clef/maia`
2. `cd maia`
3. `git submodule update --init` (needed for dependencies)
4. Use `cmake` to configure, build and install. See the complete procedure here `doc/installation.rst`

## Documentation ##
The documentation root is file `doc/index.rst`.

## Examples ##
Loading an HDF5/CGNS file in parallel
(TODO)

``` Python
import maia
# some function loading dist_tree
# some function loading dist_tree/part_tree
# transfer
# merge and save
```

Some converting functions :

``` Python
import maia
std_element_to_ngon
convert_s_to_u
```

## Contributing ##
`Maia` is open-source software. Contributions are welcome. See `Contributing`.

## License ##
`Maia` is available under the MPL-2.0 license (https://mozilla.org/MPL/2.0/).
