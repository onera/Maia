**Maia** is a Python and C++ library for parallel algorithms and manipulations over CGNS meshes. Maia introduces a parallel representation of the CGNS trees and uses [ParaDiGM](https://gitlab.onera.net/numerics/mesh/paradigm/) as a back-end to provide various functions applicable to these trees.

## Getting started ##

### Onera users 
User documentation is deployed on the Gitlab pages server: https://numerics.gitlab-pages.onera.net/mesh/maia/index.html.

Stable installations are provided on Spiro and Sator clusters: for example, on Spiro-EL8 partition, Maia environment can be loaded with the following lines:

```bash
source /scratchm/sonics/dist/source.sh --env maia
module load maia/dev-default
```

Additional environments are provided in the [Quick start](https://numerics.gitlab-pages.onera.net/mesh/maia/quick_start.html) page of the documention.

## Other users 

See the next section to build your own version of Maia.

## Build and install ##

Follow these steps to build Maia from the sources:

1. `git clone git@gitlab.onera.net:numerics/mesh/maia.git`
2. `cd maia`
3. `git submodule update --init` (needed for dependencies)
4. `(cd external/paradigm && git submodule update --init extensions/paradigma)` (enable advanced features)
5. Use `cmake` to configure, build and install. See the complete procedure here `doc/installation.rst`

Documentation can be build with `cmake` flag `-Dmaia_ENABLE_DOCUMENTATION=ON`

## Contributing ##
`Maia` is open-source software. Contributions are welcome. See `Contributing`.

Issues can be reported directly on [the Issues section](https://gitlab.onera.net/numerics/mesh/maia/-/issues).

## License ##
`Maia` is available under the MPL-2.0 license (https://mozilla.org/MPL/2.0/).

<p align="center">
  <img src="./doc/_static/logo_maia.svg" alt="logo-maia" width=25%>
</p>