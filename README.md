Scotty
======

Beam tracing for tokamak DBS diagnostics


Install
-------

Scotty can be installed using `pip`:

```
pip install scotty-beam-tracing
```

Note that the package to install is called `scotty-beam-tracing`, but
the python module is just `scotty`.

Versions
--------

Two versions of Scotty are currently maintained:

- **Scotty v3** (stable): the `v3.x` branch. Only bug fixes go here.
- **Scotty v4** (development): the `master` branch. v4 will replace v3,
  but its interface may still change between pre-releases.

Only one version can be installed in a Python environment at a time. To
install v3:

```
pip install "scotty-beam-tracing<4"
```

To install the latest v4 pre-release:

```
pip install --pre scotty-beam-tracing
```

v4 requires Python 3.10 or newer. It is not a drop-in replacement for v3:
several modules have been removed or reorganised (for example `outplot`,
`compare_plot`, `fun_evolution`, `check_input` and `check_output`), so
scripts written for v3 may need updating.

To install the development version of `scotty` straight from Github:

```
pip install git+https://github.com/beam-tracing/Scotty
```

or from a local clone:

```
pip install .
```

when developing, use:

```
pip install -e .
```

Documentation
-------
https://scotty.readthedocs.io/

Background
-------
Description of the Scotty [[1]](#1), derivation of beam tracing given in Appendix A.

## References
<a id="1">[1]</a> 
Hall-Chen, V. H., Parra, F. I., & Hillesheim, J. C. (2022). 
[Beam model of Doppler backscattering](https://iopscience.iop.org/article/10.1088/1361-6587/ac57a1). 
*Plasma Physics and Controlled Fusion*, **64**(9), 095002.
