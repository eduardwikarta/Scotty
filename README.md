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

### Changes from v3 that affect existing scripts

- `beam_me_up` requires `geometry` (`"cylindrical"` or `"cartesian"`),
  `launch_flag` (`"vacuum"` or `"plasma"`) and `boundary_flag`
  (`"continuous"`, `"discontinuous"` or `None`). These replace v3's
  `vacuumLaunch_flag`, `vacuum_propagation_flag` and `Psi_BC_flag`, which
  are now ignored. A v3 launch from vacuum with `Psi_BC_flag="continuous"`
  is `geometry="cylindrical", launch_flag="vacuum",
  boundary_flag="continuous"`, as in the built-in diagnostic presets.
- Launch angles passed to the launch functions are in degrees (TORBEAM
  convention), not radians.
- Output is an `xarray.DataTree` with `inputs`, `solver_output`,
  `analysis` and `dbs_analysis` nodes, and several variables are renamed
  (for example `q_R` is now `solver_output/q_R_output`, and
  `normalised_plasma_freqs` is `analysis/normalised_omega_pe`).
- The default `interp_smoothing` is 5 (it was 0 in v3), so results with
  interpolated fields can differ from v3 for the same inputs.
- The analytical fields use exact poloidal flux derivatives by default;
  pass `create_magnetic_geometry={"derivative_method": "numerical"}` to
  `beam_me_up` for finite differences.

### Known issues in v4

These are marked as expected failures (`xfail`) or skipped in the tests:

- Temperature profiles are not implemented yet, so there are no
  relativistic corrections.
- X-mode: `K_R` differs from v3 by about 2 (on values of order 1000) where
  it crosses zero near the cutoff; elsewhere it agrees to about 0.3%.
- Inboard launches: `find_plasma_entry_position` only searches along the
  ray up to `max(q_launch_cartesian)`, which is too short to reach the
  plasma from the inboard side.
- `find_B_method="omfit"` loads the data but then raises "not supported"
  (in `get_profile_data`, the TORBEAM branch should be an `elif`).
- `find_B_method` values `"test"`, `"test_notime"`, `"UDA_saved"`,
  `"EFITpp"` and `"curvy_slab"` from v3 have not been ported to v4 yet.
- `profile_fit` no longer accepts `ProfileFit` objects directly.
- The scripts in `Examples/` still use the v3 launch flags.

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
