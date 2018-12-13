# Models

  - Base class `Model`, subclasses `DustModel` etc. Customization
    by:
      - tweaking parameters of methods
      - create subclass of `DustModel`
      - new subclass of `Model`
  - Arguments specified, not from dictionary
  - Add new classes for multi component models, not list of dictionaries
    as inputs.

## Model Setup:

  - Model functions will still exist to construct e.g. 'd4' from the
    component classes. Will return `Sky` objects.
  - Within
    <span class="underline"><span class="underline">init</span></span>
    of each component do the reading of maps etc.

# Parallelism:

  - Combination of MPI and Dask? How to interface with toast at same
    time.
  - Assuming no Dask, use MPI4Py.
  - Inputs read per process or broadcast?
      - ordering issues: don't convert ring and nest in one process.
      - conflicts between processes reading same file.
  - Distributing maps in libsharp
      - scatterv and gatherv
  - keep get<sub>emission</sub> mostly local, also has access to
    communicator

# Object oriented structure

Reduce levels of abstraction. Initialize components from string. Keep
`pysm.Sky` as convenience initialized from string or from component
objects.

# Interface with `Instrument`

Most important factor is bandpass integration.

  - Try as `Component.get<sub>emission</sub>(bandpass)` and
    `Sky.get<sub>emission</sub>(bandpass)`.
  - Smoothing via libsharp, and trivial healpy routine separate to
    `Instrument` object, and called from `Sky`.

# Extra Galactic

Current maps 2048, each 192 MB. Code exists for spline interpolation.
Available for validation, determine the type of interpolation as a
study. In the first instance:

  - CIB ready to go at 2048 for validation.
  - Incorporate into PySM kSZ and Compton-y.
  - Need to code from scratch kSZ and Compton-y.

In second stage:

  - Lenspix / taylens used on kappa map to lensed primary

# Maintain Example usage:

``` python
sky = pysm.get_model("s1", "d3")
cmb = pysm.cmb("c1")
sky.get_emission(nu)
sky.apply_bandpass(instrument_settings)
sky.apply_smoothing(instrument_settings)
```
