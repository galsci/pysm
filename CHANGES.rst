3.4.0 (unreleased)
==================

- Documentation about the processing of GNILC dust `input templates <https://github.com/galsci/pysm/pull/97>`_
- Planck GNILC based templates `d9`, `d10`, `d11` `PR 108 <https://github.com/galsci/pysm/pull/108>`_
- CO Lines models `co1`, `co2`, `co3` `PR 86 <https://github.com/galsci/pysm/pull/86>`_
- 3D MKD Dust model with 6 layers `d12` `PR 87 <https://github.com/galsci/pysm/pull/87>`_

3.3.2 (2021-10-29)
==================

- `Improvements to install documentation <https://github.com/galsci/pysm/pull/93>`_
- Moved Github repository from `healpy/pysm` to `galsci/pysm`, under the Panexperiment Galactic Science group organization
- Changes in this release are related to the `JOSS Review <https://github.com/openjournals/joss-reviews/issues/3783>`_
- Turned `UserWarning` into `logging`, `Pull Request 88 <https://github.com/galsci/pysm/pull/88>`_

3.3.1 (2021-06-30)
==================

- Packaging: removed Poetry, using new Astropy package template and Github actions, `Pull Request 76 <https://github.com/galsci/pysm/pull/76>`_ and `77 <https://github.com/galsci/pysm/pull/77>`_.
- Docs: Reproduce PySM 2 template preprocessing for Synchrotron, `Pull Request 71 <https://github.com/galsci/pysm/pull/71>`_
- Docs: Reproduce PySM 2 template preprocessing for Dust, `Pull Request 66 <https://github.com/galsci/pysm/pull/66>`_

3.3.0 (2020-09-12)
==================

- Avoid an imcompatibility issue with ``numba``, see `Pull Request 63 <https://github.com/galsci/pysm/pull/63>`_
- Fix a severe bug in unit conversion with bandpass integration, which can give an overall scale error of a few percent at high frequency for all components, see `Issue 59 <https://github.com/galsci/pysm/issues/59>`_, also imported all bandpass integration tests from PySM 2 and added a comparison with the `tod2flux` tool by @keskitalo
- Removed support for `has_polarization` in interpolator, always return IQU map

3.2.2 (2020-06-23)
==================

- Fix packaging issue `importlib-resources` for python 3.6 was missing

3.2.1 (2020-06-05)
==================

- Renamed the package to `pysm3`, therefore now need to `import pysm3`
- Using `poetry` to build package and manage dependencies `PR 56 <https://github.com/galsci/pysm/pull/56>`_

3.2.0 (2020-04-15)
==================

First version with all models available in PySM 2

- Implemented HD2017 `d7` dust model `PR 37 <https://github.com/galsci/pysm/pull/37>`_
- Implemented HD2017 `d5` and `d8` dust models `PR 51 <https://github.com/galsci/pysm/pull/51>`_
- Improved documentation about Sky
- Implement local data folder `PR 53 <https://github.com/galsci/pysm/pull/53>`_

3.1.2 (2020-03-27)
==================

HD2017 `d7` dust model still being implemented

- Updated build/test setup to latest Astropy template `PR 47 <https://github.com/galsci/pysm/pull/47>`_
- Bugfix: `d6` model `PR 43 <https://github.com/galsci/pysm/pull/43>`_
- Bugfix: units other than GHz `PR 45 <https://github.com/galsci/pysm/pull/45>`_

3.1.0 (2019-12-11)
==================

- All emissions implemented except HD2017 `d7` dust

3.0.0 (2019-09-23)
==================

- Development release
