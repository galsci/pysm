3.4.0 (unreleased)
==================

3.4.0b4 (2022-11-21)
====================

- Brand new implementation of small scales injection for Synchrotron `PR 134 <https://github.com/galsci/pysm/pull/134>`_, affects `s4`, `s5`, `s6`, `s7`
- Brand new implementation of small scales injection for GNILC Dust `PR 133 <https://github.com/galsci/pysm/pull/133>`_, affects `d9`, `d10`, `d11`
- Fix bug in `InterpolatingComponent`, when the user requested a frequency between 2 available points, the weighting of the 2 relevant maps was switched, see `PR 129 <https://github.com/galsci/pysm/pull/129>`_
- Implemented a proper unit test of the running `trapz` implementation used for bandpass integration against `np.trapz`, see `PR 129 <https://github.com/galsci/pysm/pull/129>`_
- Imported WebSky extralactic components from `so_pysm_models`, now version 0.4, it also includes SPT based correction for CIB `PR 129 <https://github.com/galsci/pysm/pull/129>`_
- `apply_smoothing_and_coord_transform` now supports a different output resolution and supports doing both HEALPix and CAR in the same execution, also added best practices for dealing with resolution in the docs `PR 125 <https://github.com/galsci/pysm/pull/125>`_
- Model has `max_nside` attribute which specifies its max resolution `PR 124 <https://github.com/galsci/pysm/pull/124>`_

3.4.0b3 (2022-03-28)
====================

- Implementation of Synchrotron with curvature `s7` `based on ARCADE <https://github.com/galsci/pysm/pull/115>`_

3.4.0b2 (2022-03-15)
====================

- bugfix: fixed units error recently introduced in the CO Lines models, `reported by @mousset <https://github.com/galsci/pysm/issues/113>`_
- Synchrotron high resolution models `s4`, `s5`, `s6`, `based on Haslam and WMAP <https://github.com/galsci/pysm/pull/106>`_

3.4.0b1 (2022-03-03)
====================

- Galactic plane fix for `d9`, `d10`, `d11` `PR 111 <https://github.com/galsci/pysm/pull/111>`_
- Documentation about the processing of GNILC dust `input templates <https://github.com/galsci/pysm/pull/97>`, `spectral index and dust temperature <https://github.com/galsci/pysm/pull/104>`_
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
