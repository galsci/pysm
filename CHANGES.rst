3.4.2 (2025-06-17)
==================
- Allow the CMBMap class to accept in-memory arrays (with astropy units) as input for the CMB template, in addition to file paths. Adds validation for units and shape, and includes tests to ensure errors are raised for missing units or incorrect shapes. Also adds a test for correct in-memory usage. See https://github.com/galsci/pysm/issues/222 and PR #223: https://github.com/galsci/pysm/pull/223.

3.4.1 (2025-04-15)
==================
- Fix bug in `cib1` at frequencies below 18.7 GHz, created scaled maps with Modified Black Body spectrum down to 1 GHz https://github.com/galsci/pysm/issues/210
- Make sure models are evaluated in double precision https://github.com/galsci/pysm/pull/196
- Fix bug in `cib1` when frequency goes below 18.7 GHz, emission too large https://github.com/galsci/pysm/pull/195
- Finalized Point Source Catalog component and backgroud component https://github.com/galsci/pysm/pull/191
- Configure verbosity easily with `set_verbosity()`
- Updated `pixell` from 0.17.3 to 0.26.0 https://github.com/galsci/pysm/pull/183
- Initial implementation of a point source catalog component emission https://github.com/galsci/pysm/pull/187
- Switch the build system to Hatch https://github.com/galsci/pysm/pull/189
- Fix shape error in `CMBLensed` preventing use of `apply_delens=True` https://github.com/galsci/pysm/pull/214
- Fix bug in `CMBLensed` to read spectra that include monopole and dipole  https://github.com/galsci/pysm/pull/215
- Specify in the docs that the native resolution of free-free and AME is nside=512

3.4.0 (2023-12-11)
==================

- No model changes compared to `b9`
- Added `pysm_tag_filename` script to add datestamp to filenames to avoid caching errors
- Added datestamp to Dust and Synchrotron templates to avoid caching errors if users had previously used a beta version of PySM 3.4.0 see commit 694653e6a582a4051e8cc7629c69f2ccd0e195ff

3.4.0b9 (2023-06-27)
====================

- Bugfix: Fix dust beta and Td ellmax that were erroneously set to 2048 https://github.com/galsci/pysm/pull/164
- Print warning if templates are not found suggesting to update PySM https://github.com/galsci/pysm/pull/165
- Document features in the spectrum of d9 d10 d11 s4 s5 s6 and s7 due to the "Galactic plane fix" https://github.com/galsci/pysm/pull/168/ and https://github.com/galsci/pysm/pull/170

3.4.0b8 (2023-03-22)
====================

- Set Fejer1 CAR variant by default, as advised by Simons Observatory, requires `pixell` 0.17.3 https://github.com/galsci/pysm/pull/157
- Bug fix in `ksz` and `tsz`, conversion was broken for single frequency channel, was fine for bandpass https://github.com/galsci/pysm/pull/158/

3.4.0b7 (2023-02-25)
====================

- Bug fix of small scale modulation for Synchrotron models, impacts `s4`, `s5`, `s6`, `s7` https://github.com/galsci/pysm/pull/154

3.4.0b6 (2023-02-17)
====================

- New implementation of small scale modulation for Synchrotron models, impacts `s4`, `s5`, `s6`, `s7` https://github.com/galsci/pysm/pull/152
- New implementation of small scale modulation for GNILC dust models, impacts `d9`, `d10`, `d11` https://github.com/galsci/pysm/pull/150
- Support specifying number of iterations in `map2alm`, default is 10, 0 is for standard `map2alm` https://github.com/galsci/pysm/pull/144

3.4.0b5 (2022-12-05)
====================

- Implementation of color correction (multiply by a factor of 0.911) in `d12` https://github.com/galsci/pysm/pull/141

3.4.0b4 (2022-11-21)
====================

- **Known issue**: `d12` was missing color correction, see https://github.com/galsci/pysm/issues/128#issuecomment-1332843288
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
