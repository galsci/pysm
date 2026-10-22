---
title: "Adding a new model"
description: "How to add a new emission model to PySM and document it in the current structure."
---

## Model implementation

Each emission component is a class in `src/pysm3/models/` (see the existing
`dust`, `synchrotron`, `freefree`, `dipole`, ... modules for examples). A new
model:

1. Subclasses `pysm3.Model` and implements `get_emission` (see the
   [API reference](../api/pysm3/) and the `pysm3.Model` documentation), or
2. Composes existing components, typically through the `Component` classes
   (`InterpolatingComponent`, `PowerLaw`, ...) for template maps and spectral
   models.

The simplest entry point for testing a custom component on the fly is the
`pysm3.Sky` constructor with a custom component dictionary or a TOML file (see
the [Customize PySM models tutorial](../../notebooks/customize_components/)).

## Presets

Predefined simulation configurations live in `pysm3/data/presets.cfg`. Add the
new preset there with a unique letter + number identifier (for example `d13` or
`s8`) following the convention of the existing entries, and describe it in the
[Models](models/) page of this site.

## Tests and quality

- Add unit tests in `src/pysm3/tests/` with the coverage expected by the
  repository (the CI runs the full test suite on every pull request).
- Add a line to `CHANGES.rst` describing the change with a link to the pull
  request.
- If the model needs a data pipeline, document it as a notebook in the
  `notebooks/` folder of the repository: every notebook there is rendered
  automatically on the documentation site (including the
  [template preprocessing notebooks](../../notebooks/preprocess-templates/)),
  so no extra configuration is needed.

## Documentation structure

Everything the documentation site shows is generated from the repository:

- `docs-site/src/content/docs/` is the committed Markdown/MDX of the site;
  edit it directly to update prose pages.
- `notebooks/` holds every Jupyter notebook; they are rendered to HTML pages
  at build time by `docs-site/scripts/generate-notebooks.mjs`.
- The `pysm3` API reference is generated from `src/` at build time by
  `starlight-pydocs`; no manual API pages.

## Pull request

1. Open a pull request with the model implementation, preset, tests and docs.
2. Follow the [Contribution guide](contributing/): the maintainers review pull
   requests (at least 2 weeks) before merging.
