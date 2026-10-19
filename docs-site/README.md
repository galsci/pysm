# PySM Starlight documentation site

Starlight (Astro) documentation site for PySM, built and hosted on
[Read the Docs](https://pysm3.readthedocs.io/).

The site is generated from the Starlight pages in `src/content/docs/`, the
Jupyter notebook tutorials in `../docs/*.ipynb` (Git LFS, rendered at build
time by `scripts/generate-notebooks.mjs`) and the `pysm3` API reference via
`starlight-pydocs` (extracted from `../src` at build time with griffe).

Read the Docs provides the version handling: each git tag or branch is built
by Read the Docs on its own commit, and the built-in version selector picks
between them. Older releases keep their original Sphinx documentation; this
Starlight site is the documentation for the current development version.

## Commands

```bash
npm install              # install dependencies
npm run dev              # generate notebooks, then start the dev server
npm run build            # generate notebooks, then build to dist/
npm run generate:notebooks   # regenerate notebook pages from ../docs/*.ipynb
```

## Read the Docs build

`.readthedocs.yaml` at the repository root drives the build:

1. `post_checkout` downloads the Git LFS binary and pulls the notebook content.
2. `install` installs griffe (for the API extraction) and runs `npm ci`.
3. `build` runs `npm run build` and copies `docs-site/dist/` into the
   `$READTHEDOCS_OUTPUT/html` directory.

## Environment variables

Read the Docs sets `READTHEDOCS_CANONICAL_URL` (used for the Astro `site` and
`base` so assets and links resolve under the version path) and
`READTHEDOCS_OUTPUT` (where the HTML output is published).

## Content

- `src/content/docs/` is the Starlight documentation (port of the classic
  Sphinx `docs/` to Markdown).
- `src/content/docs/notebooks/` is generated from `../docs/*.ipynb` (Git LFS)
  by `scripts/generate-notebooks.mjs`; it is gitignored and never committed.

## Notes

- LaTeX math is rendered with KaTeX via a `unified` markdown processor
  (`remark-math` + `rehype-katex`), which requires `@astrojs/markdown-remark`.
