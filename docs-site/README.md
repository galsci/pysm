# PySM Starlight documentation site

Starlight (Astro) documentation site for PySM, built and hosted on
[Read the Docs](https://pysm3.readthedocs.io/).

**Source of truth:** this is it. The documentation pages live in
`src/content/docs/` as plain Markdown/MDX, committed and reviewed like any
other code. Jupyter tutorials are authored as notebooks in `notebooks/` at the
repository root (Git LFS) and rendered at build time by
`scripts/generate-notebooks.mjs` (pages and images extracted); the `pysm3` API
reference is built at build time by `starlight-pydocs` from `../src` with
griffe.

There is no generated copy of the prose documentation: edit the Markdown in
`src/content/docs/` and rebuild; the notebooks are the only content rendered
from another format, on purpose.

## Commands

```bash
npm install              # install dependencies
npm run dev              # generate notebooks, then start the dev server
npm run build            # generate notebooks, then build to dist/
npm run generate:notebooks   # regenerate notebook pages from ../notebooks/*.ipynb
```

## Read the Docs build

`.readthedocs.yaml` at the repository root drives the build:

1. `post_checkout` downloads the Git LFS binary and pulls the notebook content.
2. `install` installs griffe (API reference), then runs `npm ci`.
3. `build` runs `npm run build` and copies `docs-site/dist/` into
   `$READTHEDOCS_OUTPUT/html`.

## Environment variables

Read the Docs sets `READTHEDOCS_CANONICAL_URL` (used for the Astro `site` and
`base` so assets and links resolve under the version path) and
`READTHEDOCS_OUTPUT` (where the HTML output is published).

## Layout

- `src/content/docs/` is the committed documentation (Markdown/MDX).
- `src/content/docs/notebooks/` is generated from `../notebooks/*.ipynb`
  (Git LFS); gitignored.
- `src/content/docs/images/` and `src/content/docs/../public/{images,model-data}`
  hold the images and data files referenced by the pages.
- `src/styles/custom.css` is the theme.
- `src/components/Footer.astro` adds the Read the Docs flyout mount point.

## Notes

- LaTeX math is rendered with KaTeX via a `unified` markdown processor
  (`remark-math` + `rehype-katex`), which requires `@astrojs/markdown-remark`.
- Read the Docs handles versions: each tag or branch is built from its own
  commit, so older releases keep their original documentation while the
  current development version uses this site.
