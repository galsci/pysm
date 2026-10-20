# PySM Starlight documentation site

Starlight (Astro) documentation site for PySM, built and hosted on
[Read the Docs](https://pysm3.readthedocs.io/).

**Single source of truth:** all documentation content lives in `../docs/`
(`docs/*.rst` and `docs/*.ipynb`). This directory contains only the site
presentation (theme, layout, hero/card configuration) and the build-time
generators:

- `scripts/generate-docs.mjs` converts `../docs/*.rst` to the Starlight pages
  in `src/content/docs/` (gitignored, never committed) with `rst-to-myst`,
- `scripts/generate-notebooks.mjs` renders `../docs/*.ipynb` to the tutorial
  pages including the tutorials index,
- `starlight-pydocs` builds the `pysm3` API reference from `../src` with
  griffe at build time.

There is no committed copy of the documentation Markdown: edit the RST or the
notebooks and rebuild, the site updates.

## Commands

```bash
npm install              # install dependencies
npm run dev              # generate docs + notebooks, then start the dev server
npm run build            # generate docs + notebooks, then build to dist/
npm run generate:docs    # regenerate html pages from ../docs/*.rst
npm run generate:notebooks   # regenerate notebook pages from ../docs/*.ipynb
```

## Read the Docs build

`.readthedocs.yaml` at the repository root drives the build:

1. `post_checkout` downloads the Git LFS binary and pulls the notebook content.
2. `install` installs griffe and rst-to-myst, then runs `npm ci`.
3. `build` runs `npm run build` and copies `docs-site/dist/` into
   `$READTHEDOCS_OUTPUT/html`.

## Environment variables

Read the Docs sets `READTHEDOCS_CANONICAL_URL` (used for the Astro `site` and
`base` so assets and links resolve under the version path) and
`READTHEDOCS_OUTPUT` (where the HTML output is published).

## Site configuration

- `site.config.json` holds the hero (title, tagline, image, actions) and the
  introduction cards; it is used by `scripts/generate-docs.mjs`.
- `site-chrome/404.md` is the 404 page template.
- `src/styles/custom.css` is the theme (colors, card and hero styling).
- `src/components/Footer.astro` adds the Read the Docs flyout mount point.

## Notes

- LaTeX math is rendered with KaTeX via a `unified` markdown processor
  (`remark-math` + `rehype-katex`), which requires `@astrojs/markdown-remark`.
- Read the Docs handles versions: each tag or branch is built from its own
  commit, so old releases keep their original Sphinx documentation while the
  current development version uses this site.
