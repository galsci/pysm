# PySM Starlight documentation site

Starlight (Astro) documentation site for PySM, deployed to GitHub Pages at
<https://galsci.github.io/pysm/>.

This is the experimental site; the canonical documentation remains on
Read the Docs at <https://pysm3.readthedocs.io/>.

## Commands

```bash
npm install              # install dependencies
npm run dev              # generate notebooks, then start the dev server
npm run build            # generate notebooks, then build to dist/
npm run generate:notebooks   # regenerate notebook pages from ../docs/*.ipynb
node scripts/archive-versions.mjs  # re-create archived versions (3-4-4/3-4-5/3-4-6)
```

The production build is driven by `.github/workflows/docs-pages.yml`: it builds
`main` with base `/pysm/` and every open PR that contains `docs-site/` with base
`/pysm/preview/pr-<n>/`, then deploys a single artifact and posts a preview
comment on the PR.

## Environment variables

| Variable    | Default                            | Used for                         |
| ----------- | ---------------------------------- | -------------------------------- |
| `DOCS_SITE` | `https://galsci.github.io/pysm`    | Astro `site`                     |
| `DOCS_BASE` | `/pysm/`                           | Astro `base` (e.g. `/pysm/preview/pr-278/`) |

## Content

- `src/content/docs/` is the current (latest) documentation.
- `src/content/docs/<version>/` are archived versions used by `starlight-versions`.
- `src/content/versions/<version>.json` holds each archived version's sidebar.
- `src/content/docs/notebooks/` is generated from `../docs/*.ipynb` (Git LFS) by
  `scripts/generate-notebooks.mjs`; it is gitignored and never committed.
- `dumps/` holds pre-generated Griffe dumps of the `pysm3` API for the archived
  versions (used by `starlight-pydocs`; the current API is extracted from
  `../src` at build time).

To archive a new version: add it to `versions` in `astro.config.mjs` and to the
`versions` array in `scripts/archive-versions.mjs`, then run
`node scripts/archive-versions.mjs`.

## Notes

- Version slugs must not contain dots: Astro strips them from route slugs
  (the display label in the picker can still contain dots).
- LaTeX math is rendered with KaTeX via a `unified` markdown processor
  (`remark-math` + `rehype-katex`), which requires `@astrojs/markdown-remark`.
