# docs/ folder role

This folder is the **single source of truth** for the PySM documentation
content. Everything the documentation sites show is generated from here at
build time:

- `docs/*.ipynb` (Git LFS) are the canonical Jupyter tutorial notebooks. The
  Starlight site at `docs-site/` renders them to pages (with extracted images
  and KaTeX formulas) via `docs-site/scripts/generate-notebooks.mjs`, and every
  release tag carries its own copy so each version keeps its own tutorials.
- `docs/*.rst` are the legacy Sphinx sources, still used to build the
  documentation of the older release versions on Read the Docs (each tag has
  its own copy in its own commit). On `main` they are not built by Sphinx
  anymore: the Starlight site is generated from them (plus the notebooks) at
  build time by `docs-site/scripts/generate-docs.mjs`.
- Images and data files referenced by the RST (`websky_tensor_plot.png`,
  `cmb_c1_params.ini`, ...) are copied to the built site automatically.

How to add or change documentation:

1. Edit the corresponding `.rst` (or add a new `.rst` file), or add/replace a
   tutorial `.ipynb` in this folder.
2. Rebuild the docs site (see `docs-site/README.md`); the new pages appear
   automatically, no other change is needed.

Do not maintain a second copy of this content anywhere else: the Starlight
site's content directory is generated and never committed.
