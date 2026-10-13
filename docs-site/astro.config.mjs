// @ts-check
import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import starlightVersions from "starlight-versions";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const site = process.env.DOCS_SITE ?? "https://galsci.github.io/pysm";
const base = process.env.DOCS_BASE ?? "/pysm/";

export default defineConfig({
  site,
  base,
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [
    starlight({
      title: "PySM",
      description:
        "Python Sky Model: full-sky simulations of Galactic foreground emissions",
      customCss: ["katex/dist/katex.min.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/galsci/pysm",
        },
      ],
      plugins: [
        starlightVersions({
          versions: [
            { slug: "3.4.6" },
            { slug: "3.4.5" },
            { slug: "3.4.4" },
          ],
          current: { label: "Latest" },
          exclude: ["notebooks/**"],
        }),
      ],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Introduction", slug: "index" },
            { label: "Installation", slug: "guides/installation" },
            { label: "Usage", slug: "guides/usage" },
          ],
        },
        {
          label: "Notebooks",
          items: [{ autogenerate: { directory: "notebooks" } }],
        },
      ],
    }),
  ],
});
