// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { currentSidebar } from './scripts/sidebar.mjs';

// Read the Docs serves the site under a version path (e.g. /en/latest/): use the
// canonical URL provided by Read the Docs so links and assets resolve correctly.
// See https://docs.readthedocs.com/platform/stable/intro/starlight.html
const site = process.env.READTHEDOCS_CANONICAL_URL;
const base = site ? new URL(site).pathname : '/';

const sourceLink = (ref) => ({ host: 'github', repo: 'galsci/pysm', ref, root: 'src' });

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
			title: 'PySM',
			description: 'Python Sky Model: full-sky simulations of Galactic foreground emissions',
			customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
			favicon: '/favicon.jpg',
			logo: {
				src: './src/assets/logo.jpg',
				alt: 'PySM',
				replacesTitle: false,
			},
			head: [
				{
					tag: 'meta',
					attrs: { property: 'og:image', content: 'https://avatars.githubusercontent.com/u/81497054?s=512&v=4' },
				},
				{
					tag: 'meta',
					attrs: { name: 'twitter:card', content: 'summary_large_image' },
				},
				{
					tag: 'meta',
					attrs: { name: 'twitter:image', content: 'https://avatars.githubusercontent.com/u/81497054?s=512&v=4' },
				},
			],
			editLink: {
				baseUrl: 'https://github.com/galsci/pysm/tree/main/docs-site/',
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/galsci/pysm',
				},
			],
			plugins: [
				starlightPydocs({
					packages: [{ name: 'pysm3', search: ['../src'], sourceLink: sourceLink('main') }],
				}),
				starlightLinksValidator({ errorOnRelativeLinks: false }),
			],
			sidebar: currentSidebar({ label: 'API reference', items: [pydocsSidebarGroup] }),
		}),
	],
});
