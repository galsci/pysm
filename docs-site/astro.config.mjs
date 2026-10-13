// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightVersions from 'starlight-versions';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const site = process.env.DOCS_SITE ?? 'https://galsci.github.io/pysm';
const base = process.env.DOCS_BASE ?? '/pysm/';

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
			customCss: ['katex/dist/katex.min.css'],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/galsci/pysm',
				},
			],
			plugins: [
				starlightPydocs({
					packages: [{ name: 'pysm3', search: ['../src'] }],
				}),
				starlightVersions({
					versions: [
						{ slug: '3-4-6', label: '3.4.6' },
						{ slug: '3-4-5', label: '3.4.5' },
						{ slug: '3-4-4', label: '3.4.4' },
					],
					current: { label: 'Latest' },
					exclude: ['notebooks/**'],
				}),
			],
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Introduction', slug: 'index' },
						{ label: 'Installation', slug: 'guides/installation' },
						{ label: 'Usage', slug: 'guides/usage' },
						{ label: 'Models', slug: 'models' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Best practices', slug: 'guides/best_practices' },
						{ label: 'Pre-applied beam', slug: 'pre_applied_beam' },
						{ label: 'CO line emission', slug: 'co_lines' },
						{ label: 'WebSky models', slug: 'websky' },
					],
				},
				{
					label: 'Notebooks',
					items: [{ autogenerate: { directory: 'notebooks' } }],
				},
				{
					label: 'API reference',
					items: [pydocsSidebarGroup],
				},
				{
					label: 'Contributing',
					items: [
						{ label: 'Contribution guide', slug: 'contributing' },
						{ label: 'Methods paper contributions', slug: 'pysm_methods_author_contributions' },
					],
				},
			],
		}),
	],
});
