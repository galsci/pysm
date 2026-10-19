// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightVersions from 'starlight-versions';
import starlightLinksValidator from 'starlight-links-validator';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { currentSidebar } from './scripts/sidebar.mjs';

const site = process.env.DOCS_SITE ?? 'https://galsci.github.io/pysm';
const base = process.env.DOCS_BASE ?? '/pysm/';

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
					packages: [
						{ name: 'pysm3', search: ['../src'], sourceLink: sourceLink('main') },
						{
							name: 'pysm3',
							base: '3-4-6/api/pysm3',
							label: 'pysm3 3.4.6',
							source: { file: './dumps/pysm3-3.4.6.json' },
							sourceLink: sourceLink('3.4.6'),
						},
						{
							name: 'pysm3',
							base: '3-4-5/api/pysm3',
							label: 'pysm3 3.4.5',
							source: { file: './dumps/pysm3-3.4.5.json' },
							sourceLink: sourceLink('3.4.5'),
						},
						{
							name: 'pysm3',
							base: '3-4-4/api/pysm3',
							label: 'pysm3 3.4.4',
							source: { file: './dumps/pysm3-3.4.4.json' },
							sourceLink: sourceLink('3.4.4'),
						},
					],
				}),
				starlightLinksValidator({ errorOnRelativeLinks: false }),
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
			sidebar: currentSidebar({ label: 'API reference', items: [pydocsSidebarGroup] }),
		}),
	],
});
