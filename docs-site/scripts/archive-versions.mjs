import { readdirSync, cpSync, writeFileSync, rmSync, mkdirSync, readFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(scriptDir, '..');
const docsDir = join(rootDir, 'src/content/docs');
const versionsDir = join(rootDir, 'src/content/versions');
const versions = ['3-4-6', '3-4-5', '3-4-4'];
const excludedGlob = 'notebooks/**';

const DOC_EXTENSIONS = new Set(['.md', '.mdx']);
const SKIP_DIRS = new Set(['notebooks', ...versions]);

const sidebar = [
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
    label: 'Contributing',
    items: [
      { label: 'Contribution guide', slug: 'contributing' },
      { label: 'Methods paper contributions', slug: 'pysm_methods_author_contributions' },
    ],
  },
];

function excludedSlugs(dir) {
  const out = [];
  for (const entry of readdirSync(join(dir, 'notebooks'), { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    if (!DOC_EXTENSIONS.has(entry.name.slice(entry.name.lastIndexOf('.')))) continue;
    out.push(`notebooks/${entry.name.replace(/\.[^.]+$/, '')}`);
  }
  return out;
}

function copyDocs(fromDir, toDir) {
  mkdirSync(toDir, { recursive: true });
  const cfg = readdirSync(fromDir, { withFileTypes: true });
  for (const entry of cfg) {
    const src = join(fromDir, entry.name);
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      copyDocs(src, join(toDir, entry.name));
    } else {
      if (entry.name.startsWith('.')) continue;
      cpSync(src, join(toDir, entry.name));
    }
  }
}

mkdirSync(versionsDir, { recursive: true });
const excluded = excludedSlugs(docsDir);

for (const version of versions) {
  const target = join(docsDir, version);
  if (existsSync(target)) rmSync(target, { recursive: true, force: true });
  copyDocs(docsDir, target);
  writeFileSync(
    join(versionsDir, `${version}.json`),
    JSON.stringify({ sidebar, excluded }, null, 2) + '\n',
  );
  console.log(`Archived version '${version}' (${excluded.length} excluded notebook pages).`);
}
