// Regenerates archived version documentation from the git tags of the repo.
//
// For each version tag (3.4.4, 3.4.5, 3.4.6):
// - extracts the tag's `docs/` (ReStructuredText + images) with `git archive`,
// - converts the RST files to Markdown with `rst-to-myst` (via `uvx`),
// - post-processes the converted pages (tables, roles, math, toctree),
// - writes the pages to `src/content/docs/<slug>/` (gitignored),
// - writes the version sidebar to `src/content/versions/<slug>.json` (gitignored),
// - extracts the tag's `src/` to `.build/<slug>/` so `starlight-pydocs` can
//   build the per-version API reference at build time (no dumps committed).
//
// Requires: git (with the version tags available) and `uvx` on PATH.
// The CI workflow checks out the full history, and `uv` is set up there.

import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, readdirSync, readFileSync, writeFileSync, existsSync, cpSync } from 'node:fs';
import { resolve, dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const siteDir = resolve(scriptDir, '..');
const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], { cwd: siteDir }).toString().trim();

const versions = [
  { slug: '3-4-6', tag: '3.4.6' },
  { slug: '3-4-5', tag: '3.4.5' },
  { slug: '3-4-4', tag: '3.4.4' },
];

const contentDir = join(siteDir, 'src/content/docs');
const versionsDir = join(siteDir, 'src/content/versions');
const buildDir = join(siteDir, '.build');

function versionSidebar() {
  return [
    {
      label: 'Start here',
      items: [
        { label: 'Introduction', slug: 'index' },
        { label: 'Models', slug: 'models' },
      ],
    },
    {
      label: 'Guides',
      items: [
        { label: 'CO line emission', slug: 'co_lines' },
        { label: 'WebSky models', slug: 'websky' },
      ],
    },
    {
      label: 'API reference',
      items: [{ label: 'pysm3 API', link: 'api/pysm3/' }],
    },
    {
      label: 'Contributing',
      items: [
        { label: 'Contribution guide', slug: 'contributing' },
        { label: 'Methods paper contributions', slug: 'pysm_methods_author_contributions' },
      ],
    },
  ];
}

// --- Markdown post-processing ------------------------------------------------

function cleanInline(text) {
  return text
    .replace(/\{py:(?:class|func|meth|attr|obj|mod|data|exc|const|var)\}`~?([^`]+)`/g, '`$1`')
    .replace(/\{ref\}`([^`]+)`/g, (_, label) => `[${label}](${label}/)`)
    .replace(/\{doc\}`([^`]+?) <([^>]+)>`/g, (_, title, page) => `[${title}](${page}/)`)
    .replace(/\{download\}`([^`]+?) <([^>]+)>`/g, '[$1]($2)')
    .replace(/\{download\}`([^`]+)`/g, '[$1]($1)')
    .replace(/:math:`([^`]+)`/g, (_, m) => `$${cleanMath(m)}$`)
    .replace(/:sup:`([^`]+)`/g, '<sup>$1</sup>')
    .replace(/:sub:`([^`]+)`/g, '<sub>$1</sub>')
    .replace(/\\\s/g, ' ');
}

function cleanMath(m) {
  return m.replace(/\\([_{}])/g, '$1').replace(/\\\\/g, '\\').trim();
}

function convertEvalRst(inner) {
  if (/\.\. automodapi::/.test(inner)) {
    return 'See the [pysm3 API reference](api/pysm3/).';
  }
  if (/\.\. list-table::/.test(inner)) {
    return convertListTable(inner);
  }
  return '';
}

function convertListTable(inner) {
  const rows = [];
  let current = null;
  for (const rawLine of inner.split('\n')) {
    const line = rawLine.replace(/\s+$/, '');
    if (/^\s*\* -/.test(line)) {
      current = [line.replace(/^\s*\* -\s*/, '')];
      rows.push(current);
    } else if (/^\s*- /.test(line)) {
      if (current) current.push(line.replace(/^\s*- /, ''));
    } else if (/^\s*$/.test(line)) {
      // blank line: keep going
    } else if (current) {
      current[current.length - 1] += ' ' + line.trim();
    }
  }
  const cells = rows.map((row) => row.map((cell) => cleanInline(cell).replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim()));
  if (cells.length === 0) return '';
  const ncols = Math.max(...cells.map((r) => r.length));
  const header = cells[0];
  const body = cells.slice(1);
  const line = (row) => '| ' + Array.from({ length: ncols }, (_, i) => (row[i] ?? '')).join(' | ') + ' |';
  const sep = '| ' + Array.from({ length: ncols }, () => '---').join(' | ') + ' |';
  return [line(header), sep, ...body.map(line)].join('\n');
}

function unescapeInlineMath(text) {
  return text.replace(/\\\$([^$\n]{1,200}?)\\\$/g, (_, m) => `$${cleanMath(m)}$`);
}

function postprocess(md) {
  let out = md
    .replace(/^([ \t]*)```\{image\} ([^\n]+)\n?([\s\S]*?)```/gm, (_, indent, src, opts) => {
      const name = src.trim();
      const altMatch = opts.match(/^\s*(?:alt|name|target):\s*(.+)$/m);
      const alt = altMatch ? altMatch[1].trim() : basename(name).replace(/\.[^.]+$/, '');
      return `${indent}![${alt.replaceAll('"', '\\"')}](images/${name})`;
    })
    .replace(/^([ \t]*)```\{toctree\}[\s\S]*?```\n?/gm, '') // toctree handled by the sidebar
    .replace(/^([ \t]*)```\{eval-rst\}\n([\s\S]*?)```/gm, (_, indent, inner) =>
      convertEvalRst(inner)
        .split('\n')
        .map((l) => indent + l)
        .join('\n')
    )
    .replace(/^\([a-z0-9_-]+\)=\s*$/gm, ''); // MyST anchor lines from RST labels
  out = cleanInline(out);
  out = unescapeInlineMath(out);
  return out;
}

function frontmatter(md, file) {
  const titleMatch = md.match(/^# (.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : basename(file, '.md');
  const body = md.replace(/^# .+$/m, '');
  const firstPara = body
    .split('\n')
    .map((l) => l.trim())
    .find((l) => l && !l.startsWith('#') && !l.startsWith('|') && !l.startsWith('`'));
  const description = (firstPara ?? '').slice(0, 160);
  return ['---', `title: "${title.replaceAll('"', '\\"')}"`, `description: "${description.replaceAll('"', '\\"')}"`, '---', ''].join('\n');
}

// --- Main ---------------------------------------------------------------------

for (const version of versions) {
  console.log(`Generating archived version '${version.slug}' from tag '${version.tag}'...`);

  const workDir = join(buildDir, version.slug);
  rmSync(workDir, { recursive: true, force: true });
  mkdirSync(workDir, { recursive: true });

  execFileSync('bash', ['-c', `cd "${repoRoot}" && git archive ${version.tag} docs src | tar -x -C "${workDir}"`]);

  const docsWork = join(workDir, 'docs');
  const outVersionDir = join(contentDir, version.slug);
  rmSync(outVersionDir, { recursive: true, force: true });
  mkdirSync(outVersionDir, { recursive: true });

  const rstFiles = readdirSync(docsWork).filter((f) => f.endsWith('.rst'));
  for (const rst of rstFiles) {
    const mdFile = join(docsWork, rst.replace(/\.rst$/, '.md'));
    execFileSync('uvx', ['--from', 'rst-to-myst', 'rst2myst', 'convert', '--no-sphinx', rst], { cwd: docsWork, stdio: 'ignore' });
    let md = readFileSync(mdFile, 'utf8');
    md = postprocess(md);
    // image references in the tag's docs point at files next to the RST
    md = md.replace(/\]\((websky_tensor_plot\.png)\)/g, '](images/$1)');
    const outFile = join(outVersionDir, rst.replace(/\.rst$/, '.md'));
    writeFileSync(outFile, frontmatter(md, basename(outFile)) + md + '\n');
  }

  for (const img of readdirSync(docsWork).filter((f) => f.endsWith('.png'))) {
    const imgDir = join(outVersionDir, 'images');
    mkdirSync(imgDir, { recursive: true });
    cpSync(join(docsWork, img), join(imgDir, img));
  }

  for (const dataFile of readdirSync(docsWork).filter((f) => f.endsWith('.ini'))) {
    cpSync(join(docsWork, dataFile), join(outVersionDir, dataFile));
  }

  mkdirSync(versionsDir, { recursive: true });
  writeFileSync(
    join(versionsDir, `${version.slug}.json`),
    JSON.stringify({ sidebar: versionSidebar(), excluded: [] }, null, 2) + '\n'
  );

  // Extract the tag's Python package source for the API reference (starlight-pydocs search)
  const srcWork = join(workDir, 'src');
  if (existsSync(join(srcWork, 'pysm3'))) {
    mkdirSync(join(buildDir, version.slug, 'pysm3'), { recursive: true });
    cpSync(join(srcWork, 'pysm3'), join(buildDir, version.slug, 'pysm3'), { recursive: true });
  }

  console.log(`  -> ${rstFiles.length} doc pages and API source in ${buildDir}/${version.slug}`);
}

console.log('Done.');
