// Renders every Jupyter notebook in the repository (notebooks/**/*.ipynb,
// Git LFS) into Starlight pages, including the preprocessing/template
// notebooks. Images embedded in outputs are extracted to files, and standalone
// images next to a notebook are copied alongside it and referenced relative to
// the page. The tutorials index (notebooks/index.md) lists the main tutorials
// at the top level.
//
// Requirements: node only. The notebook sources are in ../notebooks/ (Git LFS,
// pulled at build time by the Read the Docs config and the CI workflow).

import { readFileSync, writeFileSync, mkdirSync, readdirSync, rmSync, statSync, cpSync } from 'node:fs';
import { resolve, dirname, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const notebooksSourceDir = resolve(scriptDir, '../../notebooks');
const outDir = resolve(scriptDir, '../src/content/docs/notebooks');

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']);

function collectNotebooks(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...collectNotebooks(full));
    else if (entry.name.endsWith('.ipynb')) out.push(full);
  }
  return out;
}

function collectSidecarImages(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isFile() && IMAGE_EXTS.has(entry.name.slice(entry.name.lastIndexOf('.')))) out.push(full);
  }
  return out;
}

function renderCellRendered(cell, state) {
  const parts = [];
  const outputs = cell.outputs ?? [];
  if (cell.cell_type === 'code') {
    const code = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    parts.push(`\`\`\`python\n${code}\n\`\`\``);
    for (const output of outputs) {
      if (output.output_type === 'stream') {
        const text = Array.isArray(output.text) ? output.text.join('') : output.text ?? '';
        if (text.trim()) parts.push(`\`\`\`\n${text.trimEnd()}\n\`\`\``);
      } else if (output.output_type === 'error') {
        const trace = Array.isArray(output.traceback) ? output.traceback.join('\n') : (output.traceback ?? '');
        parts.push(`\`\`\`\n${output.ename ?? 'Error'}: ${output.evalue ?? ''}\n\n${trace}\n\`\`\``);
      } else if (output.output_type === 'display_data' || output.output_type === 'execute_result') {
        const data = output.data ?? {};
        const mime = Object.keys(data).find((k) => ![ 'text/plain', 'text/html', 'text/markdown' ].includes(k));
        if (mime && mime.startsWith('image/')) {
          const b64 = Array.isArray(data[mime]) ? data[mime].join('') : data[mime];
          const ext = mime === 'image/jpeg' ? 'jpg' : mime === 'image/svg+xml' ? 'svg' : 'png';
          state.imageIndex += 1;
          const filename = `img-${state.imageIndex}.${ext}`;
          writeFileSync(join(state.assetsDir, filename), Buffer.from(b64, 'base64'));
          parts.push(`![Output](${state.assetRefPrefix}/${filename})`);
        }
        if (data['text/html']) {
          const html = Array.isArray(data['text/html']) ? data['text/html'].join('') : data['text/html'];
          parts.push(html);
        }
        if (data['text/plain'] && !mime?.startsWith('image/')) {
          const text = Array.isArray(data['text/plain']) ? data['text/plain'].join('') : data['text/plain'];
          if (text.trim()) parts.push(`\`\`\`\n${text.trimEnd()}\n\`\`\``);
        }
      }
    }
  } else if (cell.cell_type === 'markdown') {
    let source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    if (!state.titleStripped) {
      source = source.replace(/^#\s+[^\n]+\n?/, '');
      state.titleStripped = true;
    }
    source = source.replace(/\]\(([^)]*?)\.ipynb\)/g, ']($1/)');
    // Standalone images next to the notebook: point at the copied assets.
    for (const asset of state.sidecarImages) {
      source = source.replace(new RegExp(`\\]\\(${asset.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`, 'g'), `](${state.assetRefPrefix}/${asset.name})`);
    }
    parts.push(source);
  }
  return parts.join('\n\n');
}

function titleFor(nb) {
  for (const cell of nb.cells) {
    if (cell.cell_type === 'markdown') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      const match = source.match(/^#\s+(.+)$/m);
      if (match) return match[1].trim();
    }
    if (cell.cell_type === 'code') break;
  }
  return '';
}

function descriptionFor(nb) {
  for (const cell of nb.cells) {
    if (cell.cell_type !== 'markdown') continue;
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
    for (const line of source.split('\n')) {
      if (/^\s*#/.test(line)) continue;
      const text = line
        .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
        .replace(/`([^`]*)`/g, '$1')
        .replace(/[*_~]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
      if (text.length >= 12) {
        return text.length > 180 ? `${text.slice(0, 177).trimEnd()}...` : text;
      }
    }
  }
  return '';
}

const notebooks = collectNotebooks(notebooksSourceDir).sort();
const topLevel = notebooks.filter((f) => !f.includes('/preprocess-templates/'));
const preprocess = notebooks.filter((f) => f.includes('/preprocess-templates/'));

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

let count = 0;
for (const notebook of notebooks) {
  const nb = JSON.parse(readFileSync(notebook, 'utf8'));
  const rel = notebook.slice(notebooksSourceDir.length + 1).replace(/\.ipynb$/, '');
  const pageRel = rel.toLowerCase(); // Astro routes are case-sensitive; keep slugs lowercase
  const pageDir = join(outDir, pageRel);
  const assetsDir = join(pageDir, 'assets');
  rmSync(assetsDir, { recursive: true, force: true });
  mkdirSync(assetsDir, { recursive: true });

  const sidecarImages = collectSidecarImages(dirname(notebook)).map((p) => ({ name: basename(p), path: p }));
  for (const img of sidecarImages) cpSync(img.path, join(assetsDir, img.name));

  const state = { titleStripped: false, assetsDir, imageIndex: 0, sidecarImages, assetRefPrefix: `${basename(pageRel)}/assets` };
  const body = nb.cells.map((cell) => renderCellRendered(cell, state)).filter(Boolean).join('\n\n');
  const title = titleFor(nb) || basename(rel);
  const description = descriptionFor(nb) || `${title} (PySM notebook)`;
  const frontmatter = [
    '---',
    `title: "${title.replaceAll('"', '\\"')}"`,
    `description: "${description.replaceAll('"', '\\"')}"`,
    '---',
    '',
  ].join('\n');
  writeFileSync(join(outDir, `${pageRel}.md`), frontmatter + body + '\n');
  count += 1;
}

// Tutorials index: top-level tutorials only.
const indexLines = [
  '---',
  'title: "Tutorials"',
  'description: "PySM Jupyter notebook tutorials, rendered in full on this site"',
  '---',
  '',
  'Practical tutorials for every component, from the first simulated map to bandpass integration and MPI:',
  '',
  ...[...topLevel].map((notebook) => {
    const nb = JSON.parse(readFileSync(notebook, 'utf8'));
    const slug = basename(notebook, '.ipynb').replace(/^_/, '');
    const title = titleFor(nb) || slug;
    const description = descriptionFor(nb);
    return `- [${title}](${slug}/): ${description}`;
  }),
  '',
  'For the data-processing workflows used to build the high-resolution templates shipped with PySM, see the [template preprocessing notebooks](preprocess-templates/).',
  '',
].join('\n');
writeFileSync(join(outDir, 'index.md'), indexLines);

// Preprocessing index.
const preIndex = [
  '---',
  'title: "Template preprocessing"',
  'description: "Notebooks documenting the data-processing workflow used to build the high-resolution templates shipped with PySM"',
  '---',
  '',
  'These notebooks document the data-processing workflows used to build the high-resolution templates shipped with PySM; they require the large input datasets:',
  '',
  ...[...preprocess].map((notebook) => {
    const nb = JSON.parse(readFileSync(notebook, 'utf8'));
    const rel = notebook.slice(notebooksSourceDir.length + 1).replace(/\.ipynb$/, '').toLowerCase();
    const title = titleFor(nb) || basename(rel);
    return `- [${title}](${rel.replace(/^preprocess-templates\//, '')}/)`;
  }),
  '',
].join('\n');
writeFileSync(join(outDir, 'preprocess-templates/index.md'), preIndex);

console.log(`Generated ${count} notebook pages in ${outDir}`);
