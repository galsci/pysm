import { readFileSync, writeFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { resolve, dirname, basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const docsSourceDir = resolve(scriptDir, '../../notebooks');
const outDir = resolve(scriptDir, '../src/content/docs/notebooks');

const notebooks = readdirSync(docsSourceDir).filter((f) => f.endsWith('.ipynb'));

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
          parts.push(`![Output](assets/${state.slug}/${filename})`);
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
    parts.push(source);
  }
  return parts.join('\n\n');
}

function titleFor(notebook) {
  const nb = JSON.parse(readFileSync(resolve(docsSourceDir, notebook), 'utf8'));
  for (const cell of nb.cells) {
    if (cell.cell_type === 'markdown') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      const match = source.match(/^#\s+(.+)$/m);
      if (match) return match[1].trim();
    }
    if (cell.cell_type === 'code') break;
  }
  return basename(notebook, '.ipynb').replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function descriptionFor(notebook) {
  const nb = JSON.parse(readFileSync(resolve(docsSourceDir, notebook), 'utf8'));
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
  return `PySM tutorial: ${basename(notebook, '.ipynb').replaceAll('_', ' ')}`;
}

mkdirSync(outDir, { recursive: true });

for (const notebook of [...notebooks].sort()) {
  const nb = JSON.parse(readFileSync(resolve(docsSourceDir, notebook), 'utf8'));
  const title = titleFor(notebook);
  const slug = basename(notebook, '.ipynb').replace(/^_/, '');
  const assetsDir = join(outDir, 'assets', slug);
  rmSync(assetsDir, { recursive: true, force: true });
  mkdirSync(assetsDir, { recursive: true });
  const state = { titleStripped: false, assetsDir, slug, imageIndex: 0 };
  const body = nb.cells.map((cell) => renderCellRendered(cell, state)).filter(Boolean).join('\n\n');
  const frontmatter = [
    '---',
    `title: "${title.replaceAll('"', '\\"')}"`,
    `description: "${descriptionFor(notebook).replaceAll('"', '\\"')}"`,
    '---',
    '',
  ].join('\n');
  writeFileSync(resolve(outDir, `${slug}.md`), frontmatter + body + '\n');
}

const indexLines = [
  '---',
  'title: "Tutorials"',
  'description: "PySM Jupyter notebook tutorials, rendered in full on this site"',
  '---',
  '',
  'Practical tutorials for every component, from the first simulated map to bandpass integration and MPI:',
  '',
  ...[...notebooks].sort().map((notebook) => {
    const slug = basename(notebook, '.ipynb').replace(/^_/, '');
    const title = titleFor(notebook);
    return `- [${title}](${slug}/): ${descriptionFor(notebook)}`;
  }),
  '',
].join('\n');
writeFileSync(resolve(outDir, 'index.md'), indexLines);

console.log(`Generated ${notebooks.length} notebook pages in ${outDir}`);
