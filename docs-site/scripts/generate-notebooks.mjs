import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const docsSourceDir = resolve(scriptDir, '../../docs');
const outDir = resolve(scriptDir, '../src/content/docs/notebooks');

const notebooks = readdirSync(docsSourceDir).filter((f) => f.endsWith('.ipynb'));

function renderCellRendered(cell) {
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
          const mimeType = mime === 'image/svg+xml' ? 'image/svg+xml' : mime;
          parts.push(`<img src="data:${mimeType};base64,${b64}" alt="Output" />`);
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
    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
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

mkdirSync(outDir, { recursive: true });

for (const notebook of notebooks.sort()) {
  const nb = JSON.parse(readFileSync(resolve(docsSourceDir, notebook), 'utf8'));
  const title = titleFor(notebook);
  const slug = basename(notebook, '.ipynb').replace(/^_/, '');
  const body = nb.cells.map(renderCellRendered).filter(Boolean).join('\n\n');
  const frontmatter = [
    '---',
    `title: "${title.replaceAll('"', '\\"')}"`,
    `description: "PySM tutorial rendered from ${basename(notebook)}"`,
    '---',
    '',
  ].join('\n');
  writeFileSync(resolve(outDir, `${slug}.md`), frontmatter + body + '\n');
}

console.log(`Generated ${notebooks.length} notebook pages in ${outDir}`);
