#!/usr/bin/env node
// highlight-deq.mjs
//
// Generates syntax-highlighted HTML from a .deq file using Shiki
// and the existing VS Code TextMate grammar.
//
// Usage:
//   node scripts/highlight-deq.mjs <input.deq> [--theme light|dark]
//                                              [--start N] [--end N]
//
// --start and --end select a 1-indexed inclusive line range.  The whole
// file is always highlighted first (so the TextMate grammar sees full
// cross-line context, including containing GADGET / COMPOSE / Mako
// blocks); the resulting HTML is then sliced down to just the requested
// per-line <span class="line"> entries emitted by Shiki.
//
// Prints the highlighted HTML to stdout.

import { createHighlighter } from 'shiki';
import { readFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const tutorialDir = resolve(__dirname, '..');
const grammarPath = resolve(tutorialDir, '../../deq/circuit/vscode-deq/syntaxes/deq.tmLanguage.json');

// Parse CLI args
const args = process.argv.slice(2);

function optionValue(flag) {
  const idx = args.indexOf(flag);
  return idx >= 0 ? args[idx + 1] : null;
}

const themeArg = optionValue('--theme') ?? 'light';
const themeName = themeArg === 'light' ? 'light-plus' : 'github-dark';
const startArg = optionValue('--start');
const endArg = optionValue('--end');

const flagValueIndices = new Set();
for (const flag of ['--theme', '--start', '--end']) {
  const idx = args.indexOf(flag);
  if (idx >= 0) flagValueIndices.add(idx + 1);
}

const inputFile = args.find(
  (a, i) => !a.startsWith('--') && !flagValueIndices.has(i)
);

if (!inputFile) {
  console.error('Usage: node highlight-deq.mjs <file.deq> [--theme light|dark] [--start N] [--end N]');
  process.exit(1);
}

const deqGrammar = JSON.parse(readFileSync(grammarPath, 'utf-8'));

const highlighter = await createHighlighter({
  themes: ['light-plus', 'github-dark'],
  langs: [
    'python',  // needed for source.python references in Mako blocks
    { ...deqGrammar, name: 'deq', scopeName: 'source.deq' },
  ],
});

let code = readFileSync(resolve(inputFile), 'utf-8').replace(/\n+$/, '');
const totalLines = code.split('\n').length;

const start = startArg !== null ? parseInt(startArg, 10) : null;
const end = endArg !== null ? parseInt(endArg, 10) : null;
if (
  (start !== null && (Number.isNaN(start) || start < 1)) ||
  (end !== null && (Number.isNaN(end) || end < 1)) ||
  (start !== null && end !== null && start > end)
) {
  console.error(`Invalid --start/--end range: ${startArg}..${endArg}`);
  process.exit(1);
}

const html = highlighter.codeToHtml(code, { lang: 'deq', theme: themeName });
const sliced =
  start !== null || end !== null
    ? sliceHighlightedLines(html, start ?? 1, end ?? totalLines)
    : html;
process.stdout.write(sliced);
highlighter.dispose();

// Extract only the requested 1-indexed inclusive [start, end] range from
// Shiki's HTML output.  Shiki emits one ``<span class="line">…</span>``
// per source line inside ``<pre …><code>…</code></pre>``.  Because token
// spans nested inside a line are also ``<span>…</span>``, we track
// nesting depth to find the matching close.
function sliceHighlightedLines(fullHtml, sliceStart, sliceEnd) {
  const codeOpenTag = '<code>';
  const codeCloseTag = '</code></pre>';
  const codeOpenIdx = fullHtml.indexOf(codeOpenTag);
  const codeCloseIdx = fullHtml.lastIndexOf(codeCloseTag);
  if (codeOpenIdx < 0 || codeCloseIdx < 0) {
    return fullHtml;
  }
  const header = fullHtml.slice(0, codeOpenIdx + codeOpenTag.length);
  const inner = fullHtml.slice(codeOpenIdx + codeOpenTag.length, codeCloseIdx);
  const footer = fullHtml.slice(codeCloseIdx);

  const lineOpenTag = '<span class="line">';
  const spanClose = '</span>';
  const lines = [];
  let i = 0;
  while (i < inner.length) {
    const nextLine = inner.indexOf(lineOpenTag, i);
    if (nextLine < 0) break;
    let depth = 1;
    let j = nextLine + lineOpenTag.length;
    while (j < inner.length && depth > 0) {
      if (inner.startsWith('<span', j)) {
        depth++;
        const gt = inner.indexOf('>', j);
        j = gt < 0 ? inner.length : gt + 1;
      } else if (inner.startsWith(spanClose, j)) {
        depth--;
        j += spanClose.length;
      } else {
        j++;
      }
    }
    lines.push(inner.slice(nextLine, j));
    i = j;
  }

  const clampedStart = Math.max(1, sliceStart);
  const clampedEnd = Math.min(lines.length, sliceEnd);
  const selected = lines.slice(clampedStart - 1, clampedEnd);
  return header + selected.join('\n') + footer;
}
