#!/usr/bin/env node
// highlight-deq.mjs
//
// Generates syntax-highlighted HTML from a .deq file using Shiki
// and the existing VS Code TextMate grammar.
//
// Usage:
//   node scripts/highlight-deq.mjs <input.deq> [--theme light|dark]
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
const themeArg = args.includes('--theme') ? args[args.indexOf('--theme') + 1] : 'light';
const themeName = themeArg === 'light' ? 'light-plus' : 'github-dark';
const inputFile = args.find(a => !a.startsWith('--') && args[args.indexOf(a) - 1] !== '--theme');

if (!inputFile) {
  console.error('Usage: node highlight-deq.mjs <file.deq> [--theme light|dark]');
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

const code = readFileSync(resolve(inputFile), 'utf-8').trimEnd();
const html = highlighter.codeToHtml(code, { lang: 'deq', theme: themeName });
process.stdout.write(html);
highlighter.dispose();
