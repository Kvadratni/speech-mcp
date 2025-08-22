import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const webDir = __dirname;
const outDir = resolve(webDir, '../resources/ui');

const htmlTpl = readFileSync(resolve(webDir, 'panel.html'), 'utf8');
const css = readFileSync(resolve(outDir, 'panel.css'), 'utf8');
const js = readFileSync(resolve(outDir, 'panel.js'), 'utf8');

const bundled = htmlTpl
  .replace('{{CSS}}', `<style>${css}</style>`)
  .replace('{{JS}}', `<script>${js}</script>`);

writeFileSync(resolve(outDir, 'panel.bundled.html'), bundled, 'utf8');
console.log('Wrote', resolve(outDir, 'panel.bundled.html'));

