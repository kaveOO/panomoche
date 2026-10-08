'use strict';
const $ = id => document.getElementById(id);
const entries = [];
let selected = null;
let activeView = 'overlay';
let running = false;
let nextId = 1;

function settings() {
  return { decision_threshold: Number($('threshold').value), minimum_bending_percent: Number($('bending').value) };
}

function say(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}

function withoutImages(result) {
  const { original, overlay, edge_map, ...summary } = result;
  return summary;
}

async function download(data, name, mime = 'application/json') {
  try {
    say(`Preparing ${name}…`);
    const isImage = typeof data === 'string' && data.startsWith('data:image/png');
    const response = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: name, type: isImage ? 'image/png' : mime, content: typeof data === 'string' ? data : JSON.stringify(data) }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not prepare the download.');
    const anchor = document.createElement('a');
    anchor.href = payload.url;
    anchor.download = name;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    say(`${name} is ready to download.`);
  } catch (error) { say(`Export failed: ${error.message}`, true); }
}

function safeStem(name) {
  return name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 100) || 'image';
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read this image.'));
    reader.readAsDataURL(file);
  });
}

function renderQueue() {
  $('queue').replaceChildren();
  $('count').textContent = `${entries.length} image${entries.length === 1 ? '' : 's'}`;
  $('clear').disabled = running || !entries.length;
  $('reanalyze').disabled = running || !entries.length;
  $('examples').disabled = running;
  $('download-batch').hidden = !entries.some(entry => entry.result);
  for (const entry of entries) {
    const button = document.createElement('button');
    button.className = 'queue-item' + (entry.id === selected ? ' selected' : '');
    button.setAttribute('aria-pressed', String(entry.id === selected));
    const thumbnail = document.createElement('img');
    thumbnail.src = entry.thumbnail;
    thumbnail.alt = '';
    const info = document.createElement('span');
    info.className = 'queue-info';
    const name = document.createElement('strong');
    name.textContent = entry.file.name;
    const state = document.createElement('small');
    state.textContent = entry.result ? entry.result.title : entry.status === 'error' ? 'Could not analyze' : entry.status === 'working' ? 'Analyzing…' : 'Waiting…';
    info.append(name, state);
    button.append(thumbnail, info);
    if (entry.result) {
      const score = document.createElement('span');
      score.className = 'queue-score';
      score.textContent = Math.round(entry.result.score);
      button.append(score);
    }
    button.addEventListener('click', () => {
      selected = entry.id;
      renderQueue();
      renderSelected();
    });
    $('queue').append(button);
  }
}

function renderChart(result) {
  const supported = result.edges.filter(edge => edge.supported);
  const candidates = supported.length ? supported : result.edges.filter(edge => edge.bent);
  const profiles = candidates.sort((a, b) => b.bending_percent - a.bending_percent).slice(0, 6);
  if (!profiles.length) {
    const text = document.createElement('span');
    text.className = 'no-profile';
    text.textContent = 'No bending profiles to display.';
    $('chart').replaceChildren(text);
    return;
  }
  const width = 680, height = 190, left = 42, right = 22, top = 28, bottom = 35;
  const xMax = result.analysis_size[1];
  const yMax = Math.max(3, ...profiles.flatMap(edge => edge.profile.map(point => Math.abs(point[1])))) * 1.15;
  const x = row => left + row / xMax * (width - left - right);
  const y = delta => top + (yMax - delta) / (2 * yMax) * (height - top - bottom);
  const colors = ['#da5861', '#457a65', '#b37b2d', '#5b90b0', '#9869a3', '#656861'];
  const ns = 'http://www.w3.org/2000/svg';
  function element(tag, attrs, text) {
    const el = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs || {})) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = text;
    return el;
  }
  const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Sideways edge displacement plotted against image row' });
  for (const value of [-yMax, 0, yMax]) {
    svg.append(element('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), stroke: '#dce0d5', 'stroke-dasharray': value ? '3 4' : '0' }));
    svg.append(element('text', { x: left - 8, y: y(value) + 3, 'text-anchor': 'end', fill: '#778174', 'font-size': 9 }, value.toFixed(0)));
  }
  for (const row of [0, Math.round(xMax / 2), xMax]) {
    svg.append(element('text', { x: x(row), y: height - 16, 'text-anchor': 'middle', fill: '#778174', 'font-size': 9 }, String(row)));
  }
  svg.append(element('text', { x: width - right, y: height - 2, 'text-anchor': 'end', fill: '#778174', 'font-size': 8 }, 'Image row ↓'));
  profiles.forEach((edge, index) => {
    svg.append(element('polyline', { points: edge.profile.map(p => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' '), fill: 'none', stroke: colors[index], 'stroke-width': 1.8, 'stroke-linejoin': 'round' }));
    svg.append(element('line', { x1: left + index * 88, x2: left + 14 + index * 88, y1: 12, y2: 12, stroke: colors[index], 'stroke-width': 2 }));
    svg.append(element('text', { x: left + 20 + index * 88, y: 15, fill: '#687260', 'font-size': 9 }, `Edge ${edge.id}`));
  });
  $('chart').replaceChildren(svg);
}

function renderSelected() {
  const entry = entries.find(e => e.id === selected);
  $('loading').hidden = !entry || entry.status !== 'working';
  $('empty').hidden = Boolean(entry && entry.result);
  $('result').hidden = !entry || !entry.result;
  if (!entry || !entry.result) {
    if (entry && entry.error) say(`${entry.file.name}: ${entry.error}`, true);
    return;
  }
  const result = entry.result;
  $('filename').textContent = entry.file.name;
  $('result-title').textContent = result.title;
  $('explanation').textContent = result.explanation;
  $('score').textContent = result.score.toFixed(1);
  $('score').style.color = result.label === 'wavy' ? '#ba5058' : result.label === 'review' ? '#9a7434' : '#3c594b';
  $('confidence').textContent = `${result.confidence[0].toUpperCase() + result.confidence.slice(1)} evidence confidence`;
  $('confidence').classList.toggle('low', result.confidence === 'low');
  $('resolution').textContent = `${result.original_size[0]} × ${result.original_size[1]} original`;
  $('applied-settings').textContent = `Threshold ${result.settings.decision_threshold} · min. bend ${result.settings.minimum_bending_percent.toFixed(2)}%`;
  $('usable').textContent = result.usable_edges;
  $('coverage').textContent = result.shared_edge_coverage_percent.toFixed(1) + '%';
  $('amplitude').textContent = result.median_bending_percent.toFixed(2) + '%';
  renderChart(result);
  setView(activeView);
}

function setView(view) {
  activeView = view;
  const entry = entries.find(e => e.id === selected);
  if (entry && entry.result) {
    $('preview').src = entry.result[view];
    $('preview').alt = `${view === 'overlay' ? 'Detected edge evidence' : view === 'edge_map' ? 'Canny edge map' : 'Original image'} for ${entry.file.name}`;
  }
  document.querySelectorAll('[data-view]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.view === view)));
  $('legend').hidden = view !== 'overlay';
}

async function pump() {
  if (running) return;
  running = true;
  renderQueue();
  let entry;
  while ((entry = entries.find(e => e.status === 'pending'))) {
    entry.status = 'working';
    renderQueue();
    renderSelected();
    say(`Analyzing ${entry.file.name}…`);
    try {
      const image = await readFile(entry.file);
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image, filename: entry.file.name, settings: entry.settings }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Analysis failed.');
      entry.result = data;
      entry.status = 'done';
      entry.error = null;
    } catch (error) {
      entry.status = 'error';
      entry.error = error.message === 'Failed to fetch' ? 'Cannot reach the local server. Restart server.py and try again.' : error.message;
    }
    renderQueue();
    renderSelected();
  }
  running = false;
  renderQueue();
  renderSelected();
  const done = entries.filter(e => e.result).length;
  const failed = entries.filter(e => e.status === 'error').length;
  say(`${done} image${done === 1 ? '' : 's'} analyzed${failed ? ` · ${failed} could not be analyzed. Select a failed image for details.` : '.'}`, Boolean(failed));
}

function addFiles(files) {
  const errors = [];
  for (const file of files) {
    if (file.size > 20 * 1024 * 1024) { errors.push(`${file.name} exceeds 20 MB.`); continue; }
    if (!/\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name) && !file.type.startsWith('image/')) { errors.push(`${file.name} is not a supported image.`); continue; }
    const entry = { id: nextId++, file, thumbnail: URL.createObjectURL(file), status: 'pending', result: null, settings: settings() };
    entries.push(entry);
    if (selected === null) selected = entry.id;
  }
  renderQueue();
  renderSelected();
  if (entries.some(e => e.status === 'pending')) pump();
  if (errors.length) say(errors.join(' '), true);
}

$('files').addEventListener('change', event => { addFiles(event.target.files); event.target.value = ''; });
$('dropzone').addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); $('files').click(); } });
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => addFiles(event.dataTransfer.files));

$('examples').addEventListener('click', async () => {
  $('examples').disabled = true;
  say('Loading the four reference images…');
  try {
    const files = [];
    for (const [path, name, type] of [['1.png', 'Wavy street 1.png', 'image/png'], ['2.png', 'Wavy street 2.png', 'image/png'], ['cows.png', 'Cows.png', 'image/png'], ['straight-street.jpg', 'Straight street.jpg', 'image/jpeg']]) {
      const response = await fetch(`/examples/${path}`);
      if (!response.ok) throw new Error('The example images are missing.');
      files.push(new File([await response.blob()], name, { type }));
    }
    addFiles(files);
  } catch (error) { say(error.message, true); $('examples').disabled = false; }
});

$('clear').addEventListener('click', () => {
  if (running) return;
  entries.forEach(entry => URL.revokeObjectURL(entry.thumbnail));
  entries.length = 0;
  selected = null;
  $('preview').removeAttribute('src');
  renderQueue(); renderSelected(); say('');
});

function updateSettings() {
  $('threshold-value').textContent = `${$('threshold').value} / 100`;
  $('bending-value').textContent = Number($('bending').value).toFixed(2) + '%';
  $('settings-note').textContent = entries.length ? 'Analyze again to apply changes to all images.' : 'Settings are saved with each result.';
}
for (const id of ['threshold', 'bending']) $(id).addEventListener('input', updateSettings);
$('reset').addEventListener('click', () => { $('threshold').value = '45'; $('bending').value = '0.20'; updateSettings(); });
$('reanalyze').addEventListener('click', () => {
  if (running) return;
  for (const entry of entries) { entry.status = 'pending'; entry.result = null; entry.settings = settings(); }
  pump();
});
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
$('download-overlay').addEventListener('click', () => {
  const entry = entries.find(e => e.id === selected);
  if (entry && entry.result) download(entry.result.overlay, `${safeStem(entry.file.name)}-overlay.png`);
});
$('download-json').addEventListener('click', () => {
  const entry = entries.find(e => e.id === selected);
  if (entry && entry.result) download(withoutImages(entry.result), `${safeStem(entry.file.name)}-waviness.json`);
});
$('download-batch').addEventListener('click', () => download({ version: '0.2.0', results: entries.filter(e => e.result).map(e => withoutImages(e.result)), errors: entries.filter(e => e.error).map(e => ({ filename: e.file.name, error: e.error })) }, 'waviness-batch.json'));
