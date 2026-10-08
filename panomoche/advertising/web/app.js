'use strict';
const $ = id => document.getElementById(id);
const entries = [];
let selected = null, running = false, nextId = 1;

function say(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function threshold() { return Number($('threshold').value); }
function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read the image.'));
    reader.readAsDataURL(file);
  });
}
function rejection(error, name) {
  return { decision: 'error', accepted: false, ad_rejected: false, checked: false, reason_code: 'processing_error',
    reason: error, filename: name, ambiguous_action: 'accept', categories: [] };
}
function safeStem(name) { return name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 100) || 'image'; }

async function download(data, name) {
  try {
    const response = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: name, content: JSON.stringify(data) }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not prepare the download.');
    const anchor = document.createElement('a');
    anchor.href = payload.url;
    anchor.download = name;
    document.body.append(anchor); anchor.click(); anchor.remove();
    say(`${name} is ready to download.`);
  } catch (error) { say(`Export failed: ${error.message}`, true); }
}

function renderQueue() {
  $('queue').replaceChildren();
  const accepted = entries.filter(entry => entry.result && entry.result.accepted).length;
  const rejected = entries.filter(entry => entry.result && entry.result.ad_rejected).length;
  const errors = entries.filter(entry => entry.result && entry.result.decision === 'error').length;
  $('count').textContent = `${entries.length} images${accepted || rejected || errors ? ` · ${accepted} accepted / ${rejected} rejected${errors ? ` / ${errors} errors` : ''}` : ''}`;
  $('clear').disabled = running || !entries.length;
  $('reanalyze').disabled = running || !entries.length;
  $('examples').disabled = running;
  $('download-batch').hidden = !entries.some(entry => entry.result);
  for (const entry of entries) {
    const button = document.createElement('button');
    button.className = 'queue-item' + (entry.id === selected ? ' selected' : '');
    button.setAttribute('aria-pressed', String(entry.id === selected));
    const thumbnail = document.createElement('img'); thumbnail.src = entry.thumbnail; thumbnail.alt = '';
    const info = document.createElement('span'); info.className = 'queue-info';
    const name = document.createElement('strong'); name.textContent = entry.file.name;
    const state = document.createElement('small');
    state.textContent = entry.result ? (entry.result.reason_code === 'processing_error' ? 'Not processed · input or connection error' : entry.result.accepted ? entry.result.checked === false ? 'Passed · check incomplete' : 'No obvious advertisement' : 'Clear advertising detected') : entry.status === 'working' ? 'Checking…' : 'Waiting…';
    info.append(name, state); button.append(thumbnail, info);
    if (entry.result) {
      const tag = document.createElement('span');
      tag.className = 'queue-score' + (entry.result.decision === 'error' ? ' error' : entry.result.accepted ? '' : ' reject');
      tag.textContent = entry.result.decision === 'error' ? 'ERROR' : entry.result.accepted ? 'ACCEPT' : 'REJECT';
      button.append(tag);
    }
    button.addEventListener('click', () => { selected = entry.id; renderQueue(); renderSelected(); });
    $('queue').append(button);
  }
}

function categoryLabel(result, id) {
  const category = (result.categories || []).find(item => item.id === id);
  return category ? category.label : 'Unavailable';
}

function renderSelected() {
  const entry = entries.find(item => item.id === selected);
  $('loading').hidden = !entry || entry.status !== 'working';
  $('empty').hidden = Boolean(entry && entry.result);
  $('result').hidden = !entry || !entry.result;
  if (!entry || !entry.result) return;
  const result = entry.result;
  $('filename').textContent = entry.file.name;
  $('result-title').textContent = result.decision === 'error' ? 'Could not process' : result.accepted ? 'Accepted' : 'Advertisement rejected';
  $('result-title').style.color = result.accepted ? '#3c594b' : '#a34842';
  $('explanation').textContent = result.reason;
  $('score').textContent = Number.isFinite(result.ad_score) ? result.ad_score.toFixed(1) : '—';
  $('score').style.color = result.accepted ? '#3c594b' : '#a34842';
  $('decision').textContent = result.decision === 'error' ? 'No content decision' : result.accepted ? (result.checked === false ? 'Passed by default · check incomplete' : 'Passed advertisement check') : 'Clear advertisement';
  $('decision').classList.toggle('reject', result.ad_rejected === true);
  $('decision').classList.toggle('error', result.decision === 'error');
  $('resolution').textContent = result.original_size ? `${result.original_size[0]} × ${result.original_size[1]}${Number.isFinite(result.seconds) ? ` · ${result.seconds.toFixed(1)}s` : ''}` : 'Image could not be analyzed';
  $('applied-settings').textContent = `Reject at ${result.threshold ?? entry.threshold} · uncertain → accept`;
  $('preview').src = entry.thumbnail;
  $('preview').alt = `Content check for ${entry.file.name}`;
  const component = key => Number.isFinite(result.score_components?.[key]) ? result.score_components[key].toFixed(1) : '—';
  $('place-category').textContent = component('visual_layout');
  $('other-category').textContent = component('text_layout');
  $('time').textContent = component('commercial_text');
  $('categories').replaceChildren();
  const ranking = (result.categories || []).slice(0, 6);
  const max = Math.max(0.001, ...ranking.map(item => item.similarity));
  for (const item of ranking) {
    const row = document.createElement('div'); row.className = 'category-row';
    const label = document.createElement('span'); label.textContent = item.label;
    const track = document.createElement('div'); track.className = 'category-track';
    const fill = document.createElement('div'); fill.className = 'category-fill' + (item.advertising ? ' blocked' : '');
    fill.style.width = `${Math.max(0, item.similarity / max * 100)}%`;
    track.append(fill);
    const value = document.createElement('span'); value.className = 'category-value'; value.textContent = item.similarity.toFixed(3);
    row.append(label, track, value); $('categories').append(row);
  }
  if (!ranking.length) $('categories').textContent = 'No model evidence was produced. This is not an advertisement rejection.';
  const ocr = result.ocr;
  $('ocr').textContent = !ocr ? 'Text checks were not completed.' : !ocr.available ? 'Text evidence is unavailable. The visual score is retained and the image passes by default.'
    : ocr.promotional_terms.length ? `Commercial signals: ${ocr.promotional_terms.join(', ')}. Text covers approximately ${ocr.text_area_percent.toFixed(1)}% of the image. These signals support the layout evidence; text alone cannot reject a photo.`
    : `No commercial wording recognized. Text covers approximately ${ocr.text_area_percent.toFixed(1)}% of the image. Visual and text coverage evidence still contribute to the score.`;

}

async function pump() {
  if (running) return;
  running = true;
  renderQueue();
  let entry;
  while ((entry = entries.find(item => item.status === 'pending'))) {
    entry.status = 'working'; renderQueue(); renderSelected(); say(`Checking ${entry.file.name}…`);
    try {
      const image = await readFile(entry.file);
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image, filename: entry.file.name, ad_threshold: entry.threshold }) });
      const result = await response.json();
      if (!['accept', 'reject', 'error'].includes(result.decision)) throw new Error(result.error || 'The server returned an invalid decision.');
      if (!response.ok) result.accepted = false;
      entry.result = result;
    } catch (error) { entry.result = rejection(error.message === 'Failed to fetch' ? 'Cannot reach the local server. No advertisement decision was made.' : error.message, entry.file.name); }
    entry.status = 'done'; renderQueue(); renderSelected();
  }
  running = false; renderQueue(); renderSelected();
  const accepted = entries.filter(item => item.result && item.result.accepted).length;
  const rejected = entries.filter(item => item.result && item.result.ad_rejected).length;
  const errors = entries.filter(item => item.result && item.result.decision === 'error').length;
  say(`${accepted} accepted · ${rejected} advertisements rejected${errors ? ` · ${errors} processing errors` : ''}. Uncertain content passes.`);
}

function addFiles(files) {
  const placeholder = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="320"><rect width="480" height="320" fill="#eee5df"/><text x="240" y="165" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#a34842">Cannot process input</text></svg>');
  for (const file of files) {
    let error = null;
    if (file.size > 20 * 1024 * 1024) error = 'File exceeds the 20 MB limit.';
    else if (!/\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name) && !file.type.startsWith('image/')) error = 'Unsupported file format.';
    const entry = { id: nextId++, file, thumbnail: error ? placeholder : URL.createObjectURL(file), status: error ? 'done' : 'pending', threshold: threshold(), result: error ? rejection(error, file.name) : null, clientError: error };
    entries.push(entry); if (selected === null) selected = entry.id;
  }
  renderQueue(); renderSelected();
  if (entries.some(item => item.status === 'pending')) pump();
  else say(`${entries.filter(item => item.result && !item.result.accepted).length} inputs could not be processed.`);
}

$('files').addEventListener('change', event => { addFiles(event.target.files); event.target.value = ''; });
$('dropzone').addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); $('files').click(); } });
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => addFiles(event.dataTransfer.files));
$('examples').addEventListener('click', async () => {
  $('examples').disabled = true; say('Loading reference images…');
  try {
    const response = await fetch('/api/examples');
    if (!response.ok) throw new Error('Cannot load examples.');
    const list = await response.json();
    const files = [];
    for (const example of list) {
      const image = await fetch(example.url);
      if (!image.ok) throw new Error('An example image is missing.');
      const blob = await image.blob();
      files.push(new File([blob], example.name, { type: blob.type }));
    }
    addFiles(files);
  } catch (error) { $('examples').disabled = false; say(error.message, true); }
});
$('clear').addEventListener('click', () => {
  if (running) return;
  entries.forEach(entry => URL.revokeObjectURL(entry.thumbnail)); entries.length = 0; selected = null;
  $('preview').removeAttribute('src'); renderQueue(); renderSelected(); say('');
});
function updateSettings() {
  $('threshold-value').textContent = `${$('threshold').value} / 100`;
  $('settings-note').textContent = entries.length ? 'Check again to apply changes to all images.' : 'Settings are recorded with each result.';
}
$('threshold').addEventListener('input', updateSettings);
$('reset').addEventListener('click', () => { $('threshold').value = '90'; updateSettings(); });
$('reanalyze').addEventListener('click', () => { if (running) return; for (const entry of entries) { entry.threshold = threshold(); if (entry.clientError) { entry.result = rejection(entry.clientError, entry.file.name); entry.status = 'done'; } else { entry.result = null; entry.status = 'pending'; } } pump(); });
$('download-json').addEventListener('click', () => { const entry = entries.find(item => item.id === selected); if (entry && entry.result) download(entry.result, `${safeStem(entry.file.name)}-content.json`); });
$('download-batch').addEventListener('click', () => download({ version: '0.4.0', scope: 'obvious_advertisements_only', ambiguous_action: 'accept', results: entries.filter(entry => entry.result).map(entry => entry.result) }, 'content-batch.json'));
fetch('/api/health').then(response => response.json()).then(state => { $('model-state').textContent = state.ready ? 'SigLIP 2 ready · runs locally' : 'Model unavailable · passes unchecked'; }).catch(() => { $('model-state').textContent = 'Local server unavailable'; });
