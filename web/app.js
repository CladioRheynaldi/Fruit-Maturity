let currentFruit = 'banana';
let data, upload, errorRows = [], visible = 30;

const $ = id => document.getElementById(id);
const pct = x => (100 * x).toFixed(1) + '%';
const escapeHTML = x => String(x).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));

const STAGE_COLORS = {
    'unripe': '#7ca942',
    'semi-ripe': '#c4bb3b',
    'ripe': '#dca132',
    'overripe': '#9b643a',
    'rotten': '#4d3b32'
};

async function request(url, options) {
    const res = await fetch(url, options);
    const body = await res.json();
    if (!res.ok) throw Error(body.error || 'Request failed');
    return body;
}

document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tab,.panel').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    $(b.dataset.tab).classList.add('active');
});

function selectFile(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
        $('file-note').textContent = 'Please choose an image smaller than 10 MB.';
        return;
    }
    upload = file;
    if ($('preview').src.startsWith('blob:')) URL.revokeObjectURL($('preview').src);
    $('preview').src = URL.createObjectURL(file);
    $('preview').hidden = false;
    $('analyze').disabled = false;
    $('file-note').textContent = file.name || 'Held-out test sample';
    $('result').innerHTML = '<h3>Photo ready.</h3><p>Select Analyze maturity to see its prediction.</p>';
}

$('file').onchange = e => selectFile(e.target.files[0]);
$('drop').ondragover = e => e.preventDefault();
$('drop').ondrop = e => {
    e.preventDefault();
    selectFile(e.dataTransfer.files[0]);
};

$('sample').onclick = async () => {
    try {
        const r = await fetch(`/sample/0?fruit=${encodeURIComponent(currentFruit)}`);
        if (!r.ok) throw Error('Test samples unavailable');
        const blob = await r.blob();
        selectFile(new File([blob], `${currentFruit}-test-sample.jpg`, { type: blob.type || 'image/jpeg' }));
        $('file-note').textContent = `Sample from the ${currentFruit} held-out test set; also included in the reported evaluation.`;
    } catch (e) {
        $('file-note').textContent = e.message;
    }
};

$('analyze').onclick = async () => {
    if (!upload) return;
    $('analyze').disabled = true;
    $('result').innerHTML = '<h3>Inspecting your image…</h3>';
    try {
        const url = `/api/predict?fruit=${encodeURIComponent(currentFruit)}&model=${encodeURIComponent($('model').value)}`;
        const r = await request(url, {
            method: 'POST',
            body: upload,
            headers: { 'Content-Type': upload.type || 'application/octet-stream' }
        });
        $('result').innerHTML = `
            <div class="prediction">${escapeHTML(r.label)}</div>
            <p style="text-align:left">${pct(r.confidence)} model score ${r.uncertain ? '· uncertain prediction' : ''}</p>
            ${Object.entries(r.probabilities).map(([k, v]) => `
                <div class="score-line"><span>${escapeHTML(k)}</span><span>${pct(v)}</span></div>
                <div class="bar"><i style="width:${100 * v}%"></i></div>
            `).join('')}
        `;
    } catch (e) {
        $('result').innerHTML = `<p class="error">${escapeHTML(e.message)}</p>`;
    } finally {
        $('analyze').disabled = false;
    }
};

function training() {
    const r = data.experiments.find(x => x.config.id === $('training-model').value);
    if (!r) return;
    const base = data.artifact_base || '/artifacts';
    $('curves').src = `${base}/${r.config.id}/curves.png`;
    $('diagnostics').innerHTML = r.diagnostics.map(d => `
        <div class="card">
            <h3>${escapeHTML(d.topic)}</h3>
            <span class="badge ${d.flag ? 'flag' : ''}">${d.flag ? 'Indicator present' : 'Indicator absent'}</span>
            <p>${escapeHTML(d.detail)}</p>
        </div>
    `).join('');
    $('epoch-table').innerHTML = `
        <p class="hint">Best checkpoint: epoch ${r.best_epoch} · ${r.config.epochs} epochs · ${r.training_seconds.toFixed(1)} seconds · Adam · batch ${r.config.batch_size}</p>
        <table>
            <thead><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Train accuracy</th><th>Val accuracy</th><th>Train F1</th><th>Val F1</th><th>Gradient norm</th></tr></thead>
            <tbody>${r.history.map(h => `
                <tr>
                    <td>${h.epoch}</td>
                    <td>${h.train_loss.toFixed(4)}</td>
                    <td>${h.val_loss.toFixed(4)}</td>
                    <td>${pct(h.train_accuracy)}</td>
                    <td>${pct(h.val_accuracy)}</td>
                    <td>${h.train_f1.toFixed(4)}</td>
                    <td>${h.val_f1.toFixed(4)}</td>
                    <td>${h.gradient_mean.toFixed(3)}</td>
                </tr>
            `).join('')}</tbody>
        </table>
    `;
}

function renderErrors() {
    const rows = errorRows.map((r, index) => ({ ...r, index })).filter(r => !$('errors-only').checked || !r.correct);
    $('error-count').textContent = `${rows.length} matching samples out of ${errorRows.length} test images. Showing ${Math.min(visible, rows.length)}.`;
    $('error-rows').innerHTML = rows.slice(0, visible).map(r => `
        <tr>
            <td><a href="/sample/${r.index}?fruit=${encodeURIComponent(currentFruit)}" target="_blank" rel="noopener"><img class="thumb" loading="lazy" src="/sample/${r.index}?fruit=${encodeURIComponent(currentFruit)}" alt="${escapeHTML(r.true_label)} test sample"></a></td>
            <td>${escapeHTML(r.true_label)}</td>
            <td>${escapeHTML(r.prediction)}</td>
            <td>${pct(r.confidence)}</td>
            <td>${escapeHTML(r.analysis)}</td>
        </tr>
    `).join('');
    $('more').hidden = visible >= rows.length;
}

async function errors() {
    const id = $('error-model').value;
    const r = data.experiments.find(x => x.config.id === id);
    if (!r) return;
    const base = data.artifact_base || '/artifacts';
    $('confusion').src = `${base}/${id}/confusion_matrix.png`;
    $('predictions-download').href = `${base}/${id}/test_predictions.csv`;
    $('class-metrics').innerHTML = `
        <table>
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>N</th></tr></thead>
            <tbody>${data.classes.map(c => {
                const m = r.test.per_class[c];
                return `<tr><td>${escapeHTML(c)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${m['f1-score'].toFixed(3)}</td><td>${m.support}</td></tr>`;
            }).join('')}</tbody>
        </table>
    `;
    try {
        const rows = await request(`/api/errors?fruit=${encodeURIComponent(currentFruit)}&model=${encodeURIComponent(id)}`);
        if ($('error-model').value !== id) return;
        errorRows = rows;
        visible = 30;
        renderErrors();
    } catch (e) {
        $('error-count').textContent = e.message;
    }
}

$('training-model').onchange = training;
$('error-model').onchange = errors;
$('errors-only').onchange = () => { visible = 30; renderErrors(); };
$('more').onclick = () => { visible += 30; renderErrors(); };

function renderHeroStages(classes) {
    if ($('stage-dots')) {
        $('stage-dots').innerHTML = classes.map(c => {
            const col = STAGE_COLORS[c] || '#888';
            return `<i style="--c:${col}" title="${escapeHTML(c)}"></i>`;
        }).join('');
    }
    if ($('stage-labels')) {
        $('stage-labels').innerHTML = classes.map(c => `<span>${escapeHTML(c)}</span>`).join('');
    }
}

async function loadFruit(fruit) {
    currentFruit = fruit;
    try {
        $('status').textContent = 'Loading fruit data…';
        data = await request(`/api/summary?fruit=${encodeURIComponent(currentFruit)}`);
        $('status').textContent = data.training ? 'Training in progress' : `${data.experiments.length} trained experiments (${currentFruit})`;
        
        renderHeroStages(data.classes);
        
        const base = data.artifact_base || '/artifacts';
        if ($('metrics-download')) {
            $('metrics-download').href = `${base}/comparison.csv`;
        }
        
        if (!data.experiments.length) {
            $('file-note').textContent = `Run python train.py --fruit ${currentFruit} to create trained models.`;
            return;
        }

        const options = data.experiments.map(r => `
            <option value="${r.config.id}">${escapeHTML(r.config.name)}${r.config.id === data.selected_model ? ' · selected' : ''}</option>
        `).join('');
        for (const id of ['model', 'training-model', 'error-model']) {
            $(id).innerHTML = options;
            $(id).value = data.selected_model || data.experiments[0].config.id;
        }

        const a = data.audit, b = data.experiments[0];
        const testCount = a.counts && a.counts.test ? Object.values(a.counts.test).reduce((x, y) => x + y, 0) : 0;
        $('stats').innerHTML = [
            ['Images in experiment', a.used ?? 'N/A', 'Frozen cleaned subset'],
            ['Test images', testCount, 'Never used for optimization'],
            ['Model selection', 'Macro F1', 'Validation set only'],
            ['Maturity classes', String(data.classes.length).padStart(2, '0'), data.classes.join(' → ')]
        ].map(([k, v, n]) => `
            <div class="stat"><span>${escapeHTML(k)}</span><strong>${escapeHTML(String(v))}</strong><span>${escapeHTML(n)}</span></div>
        `).join('');

        $('comparison').innerHTML = data.experiments.map(r => `
            <tr class="${r.config.id === data.selected_model ? 'selected' : ''}">
                <td><strong>${escapeHTML(r.config.name)}</strong>${r.config.id === data.selected_model ? ' <span class="badge">Selected</span>' : ''}</td>
                <td>${r.config.lr}</td>
                <td>${pct(r.test.accuracy)}</td>
                <td>${pct(r.test.precision)}</td>
                <td>${pct(r.test.recall)}</td>
                <td>${r.test.f1.toFixed(4)}</td>
                <td>${((r.test.f1 - b.test.f1) * 100).toFixed(2)} pp</td>
            </tr>
        `).join('');

        $('conditions').innerHTML = data.experiments.map(r => `
            <div class="card">
                <h3>${escapeHTML(r.config.name)}</h3>
                <span class="badge">${r.parameters.toLocaleString()} parameters</span>
                <p><strong>Changed:</strong> ${escapeHTML(r.config.changed)}</p>
                <p><strong>Why:</strong> ${escapeHTML(r.config.why)}</p>
                <p><strong>Measured effect:</strong> ${((r.test.f1 - b.test.f1) * 100).toFixed(2)} percentage points in test macro F1 versus baseline.</p>
            </div>
        `).join('');

        $('dataset').innerHTML = `
            <h3>Data preparation & imbalance (${escapeHTML(currentFruit.toUpperCase())})</h3>
            <p>${(a.downloaded_images || 0).toLocaleString()} source images → ${(a.retained_after_cleaning || 0).toLocaleString()} retained after cleaning → ${(a.used || 0).toLocaleString()} used in this run.</p>
            <table>
                <thead><tr><th>Class</th><th>Original train</th><th>Used train</th><th>Validation</th><th>Test</th></tr></thead>
                <tbody>${data.classes.map(c => `
                    <tr>
                        <td>${escapeHTML(c)}</td>
                        <td>${(a.original_counts && a.original_counts.train && a.original_counts.train[c]) ?? 'N/A'}</td>
                        <td>${(a.counts && a.counts.train && a.counts.train[c]) ?? 'N/A'}</td>
                        <td>${(a.counts && a.counts.val && a.counts.val[c]) ?? 'N/A'}</td>
                        <td>${(a.counts && a.counts.test && a.counts.test[c]) ?? 'N/A'}</td>
                    </tr>
                `).join('')}</tbody>
            </table>
            <p class="hint">${escapeHTML(a.preprocessing || '')} ${escapeHTML(a.missing_data || '')}</p>
            <p class="hint">${escapeHTML(a.limitation || '')}</p>
            <p class="hint">Split policy: ${escapeHTML(a.split_policy || '')}</p>
            <p class="hint"><a href="${a.source || '#'}" target="_blank" rel="noopener">Dataset & attribution</a> · <a href="${base}/REPORT.md" target="_blank" rel="noopener">Full experiment report (Markdown)</a></p>
        `;

        training();
        await errors();
    } catch (e) {
        $('status').textContent = 'Results unavailable';
        $('file-note').textContent = e.message;
    }
}

async function init() {
    try {
        const fruits = await request('/api/fruits');
        const fruitSelect = $('fruit');
        if (fruitSelect) {
            fruitSelect.innerHTML = fruits.map(f => `
                <option value="${f.id}" ${f.id === currentFruit ? 'selected' : ''}>${escapeHTML(f.name)} ${f.ready ? '✓' : '(untrained)'}</option>
            `).join('');
            fruitSelect.disabled = false;
            fruitSelect.onchange = e => {
                loadFruit(e.target.value);
            };
        }
        await loadFruit(currentFruit);
    } catch (e) {
        $('status').textContent = 'Error initializing fruit selector';
        await loadFruit('banana');
    }
}

init();
