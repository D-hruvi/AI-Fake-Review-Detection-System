// ---------- tab switching ----------
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});
document.querySelectorAll('[data-goto]').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.goto));
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
  if (name === 'dashboard') loadDashboard();
}

// ---------- samples ----------
// Real examples pulled from the training corpus (not hand-written) so the
// demo actually reflects what the model was trained to catch: AI/template-
// generated review text, not just "salesy-sounding" language.
const FAKE_SAMPLE = "This is a great bag. I love the look and feel of it, and the size is perfect. I had to get a size up from what I normally wear but it fits great now.";
const REAL_SAMPLE = "Works wonderfully. We have had for several months now and it is still going strong. I love the length of it, easy to carry around and it holds a good amount of charge, though it does take a while to fully charge from empty.";

document.getElementById('sampleFake').addEventListener('click', e => {
  e.preventDefault();
  document.getElementById('reviewInput').value = FAKE_SAMPLE;
});
document.getElementById('sampleReal').addEventListener('click', e => {
  e.preventDefault();
  document.getElementById('reviewInput').value = REAL_SAMPLE;
});

// ---------- analyze ----------
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const text = document.getElementById('reviewInput').value;
  const zone = document.getElementById('verdictZone');
  if (!text.trim()) { zone.innerHTML = errorBox('Paste a review first.'); return; }
  zone.innerHTML = '<div class="hint">Analyzing…</div>';
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (!res.ok) { zone.innerHTML = errorBox(data.error); return; }
    zone.innerHTML = verdictCard(data);
  } catch (e) {
    zone.innerHTML = errorBox('Request failed: ' + e.message);
  }
});

function verdictCard(d) {
  const isFake = d.label === 'fake';
  const chips = d.signals.map(s =>
    `<span class="chip chip--${s.pushes}">${s.word}</span>`).join('');
  return `
    <div class="verdict-card">
      <div class="stamp stamp--${isFake ? 'fake' : 'genuine'}">${isFake ? 'FLAGGED · FAKE' : 'CLEAR · GENUINE'}</div>
      <div class="verdict-meta">
        <div><span class="k">CONFIDENCE</span><span class="v">${d.confidence}%</span></div>
        <div><span class="k">FAKE PROBABILITY</span><span class="v">${d.fake_probability}%</span></div>
        <div><span class="k">WORD COUNT</span><span class="v">${d.word_count}</span></div>
      </div>
      <div class="signals-label">EVIDENCE — WORDS THAT SWUNG THE VERDICT</div>
      <div class="signal-chips">${chips || '<span class="hint">no strong lexical signal — verdict driven by overall pattern</span>'}</div>
    </div>`;
}

function errorBox(msg) {
  return `<div class="error-box">${msg}</div>`;
}

// ---------- scan URL ----------
document.getElementById('scanBtn').addEventListener('click', async () => {
  const url = document.getElementById('urlInput').value;
  const zone = document.getElementById('scanResults');
  if (!url.trim()) { zone.innerHTML = errorBox('Paste a product URL first.'); return; }
  zone.innerHTML = '<div class="hint">Fetching and scanning reviews…</div>';
  try {
    const res = await fetch('/api/scrape', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url})
    });
    const data = await res.json();
    if (!res.ok) { zone.innerHTML = errorBox(data.error); return; }
    zone.innerHTML = scanResultsHTML(data);
  } catch (e) {
    zone.innerHTML = errorBox('Request failed: ' + e.message);
  }
});

function scanResultsHTML(d) {
  const items = d.results.map(r => `
    <div class="scan-item">
      <div class="scan-item__text">${escapeHtml(r.text)}</div>
      <div class="scan-item__tag tag--${r.label}">${r.label.toUpperCase()} · ${r.confidence}%</div>
    </div>`).join('');
  return `
    <div class="scan-summary">
      <div><span class="v">${d.total}</span><span class="k"> reviews scanned</span></div>
      <div><span class="v" style="color:var(--rust)">${d.fake_pct}%</span><span class="k"> flagged fake</span></div>
    </div>
    ${items}`;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

// ---------- dashboard ----------
let pieChart, trendChart, sourceChart;

async function loadDashboard() {
  const res = await fetch('/api/stats');
  const d = await res.json();

  document.getElementById('statTotal').textContent = d.total_analyzed;
  document.getElementById('statFakePct').textContent = d.fake_pct + '%';
  document.getElementById('statConf').textContent = d.avg_confidence + '%';

  const paper = '#EDEAE1', rust = '#B8451F', teal = '#2E6E58', mute = '#8A8F9C';

  if (pieChart) pieChart.destroy();
  pieChart = new Chart(document.getElementById('pieChart'), {
    type: 'doughnut',
    data: {
      labels: ['Genuine', 'Fake'],
      datasets: [{data: [d.genuine_count, d.fake_count], backgroundColor: [teal, rust], borderWidth: 0}]
    },
    options: {plugins: {legend: {labels: {color: paper, font: {family: 'Space Mono'}}}}}
  });

  if (trendChart) trendChart.destroy();
  trendChart = new Chart(document.getElementById('trendChart'), {
    type: 'bar',
    data: {
      labels: d.by_day.map(x => x.day.slice(5)),
      datasets: [
        {label: 'Fake', data: d.by_day.map(x => x.fake), backgroundColor: rust, stack: 's'},
        {label: 'Genuine', data: d.by_day.map(x => x.total - x.fake), backgroundColor: teal, stack: 's'}
      ]
    },
    options: {
      scales: {
        x: {stacked: true, ticks: {color: mute}, grid: {display: false}},
        y: {stacked: true, ticks: {color: mute}, grid: {color: '#2A2F3D'}}
      },
      plugins: {legend: {labels: {color: paper}}}
    }
  });

  if (sourceChart) sourceChart.destroy();
  sourceChart = new Chart(document.getElementById('sourceChart'), {
    type: 'bar',
    data: {
      labels: d.by_source.map(x => x.source),
      datasets: [{label: 'Total scanned', data: d.by_source.map(x => x.total), backgroundColor: '#5B6478'}]
    },
    options: {
      indexAxis: 'y',
      scales: {
        x: {ticks: {color: mute}, grid: {color: '#2A2F3D'}},
        y: {ticks: {color: paper}, grid: {display: false}}
      },
      plugins: {legend: {display: false}}
    }
  });
}
