/* VaniStream charts: small dependency-free SVG charts for the analytics dashboard.
   Series colours follow the entity in a fixed categorical order (validated for colour-vision
   deficiency in light and dark mode); text always uses the text tokens. Every chart has a
   hover tooltip, a legend for two or more series, a table view and an empty state. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const esc = t => String(t == null ? '' : t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const fmtNum = v => v == null ? '–' : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (Number.isInteger(v) ? String(v) : v.toFixed(1)));

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  function shortLabel(l) {
    if (/^\d{4}-\d{2}-\d{2}$/.test(l)) { const d = new Date(l + 'T00:00:00'); return d.toLocaleDateString([], { day: 'numeric', month: 'short' }); }
    if (/^\d{4}-\d{2}$/.test(l)) { const d = new Date(l + '-01T00:00:00'); return d.toLocaleDateString([], { month: 'short', year: '2-digit' }); }
    return l;
  }

  /* Shared tooltip */
  let tip;
  function tooltip() {
    if (!tip) { tip = document.createElement('div'); tip.className = 'viz-tip'; tip.setAttribute('role', 'status'); document.body.appendChild(tip); }
    return tip;
  }
  function showTip(evt, html) {
    const t = tooltip(); t.innerHTML = html; t.style.display = 'block';
    const r = t.getBoundingClientRect(), pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = Math.max(8, x) + 'px'; t.style.top = Math.max(8, y) + 'px';
  }
  function hideTip() { if (tip) tip.style.display = 'none'; }
  addEventListener('scroll', hideTip, { passive: true });

  function legend(host, series) {
    if (series.length < 2) return;
    const lg = document.createElement('div'); lg.className = 'viz-legend';
    lg.innerHTML = series.map(s => `<span><i style="background:var(${s.color})"></i>${esc(s.name)}</span>`).join('');
    host.appendChild(lg);
  }
  function table(host, head, rows) {
    const wrap = document.createElement('div'); wrap.className = 'viz-table'; wrap.hidden = true;
    wrap.innerHTML = '<table><thead><tr>' + head.map(h => `<th>${esc(h)}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map(r => '<tr>' + r.map((c, i) => `<td${i ? ' class="tnum"' : ''}>${esc(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    host.appendChild(wrap);
    return wrap;
  }
  function frame(host, opts) {
    host.innerHTML = '';
    host.classList.add('viz');
    const body = document.createElement('div'); body.className = 'viz-body'; host.appendChild(body);
    return body;
  }
  function empty(host, msg) { host.innerHTML = `<div class="viz-empty">${esc(msg || 'No data for these filters.')}</div>`; }

  /* Vertical bars over time (stacked when several series). data: {labels, series:[{name, values, color}]} */
  function columns(host, data, opts = {}) {
    const series = data.series.filter(s => s.values.some(v => v));
    const total = data.labels.map((_, i) => series.reduce((a, s) => a + (s.values[i] || 0), 0));
    if (!series.length) return empty(host, opts.empty);
    const body = frame(host);
    const W = Math.max(320, host.clientWidth || 640), H = opts.height || 220, m = { t: 10, r: 8, b: 26, l: 40 };
    const n = data.labels.length, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const max = niceMax(Math.max(...total));
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img', 'aria-label': opts.title || 'chart' }, body);
    for (let k = 0; k <= 4; k++) {
      const y = m.t + ih - ih * k / 4;
      el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: 'viz-grid' }, svg);
      el('text', { x: m.l - 6, y: y + 4, class: 'viz-axis', 'text-anchor': 'end' }, svg).textContent = fmtNum(max * k / 4);
    }
    const step = iw / n, bw = Math.max(1, Math.min(28, step - (step > 6 ? 2 : 0)));
    const every = Math.ceil(n / Math.max(1, Math.floor(iw / 64)));
    data.labels.forEach((l, i) => {
      const x = m.l + step * i + (step - bw) / 2;
      let y0 = m.t + ih;
      series.forEach((s, si) => {
        const v = s.values[i] || 0;
        if (!v) return;
        const h = ih * v / max, gap = si > 0 && h > 3 ? 2 : 0;
        const top = si === series.filter(x => x.values[i]).length - 1;
        el('rect', { x, y: y0 - h, width: bw, height: Math.max(0, h - gap), rx: top && bw > 6 ? 3 : 0, fill: `var(${s.color})` }, svg);
        y0 -= h;
      });
      if (i % every === 0) el('text', { x: m.l + step * i + step / 2, y: H - 8, class: 'viz-axis', 'text-anchor': 'middle' }, svg).textContent = shortLabel(l);
      const hit = el('rect', { x: m.l + step * i, y: m.t, width: step, height: ih, fill: 'transparent' }, svg);
      hit.addEventListener('mousemove', e => showTip(e, `<b>${esc(opts.labelFmt ? opts.labelFmt(l) : l)}</b>` +
        series.map(s => `<div><i style="background:var(${s.color})"></i>${esc(s.name)} <span class="tnum">${fmtNum(s.values[i] || 0)}</span></div>`).join('') +
        (series.length > 1 ? `<div class="sum">Total <span class="tnum">${fmtNum(total[i])}</span></div>` : '')));
      hit.addEventListener('mouseleave', hideTip);
    });
    legend(host, series);
    host._table = table(host, ['Period', ...series.map(s => s.name)], data.labels.map((l, i) => [l, ...series.map(s => s.values[i] || 0)]));
  }

  /* Lines over time (latency). Null values break the line. */
  function lines(host, data, opts = {}) {
    const series = data.series.filter(s => s.values.some(v => v != null));
    if (!series.length) return empty(host, opts.empty);
    const body = frame(host);
    const W = Math.max(320, host.clientWidth || 640), H = opts.height || 220, m = { t: 10, r: 10, b: 26, l: 48 };
    const n = data.labels.length, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const max = niceMax(Math.max(...series.flatMap(s => s.values.filter(v => v != null))));
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img', 'aria-label': opts.title || 'chart' }, body);
    for (let k = 0; k <= 4; k++) {
      const y = m.t + ih - ih * k / 4;
      el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: 'viz-grid' }, svg);
      el('text', { x: m.l - 6, y: y + 4, class: 'viz-axis', 'text-anchor': 'end' }, svg).textContent = fmtNum(max * k / 4) + (opts.unit || '');
    }
    const X = i => m.l + (n === 1 ? iw / 2 : iw * i / (n - 1)), Y = v => m.t + ih - ih * v / max;
    const every = Math.ceil(n / Math.max(1, Math.floor(iw / 64)));
    data.labels.forEach((l, i) => { if (i % every === 0) el('text', { x: X(i), y: H - 8, class: 'viz-axis', 'text-anchor': 'middle' }, svg).textContent = shortLabel(l); });
    series.forEach(s => {
      let d = '', pen = false;
      s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1); pen = true; });
      el('path', { d, fill: 'none', stroke: `var(${s.color})`, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      s.values.forEach((v, i) => { if (v != null) el('circle', { cx: X(i), cy: Y(v), r: 4, fill: `var(${s.color})`, stroke: 'var(--surface)', 'stroke-width': 2 }, svg); });
    });
    const cross = el('line', { y1: m.t, y2: m.t + ih, class: 'viz-cross', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', e => {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / (n === 1 ? 1 : iw / (n - 1)))));
      cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('visibility', 'visible');
      showTip(e, `<b>${esc(data.labels[i])}</b>` + series.map(s => `<div><i style="background:var(${s.color})"></i>${esc(s.name)} <span class="tnum">${s.values[i] == null ? 'no queries' : fmtNum(s.values[i]) + (opts.unit || '')}</span></div>`).join(''));
    });
    hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    legend(host, series);
    host._table = table(host, ['Period', ...series.map(s => s.name + (opts.unit ? ' (' + opts.unit.trim() + ')' : ''))],
      data.labels.map((l, i) => [l, ...series.map(s => s.values[i] == null ? '–' : fmtNum(s.values[i]))]));
  }

  /* Horizontal bars: items [{name, value}], one colour (magnitude), direct value labels. */
  function hbars(host, items, opts = {}) {
    items = items.filter(x => x.value);
    if (!items.length) return empty(host, opts.empty);
    const body = frame(host);
    const max = Math.max(...items.map(x => x.value));
    const tot = items.reduce((a, x) => a + x.value, 0);
    body.innerHTML = '<div class="viz-hbars">' + items.map((x, i) => {
      const color = opts.colors ? opts.colors[i] : (opts.color || '--series-1');
      return `<div class="viz-hrow" data-i="${i}"><span class="lbl">${esc(x.name)}</span><span class="trk"><span class="fil" style="width:${(x.value / max * 100).toFixed(1)}%;background:var(${color})"></span></span><span class="val tnum">${fmtNum(x.value)}${opts.share ? ' <small>' + (x.value / tot * 100).toFixed(0) + '%</small>' : ''}</span></div>`;
    }).join('') + '</div>';
    body.querySelectorAll('.viz-hrow').forEach(row => {
      const x = items[+row.dataset.i];
      row.addEventListener('mousemove', e => showTip(e, `<b>${esc(x.name)}</b><div>${esc(opts.unit || 'Count')} <span class="tnum">${fmtNum(x.value)}</span> · ${(x.value / tot * 100).toFixed(1)}% of shown</div>` + (x.note ? `<div class="sum">${esc(x.note)}</div>` : '')));
      row.addEventListener('mouseleave', hideTip);
    });
    host._table = table(host, [opts.labelHead || 'Item', opts.unit || 'Count', 'Share'], items.map(x => [x.name, x.value, (x.value / tot * 100).toFixed(1) + '%']));
  }

  /* Funnel: ordered stages [{stage, value, na}] drawn as bars against the first stage. */
  function funnel(host, stages, opts = {}) {
    const real = stages.filter(s => !s.na);
    if (!real.some(s => s.value)) return empty(host, opts.empty);
    const body = frame(host);
    // Stages counted in different units (documents vs passages) are scaled within their own unit.
    const maxBy = {};
    real.forEach(s => { const u = s.unit || ''; maxBy[u] = Math.max(maxBy[u] || 1, s.value); });
    body.innerHTML = '<div class="viz-hbars">' + stages.map((s, i) => s.na
      ? `<div class="viz-hrow na"><span class="lbl">${esc(s.stage)}</span><span class="trk na">Does not apply</span><span class="val">–</span></div>`
      : `<div class="viz-hrow" data-i="${i}"><span class="lbl">${esc(s.stage)}${s.unit ? ' <small style="color:var(--muted)">' + esc(s.unit) + '</small>' : ''}</span><span class="trk"><span class="fil" style="width:${(s.value / maxBy[s.unit || ''] * 100).toFixed(1)}%;background:var(${s.unit ? (opts.color2 || '--series-3') : (opts.color || '--series-1')})"></span></span><span class="val tnum">${fmtNum(s.value)}</span></div>`).join('') + '</div>';
    body.querySelectorAll('.viz-hrow[data-i]').forEach(row => {
      const s = stages[+row.dataset.i];
      row.addEventListener('mousemove', e => showTip(e, `<b>${esc(s.stage)}</b><div><span class="tnum">${fmtNum(s.value)}</span>${s.hint ? ' ' + esc(s.hint) : ''}</div>`));
      row.addEventListener('mouseleave', hideTip);
    });
    host._table = table(host, ['Stage', 'Count'], stages.map(s => [s.stage + (s.unit ? ' (' + s.unit + ')' : ''), s.na ? 'does not apply' : s.value]));
  }

  function toggleTable(host, btn) {
    if (!host._table) return;
    const show = host._table.hidden;
    host._table.hidden = !show;
    host.querySelectorAll('.viz-body').forEach(b => b.hidden = show);
    host.querySelectorAll('.viz-legend').forEach(b => b.hidden = show);
    if (btn) { btn.textContent = show ? 'Chart' : 'Table'; btn.setAttribute('aria-pressed', show); }
  }

  window.VizCharts = { columns, lines, hbars, funnel, empty, toggleTable, fmtNum };
})();
