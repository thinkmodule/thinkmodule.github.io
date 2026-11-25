/**
 * ThinkModule — Module 03: Application Controller, Live Log-Loss Chart & Collapsible HUD
 */

import { MAX_STEPS, OPTIMIZERS, LossEngine } from './math-engine.js';
import { LossStudioScene } from './scene-3d.js';

const engine = new LossEngine();
const studio = new LossStudioScene(document.getElementById('viewport'));
const optIds = Object.keys(OPTIMIZERS);

// Playback State (Continuous float step for silky 60 FPS motion)
let isPlaying = true;
let continuousStep = 0;
const speedPresets = [
  { label: '0.5×', stepsPerSec: 8 },
  { label: '1×', stepsPerSec: 16 },
  { label: '2×', stepsPerSec: 32 }
];
let speedIdx = 1;
let lastFrameTime = performance.now();
let lastHudStep = -1;

// DOM Elements
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');
const fmt = (n, d = 4) => (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(d);

// ============================================================================
// 1. HUD RENDERING (OPTIMIZER LIST, CHART & EXACT MATH)
// ============================================================================
function buildOptimizerList() {
  const list = document.getElementById('optList');
  list.innerHTML = '';

  for (const id of optIds) {
    const meta = OPTIMIZERS[id];
    const row = document.createElement('div');
    row.className = `opt-row ${engine.focusedId === id ? 'focused' : ''}`;
    row.dataset.id = id;
    row.innerHTML = `
      <div class="opt-left">
        <i class="opt-swatch" style="background:${meta.color}"></i>
        <span class="opt-name">${meta.name}</span>
      </div>
      <div class="opt-meta">
        <span class="opt-loss" id="loss-${id}">—</span>
        <span class="opt-state" id="state-${id}">Start</span>
      </div>
    `;
    row.addEventListener('click', () => {
      engine.focusedId = id;
      buildOptimizerList();
      updateHUD(Math.floor(continuousStep));
      studio.syncStep(engine, continuousStep, performance.now());
    });
    list.appendChild(row);
  }
}

function drawLossChart(stepIdx) {
  const W = lossCanvas.width;
  const H = lossCanvas.height;
  const pad = { l: 58, r: 16, t: 18, b: 34 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  lossCtx.fillStyle = '#03060c';
  lossCtx.fillRect(0, 0, W, H);

  let minLog = Infinity;
  let maxLog = -Infinity;
  for (const id of optIds) {
    for (const f of engine.history[id]) {
      const lg = Math.log10(Math.max(0.01, f.eval.loss));
      if (lg < minLog) minLog = lg;
      if (lg > maxLog) maxLog = lg;
    }
  }
  minLog = Math.floor(minLog * 2) / 2 - 0.1;
  maxLog = Math.ceil(maxLog * 2) / 2 + 0.1;

  lossCtx.strokeStyle = 'rgba(255,255,255,0.08)';
  lossCtx.lineWidth = 1;
  lossCtx.fillStyle = '#7e90ad';
  lossCtx.font = '16px ui-monospace, monospace';

  for (let i = 0; i <= 3; i++) {
    const f = i / 3;
    const y = pad.t + f * ph;
    const v = Math.pow(10, maxLog - f * (maxLog - minLog));
    lossCtx.beginPath();
    lossCtx.moveTo(pad.l, y);
    lossCtx.lineTo(W - pad.r, y);
    lossCtx.stroke();
    lossCtx.textAlign = 'right';
    lossCtx.fillText(v >= 1 ? v.toFixed(1) : v.toFixed(2), pad.l - 8, y + 5);
  }

  for (const id of optIds) {
    const hist = engine.history[id];
    const meta = OPTIMIZERS[id];
    const isFocused = id === engine.focusedId;

    lossCtx.strokeStyle = meta.color;
    lossCtx.globalAlpha = isFocused ? 1.0 : 0.55;
    lossCtx.lineWidth = isFocused ? 3.5 : 2.0;
    lossCtx.beginPath();

    for (let i = 0; i <= stepIdx; i++) {
      const x = pad.l + (i / MAX_STEPS) * pw;
      const lg = Math.log10(Math.max(0.01, hist[i].eval.loss));
      const y = pad.t + ((maxLog - lg) / (maxLog - minLog)) * ph;
      if (i === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    }
    lossCtx.stroke();
  }
  lossCtx.globalAlpha = 1.0;

  // Vertical current-step cursor line
  const cx = pad.l + (stepIdx / MAX_STEPS) * pw;
  lossCtx.strokeStyle = 'rgba(255,255,255,0.28)';
  lossCtx.setLineDash([4, 4]);
  lossCtx.beginPath();
  lossCtx.moveTo(cx, pad.t);
  lossCtx.lineTo(cx, H - pad.b);
  lossCtx.stroke();
  lossCtx.setLineDash([]);
}

function updateHUD(stepIdx) {
  const sc = engine.scenario;
  const fid = engine.focusedId;
  const meta = OPTIMIZERS[fid];
  const f = engine.history[fid][stepIdx];
  const ev = f.eval;

  document.getElementById('scenarioTitle').textContent = sc.title;
  document.getElementById('scenarioStory').textContent = sc.story;
  document.getElementById('epochLabel').textContent = `Step ${stepIdx} / ${MAX_STEPS}`;
  document.getElementById('timelineSlider').value = stepIdx;

  document.getElementById('inspectedHeader').textContent = `Inspecting: ${meta.name}`;
  document.getElementById('optTakeaway').textContent = meta.desc;
  document.getElementById('surfaceFormula').textContent = sc.formula;
  document.getElementById('mathRuleTag').textContent = meta.name;

  // Regime banner
  const banner = document.getElementById('regimeBanner');
  banner.className = `regime-banner ${ev.regimeClass}`;
  document.getElementById('regimeName').textContent = ev.regime.toUpperCase();
  document.getElementById('currentLossBadge').textContent = `L = ${ev.loss.toFixed(4)}`;

  // Per-optimizer rows
  for (const id of optIds) {
    const of = engine.history[id][stepIdx];
    const lEl = document.getElementById(`loss-${id}`);
    const sEl = document.getElementById(`state-${id}`);
    if (lEl) lEl.textContent = of.eval.loss.toFixed(4);
    if (sEl) {
      sEl.textContent = of.status;
      sEl.className = `opt-state ${of.converged ? 'done' : ''}`;
    }
  }

  const leader = engine.getLeader(stepIdx);
  document.getElementById('leaderBadge').textContent = `Best: ${OPTIMIZERS[leader.id].name} (${leader.loss.toFixed(3)})`;

  // Exact Math Tab
  document.getElementById('mathGrad').innerHTML =
    `θₜ = [ ${fmt(f.w1)} ]  w₁   L(θ) = <span class="pos">${ev.loss.toFixed(4)}</span>\n` +
    `     [ ${fmt(f.w2)} ]  w₂\n` +
    `∇L = [ ${fmt(ev.g1)} ]  ∂₁   ‖∇L‖ = ${ev.gradNorm.toFixed(4)}\n` +
    `     [ ${fmt(ev.g2)} ]  ∂₂`;

  const detCls = ev.det > 0 ? 'pos' : 'neg';
  document.getElementById('mathHess').innerHTML =
    `H  = [ ${fmt(ev.h11, 3)}   ${fmt(ev.h12, 3)} ]\n` +
    `     [ ${fmt(ev.h12, 3)}   ${fmt(ev.h22, 3)} ]\n` +
    `tr(H) = ${fmt(ev.trace, 3)}   det(H) = <span class="${detCls}">${fmt(ev.det, 3)}</span>`;

  const l1Cls = ev.l1 >= 0 ? 'pos' : 'neg';
  const l2Cls = ev.l2 >= 0 ? 'pos' : 'neg';
  document.getElementById('mathEigen').innerHTML =
    `λ₁ = <span class="${l1Cls}">${fmt(ev.l1, 3)}</span>  e₁ = [${fmt(ev.e1[0], 2)}, ${fmt(ev.e1[1], 2)}]\n` +
    `λ₂ = <span class="${l2Cls}">${fmt(ev.l2, 3)}</span>  e₂ = [${fmt(ev.e2[0], 2)}, ${fmt(ev.e2[1], 2)}]\n` +
    `κ  = |λ_max/λ_min| = <span class="warn">${ev.kappa.toFixed(2)}</span>`;

  let upd = `ηₜ = ${f.eta.toFixed(4)}   (${meta.rule})\nΔθ = [ ${fmt(f.dw1)}, ${fmt(f.dw2)} ]`;
  if (fid === 'adamw') {
    upd = `ηₜ = ${f.eta.toFixed(4)}   m̂ₜ = [${fmt(f.mHat1, 2)}, ${fmt(f.mHat2, 2)}]\n` +
          `v̂ₜ = [${f.vHat1.toFixed(3)}, ${f.vHat2.toFixed(3)}]\n` +
          `Δθ = [ ${fmt(f.dw1)}, ${fmt(f.dw2)} ]`;
  } else if (fid === 'momentum') {
    upd = `ηₜ = ${f.eta.toFixed(4)}   vₜ = [${fmt(f.mHat1, 2)}, ${fmt(f.mHat2, 2)}]\n` +
          `Δθ = [ ${fmt(f.dw1)}, ${fmt(f.dw2)} ]`;
  }
  document.getElementById('mathUpdate').innerHTML = upd;

  drawLossChart(stepIdx);
}

// ============================================================================
// 2. UI INTERACTIONS
// ============================================================================
document.querySelectorAll('.scen-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.scen-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    engine.setScenario(btn.dataset.scen);
    studio.rebuildScenario(engine);
    continuousStep = 0;
    lastHudStep = -1;
    isPlaying = true;
    document.getElementById('playBtn').textContent = 'Pause';
    buildOptimizerList();
    updateHUD(0);
    studio.syncStep(engine, 0, performance.now());
  });
});

document.querySelectorAll('[data-cam]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-cam]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    studio.flyToPreset(btn.dataset.cam);
  });
});

document.getElementById('orbitBtn').addEventListener('click', (e) => {
  const active = e.target.classList.toggle('active');
  studio.setAutoOrbit(active);
});

document.getElementById('playBtn').addEventListener('click', (e) => {
  if (!isPlaying && continuousStep >= MAX_STEPS) {
    continuousStep = 0;
  }
  isPlaying = !isPlaying;
  e.target.textContent = isPlaying ? 'Pause' : 'Play';
});

document.getElementById('restartBtn').addEventListener('click', () => {
  continuousStep = 0;
  lastHudStep = -1;
  isPlaying = true;
  document.getElementById('playBtn').textContent = 'Pause';
  updateHUD(0);
  studio.syncStep(engine, 0, performance.now());
});

document.getElementById('speedBtn').addEventListener('click', (e) => {
  speedIdx = (speedIdx + 1) % speedPresets.length;
  e.target.textContent = speedPresets[speedIdx].label;
});

document.getElementById('timelineSlider').addEventListener('input', (e) => {
  isPlaying = false;
  document.getElementById('playBtn').textContent = 'Play';
  continuousStep = parseFloat(e.target.value);
  const idx = Math.floor(continuousStep);
  lastHudStep = idx;
  updateHUD(idx);
  studio.syncStep(engine, continuousStep, performance.now());
});

document.getElementById('lrScaleSlider').addEventListener('input', (e) => {
  const scale = parseFloat(e.target.value);
  document.getElementById('lrScaleVal').textContent = `${scale.toFixed(2)}×`;
  engine.setLrScale(scale);
  studio.rebuildTrails(engine);
  const idx = Math.floor(continuousStep);
  updateHUD(idx);
  studio.syncStep(engine, continuousStep, performance.now());
});

document.getElementById('vectorsToggle').addEventListener('click', (e) => {
  const on = e.target.classList.toggle('active');
  studio.setVectorsVisible(on);
});

document.getElementById('markersToggle').addEventListener('click', (e) => {
  const on = e.target.classList.toggle('active');
  studio.setMarkersVisible(on);
});

// Collapsible Side Panels
const wireCollapse = (panelId, btnId) => {
  const panel = document.getElementById(panelId);
  const btn = document.getElementById(btnId);
  btn.addEventListener('click', () => {
    const collapsed = panel.classList.toggle('collapsed');
    btn.setAttribute('aria-expanded', String(!collapsed));
  });
};
wireCollapse('controlsPanel', 'toggleControlsBtn');
wireCollapse('inspectorPanel', 'toggleInspectorBtn');

// Right Tabs
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// ============================================================================
// 3. BOOTSTRAP & 60-FPS LOOP
// ============================================================================
studio.rebuildScenario(engine);
buildOptimizerList();
updateHUD(0);
studio.syncStep(engine, 0, performance.now());

requestAnimationFrame(() => {
  document.getElementById('loadingOverlay').classList.add('hidden');
});

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  if (isPlaying) {
    continuousStep += dt * speedPresets[speedIdx].stepsPerSec;
    if (continuousStep >= MAX_STEPS) {
      continuousStep = MAX_STEPS;
      isPlaying = false;
      document.getElementById('playBtn').textContent = 'Play';
    }
  }

  studio.syncStep(engine, continuousStep, now);
  const stepIdx = Math.floor(continuousStep);
  if (stepIdx !== lastHudStep) {
    lastHudStep = stepIdx;
    updateHUD(stepIdx);
  }

  studio.render();
}
requestAnimationFrame(loop);