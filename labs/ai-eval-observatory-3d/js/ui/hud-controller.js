import { CLUSTERS, SCENARIOS, DEFAULT_STATE } from '../core/config.js';

const ICON_PAUSE = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
const ICON_PLAY  = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg>';

export class HUDController {
  constructor(store, callbacks) {
    this.store = store;
    this.callbacks = callbacks;

    this.el = {
      scenarioPill: document.getElementById('active-scenario-pill'),
      fpsBadge: document.getElementById('fps-badge'),
      btnZen: document.getElementById('btn-zen-mode'),
      btnZenExit: document.getElementById('zen-exit-btn'),
      dockLeft: document.getElementById('dock-left'),
      dockRight: document.getElementById('dock-right'),
      btnToggleLeft: document.getElementById('btn-toggle-left'),
      btnToggleRight: document.getElementById('btn-toggle-right'),

      sliderTopK: document.getElementById('slider-topk'),
      valTopK: document.getElementById('val-topk'),
      sliderTemp: document.getElementById('slider-temp'),
      valTemp: document.getElementById('val-temp'),
      sliderAlignment: document.getElementById('slider-alignment'),
      valAlignment: document.getElementById('val-alignment'),
      sliderBloom: document.getElementById('slider-bloom'),
      valBloom: document.getElementById('val-bloom'),

      toggleHNSW: document.getElementById('toggle-hnsw'),
      toggleHull: document.getElementById('toggle-hull'),
      toggleRewardField: document.getElementById('toggle-reward-field'),
      toggleSafetyPlane: document.getElementById('toggle-safety-plane'),
      toggleHalos: document.getElementById('toggle-halos'),
      toggleAutoRotate: document.getElementById('toggle-autorotate'),

      btnRerun: document.getElementById('btn-rerun'),
      btnResetDefaults: document.getElementById('btn-reset-defaults'),
      btnPlayPause: document.getElementById('btn-play-pause'),
      btnReplay: document.getElementById('btn-replay'),

      clusterLegend: document.getElementById('cluster-legend'),
      verdictBadge: document.getElementById('eval-verdict-badge'),
      overallScore: document.getElementById('eval-overall-score'),
      queryPreview: document.getElementById('scenario-query-preview'),

      faithVal: document.getElementById('metric-faithfulness-val'),
      faithBar: document.getElementById('metric-faithfulness-bar'),
      relVal: document.getElementById('metric-relevance-val'),
      relBar: document.getElementById('metric-relevance-bar'),
      ndcgVal: document.getElementById('metric-ndcg-val'),
      ndcgBar: document.getElementById('metric-ndcg-bar'),
      safetyVal: document.getElementById('metric-safety-val'),
      safetyBar: document.getElementById('metric-safety-bar'),
      klVal: document.getElementById('metric-kldiv-val'),
      klBar: document.getElementById('metric-kldiv-bar'),

      driftVal: document.getElementById('metric-drift-val'),
      hopsVal: document.getElementById('metric-hops-val'),
      precisionVal: document.getElementById('metric-precision-val'),

      traceList: document.getElementById('judge-trace-list'),
      tooltip: document.getElementById('node-tooltip'),
      scenarioBtns: Array.from(document.querySelectorAll('.scenario-btn')),
      camPresetBtns: Array.from(document.querySelectorAll('.cam-preset-btn'))
    };

    this._renderClusterLegend();
    this._bindEvents();
    this._bindKeyboardShortcuts();
    this.syncControlsFromState(this.store.get());
  }

  _renderClusterLegend() {
    this.el.clusterLegend.innerHTML = CLUSTERS.map(c => `
      <span class="legend-chip">
        <span class="legend-dot" style="background:${c.hex}; box-shadow:0 0 6px ${c.hex};"></span>
        <span>${c.shortName}</span>
      </span>
    `).join('');
  }

  _bindEvents() {
    // Collapsible Side Docks (zero obstruction)
    this.el.btnToggleLeft.addEventListener('click', () => {
      this.el.dockLeft.classList.toggle('collapsed');
    });
    this.el.btnToggleRight.addEventListener('click', () => {
      this.el.dockRight.classList.toggle('collapsed');
    });

    // Zen Mode
    const toggleZen = () => {
      const isZen = document.body.classList.toggle('zen-mode');
      this.el.btnZenExit.classList.toggle('hidden', !isZen);
    };
    this.el.btnZen.addEventListener('click', toggleZen);
    this.el.btnZenExit.addEventListener('click', toggleZen);
    this._toggleZen = toggleZen;

    // Sliders
    this.el.sliderTopK.addEventListener('input', (e) => {
      const v = parseInt(e.target.value, 10);
      this.el.valTopK.textContent = String(v);
      this.store.set({ topK: v });
    });

    this.el.sliderTemp.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      this.el.valTemp.textContent = v.toFixed(2);
      this.store.set({ temperature: v });
    });

    this.el.sliderAlignment.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      this.el.valAlignment.textContent = v.toFixed(2);
      this.store.set({ alignment: v });
    });

    this.el.sliderBloom.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      this.el.valBloom.textContent = v.toFixed(2);
      this.store.set({ bloomStrength: v });
    });

    // Visual Layer Toggles
    this.el.toggleHNSW.addEventListener('change', (e) => this.store.set({ showHNSW: e.target.checked }));
    this.el.toggleHull.addEventListener('change', (e) => this.store.set({ showHull: e.target.checked }));
    this.el.toggleRewardField.addEventListener('change', (e) => this.store.set({ showRewardField: e.target.checked }));
    this.el.toggleSafetyPlane.addEventListener('change', (e) => this.store.set({ showSafetyPlane: e.target.checked }));
    this.el.toggleHalos.addEventListener('change', (e) => this.store.set({ showHalos: e.target.checked }));
    this.el.toggleAutoRotate.addEventListener('change', (e) => this.store.set({ autoRotate: e.target.checked }));

    // Scenario Switcher
    this.el.scenarioBtns.forEach(btn => {
      btn.addEventListener('click', () => this.selectScenario(btn.dataset.scenario));
    });

    // Camera Presets
    this.el.camPresetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.dataset.preset;
        this._setActivePresetButton(preset);
        this.store.set({ cameraPreset: preset });
        this.callbacks.onCameraPreset(preset);
      });
    });

    // Action Buttons
    this.el.btnRerun.addEventListener('click', () => {
      this.store.set({ seedOffset: this.store.get('seedOffset') + 1 });
    });

    this.el.btnResetDefaults.addEventListener('click', () => {
      this.store.set({ ...DEFAULT_STATE });
      this.syncControlsFromState(this.store.get());
      this.callbacks.onCameraPreset(DEFAULT_STATE.cameraPreset);
    });

    this.el.btnPlayPause.addEventListener('click', () => {
      const next = !this.store.get('isPlaying');
      this.store.set({ isPlaying: next });
      this.el.btnPlayPause.innerHTML = next ? ICON_PAUSE : ICON_PLAY;
    });

    this.el.btnReplay.addEventListener('click', () => {
      this.store.set({ isPlaying: true });
      this.el.btnPlayPause.innerHTML = ICON_PAUSE;
      this.callbacks.onReplay();
    });
  }

  _bindKeyboardShortcuts() {
    const scenarioOrder = ['rag_grounded', 'hallucination', 'rlhf_steering', 'adversarial_guard'];
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      const key = e.key.toLowerCase();
      if (key === 'h') {
        this._toggleZen();
      } else if (key === '[') {
        this.el.dockLeft.classList.toggle('collapsed');
      } else if (key === ']') {
        this.el.dockRight.classList.toggle('collapsed');
      } else if (key === ' ') {
        e.preventDefault();
        this.el.btnPlayPause.click();
      } else if (key === 'r') {
        this.el.btnReplay.click();
      } else if (['1', '2', '3', '4'].includes(key)) {
        this.selectScenario(scenarioOrder[parseInt(key, 10) - 1]);
      }
    });
  }

  selectScenario(scenarioId) {
    const sc = SCENARIOS[scenarioId];
    if (!sc) return;

    this.store.set({
      scenarioId,
      topK: sc.defaultTopK,
      temperature: sc.defaultTemp,
      alignment: sc.defaultAlignment,
      cameraPreset: sc.cameraPreset,
      isPlaying: true
    });
    this.syncControlsFromState(this.store.get());
    this.callbacks.onCameraPreset(sc.cameraPreset);
  }

  _setActivePresetButton(preset) {
    this.el.camPresetBtns.forEach(b => b.classList.toggle('active', b.dataset.preset === preset));
  }

  syncControlsFromState(state) {
    this.el.sliderTopK.value = state.topK;
    this.el.valTopK.textContent = String(state.topK);

    this.el.sliderTemp.value = state.temperature;
    this.el.valTemp.textContent = state.temperature.toFixed(2);

    this.el.sliderAlignment.value = state.alignment;
    this.el.valAlignment.textContent = state.alignment.toFixed(2);

    this.el.sliderBloom.value = state.bloomStrength;
    this.el.valBloom.textContent = state.bloomStrength.toFixed(2);

    this.el.toggleHNSW.checked = state.showHNSW;
    this.el.toggleHull.checked = state.showHull;
    this.el.toggleRewardField.checked = state.showRewardField;
    this.el.toggleSafetyPlane.checked = state.showSafetyPlane;
    this.el.toggleHalos.checked = state.showHalos;
    this.el.toggleAutoRotate.checked = state.autoRotate;

    this.el.btnPlayPause.innerHTML = state.isPlaying ? ICON_PAUSE : ICON_PLAY;
    this.el.scenarioBtns.forEach(b => b.classList.toggle('active', b.dataset.scenario === state.scenarioId));
    this._setActivePresetButton(state.cameraPreset);
  }

  updateEvaluation(evalResult) {
    const { scenario, metrics, verdict, traceSteps } = evalResult;

    this.el.scenarioPill.textContent = scenario.title;
    this.el.queryPreview.textContent = scenario.queryText;

    this.el.verdictBadge.textContent = verdict.label;
    this.el.verdictBadge.className = `verdict-badge ${verdict.cssClass}`;
    this.el.overallScore.textContent = metrics.compositeScore.toFixed(1);

    this._setMetric(this.el.faithVal, this.el.faithBar, metrics.faithfulness);
    this._setMetric(this.el.relVal, this.el.relBar, metrics.relevance);
    this._setMetric(this.el.ndcgVal, this.el.ndcgBar, metrics.ndcgAtK);
    this._setMetric(this.el.safetyVal, this.el.safetyBar, metrics.safetyMargin);
    this._setMetric(this.el.klVal, this.el.klBar, metrics.klDivergence);

    this.el.driftVal.textContent = `${metrics.driftDist.toFixed(1)}u`;
    this.el.hopsVal.textContent = String(metrics.hops);
    this.el.precisionVal.textContent = `${Math.round(metrics.precisionAtK * 100)}%`;

    this.el.traceList.innerHTML = traceSteps.map(t => `
      <div class="trace-item ${t.status}">
        <div class="trace-head">
          <span>${t.step}</span>
          <span>${t.metric}</span>
        </div>
        <div class="trace-msg">${t.message}</div>
      </div>
    `).join('');
  }

  _setMetric(valEl, barEl, raw) {
    const clamped = Math.max(0, Math.min(1, raw));
    valEl.textContent = `${(clamped * 100).toFixed(1)}%`;
    barEl.style.width = `${(clamped * 100).toFixed(1)}%`;
  }

  updateFPS(fps) {
    this.el.fpsBadge.textContent = `${fps} FPS · WebGL2`;
  }

  showNodeTooltip(node, clientX, clientY, cosSim, isTopK) {
    if (!node) {
      this.el.tooltip.classList.add('hidden');
      return;
    }

    const tt = this.el.tooltip;
    tt.innerHTML = `
      <div class="tt-header">
        <span>VEC #${String(node.id).padStart(4, '0')} · L${node.layer} HNSW</span>
        <span>${isTopK ? '★ TOP-K HIT' : node.clusterName}</span>
      </div>
      <div class="tt-title">${node.label}</div>
      <div class="tt-grid">
        <span>Domain</span><strong style="color:${node.hex}">${node.clusterName}</strong>
        <span>Cosine Sim</span><strong>${cosSim.toFixed(3)}</strong>
        <span>Chunk Size</span><strong>${node.tokenCount} tokens</strong>
        <span>Latent XYZ</span><strong>(${node.pos[0].toFixed(1)}, ${node.pos[1].toFixed(1)}, ${node.pos[2].toFixed(1)})</strong>
      </div>
    `;

    const pad = 16;
    const maxX = window.innerWidth - 275;
    const maxY = window.innerHeight - 155;
    tt.style.left = `${Math.min(maxX, clientX + pad)}px`;
    tt.style.top = `${Math.min(maxY, clientY + pad)}px`;
    tt.classList.remove('hidden');
  }
}
