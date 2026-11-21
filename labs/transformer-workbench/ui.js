/**
 * ThinkModule Lab — Interactive UI, Canvas Telemetry, Attention Arcs & Heatmap Controller
 * Connects TransformerEngine (engine.js) to all 4 interactive workbench stages.
 */
(function () {
  "use strict";

  const Engine = window.TransformerEngine;
  if (!Engine) {
    console.error("TransformerEngine not loaded.");
    return;
  }

  // ============================================================================
  // APPLICATION STATE
  // ============================================================================

  const PRESETS = {
    strict: {
      strategy: "greedy",
      temperature: 0.2,
      topP: 0.85,
      topK: 4,
      minP: 0.1,
      repetitionPenalty: 1.08
    },
    balanced: {
      strategy: "sampling",
      temperature: 0.75,
      topP: 0.9,
      topK: 6,
      minP: 0.05,
      repetitionPenalty: 1.05
    },
    creative: {
      strategy: "sampling",
      temperature: 1.65,
      topP: 0.98,
      topK: 6,
      minP: 0.0,
      repetitionPenalty: 1.02
    },
    loop: {
      strategy: "sampling",
      temperature: 0.85,
      topP: 0.95,
      topK: 6,
      minP: 0.0,
      repetitionPenalty: 0.72
    }
  };

  const BPE_EXAMPLES = {
    strawberry: "How many 'r's are in strawberry? Wait: straw + berry!",
    python: "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    attention: "Transformer attention models cache tokens in O(n) space.",
    numbers: "Calculate 1000000 + 2^10 == 1001024 -> exact complexity."
  };

  const state = {
    scenarioId: Engine.SCENARIOS[0].id,
    params: { ...Engine.DEFAULT_PARAMS },
    activePreset: "balanced",
    trace: null,
    selectedStep: 0,
    activeHead: 0,
    causalMask: true,
    matrixSource: "trace", // "trace" | "bpe"
    streamAttention: null,
    matrixAttention: null,
    matrixLayout: null,
    bpeResult: null,
    bpeSelectedIdx: 0,
    playTimer: null
  };

  // ============================================================================
  // DOM REFERENCES
  // ============================================================================

  const $ = id => document.getElementById(id);

  const dom = {
    // Controls
    scenarioSelect: $("scenarioSelect"),
    scenarioCategory: $("scenarioCategory"),
    scenarioBadge: $("scenarioBadge"),
    presetGrid: $("presetGrid"),
    strategySelect: $("strategySelect"),
    tempSlider: $("tempSlider"),
    tempVal: $("tempVal"),
    topPSlider: $("topPSlider"),
    topPVal: $("topPVal"),
    topKSlider: $("topKSlider"),
    topKVal: $("topKVal"),
    minPSlider: $("minPSlider"),
    minPVal: $("minPVal"),
    repPenSlider: $("repPenSlider"),
    repPenVal: $("repPenVal"),
    generateBtn: $("generateBtn"),
    replayBtn: $("replayBtn"),
    resetBtn: $("resetBtn"),

    // Stage 3: Token Stream & Scrubber
    statusPills: $("statusPills"),
    promptDisplay: $("promptDisplay"),
    streamHeadTabs: $("streamHeadTabs"),
    arcsViewport: $("arcsViewport"),
    arcsSvg: $("arcsSvg"),
    tokenStream: $("tokenStream"),
    stepPrevBtn: $("stepPrevBtn"),
    stepPlayBtn: $("stepPlayBtn"),
    stepNextBtn: $("stepNextBtn"),
    stepEndBtn: $("stepEndBtn"),
    timelineSlider: $("timelineSlider"),
    timelineMeta: $("timelineMeta"),

    // Inspector
    candHeaderMeta: $("candHeaderMeta"),
    candList: $("candList"),
    kpiEntropy: $("kpiEntropy"),
    kpiProb: $("kpiProb"),
    kpiLogProb: $("kpiLogProb"),
    kpiPpl: $("kpiPpl"),
    diagnosisText: $("diagnosisText"),
    attendedHeadName: $("attendedHeadName"),
    attendedList: $("attendedList"),

    // Stage 4: Telemetry
    telemetryCanvas: $("telemetryCanvas"),
    telemetryMeta: $("telemetryMeta"),

    // Stage 2: Attention Matrix
    matrixHeadTabs: $("matrixHeadTabs"),
    causalMaskToggle: $("causalMaskToggle"),
    matrixSourceSelect: $("matrixSourceSelect"),
    attentionCanvas: $("attentionCanvas"),
    matrixTooltip: $("matrixTooltip"),
    headDescTitle: $("headDescTitle"),
    headDescText: $("headDescText"),
    headAvgEntropy: $("headAvgEntropy"),
    headDimVal: $("headDimVal"),

    // Stage 1: BPE Tokenizer
    bpeInput: $("bpeInput"),
    bpePresets: $("bpePresets"),
    bpeChars: $("bpeChars"),
    bpeBytes: $("bpeBytes"),
    bpeTokensCount: $("bpeTokensCount"),
    bpeRatio: $("bpeRatio"),
    bpeTokensBox: $("bpeTokensBox"),
    bpeInspector: $("bpeInspector"),
    bpeMergeSlider: $("bpeMergeSlider"),
    bpeMergeLabel: $("bpeMergeLabel"),
    bpeMergePreview: $("bpeMergePreview")
  };

  // ============================================================================
  // UTILITIES
  // ============================================================================

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatTokLabel(tok) {
    if (tok === " ") return "␣";
    return tok.replace(/\n/g, "↵");
  }

  function setupHiDpiCanvas(canvas, cssHeight) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 300);
    const height = cssHeight || Math.max(Math.floor(rect.height), 180);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function hexToRgba(hex, alpha) {
    const clean = hex.replace("#", "");
    const num = parseInt(clean, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  // ============================================================================
  // CONTROL SYNC & GENERATION LIFECYCLE
  // ============================================================================

  function stopPlayback() {
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    dom.stepPlayBtn.classList.remove("playing");
    dom.stepPlayBtn.textContent = "▶";
  }

  function startPlayback(fromStart = false) {
    if (!state.trace || !state.trace.steps.length) return;
    stopPlayback();

    if (fromStart || state.selectedStep >= state.trace.steps.length - 1) {
      state.selectedStep = 0;
      renderStepSelection();
    }

    dom.stepPlayBtn.classList.add("playing");
    dom.stepPlayBtn.textContent = "⏸";

    state.playTimer = setInterval(() => {
      if (!state.trace || state.selectedStep >= state.trace.steps.length - 1) {
        stopPlayback();
        return;
      }
      state.selectedStep += 1;
      renderStepSelection();
    }, 150);
  }

  function syncControlsFromState() {
    const p = state.params;
    dom.strategySelect.value = p.strategy;
    dom.tempSlider.value = p.temperature;
    dom.topPSlider.value = p.topP;
    dom.topKSlider.value = p.topK;
    dom.minPSlider.value = p.minP;
    dom.repPenSlider.value = p.repetitionPenalty;

    dom.tempVal.textContent = p.temperature.toFixed(2);
    dom.tempVal.className = "field-val" + (p.temperature >= 1.4 ? " warn" : "");

    dom.topPVal.textContent = p.topP.toFixed(2);
    dom.topKVal.textContent = String(p.topK);
    dom.minPVal.textContent = p.minP.toFixed(2);

    dom.repPenVal.textContent = p.repetitionPenalty.toFixed(2);
    dom.repPenVal.className = "field-val" + (p.repetitionPenalty < 0.95 ? " danger" : "");

    const isGreedy = p.strategy === "greedy";
    const isMinPOnly = p.strategy === "min_p";
    dom.tempSlider.disabled = isGreedy;
    dom.topPSlider.disabled = isGreedy || isMinPOnly;
    dom.topKSlider.disabled = isGreedy || isMinPOnly;
    dom.minPSlider.disabled = isGreedy;

    // Preset highlight
    dom.presetGrid.querySelectorAll(".preset-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.preset === state.activePreset);
    });

    // Scenario metadata
    const sc = Engine.SCENARIOS.find(s => s.id === state.scenarioId);
    if (sc) {
      dom.scenarioCategory.textContent = sc.category;
      dom.scenarioBadge.textContent = sc.badge;
    }
  }

  function runNewGeneration(autoPlay = false) {
    stopPlayback();
    state.trace = Engine.generateTrace(state.scenarioId, state.params);
    state.selectedStep = autoPlay ? 0 : Math.max(0, state.trace.steps.length - 1);
    recomputeAttentionAndRender();
    if (autoPlay) startPlayback(true);
  }

  function triggerCounterfactualBranch(overrideToken) {
    if (!state.trace) return;
    stopPlayback();
    const forkStep = state.selectedStep;
    state.trace = Engine.branchFromStep(state.trace, forkStep, overrideToken, state.params);
    state.selectedStep = Math.min(forkStep, state.trace.steps.length - 1);
    recomputeAttentionAndRender();
  }

  // ============================================================================
  // STAGE 3: TOKEN STREAM, ATTENTION ARCS & INSPECTOR
  // ============================================================================

  function renderHeadTabs() {
    const html = Engine.ATTENTION_HEADS.map(h => {
      const active = h.id === state.activeHead ? "active" : "";
      const style = active ? `style="background:${h.color};border-color:${h.color};color:#061220;"` : "";
      return `<button class="head-tab ${active}" data-head="${h.id}" ${style}>H${h.id}: ${escHtml(h.shortName)}</button>`;
    }).join("");

    dom.streamHeadTabs.innerHTML = html;
    dom.matrixHeadTabs.innerHTML = html;
  }

  function recomputeAttentionAndRender() {
    if (!state.trace) return;

    const genTokens = state.trace.steps.map(s => s.token);
    state.streamAttention = Engine.computeAttention(genTokens, state.activeHead, state.causalMask);

    renderStatusPills();
    renderTokenStreamDom();
    renderStepSelection();
    renderAttentionMatrixSection();
  }

  function renderStatusPills() {
    const s = state.trace.summary;
    const pills = [
      `<span class="chip-pill">${s.totalTokens} tokens</span>`,
      `<span class="chip-pill">Avg H: ${s.avgEntropy.toFixed(2)} bits</span>`,
      `<span class="chip-pill">PPL: ${s.finalPerplexity.toFixed(2)}</span>`
    ];

    for (const tag of s.branchTags) {
      if (tag === "Repetition Loop") continue;
      pills.push(`<span class="chip-pill branch">${escHtml(tag)}</span>`);
    }

    if (s.loopDetected) {
      pills.push(`<span class="chip-pill loop">⚠ Degeneration Loop</span>`);
    }
    if (s.hallucinationCount > 0) {
      pills.push(`<span class="chip-pill halluc">⚠ ${s.hallucinationCount} Hallucination Token${s.hallucinationCount > 1 ? "s" : ""}</span>`);
    } else if (!s.loopDetected) {
      pills.push(`<span class="chip-pill ok">✓ Coherent Trajectory</span>`);
    }

    dom.statusPills.innerHTML = pills.join("");
    dom.promptDisplay.innerHTML = `<strong>Prompt</strong>${escHtml(state.trace.prompt)}`;
  }

  function renderTokenStreamDom() {
    const steps = state.trace.steps;
    dom.tokenStream.innerHTML = steps
      .map((st, idx) => {
        const cls = ["tok"];
        if (st.isCounterfactual) cls.push("tok-counterfactual");
        else if (st.isLoopStep) cls.push("tok-loop");
        else if (st.isHallucination) cls.push("tok-halluc");
        else if (st.isSurprise) cls.push("tok-surprise");

        return `<button type="button" class="${cls.join(" ")}" data-step="${idx}" title="Step #${idx + 1} | ID ${st.tokenId} | Prob ${(st.metrics.chosenProb * 100).toFixed(1)}%">${escHtml(formatTokLabel(st.token))}</button>`;
      })
      .join("");

    dom.timelineSlider.max = String(Math.max(0, steps.length - 1));
  }

  function renderStepSelection() {
    const steps = state.trace ? state.trace.steps : [];
    if (!steps.length) return;

    const idx = Math.max(0, Math.min(state.selectedStep, steps.length - 1));
    state.selectedStep = idx;
    const step = steps[idx];

    dom.timelineSlider.value = String(idx);
    dom.timelineMeta.innerHTML = `Step <strong>#${idx + 1}</strong> / ${steps.length} &bull; <code>${escHtml(step.displayToken)}</code> [${step.tokenId}]`;

    // Update token classes (future dimming, selected highlight, attended tokens)
    const attnRow = state.streamAttention && state.streamAttention.matrix[idx] ? state.streamAttention.matrix[idx] : [];
    const topAttended = attnRow
      .map((w, j) => ({ j, w }))
      .filter(x => x.j !== idx && x.w >= 0.08)
      .sort((a, b) => b.w - a.w)
      .slice(0, 6);
    const attendedSet = new Set(topAttended.map(x => x.j));

    const tokEls = dom.tokenStream.querySelectorAll(".tok");
    tokEls.forEach((el, i) => {
      el.classList.toggle("tok-selected", i === idx);
      el.classList.toggle("tok-future", i > idx);
      el.classList.toggle("tok-attended", attendedSet.has(i) && i <= idx);
    });

    // Render Inspector, Telemetry & SVG Arcs
    renderCandidateInspector(step);
    renderDiagnostics(step, attnRow);
    renderTelemetryChart();
    requestAnimationFrame(() => drawAttentionArcs(idx, attnRow));
  }

  function drawAttentionArcs(selectedIdx, attnRow) {
    const svg = dom.arcsSvg;
    const viewport = dom.arcsViewport;
    if (!svg || !viewport || !attnRow || !attnRow.length) {
      if (svg) svg.innerHTML = "";
      return;
    }

    const vpRect = viewport.getBoundingClientRect();
    const tokEls = dom.tokenStream.querySelectorAll(".tok");
    const targetEl = tokEls[selectedIdx];
    if (!targetEl) {
      svg.innerHTML = "";
      return;
    }

    const head = Engine.ATTENTION_HEADS[state.activeHead];
    const targetRect = targetEl.getBoundingClientRect();
    const x2 = targetRect.left - vpRect.left + targetRect.width / 2;
    const y2 = targetRect.top - vpRect.top;

    // Pick top 7 strongest attention weights (excluding self)
    const links = attnRow
      .map((weight, j) => ({ j, weight }))
      .filter(item => item.j !== selectedIdx && item.weight >= 0.05 && (!state.causalMask || item.j < selectedIdx))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 7);

    const maxW = links.length ? links[0].weight : 1;

    const paths = links.map(link => {
      const srcEl = tokEls[link.j];
      if (!srcEl) return "";
      const srcRect = srcEl.getBoundingClientRect();
      const x1 = srcRect.left - vpRect.left + srcRect.width / 2;
      const y1 = srcRect.top - vpRect.top;

      const dx = Math.abs(x2 - x1);
      const lift = Math.min(54, Math.max(20, dx * 0.18));
      const cx = (x1 + x2) / 2;
      const cy = Math.min(y1, y2) - lift;

      const norm = link.weight / Math.max(maxW, 0.01);
      const strokeWidth = (1.1 + norm * 2.8).toFixed(2);
      const opacity = (0.25 + norm * 0.72).toFixed(2);

      return `
        <path d="M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}"
              fill="none"
              stroke="${head.color}"
              stroke-width="${strokeWidth}"
              stroke-opacity="${opacity}"
              stroke-linecap="round" />
        <circle cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="${(2 + norm * 2).toFixed(1)}"
                fill="${head.color}" fill-opacity="${opacity}" />
      `;
    });

    svg.setAttribute("viewBox", `0 0 ${Math.max(vpRect.width, 1)} ${Math.max(vpRect.height, 1)}`);
    svg.innerHTML = paths.join("");
  }

  function renderCandidateInspector(step) {
    dom.candHeaderMeta.textContent = `Step #${step.stepIndex + 1} • ${step.metrics.activePoolSize}/${step.metrics.totalCandidates} active`;

    dom.candList.innerHTML = step.candidates
      .map((c, rank) => {
        const isChosen = c.token === step.token && rank === step.chosenRank;
        const rowCls = ["cand-row"];
        if (isChosen) rowCls.push("is-chosen");
        if (!c.included) rowCls.push("is-cut");
        if (c.isHallucination) rowCls.push("is-halluc");

        const pct = (c.finalProb * 100).toFixed(1);
        const rawPct = (c.rawProb * 100).toFixed(1);
        const barW = c.included ? Math.max(c.finalProb * 100, 1.5) : Math.max(c.rawProb * 100, 1.5);

        const miniTags = [];
        if (c.isHallucination) miniTags.push(`<span class="cand-tag-mini halluc">Halluc</span>`);
        if (c.isInjectedLoop) miniTags.push(`<span class="cand-tag-mini loop">Loop</span>`);
        else if (c.branchTag) miniTags.push(`<span class="cand-tag-mini branch">Fork</span>`);

        const actionHtml = isChosen
          ? `<span class="branch-btn current-badge">Selected</span>`
          : `<button type="button" class="branch-btn" data-branch-token="${escHtml(c.token)}" title="Force this token and regenerate downstream trajectory">Branch ↱</button>`;

        return `
          <div class="${rowCls.join(" ")}">
            <div class="cand-tok">
              <span class="cand-tok-str">"${escHtml(formatTokLabel(c.token))}"</span>
              <div class="cand-sub">
                <span>z:${c.rawLogit.toFixed(1)}</span>
                ${c.repeatCount > 0 ? `<span>rep:${c.repeatCount}×</span>` : ""}
                ${miniTags.join("")}
              </div>
            </div>
            <div class="cand-bar-wrap">
              <div class="cand-bar-track">
                <div class="cand-bar-fill" style="width:${barW.toFixed(1)}%"></div>
              </div>
            </div>
            <div class="cand-prob">
              ${c.included ? `${pct}%` : `<span class="cut-label">${escHtml(c.cutReason)} (${rawPct}%)</span>`}
            </div>
            <div>${actionHtml}</div>
          </div>
        `;
      })
      .join("");
  }

  function renderDiagnostics(step, attnRow) {
    const m = step.metrics;
    dom.kpiEntropy.textContent = `${m.entropy.toFixed(2)} bits`;
    dom.kpiEntropy.className = "kpi-value " + (m.entropy > 1.8 ? "amber" : m.entropy < 0.5 ? "green" : "accent");

    dom.kpiProb.textContent = `${(m.chosenProb * 100).toFixed(1)}%`;
    dom.kpiProb.className = "kpi-value " + (step.isHallucination ? "red" : m.chosenProb >= 0.5 ? "green" : "amber");

    dom.kpiLogProb.textContent = m.logProb.toFixed(2);
    dom.kpiPpl.textContent = m.cumPerplexity.toFixed(2);
    dom.diagnosisText.textContent = step.diagnosis;

    const head = Engine.ATTENTION_HEADS[state.activeHead];
    dom.attendedHeadName.textContent = `Top Context Attended (${head.shortName})`;

    if (!attnRow || !attnRow.length) {
      dom.attendedList.innerHTML = `<div style="font-size:0.78rem;color:var(--muted);">No attention weights yet.</div>`;
      return;
    }

    const top = attnRow
      .map((w, j) => ({ j, w, tok: state.trace.steps[j].token }))
      .filter(x => x.w > 0.01)
      .sort((a, b) => b.w - a.w)
      .slice(0, 4);

    dom.attendedList.innerHTML = top
      .map(item => {
        const pct = (item.w * 100).toFixed(1);
        const selfMark = item.j === step.stepIndex ? " (self)" : "";
        return `
          <div class="attended-item">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">#${item.j + 1} "${escHtml(formatTokLabel(item.tok))}"${selfMark}</span>
            <div class="attended-bar">
              <div class="attended-bar-fill" style="width:${Math.min(100, item.w * 100).toFixed(1)}%;background:${head.color};"></div>
            </div>
            <span style="text-align:right;color:var(--text);">${pct}%</span>
          </div>
        `;
      })
      .join("");
  }

  // ============================================================================
  // STAGE 4: LIVE TELEMETRY CANVAS (ENTROPY & PROBABILITY OVER TIME)
  // ============================================================================

  function renderTelemetryChart() {
    const canvas = dom.telemetryCanvas;
    if (!canvas || !state.trace || !state.trace.steps.length) return;

    const { ctx, width, height } = setupHiDpiCanvas(canvas, 190);
    const steps = state.trace.steps;
    const n = steps.length;

    const pad = { top: 20, right: 20, bottom: 28, left: 42 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    ctx.clearRect(0, 0, width, height);

    // Subtle grid lines
    ctx.strokeStyle = "rgba(34, 48, 79, 0.7)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#7f92b0";

    const maxEntropy = Math.max(2.6, ...steps.map(s => s.metrics.entropy));

    for (let g = 0; g <= 4; g++) {
      const y = pad.top + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();

      const labelVal = (maxEntropy * (1 - g / 4)).toFixed(1);
      ctx.fillText(`${labelVal}b`, 6, y + 3);
    }

    const xFor = i => (n === 1 ? pad.left + plotW / 2 : pad.left + (i / (n - 1)) * plotW);
    const yEnt = e => pad.top + plotH * (1 - e / maxEntropy);
    const yProb = p => pad.top + plotH * (1 - p);

    // Entropy area fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    grad.addColorStop(0, "rgba(56, 189, 248, 0.30)");
    grad.addColorStop(1, "rgba(56, 189, 248, 0.01)");

    ctx.beginPath();
    ctx.moveTo(xFor(0), height - pad.bottom);
    steps.forEach((s, i) => ctx.lineTo(xFor(i), yEnt(s.metrics.entropy)));
    ctx.lineTo(xFor(n - 1), height - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Chosen probability dashed line (emerald)
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = xFor(i);
      const y = yProb(s.metrics.chosenProb);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();

    // Entropy main line (cyan)
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = xFor(i);
      const y = yEnt(s.metrics.entropy);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Markers for Hallucinations, Loops & Counterfactual Overrides
    steps.forEach((s, i) => {
      const x = xFor(i);
      const y = yEnt(s.metrics.entropy);

      if (s.isCounterfactual) {
        ctx.fillStyle = "#a78bfa";
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      } else if (s.isHallucination) {
        ctx.fillStyle = "#f87171";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      } else if (s.isLoopStep) {
        ctx.fillStyle = "#fbbf24";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Selected step cursor
    const selX = xFor(state.selectedStep);
    const selStep = steps[state.selectedStep];
    ctx.strokeStyle = "rgba(226, 232, 240, 0.7)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(selX, pad.top);
    ctx.lineTo(selX, height - pad.bottom);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(selX, yEnt(selStep.metrics.entropy), 5, 0, Math.PI * 2);
    ctx.fill();

    // X-axis tick labels
    ctx.fillStyle = "#7f92b0";
    const tickEvery = Math.max(1, Math.ceil(n / 10));
    for (let i = 0; i < n; i += tickEvery) {
      ctx.fillText(`#${i + 1}`, xFor(i) - 8, height - 8);
    }

    dom.telemetryMeta.textContent = `Click chart to scrub • Peak Entropy: ${Math.max(...steps.map(s => s.metrics.entropy)).toFixed(2)} bits`;
  }

  function handleTelemetryScrub(evt) {
    if (!state.trace || !state.trace.steps.length) return;
    const rect = dom.telemetryCanvas.getBoundingClientRect();
    const padLeft = 42;
    const padRight = 20;
    const plotW = Math.max(1, rect.width - padLeft - padRight);
    const relX = Math.max(0, Math.min(1, (evt.clientX - rect.left - padLeft) / plotW));
    const idx = Math.round(relX * (state.trace.steps.length - 1));
    stopPlayback();
    state.selectedStep = idx;
    renderStepSelection();
  }

  // ============================================================================
  // STAGE 2: MULTI-HEAD SELF-ATTENTION HEATMAP MATRIX
  // ============================================================================

  function renderAttentionMatrixSection() {
    let tokens = [];
    if (state.matrixSource === "bpe" && state.bpeResult && state.bpeResult.tokens.length) {
      tokens = state.bpeResult.tokens.slice(0, 22).map(t => t.text);
    } else if (state.trace) {
      tokens = state.trace.steps.slice(0, 22).map(s => s.token);
    }

    state.matrixAttention = Engine.computeAttention(tokens, state.activeHead, state.causalMask);
    const attn = state.matrixAttention;

    dom.headDescTitle.textContent = attn.head.name;
    dom.headDescText.textContent = attn.head.description;
    dom.headAvgEntropy.textContent = `${attn.avgEntropy.toFixed(2)} bits`;
    dom.headDimVal.textContent = `d_k = ${attn.qkDim}`;

    drawAttentionHeatmap(attn, null);
  }

  function drawAttentionHeatmap(attn, hoverCell) {
    const canvas = dom.attentionCanvas;
    if (!canvas) return;

    const { ctx, width, height } = setupHiDpiCanvas(canvas, 410);
    ctx.clearRect(0, 0, width, height);

    const n = attn.tokens.length;
    if (!n) return;

    const padLeft = 86;
    const padTop = 76;
    const padRight = 16;
    const padBottom = 16;

    const gridSize = Math.min(width - padLeft - padRight, height - padTop - padBottom);
    const cell = gridSize / n;
    const originX = padLeft;
    const originY = padTop;

    state.matrixLayout = { originX, originY, cell, n };

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = originX + j * cell;
        const y = originY + i * cell;
        const isMasked = state.causalMask && j > i;

        if (isMasked) {
          ctx.fillStyle = "rgba(15, 23, 42, 0.7)";
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
          // Subtle diagonal hatch for masked future tokens
          ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(x + 2, y + cell - 2);
          ctx.lineTo(x + cell - 2, y + 2);
          ctx.stroke();
        } else {
          const w = attn.matrix[i][j];
          const intensity = Math.min(1, Math.pow(w, 0.65));
          ctx.fillStyle = hexToRgba(attn.head.color, 0.06 + intensity * 0.92);
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
      }
    }

    // Hover highlight
    if (hoverCell && hoverCell.i >= 0 && hoverCell.i < n && hoverCell.j >= 0 && hoverCell.j < n) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.strokeRect(
        originX + hoverCell.j * cell + 0.5,
        originY + hoverCell.i * cell + 0.5,
        cell - 1,
        cell - 1
      );
    }

    // Row (Query) and Column (Key) labels
    const fontSize = Math.max(9, Math.min(11, Math.floor(cell * 0.62)));
    ctx.font = `${fontSize}px ui-monospace, Menlo, monospace`;

    for (let i = 0; i < n; i++) {
      const label = formatTokLabel(attn.tokens[i]).trim().slice(0, 9) || "␣";
      const isHoveredRow = hoverCell && hoverCell.i === i;
      const isHoveredCol = hoverCell && hoverCell.j === i;

      // Left Query label
      ctx.fillStyle = isHoveredRow ? "#ffffff" : "#94a3b8";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(label, originX - 8, originY + i * cell + cell / 2);

      // Top Key label (rotated -45 deg)
      ctx.save();
      ctx.translate(originX + i * cell + cell / 2, originY - 8);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = isHoveredCol ? "#ffffff" : "#94a3b8";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }

    // Axis titles
    ctx.fillStyle = "#7f92b0";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillText("KEYS (Attended To) →", originX, 14);
    ctx.fillText("QUERIES ↓", 8, originY - 12);
  }

  function handleMatrixHover(evt) {
    if (!state.matrixLayout || !state.matrixAttention) return;
    const rect = dom.attentionCanvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    const { originX, originY, cell, n } = state.matrixLayout;

    const j = Math.floor((x - originX) / cell);
    const i = Math.floor((y - originY) / cell);

    if (i >= 0 && i < n && j >= 0 && j < n) {
      drawAttentionHeatmap(state.matrixAttention, { i, j });
      const qTok = formatTokLabel(state.matrixAttention.tokens[i]);
      const kTok = formatTokLabel(state.matrixAttention.tokens[j]);
      const isMasked = state.causalMask && j > i;

      if (isMasked) {
        dom.matrixTooltip.innerHTML = `<span>Query <strong>"${escHtml(qTok)}"</strong> &rarr; Key <strong>"${escHtml(kTok)}"</strong></span><span style="color:var(--red);">Masked (-&infin; future token)</span>`;
      } else {
        const w = (state.matrixAttention.matrix[i][j] * 100).toFixed(1);
        const h = state.matrixAttention.rowEntropies[i].toFixed(2);
        dom.matrixTooltip.innerHTML = `<span>Query #${i + 1} <strong>"${escHtml(qTok)}"</strong> &rarr; Key #${j + 1} <strong>"${escHtml(kTok)}"</strong></span><span>Weight: <strong style="color:${state.matrixAttention.head.color};">${w}%</strong> &bull; Row Entropy: ${h} bits</span>`;
      }
    }
  }

  // ============================================================================
  // STAGE 1: INTERACTIVE FREE-TEXT BPE TOKENIZER LAB
  // ============================================================================

  function renderBPELab() {
    const res = Engine.tokenizeBPE(dom.bpeInput.value);
    state.bpeResult = res;

    dom.bpeChars.textContent = String(res.stats.charCount);
    dom.bpeBytes.textContent = String(res.stats.byteCount);
    dom.bpeTokensCount.textContent = String(res.stats.tokenCount);
    dom.bpeRatio.textContent = `${res.stats.compressionRatio.toFixed(2)}×`;

    if (!res.tokens.length) {
      dom.bpeTokensBox.innerHTML = `<span style="color:var(--muted);font-size:0.84rem;">Type any text on the left to inspect real Byte-Pair Encoding...</span>`;
      dom.bpeInspector.textContent = "Hover or click a token chip above to inspect its Token ID and UTF-8 bytes.";
      dom.bpeMergeSlider.max = "0";
      dom.bpeMergeSlider.value = "0";
      dom.bpeMergeLabel.textContent = "Step 0 / 0";
      dom.bpeMergePreview.innerHTML = "";
      return;
    }

    dom.bpeTokensBox.innerHTML = res.tokens
      .map((t, idx) => {
        const colorCls = `bpe-c${idx % 5}`;
        return `
          <div class="bpe-chip ${colorCls}" data-bpe-idx="${idx}">
            <span class="bpe-chip-txt">${escHtml(t.display)}</span>
            <span class="bpe-chip-id">#${t.id}</span>
          </div>
        `;
      })
      .join("");

    state.bpeSelectedIdx = Math.min(state.bpeSelectedIdx, res.tokens.length - 1);
    renderBPETokenDetail(state.bpeSelectedIdx);

    const maxStep = res.mergeSteps.length;
    dom.bpeMergeSlider.max = String(maxStep);
    dom.bpeMergeSlider.value = String(maxStep);
    renderBPEMergeStep(maxStep);

    if (state.matrixSource === "bpe") {
      renderAttentionMatrixSection();
    }
  }

  function renderBPETokenDetail(idx) {
    if (!state.bpeResult || !state.bpeResult.tokens[idx]) return;
    const t = state.bpeResult.tokens[idx];
    const hexBytes = t.bytes.map(b => "0x" + b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
    dom.bpeInspector.innerHTML = `Token <strong>#${idx + 1}</strong>: <code>"${escHtml(t.display)}"</code> &bull; Vocab ID: <strong>${t.id}</strong> &bull; UTF-8 Bytes (${t.bytes.length}): <code>[${hexBytes}]</code>`;
  }

  function renderBPEMergeStep(stepNum) {
    const res = state.bpeResult;
    if (!res) return;

    const total = res.mergeSteps.length;
    if (stepNum <= 0 || total === 0) {
      const rawChars = Array.from(res.text.replace(/ /g, "Ġ").replace(/\n/g, "Ċ")).slice(0, 32);
      dom.bpeMergeLabel.textContent = `Step 0 / ${total} (Raw Characters)`;
      dom.bpeMergePreview.innerHTML = rawChars
        .map(ch => `<span class="merge-sym">${escHtml(ch)}</span>`)
        .join("");
      return;
    }

    const stepObj = res.mergeSteps[Math.min(stepNum - 1, total - 1)];
    dom.bpeMergeLabel.textContent = `Step ${stepObj.step} / ${total}: merged ("${stepObj.pair[0]}" + "${stepObj.pair[1]}") → "${stepObj.merged}" (${stepObj.tokenCountAfter} tokens)`;
    dom.bpeMergePreview.innerHTML = stepObj.snapshot
      .map(sym => {
        const cls = sym === stepObj.merged ? "merge-sym just-merged" : "merge-sym";
        return `<span class="${cls}">${escHtml(sym)}</span>`;
      })
      .join("");
  }

  // ============================================================================
  // EVENT BINDINGS & INITIALIZATION
  // ============================================================================

  function bindEvents() {
    // Scenario selector
    dom.scenarioSelect.addEventListener("change", () => {
      state.scenarioId = dom.scenarioSelect.value;
      syncControlsFromState();
      runNewGeneration(false);
    });

    // Presets
    dom.presetGrid.addEventListener("click", evt => {
      const btn = evt.target.closest(".preset-btn");
      if (!btn || !PRESETS[btn.dataset.preset]) return;
      state.activePreset = btn.dataset.preset;
      state.params = { ...state.params, ...PRESETS[btn.dataset.preset] };
      syncControlsFromState();
      runNewGeneration(false);
    });

    // Strategy & Sliders
    dom.strategySelect.addEventListener("change", () => {
      state.activePreset = null;
      state.params.strategy = dom.strategySelect.value;
      syncControlsFromState();
      runNewGeneration(false);
    });

    const sliderBindings = [
      [dom.tempSlider, "temperature", parseFloat],
      [dom.topPSlider, "topP", parseFloat],
      [dom.topKSlider, "topK", v => parseInt(v, 10)],
      [dom.minPSlider, "minP", parseFloat],
      [dom.repPenSlider, "repetitionPenalty", parseFloat]
    ];

    sliderBindings.forEach(([el, key, parser]) => {
      el.addEventListener("input", () => {
        state.activePreset = null;
        state.params[key] = parser(el.value);
        syncControlsFromState();
      });
      el.addEventListener("change", () => runNewGeneration(false));
    });

    // Action Buttons
    dom.generateBtn.addEventListener("click", () => runNewGeneration(false));
    dom.replayBtn.addEventListener("click", () => startPlayback(true));
    dom.resetBtn.addEventListener("click", () => {
      state.activePreset = "balanced";
      state.params = { ...Engine.DEFAULT_PARAMS };
      syncControlsFromState();
      runNewGeneration(false);
    });

    // Token Stream click
    dom.tokenStream.addEventListener("click", evt => {
      const btn = evt.target.closest(".tok");
      if (!btn) return;
      stopPlayback();
      state.selectedStep = parseInt(btn.dataset.step, 10);
      renderStepSelection();
    });

    // Scrubber controls
    dom.timelineSlider.addEventListener("input", () => {
      stopPlayback();
      state.selectedStep = parseInt(dom.timelineSlider.value, 10);
      renderStepSelection();
    });

    dom.stepPrevBtn.addEventListener("click", () => {
      stopPlayback();
      state.selectedStep = Math.max(0, state.selectedStep - 1);
      renderStepSelection();
    });

    dom.stepNextBtn.addEventListener("click", () => {
      stopPlayback();
      if (!state.trace) return;
      state.selectedStep = Math.min(state.trace.steps.length - 1, state.selectedStep + 1);
      renderStepSelection();
    });

    dom.stepEndBtn.addEventListener("click", () => {
      stopPlayback();
      if (!state.trace) return;
      state.selectedStep = state.trace.steps.length - 1;
      renderStepSelection();
    });

    dom.stepPlayBtn.addEventListener("click", () => {
      if (state.playTimer) stopPlayback();
      else startPlayback(false);
    });

    // Keyboard arrow scrubbing (when not typing in textarea/select)
    document.addEventListener("keydown", evt => {
      const tag = (evt.target && evt.target.tagName) || "";
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
      if (!state.trace) return;

      if (evt.key === "ArrowLeft") {
        evt.preventDefault();
        stopPlayback();
        state.selectedStep = Math.max(0, state.selectedStep - 1);
        renderStepSelection();
      } else if (evt.key === "ArrowRight") {
        evt.preventDefault();
        stopPlayback();
        state.selectedStep = Math.min(state.trace.steps.length - 1, state.selectedStep + 1);
        renderStepSelection();
      }
    });

    // Counterfactual Branch button click
    dom.candList.addEventListener("click", evt => {
      const btn = evt.target.closest(".branch-btn[data-branch-token]");
      if (!btn) return;
      triggerCounterfactualBranch(btn.dataset.branchToken);
    });

    // Head tab switching (synced across arcs & heatmap)
    const onHeadTabClick = evt => {
      const btn = evt.target.closest(".head-tab");
      if (!btn) return;
      state.activeHead = parseInt(btn.dataset.head, 10);
      renderHeadTabs();
      recomputeAttentionAndRender();
    };
    dom.streamHeadTabs.addEventListener("click", onHeadTabClick);
    dom.matrixHeadTabs.addEventListener("click", onHeadTabClick);

    // Attention Matrix controls
    dom.causalMaskToggle.addEventListener("change", () => {
      state.causalMask = dom.causalMaskToggle.checked;
      recomputeAttentionAndRender();
    });

    dom.matrixSourceSelect.addEventListener("change", () => {
      state.matrixSource = dom.matrixSourceSelect.value;
      renderAttentionMatrixSection();
    });

    dom.attentionCanvas.addEventListener("mousemove", handleMatrixHover);
    dom.attentionCanvas.addEventListener("mouseleave", () => {
      if (state.matrixAttention) drawAttentionHeatmap(state.matrixAttention, null);
    });

    // Telemetry chart interactive scrubbing
    let draggingTelemetry = false;
    dom.telemetryCanvas.addEventListener("mousedown", evt => {
      draggingTelemetry = true;
      handleTelemetryScrub(evt);
    });
    window.addEventListener("mousemove", evt => {
      if (draggingTelemetry) handleTelemetryScrub(evt);
    });
    window.addEventListener("mouseup", () => {
      draggingTelemetry = false;
    });

    // BPE Tokenizer Lab
    dom.bpeInput.addEventListener("input", renderBPELab);

    dom.bpePresets.addEventListener("click", evt => {
      const btn = evt.target.closest("[data-bpe-preset]");
      if (!btn) return;
      const key = btn.dataset.bpePreset;
      if (BPE_EXAMPLES[key]) {
        dom.bpeInput.value = BPE_EXAMPLES[key];
        renderBPELab();
      }
    });

    dom.bpeTokensBox.addEventListener("mouseover", evt => {
      const chip = evt.target.closest(".bpe-chip");
      if (!chip) return;
      state.bpeSelectedIdx = parseInt(chip.dataset.bpeIdx, 10);
      renderBPETokenDetail(state.bpeSelectedIdx);
    });

    dom.bpeMergeSlider.addEventListener("input", () => {
      renderBPEMergeStep(parseInt(dom.bpeMergeSlider.value, 10));
    });

    // Responsive redraw on window resize
    let resizeRaf = null;
    window.addEventListener("resize", () => {
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => {
        if (state.trace && state.streamAttention) {
          drawAttentionArcs(state.selectedStep, state.streamAttention.matrix[state.selectedStep]);
          renderTelemetryChart();
        }
        if (state.matrixAttention) {
          drawAttentionHeatmap(state.matrixAttention, null);
        }
      });
    });
  }

  function init() {
    // Populate scenario dropdown
    dom.scenarioSelect.innerHTML = Engine.SCENARIOS.map(
      sc => `<option value="${escHtml(sc.id)}">${escHtml(sc.title)}</option>`
    ).join("");
    dom.scenarioSelect.value = state.scenarioId;

    renderHeadTabs();
    syncControlsFromState();
    bindEvents();

    // Initialize BPE Lab first, then generate initial full trace
    dom.bpeInput.value = BPE_EXAMPLES.strawberry;
    renderBPELab();
    runNewGeneration(false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
