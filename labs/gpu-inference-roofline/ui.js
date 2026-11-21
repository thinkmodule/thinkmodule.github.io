/**
 * ThinkModule Lab — GPU Inference & Roofline Studio UI + 60 FPS Visualizations (v1.0)
 * Calm, readable hardware dataflow animation, interactive Log-Log Roofline,
 * Batch Pareto frontier, and live PagedAttention vs. Contiguous VRAM block heatmap.
 */
(function () {
  "use strict";

  const Engine = window.RooflineEngine;
  if (!Engine) {
    console.error("RooflineEngine not loaded.");
    return;
  }

  const $ = id => document.getElementById(id);

  const state = {
    cfg: { ...Engine.PRESETS.memory_wall },
    activePreset: "memory_wall",
    phase: "decode", // "decode" | "prefill"
    animSpeed: 0.55, // Calm default pace
    metrics: null,
    sweep: [],
    simTime: 0,
    lastTs: 0,
    packets: [],
    rooflineHover: null,
    rooflineGeom: null,
    paretoPoints: [],
    paretoHoverIdx: -1,
    allocatorRequests: []
  };

  const REQ_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#22d3ee"];

  const dom = {
    presetGrid: $("presetGrid"),
    gpuSelect: $("gpuSelect"),
    gpuCountRow: $("gpuCountRow"),
    gpuCountVal: $("gpuCountVal"),
    hwRidgeBadge: $("hwRidgeBadge"),
    modelSelect: $("modelSelect"),
    attnModeRow: $("attnModeRow"),
    kvPerTokBadge: $("kvPerTokBadge"),
    weightQuantRow: $("weightQuantRow"),
    kvQuantRow: $("kvQuantRow"),
    allocModeRow: $("allocModeRow"),
    batchSlider: $("batchSlider"),
    batchVal: $("batchVal"),
    promptSlider: $("promptSlider"),
    promptVal: $("promptVal"),
    genSlider: $("genSlider"),
    genVal: $("genVal"),

    heroBadges: $("heroBadges"),
    phaseDecodeBtn: $("phaseDecodeBtn"),
    phasePrefillBtn: $("phasePrefillBtn"),
    speedCalmBtn: $("speedCalmBtn"),
    speedNormalBtn: $("speedNormalBtn"),
    speedPauseBtn: $("speedPauseBtn"),

    pipelineCanvas: $("pipelineCanvas"),
    rooflineCanvas: $("rooflineCanvas"),
    paretoCanvas: $("paretoCanvas"),
    allocatorCanvas: $("allocatorCanvas"),

    kpiTtft: $("kpiTtft"),
    kpiTtftSub: $("kpiTtftSub"),
    kpiUserSpeed: $("kpiUserSpeed"),
    kpiTpotSub: $("kpiTpotSub"),
    kpiThroughput: $("kpiThroughput"),
    kpiMfu: $("kpiMfu"),
    kpiMfuSub: $("kpiMfuSub"),
    kpiBwUtil: $("kpiBwUtil"),
    kpiBwSub: $("kpiBwSub"),
    kpiCost: $("kpiCost"),

    diagnosisBox: $("diagnosisBox"),
    diagnosisTitle: $("diagnosisTitle"),
    diagnosisBody: $("diagnosisBody"),

    rooflineTooltip: $("rooflineTooltip"),
    rooflineMeta: $("rooflineMeta"),
    paretoTooltip: $("paretoTooltip"),
    paretoMeta: $("paretoMeta"),

    vramSummaryMeta: $("vramSummaryMeta"),
    segWeights: $("segWeights"),
    segKv: $("segKv"),
    segWaste: $("segWaste"),
    segAct: $("segAct"),
    vramLegend: $("vramLegend"),
    burstReqBtn: $("burstReqBtn"),
    resetBlocksBtn: $("resetBlocksBtn")
  };

  // ============================================================================
  // HELPERS
  // ============================================================================

  function setupCanvas(canvas, cssHeight) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 300);
    const height = cssHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function fmtMs(ms) {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
    if (ms >= 10) return `${ms.toFixed(0)} ms`;
    return `${ms.toFixed(1)} ms`;
  }

  function fmtNum(n) {
    if (n >= 10000) return `${(n / 1000).toFixed(1)}k`;
    if (n >= 1000) return Math.round(n).toLocaleString();
    return n.toFixed(1);
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rad = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  // ============================================================================
  // DATA PACKETS & PAGED-ATTENTION REQUEST POOL
  // ============================================================================

  function initPackets() {
    state.packets = [];
    for (let i = 0; i < 28; i++) {
      state.packets.push({
        t: i / 28,
        lane: i % 4,
        kind: i % 3 === 0 ? "kv" : "weight",
        speedJitter: 0.88 + (i % 5) * 0.06
      });
    }
  }

  function resetAllocatorRequests() {
    state.allocatorRequests = [
      { id: "Req A", color: REQ_COLORS[0], usedBlocks: 6, reservedBlocks: 14 },
      { id: "Req B", color: REQ_COLORS[1], usedBlocks: 4, reservedBlocks: 12 },
      { id: "Req C", color: REQ_COLORS[2], usedBlocks: 9, reservedBlocks: 16 },
      { id: "Req D", color: REQ_COLORS[3], usedBlocks: 5, reservedBlocks: 12 }
    ];
  }

  function addBurstRequest() {
    if (state.allocatorRequests.length >= 9) {
      state.allocatorRequests.shift();
    }
    const idx = state.allocatorRequests.length;
    const labelChar = String.fromCharCode(65 + (Math.floor(Math.random() * 20) + 4));
    const used = 3 + Math.floor(Math.random() * 7);
    const reserved = used + 5 + Math.floor(Math.random() * 7);
    state.allocatorRequests.push({
      id: `Req ${labelChar}${idx + 1}`,
      color: REQ_COLORS[idx % REQ_COLORS.length],
      usedBlocks: used,
      reservedBlocks: reserved
    });
    drawAllocatorCanvas();
  }

  // ============================================================================
  // STATE SYNC & RECOMPUTATION
  // ============================================================================

  function syncSidebarFromState() {
    const c = state.cfg;
    dom.gpuSelect.value = c.gpuId;
    dom.modelSelect.value = c.modelId;

    dom.gpuCountVal.textContent = `${c.gpuCount}×`;
    dom.gpuCountRow.querySelectorAll(".seg-btn").forEach(b => {
      b.classList.toggle("active", parseInt(b.dataset.val, 10) === c.gpuCount);
    });

    dom.attnModeRow.querySelectorAll(".seg-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.val === c.attnMode);
    });

    dom.weightQuantRow.querySelectorAll(".seg-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.val === c.weightQuant);
    });

    dom.kvQuantRow.querySelectorAll(".seg-btn").forEach(b => {
      b.classList.toggle("active-violet", b.dataset.val === c.kvQuant);
    });

    dom.allocModeRow.querySelectorAll(".seg-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.val === c.allocMode);
    });

    const bIdx = Math.max(0, Engine.BATCH_OPTIONS.indexOf(c.batchSize));
    dom.batchSlider.value = String(bIdx);
    dom.batchVal.textContent = String(c.batchSize);

    const pIdx = Math.max(0, Engine.PROMPT_OPTIONS.indexOf(c.promptLen));
    dom.promptSlider.value = String(pIdx);
    dom.promptVal.textContent = c.promptLen.toLocaleString();
    dom.promptVal.className = "field-val" + (c.promptLen >= 32768 ? " warn" : "");

    dom.genSlider.value = String(c.genLen);
    dom.genVal.textContent = c.genLen.toLocaleString();

    dom.presetGrid.querySelectorAll(".preset-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.preset === state.activePreset);
    });
  }

  function recomputeAndRenderStatic() {
    state.metrics = Engine.computeMetrics(state.cfg);
    state.sweep = Engine.computeBatchSweep(state.cfg);

    const m = state.metrics;

    // Sidebar micro-badges
    dom.hwRidgeBadge.textContent = `I* = ${m.hardware.ridgePoint.toFixed(0)} FLOP/B`;
    dom.kvPerTokBadge.textContent = `${(m.kvBytesPerToken / 1024).toFixed(0)} KB/tok`;

    // Hero status badges
    const badges = [];
    if (m.memory.isOOM) {
      badges.push(`<span class="badge oom">⚠ VRAM OOM (${m.memory.totalUsedGB.toFixed(1)} / ${m.hardware.totalVramGB} GB)</span>`);
    } else {
      badges.push(`<span class="badge">VRAM: ${m.memory.totalUsedGB.toFixed(1)} / ${m.hardware.totalVramGB} GB (${m.memory.vramUtilPct.toFixed(0)}%)</span>`);
    }
    const activePhaseObj = state.phase === "prefill" ? m.prefill : m.decode;
    const regCls = activePhaseObj.regime === "Compute-Bound" ? "comp" : "mem";
    badges.push(`<span class="badge ${regCls}">${state.phase.toUpperCase()}: ${activePhaseObj.regime}</span>`);
    badges.push(`<span class="badge">${m.gpuCount}× ${m.gpu.shortName} &bull; ${m.hardware.effPeakTflops.toFixed(0)} TFLOPs</span>`);
    dom.heroBadges.innerHTML = badges.join("");

    // KPIs
    dom.kpiTtft.textContent = m.memory.isOOM ? "OOM" : fmtMs(m.prefill.ttftMs);
    dom.kpiTtft.className = "kpi-val " + (m.memory.isOOM ? "red" : "cyan");
    dom.kpiTtftSub.textContent = `Prefill AI: ${m.prefill.ai.toFixed(0)} FLOP/B`;

    dom.kpiUserSpeed.textContent = m.memory.isOOM ? "OOM" : `${m.decode.userTokPerSec.toFixed(1)} tok/s`;
    dom.kpiUserSpeed.className = "kpi-val " + (m.memory.isOOM ? "red" : m.decode.userTokPerSec >= 30 ? "green" : "amber");
    dom.kpiTpotSub.textContent = m.memory.isOOM ? "Exceeds GPU VRAM" : `TPOT: ${fmtMs(m.decode.tpotMs)} / tok`;

    dom.kpiThroughput.textContent = m.memory.isOOM ? "OOM" : `${fmtNum(m.decode.totalTokPerSec)} tok/s`;
    dom.kpiThroughput.className = "kpi-val " + (m.memory.isOOM ? "red" : "violet");

    const curMfu = state.phase === "prefill" ? m.prefill.mfuPct : m.decode.mfuPct;
    const curBw = state.phase === "prefill" ? m.prefill.bwUtilPct : m.decode.bwUtilPct;

    dom.kpiMfu.textContent = m.memory.isOOM ? "0.0%" : `${curMfu.toFixed(1)}%`;
    dom.kpiMfu.className = "kpi-val " + (curMfu >= 45 ? "green" : curMfu >= 12 ? "amber" : "red");
    dom.kpiMfuSub.textContent = `${(state.phase === "prefill" ? m.prefill.achievedTflops : m.decode.achievedTflops).toFixed(1)} / ${m.hardware.effPeakTflops.toFixed(0)} TFLOPs`;

    dom.kpiBwUtil.textContent = m.memory.isOOM ? "STALL" : `${curBw.toFixed(0)}%`;
    dom.kpiBwUtil.className = "kpi-val " + (curBw >= 80 ? "amber" : "cyan");
    dom.kpiBwSub.textContent = `${(m.hardware.effBwGBs / 1000).toFixed(2)} TB/s ceiling`;

    dom.kpiCost.textContent = m.memory.isOOM ? "N/A" : `$${m.decode.costPer1MTok.toFixed(2)}`;
    dom.kpiCost.className = "kpi-val " + (m.decode.costPer1MTok <= 0.5 ? "green" : "text");

    // Diagnosis banner
    dom.diagnosisBox.className = `diagnosis ${m.diagnosis.severity}`;
    dom.diagnosisTitle.textContent = m.diagnosis.title;
    dom.diagnosisBody.textContent = m.diagnosis.body;

    // VRAM stacked bar
    const denom = Math.max(m.hardware.totalVramGB, m.memory.totalUsedGB);
    dom.segWeights.style.width = `${(m.memory.weightsGB / denom) * 100}%`;
    dom.segKv.style.width = `${(m.memory.logicalKvGB / denom) * 100}%`;
    dom.segWaste.style.width = `${(m.memory.wasteKvGB / denom) * 100}%`;
    dom.segAct.style.width = `${(m.memory.activationGB / denom) * 100}%`;

    dom.vramSummaryMeta.textContent = m.memory.isOOM
      ? `OVERFLOW: ${m.memory.totalUsedGB.toFixed(1)} GB required vs. ${m.hardware.totalVramGB} GB available`
      : `${m.memory.totalUsedGB.toFixed(1)} GB used of ${m.hardware.totalVramGB} GB (${m.memory.freeGB.toFixed(1)} GB headroom)`;

    dom.vramLegend.innerHTML = `
      <span class="leg-item"><span class="leg-dot" style="background:#38bdf8"></span>Weights: <strong>${m.memory.weightsGB.toFixed(1)} GB</strong></span>
      <span class="leg-item"><span class="leg-dot" style="background:#a78bfa"></span>Active KV-Cache: <strong>${m.memory.logicalKvGB.toFixed(1)} GB</strong></span>
      <span class="leg-item"><span class="leg-dot" style="background:#fbbf24"></span>Allocator Waste: <strong>${m.memory.wasteKvGB.toFixed(1)} GB (${m.memory.wastePct.toFixed(1)}%)</strong></span>
      <span class="leg-item"><span class="leg-dot" style="background:#34d399"></span>Activations/CUDA: <strong>${m.memory.activationGB.toFixed(1)} GB</strong></span>
    `;

    dom.rooflineMeta.textContent = `Ridge I* = ${m.hardware.ridgePoint.toFixed(0)} FLOP/B | Decode AI = ${m.decode.ai.toFixed(1)} | Prefill AI = ${m.prefill.ai.toFixed(0)}`;
    const maxSafeBatch = state.sweep.filter(s => !s.isOOM).pop();
    dom.paretoMeta.textContent = maxSafeBatch
      ? `Max Safe Batch: B=${maxSafeBatch.batchSize} (${fmtNum(maxSafeBatch.totalTokPerSec)} tok/s)`
      : `All batch sizes exceed current VRAM`;

    drawRooflineChart();
    drawParetoChart();
    drawAllocatorCanvas();
  }

  // ============================================================================
  // CANVAS 1: ANIMATED GPU HARDWARE PIPELINE (CALM, READABLE TEMPO)
  // ============================================================================

  function drawPipelineFrame(timeSec) {
    const m = state.metrics;
    if (!m) return;

    const { ctx, width, height } = setupCanvas(dom.pipelineCanvas, 270);
    ctx.clearRect(0, 0, width, height);

    const isPrefill = state.phase === "prefill";
    const mfu = m.memory.isOOM ? 0 : (isPrefill ? m.prefill.mfuPct : m.decode.mfuPct);
    const bwUtil = m.memory.isOOM ? 0 : (isPrefill ? m.prefill.bwUtilPct : m.decode.bwUtilPct);
    const kvShare = isPrefill ? 0.18 : m.decode.kvTrafficSharePct / 100;

    // Layout geometry (responsive across widths)
    const padX = Math.max(18, width * 0.03);
    const hbmW = Math.min(240, width * 0.23);
    const tcW = Math.min(255, width * 0.25);
    const sramW = Math.min(115, width * 0.12);
    const hbmX = padX;
    const tcX = width - padX - tcW;
    const sramX = hbmX + hbmW + (tcX - (hbmX + hbmW) - sramW) * 0.58;

    const boxY = 36;
    const boxH = height - 64;

    // Subtle background grid
    ctx.strokeStyle = "rgba(30, 47, 82, 0.35)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 36) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }

    // ---- 1. HBM VRAM BANK (LEFT) ----
    roundRect(ctx, hbmX, boxY, hbmW, boxH, 10);
    ctx.fillStyle = "#0b1426";
    ctx.fill();
    ctx.strokeStyle = m.memory.isOOM ? "#f87171" : "#2a3f6b";
    ctx.lineWidth = m.memory.isOOM ? 2 : 1.3;
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    const memLabel = m.gpu.id === "m3_max" ? "UNIFIED MEM" : m.gpu.id === "rtx_4090" ? "GDDR6X VRAM" : "HBM VRAM";
    const busLabel = m.gpu.id === "m3_max" ? "UNIFIED BUS" : m.gpu.id === "rtx_4090" ? "GDDR6X BUS" : "HBM BUS";
    const coreLabel = m.gpu.id === "m3_max" ? "GPU CORES" : "TENSOR CORES";
    ctx.fillText(`${memLabel} (${m.hardware.totalVramGB} GB)`, hbmX + 12, boxY + 20);

    // Inner VRAM fill meter (vertical bars)s
    const innerX = hbmX + 12;
    const innerY = boxY + 32;
    const innerW = hbmW - 24;
    const innerH = boxH - 64;

    roundRect(ctx, innerX, innerY, innerW, innerH, 6);
    ctx.fillStyle = "#060b16";
    ctx.fill();
    ctx.strokeStyle = "#1c2c4d";
    ctx.stroke();

    const cap = Math.max(m.hardware.totalVramGB, m.memory.totalUsedGB);
    const wH = (m.memory.weightsGB / cap) * innerH;
    const kvH = (m.memory.logicalKvGB / cap) * innerH;
    const wasteH = (m.memory.wasteKvGB / cap) * innerH;

    // Draw stacked from bottom up
    let curY = innerY + innerH;
    if (wH > 0) {
      ctx.fillStyle = "rgba(56, 189, 248, 0.78)";
      ctx.fillRect(innerX + 2, curY - wH, innerW - 4, wH);
      curY -= wH;
    }
    if (kvH > 0) {
      ctx.fillStyle = "rgba(167, 139, 250, 0.82)";
      ctx.fillRect(innerX + 2, curY - kvH, innerW - 4, kvH);
      curY -= kvH;
    }
    if (wasteH > 0) {
      ctx.fillStyle = "rgba(251, 191, 36, 0.62)";
      ctx.fillRect(innerX + 2, curY - wasteH, innerW - 4, wasteH);
    }

    // OOM red danger line
    if (m.memory.isOOM) {
      const pulse = 0.55 + 0.45 * Math.sin(timeSec * 5);
      ctx.fillStyle = `rgba(248, 113, 113, ${0.22 * pulse})`;
      ctx.fillRect(innerX, innerY, innerW, innerH);
      ctx.fillStyle = "#fca5a5";
      ctx.font = "700 12px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("⚠ OUT OF MEMORY", innerX + innerW / 2, innerY + innerH / 2);
    } else {
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "600 10px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      if (wH > 16) ctx.fillText(`Weights ${m.memory.weightsGB.toFixed(1)}G`, innerX + innerW / 2, innerY + innerH - wH / 2 + 3);
      if (kvH > 16) ctx.fillText(`KV ${m.memory.logicalKvGB.toFixed(1)}G`, innerX + innerW / 2, innerY + innerH - wH - kvH / 2 + 3);
    }

    ctx.fillStyle = "#7f92b0";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Used: ${m.memory.totalUsedGB.toFixed(1)} GB (${m.memory.vramUtilPct.toFixed(0)}%)`, hbmX + 12, boxY + boxH - 12);

    // ---- 2. HIGH-BANDWIDTH MEMORY BUS (HBM -> SRAM) ----
    const bus1StartX = hbmX + hbmW;
    const bus1EndX = sramX;
    const busCenterY = boxY + boxH / 2;
    const busHeight = 76;

    const isBusHot = !m.memory.isOOM && bwUtil >= 75;
    ctx.fillStyle = isBusHot ? "rgba(251, 191, 36, 0.08)" : "rgba(56, 189, 248, 0.06)";
    ctx.fillRect(bus1StartX, busCenterY - busHeight / 2, bus1EndX - bus1StartX, busHeight);

    ctx.strokeStyle = isBusHot ? "rgba(251, 191, 36, 0.55)" : "rgba(56, 189, 248, 0.32)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(bus1StartX, busCenterY - busHeight / 2);
    ctx.lineTo(bus1EndX, busCenterY - busHeight / 2);
    ctx.moveTo(bus1StartX, busCenterY + busHeight / 2);
    ctx.lineTo(bus1EndX, busCenterY + busHeight / 2);
    ctx.stroke();

    // Bus label
    ctx.fillStyle = isBusHot ? "#fbbf24" : "#94a3b8";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText(
      `${busLabel} • ${(m.hardware.effBwGBs / 1000).toFixed(2)} TB/s`,
      (bus1StartX + bus1EndX) / 2,
      busCenterY - busHeight / 2 - 12
    );
    ctx.fillStyle = isBusHot ? "#fde68a" : "#64748b";
    ctx.fillText(
      m.memory.isOOM ? "HALTED (OOM)" : `${bwUtil.toFixed(0)}% Bandwidth Saturated`,
      (bus1StartX + bus1EndX) / 2,
      busCenterY + busHeight / 2 + 18
    );

    // Animate calm data packets along Bus 1
    if (!m.memory.isOOM) {
      const visiblePackets = Math.max(4, Math.round((bwUtil / 100) * state.packets.length));
      for (let i = 0; i < visiblePackets; i++) {
        const p = state.packets[i];
        const px = bus1StartX + 8 + p.t * Math.max(10, bus1EndX - bus1StartX - 16);
        const py = busCenterY - 24 + p.lane * 16;
        const isKvPkt = (i / visiblePackets) < kvShare;

        ctx.fillStyle = isKvPkt ? "#a78bfa" : "#38bdf8";
        ctx.shadowColor = isKvPkt ? "rgba(167,139,250,0.65)" : "rgba(56,189,248,0.65)";
        ctx.shadowBlur = 7;
        roundRect(ctx, px - 6, py - 3.5, 12, 7, 3);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // ---- 3. L2 CACHE & ON-CHIP SRAM (MIDDLE) ----
    roundRect(ctx, sramX, boxY + 24, sramW, boxH - 48, 9);
    ctx.fillStyle = "#0e1a31";
    ctx.fill();
    ctx.strokeStyle = "#334e82";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("L2 / SRAM", sramX + sramW / 2, boxY + 46);
    ctx.fillStyle = "#64748b";
    ctx.font = "9.5px ui-monospace, Menlo, monospace";
    ctx.fillText(`${m.gpu.sramMB * m.gpuCount} MB`, sramX + sramW / 2, boxY + 62);
    ctx.fillStyle = "#38bdf8";
    ctx.fillText("FlashAttn-2", sramX + sramW / 2, boxY + boxH / 2 + 4);
    ctx.fillStyle = "#7f92b0";
    ctx.fillText("Tiled Q·Kᵀ", sramX + sramW / 2, boxY + boxH / 2 + 18);

    // ---- 4. SRAM -> TENSOR CORES CONDUIT ----
    const bus2StartX = sramX + sramW;
    const bus2EndX = tcX;
    ctx.strokeStyle = mfu > 40 ? "rgba(52, 211, 153, 0.5)" : "rgba(100, 116, 139, 0.35)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(bus2StartX, busCenterY - 22);
    ctx.lineTo(bus2EndX, busCenterY - 22);
    ctx.moveTo(bus2StartX, busCenterY + 22);
    ctx.lineTo(bus2EndX, busCenterY + 22);
    ctx.stroke();

    if (!m.memory.isOOM) {
      const streamCount = Math.max(2, Math.round((mfu / 100) * 10) + 2);
      for (let i = 0; i < streamCount; i++) {
        const t2 = (state.packets[i].t + 0.35) % 1;
        const px = bus2StartX + 6 + t2 * Math.max(8, bus2EndX - bus2StartX - 12);
        const py = busCenterY - 12 + (i % 3) * 12;
        ctx.fillStyle = mfu > 40 ? "#34d399" : "#38bdf8";
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ---- 5. TENSOR CORE ARRAY (RIGHT) ----
    roundRect(ctx, tcX, boxY, tcW, boxH, 10);
    ctx.fillStyle = "#0b1426";
    ctx.fill();
    ctx.strokeStyle = mfu >= 45 ? "#34d399" : mfu < 10 ? "#fbbf24" : "#2a3f6b";
    ctx.lineWidth = 1.4;
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${coreLabel} (${m.hardware.effPeakTflops.toFixed(0)} TF)`, tcX + 12, boxY + 20);

    // 4x4 Tensor Core Matrix Grid
    const gridPad = 14;
    const gridW = tcW - gridPad * 2;
    const gridH = boxH - 68;
    const cols = 4, rows = 4;
    const gap = 6;
    const cellW = (gridW - gap * (cols - 1)) / cols;
    const cellH = (gridH - gap * (rows - 1)) / rows;
    const activeCoreFloat = (mfu / 100) * (cols * rows);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const cx = tcX + gridPad + c * (cellW + gap);
        const cy = boxY + 32 + r * (cellH + gap);

        const isActive = !m.memory.isOOM && idx < Math.max(1, Math.ceil(activeCoreFloat));
        const intensity = !m.memory.isOOM
          ? Math.max(0, Math.min(1, activeCoreFloat - idx + 0.35))
          : 0;
        const wave = 0.72 + 0.28 * Math.sin(timeSec * 2.6 + idx * 0.7);

        roundRect(ctx, cx, cy, cellW, cellH, 5);
        if (isActive && mfu > 0.2) {
          const alpha = (0.18 + intensity * 0.65) * wave;
          ctx.fillStyle = mfu >= 45
            ? `rgba(52, 211, 153, ${alpha.toFixed(2)})`
            : `rgba(56, 189, 248, ${alpha.toFixed(2)})`;
          ctx.strokeStyle = mfu >= 45 ? "rgba(52,211,153,0.7)" : "rgba(56,189,248,0.6)";
        } else {
          ctx.fillStyle = "rgba(15, 25, 45, 0.9)";
          ctx.strokeStyle = "rgba(34, 52, 86, 0.65)";
        }
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      }
    }

    ctx.fillStyle = mfu >= 45 ? "#34d399" : mfu < 10 ? "#fbbf24" : "#94a3b8";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    const statusTxt = m.memory.isOOM
      ? "Idle — VRAM Overflow"
      : mfu < 10
        ? `Starving Compute: ${mfu.toFixed(1)}% MFU`
        : mfu >= 45
          ? `Compute Saturated: ${mfu.toFixed(1)}% MFU`
          : `Active Utilization: ${mfu.toFixed(1)}% MFU`;
    ctx.fillText(statusTxt, tcX + 12, boxY + boxH - 12);

    // Legend top-right of canvas
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "#38bdf8";
    ctx.fillText("■ Weights Stream", padX, 20);
    ctx.fillStyle = "#a78bfa";
    ctx.fillText("■ KV-Cache Stream", padX + 118, 20);
    ctx.fillStyle = "#34d399";
    ctx.fillText("■ Active Tensor Cores", padX + 245, 20);
  }

  // ============================================================================
  // CANVAS 2: LOG-LOG GPU ROOFLINE MODEL
  // ============================================================================

  function drawRooflineChart() {
    const m = state.metrics;
    if (!m) return;

    const { ctx, width, height } = setupCanvas(dom.rooflineCanvas, 330);
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 28, right: 24, bottom: 44, left: 60 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const minAI = 0.5, maxAI = 4096;
    const minTF = 0.5;
    const maxTF = Math.pow(10, Math.ceil(Math.log10(m.hardware.effPeakTflops * 1.7)));

    const logMinAI = Math.log10(minAI), logMaxAI = Math.log10(maxAI);
    const logMinTF = Math.log10(minTF), logMaxTF = Math.log10(maxTF);

    const xOf = ai => {
      const clamped = Math.max(minAI, Math.min(maxAI, ai));
      return pad.left + ((Math.log10(clamped) - logMinAI) / (logMaxAI - logMinAI)) * plotW;
    };
    const yOf = tf => {
      const clamped = Math.max(minTF, Math.min(maxTF, tf));
      return pad.top + plotH * (1 - (Math.log10(clamped) - logMinTF) / (logMaxTF - logMinTF));
    };

    state.rooflineGeom = { pad, plotW, plotH, minAI, maxAI, minTF, maxTF, logMinAI, logMaxAI, logMinTF, logMaxTF };

    const ridgeX = xOf(m.hardware.ridgePoint);
    const peakY = yOf(m.hardware.effPeakTflops);

    // Shaded Memory-Bound vs Compute-Bound regions
    ctx.fillStyle = "rgba(251, 191, 36, 0.045)";
    ctx.fillRect(pad.left, pad.top, Math.max(0, ridgeX - pad.left), plotH);
    ctx.fillStyle = "rgba(52, 211, 153, 0.045)";
    ctx.fillRect(ridgeX, pad.top, Math.max(0, pad.left + plotW - ridgeX), plotH);

    // Grid lines (powers of 10 / 2)
    ctx.strokeStyle = "rgba(30, 47, 82, 0.65)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#7f92b0";

    const xTicks = [1, 4, 16, 64, 256, 1024, 4096];
    ctx.textAlign = "center";
    xTicks.forEach(t => {
      const x = xOf(t);
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      ctx.fillText(String(t), x, pad.top + plotH + 16);
    });

    const yTicks = [1, 10, 100, 1000, 10000].filter(t => t <= maxTF);
    ctx.textAlign = "right";
    yTicks.forEach(t => {
      const y = yOf(t);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      ctx.fillText(t >= 1000 ? `${t / 1000}k` : String(t), pad.left - 8, y + 3);
    });

    // Ridge vertical dashed line
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.beginPath();
    ctx.moveTo(ridgeX, peakY);
    ctx.lineTo(ridgeX, pad.top + plotH);
    ctx.stroke();
    ctx.restore();

    // Region labels
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(251, 191, 36, 0.78)";
    ctx.fillText("MEMORY-BANDWIDTH BOUND", pad.left + 8, pad.top + 16);
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(52, 211, 153, 0.82)";
    ctx.fillText("COMPUTE BOUND", pad.left + plotW - 8, pad.top + 16);

    // Roofline ceiling path
    ctx.beginPath();
    const startTF = (m.hardware.effBwGBs * minAI) / 1000;
    ctx.moveTo(xOf(minAI), yOf(startTF));
    ctx.lineTo(ridgeX, peakY);
    ctx.lineTo(xOf(maxAI), peakY);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.6;
    ctx.stroke();

    // Ridge Point dot
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(ridgeX, peakY, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = "#bae6fd";
    ctx.fillText(`Ridge I*=${m.hardware.ridgePoint.toFixed(0)}`, ridgeX, Math.max(pad.top + 28, peakY - 10));

    // Batch-scaling trajectory curve for Decode (B=1 -> 128)
    if (state.sweep.length > 1) {
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(167, 139, 250, 0.65)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      state.sweep.forEach((pt, idx) => {
        const px = xOf(pt.decodeAI);
        const py = yOf(pt.decodeTflops);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.restore();
    }

    // Plot DECODE operating point
    const decX = xOf(m.decode.ai);
    const decY = yOf(m.decode.achievedTflops);
    ctx.fillStyle = m.memory.isOOM ? "rgba(248, 113, 113, 0.25)" : "rgba(251, 191, 36, 0.24)";
    ctx.beginPath(); ctx.arc(decX, decY, 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = m.memory.isOOM ? "#f87171" : "#fbbf24";
    ctx.beginPath(); ctx.arc(decX, decY, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = "700 10px ui-monospace, Menlo, monospace";
    ctx.textAlign = decX > width - 120 ? "right" : "left";
    ctx.fillText(
      `DECODE (B=${m.batchSize})`,
      decX + (decX > width - 120 ? -10 : 10),
      decY + 14
    );

    // Plot PREFILL operating point
    const preX = xOf(m.prefill.ai);
    const preY = yOf(m.prefill.achievedTflops);
    ctx.fillStyle = "rgba(52, 211, 153, 0.22)";
    ctx.beginPath(); ctx.arc(preX, preY, 11, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#34d399";
    ctx.beginPath(); ctx.arc(preX, preY, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.textAlign = preX > width - 110 ? "right" : "left";
    ctx.fillText(
      "PREFILL",
      preX + (preX > width - 110 ? -10 : 10),
      preY + 15
    );

    // Hover crosshair
    if (state.rooflineHover) {
      const hx = state.rooflineHover.x;
      ctx.strokeStyle = "rgba(226, 232, 240, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, pad.top + plotH);
      ctx.stroke();
    }

    // Axis titles
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10.5px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Arithmetic Intensity (FLOPs / Byte) — Log Scale", pad.left + plotW / 2, height - 10);

    ctx.save();
    ctx.translate(15, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Performance (TFLOPs/s) — Log", 0, 0);
    ctx.restore();
  }

  // ============================================================================
  // CANVAS 3: THROUGHPUT VS. LATENCY PARETO CURVE (BATCH 1 -> 128)
  // ============================================================================

  function drawParetoChart() {
    const sweep = state.sweep;
    if (!sweep.length) return;

    const { ctx, width, height } = setupCanvas(dom.paretoCanvas, 330);
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 28, right: 24, bottom: 44, left: 58 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const maxLat = Math.max(25, ...sweep.map(s => s.tpotMs)) * 1.15;
    const maxTput = Math.max(100, ...sweep.map(s => s.totalTokPerSec)) * 1.15;

    const xOf = lat => pad.left + (lat / maxLat) * plotW;
    const yOf = tp => pad.top + plotH * (1 - tp / maxTput);

    // Grid
    ctx.strokeStyle = "rgba(30, 47, 82, 0.65)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "#7f92b0";

    for (let g = 0; g <= 4; g++) {
      const y = pad.top + (plotH * g) / 4;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      const val = maxTput * (1 - g / 4);
      ctx.textAlign = "right";
      ctx.fillText(fmtNum(val), pad.left - 8, y + 3);

      const x = pad.left + (plotW * g) / 4;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round((maxLat * g) / 4)}ms`, x, pad.top + plotH + 16);
    }

    // 50ms interactive latency target band (~20 tok/s human reading comfort)
    if (50 < maxLat) {
      const targetX = xOf(50);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.5)";
      ctx.beginPath();
      ctx.moveTo(targetX, pad.top);
      ctx.lineTo(targetX, pad.top + plotH);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "rgba(52, 211, 153, 0.8)";
      ctx.textAlign = "left";
      ctx.fillText("50ms UX Target", Math.min(targetX + 5, pad.left + plotW - 95), pad.top + 14);
    }

    // Connect Pareto segments (solid violet for fits-in-VRAM, dashed red for OOM)
    for (let i = 0; i < sweep.length - 1; i++) {
      const a = sweep[i], b = sweep[i + 1];
      ctx.save();
      if (a.isOOM || b.isOOM) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(248, 113, 113, 0.55)";
      } else {
        ctx.strokeStyle = "#a78bfa";
      }
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(xOf(a.tpotMs), yOf(a.totalTokPerSec));
      ctx.lineTo(xOf(b.tpotMs), yOf(b.totalTokPerSec));
      ctx.stroke();
      ctx.restore();
    }

    // Points
    state.paretoPoints = [];
    sweep.forEach((pt, idx) => {
      const px = xOf(pt.tpotMs);
      const py = yOf(pt.totalTokPerSec);
      const isSelected = pt.batchSize === state.cfg.batchSize;
      const isHover = idx === state.paretoHoverIdx;

      state.paretoPoints.push({ px, py, pt, idx });

      if (isSelected || isHover) {
        ctx.fillStyle = pt.isOOM ? "rgba(248, 113, 113, 0.25)" : "rgba(56, 189, 248, 0.25)";
        ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI * 2); ctx.fill();
      }

      ctx.fillStyle = pt.isOOM ? "#f87171" : isSelected ? "#38bdf8" : "#a78bfa";
      ctx.beginPath();
      ctx.arc(px, py, isSelected ? 6 : 4.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = isSelected ? "#ffffff" : pt.isOOM ? "#fca5a5" : "#cbd5e1";
      ctx.font = `${isSelected ? "700" : "500"} 9.5px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.fillText(`B${pt.batchSize}`, px, py - 9);
    });

    // Axis titles
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10.5px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Per-User Latency — TPOT (ms, lower is faster)", pad.left + plotW / 2, height - 10);

    ctx.save();
    ctx.translate(15, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("System Throughput (tok/s)", 0, 0);
    ctx.restore();
  }

  // ============================================================================
  // CANVAS 4: PAGEDATTENTION (vLLM) VS. CONTIGUOUS BLOCK ALLOCATOR
  // ============================================================================

  function drawAllocatorCanvas() {
    const { ctx, width, height } = setupCanvas(dom.allocatorCanvas, 245);
    ctx.clearRect(0, 0, width, height);

    const reqs = state.allocatorRequests;
    const isPagedSelected = state.cfg.allocMode === "paged";
    const pad = 16;
    const gapMid = 20;
    const halfW = (width - pad * 2 - gapMid) / 2;

    const totalSlots = 64; // 8x8 physical KV blocks (each block = 16 tokens)
    const cols = 16;
    const rows = 4;

    // Build Contiguous slots array
    const contigSlots = new Array(totalSlots).fill(null);
    let cPtr = 0;
    let contigUsed = 0, contigWaste = 0, contigDropped = 0;

    reqs.forEach(r => {
      if (cPtr + r.reservedBlocks <= totalSlots) {
        for (let k = 0; k < r.reservedBlocks; k++) {
          const isWaste = k >= r.usedBlocks;
          contigSlots[cPtr++] = { color: r.color, id: r.id, waste: isWaste };
          if (isWaste) contigWaste++;
          else contigUsed++;
        }
      } else {
        contigDropped++;
      }
    });

    // Build PagedAttention slots array (scattered non-contiguous physical pages, 0 wasted whole blocks!)
    const pagedSlots = new Array(totalSlots).fill(null);
    // Deterministic pseudo-scattered order to illustrate non-contiguous block-table mapping
    const perm = Array.from({ length: totalSlots }, (_, i) => (i * 11 + 5) % totalSlots);
    let pPtr = 0;
    let pagedUsed = 0, pagedDropped = 0;

    reqs.forEach(r => {
      if (pPtr + r.usedBlocks <= totalSlots) {
        for (let k = 0; k < r.usedBlocks; k++) {
          pagedSlots[perm[pPtr++]] = { color: r.color, id: r.id, waste: false };
          pagedUsed++;
        }
      } else {
        pagedDropped++;
      }
    });

    function drawPanel(x, title, subtitle, slots, highlight, accentColor) {
      roundRect(ctx, x, 12, halfW, height - 24, 10);
      ctx.fillStyle = "#0a1324";
      ctx.fill();
      ctx.strokeStyle = highlight ? accentColor : "#1e2f52";
      ctx.lineWidth = highlight ? 1.8 : 1;
      ctx.stroke();

      ctx.fillStyle = highlight ? "#f8fafc" : "#cbd5e1";
      ctx.font = "700 11.5px ui-monospace, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText(title, x + 14, 34);

      ctx.fillStyle = "#8ea2c4";
      ctx.font = "10px ui-monospace, Menlo, monospace";
      ctx.fillText(subtitle, x + 14, 51);

      const gridX = x + 14;
      const gridY = 66;
      const gridW = halfW - 28;
      const gridH = height - 118;
      const cGap = 4;
      const cellW = (gridW - cGap * (cols - 1)) / cols;
      const cellH = (gridH - cGap * (rows - 1)) / rows;

      for (let i = 0; i < totalSlots; i++) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const bx = gridX + c * (cellW + cGap);
        const by = gridY + r * (cellH + cGap);
        const slot = slots[i];

        roundRect(ctx, bx, by, cellW, cellH, 3);
        if (!slot) {
          ctx.fillStyle = "rgba(15, 25, 45, 0.85)";
          ctx.strokeStyle = "rgba(34, 52, 86, 0.65)";
          ctx.fill();
          ctx.stroke();
        } else if (slot.waste) {
          ctx.fillStyle = "rgba(251, 191, 36, 0.15)";
          ctx.strokeStyle = "rgba(251, 191, 36, 0.65)";
          ctx.fill();
          ctx.stroke();
          // Diagonal hatch for wasted reservation
          ctx.beginPath();
          ctx.moveTo(bx + 2, by + cellH - 2);
          ctx.lineTo(bx + cellW - 2, by + 2);
          ctx.strokeStyle = "rgba(251, 191, 36, 0.55)";
          ctx.stroke();
        } else {
          ctx.fillStyle = slot.color;
          ctx.globalAlpha = 0.82;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      // Footer legend per panel
      ctx.font = "10px ui-monospace, Menlo, monospace";
      ctx.fillStyle = "#94a3b8";
      ctx.fillText("1 block = 16 KV tokens  •  Hatched = Wasted reservation", x + 14, height - 24);
    }

    const contigWastePct = contigUsed + contigWaste > 0
      ? ((contigWaste / (contigUsed + contigWaste)) * 100).toFixed(0)
      : "0";

    drawPanel(
      pad,
      `CONTIGUOUS KV-CACHE ${!isPagedSelected ? "(ACTIVE)" : ""}`,
      `Used: ${contigUsed} blk | Wasted: ${contigWaste} blk (${contigWastePct}%) | Queued/OOM: ${contigDropped} reqs`,
      contigSlots,
      !isPagedSelected,
      "#fbbf24"
    );

    drawPanel(
      pad + halfW + gapMid,
      `PAGEDATTENTION / vLLM ${isPagedSelected ? "(ACTIVE)" : ""}`,
      `Used: ${pagedUsed} blk | Internal Waste: <3% | Free: ${totalSlots - pagedUsed} blk | Queued: ${pagedDropped}`,
      pagedSlots,
      isPagedSelected,
      "#38bdf8"
    );
  }

  // ============================================================================
  // ANIMATION LOOP (CLAMPED DELTA, CALM PACE)
  // ============================================================================

  function tick(ts) {
    if (!state.lastTs) state.lastTs = ts;
    const rawDt = (ts - state.lastTs) / 1000;
    state.lastTs = ts;
    const dt = Math.min(rawDt, 0.05) * state.animSpeed;

    state.simTime += dt;

    if (state.metrics && !state.metrics.memory.isOOM && state.animSpeed > 0) {
      const bwFactor = state.phase === "prefill"
        ? state.metrics.prefill.bwUtilPct / 100
        : state.metrics.decode.bwUtilPct / 100;
      const baseVel = 0.16 + bwFactor * 0.24; // Deliberately calm, easy-to-follow flow

      for (let i = 0; i < state.packets.length; i++) {
        const p = state.packets[i];
        p.t = (p.t + dt * baseVel * p.speedJitter) % 1;
      }
    }

    drawPipelineFrame(state.simTime);
    requestAnimationFrame(tick);
  }

  // ============================================================================
  // EVENT BINDINGS & INIT
  // ============================================================================

  function bindEvents() {
    dom.presetGrid.addEventListener("click", evt => {
      const btn = evt.target.closest(".preset-btn");
      if (!btn || !Engine.PRESETS[btn.dataset.preset]) return;
      state.activePreset = btn.dataset.preset;
      state.cfg = { ...Engine.PRESETS[btn.dataset.preset] };
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.gpuSelect.addEventListener("change", () => {
      state.activePreset = null;
      state.cfg.gpuId = dom.gpuSelect.value;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.modelSelect.addEventListener("change", () => {
      state.activePreset = null;
      state.cfg.modelId = dom.modelSelect.value;
      // Auto-select native attention mode for GPT-3 vs modern GQA models
      state.cfg.attnMode = state.cfg.modelId === "gpt3_175b" ? "mha" : "gqa";
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.gpuCountRow.addEventListener("click", evt => {
      const btn = evt.target.closest(".seg-btn");
      if (!btn) return;
      state.activePreset = null;
      state.cfg.gpuCount = parseInt(btn.dataset.val, 10);
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.attnModeRow.addEventListener("click", evt => {
      const btn = evt.target.closest(".seg-btn");
      if (!btn) return;
      state.activePreset = null;
      state.cfg.attnMode = btn.dataset.val;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.weightQuantRow.addEventListener("click", evt => {
      const btn = evt.target.closest(".seg-btn");
      if (!btn) return;
      state.activePreset = null;
      state.cfg.weightQuant = btn.dataset.val;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.kvQuantRow.addEventListener("click", evt => {
      const btn = evt.target.closest(".seg-btn");
      if (!btn) return;
      state.activePreset = null;
      state.cfg.kvQuant = btn.dataset.val;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.allocModeRow.addEventListener("click", evt => {
      const btn = evt.target.closest(".seg-btn");
      if (!btn) return;
      state.activePreset = null;
      state.cfg.allocMode = btn.dataset.val;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.batchSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.batchSize = Engine.BATCH_OPTIONS[parseInt(dom.batchSlider.value, 10)] || 1;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.promptSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.promptLen = Engine.PROMPT_OPTIONS[parseInt(dom.promptSlider.value, 10)] || 2048;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    dom.genSlider.addEventListener("input", () => {
      state.activePreset = null;
      state.cfg.genLen = parseInt(dom.genSlider.value, 10) || 512;
      syncSidebarFromState();
      recomputeAndRenderStatic();
    });

    // Phase toggle (Decode vs Prefill)
    [dom.phaseDecodeBtn, dom.phasePrefillBtn].forEach(btn => {
      btn.addEventListener("click", () => {
        state.phase = btn.dataset.phase;
        dom.phaseDecodeBtn.classList.toggle("active", state.phase === "decode");
        dom.phasePrefillBtn.classList.toggle("active", state.phase === "prefill");
        recomputeAndRenderStatic();
      });
    });

    // Animation speed controls
    [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedPauseBtn].forEach(btn => {
      btn.addEventListener("click", () => {
        state.animSpeed = parseFloat(btn.dataset.speed);
        [dom.speedCalmBtn, dom.speedNormalBtn, dom.speedPauseBtn].forEach(b => {
          b.classList.toggle("active", b === btn);
        });
      });
    });

    // Roofline hover inspector
    dom.rooflineCanvas.addEventListener("mousemove", evt => {
      if (!state.rooflineGeom || !state.metrics) return;
      const rect = dom.rooflineCanvas.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const g = state.rooflineGeom;
      if (x >= g.pad.left && x <= g.pad.left + g.plotW) {
        const frac = (x - g.pad.left) / g.plotW;
        const ai = Math.pow(10, g.logMinAI + frac * (g.logMaxAI - g.logMinAI));
        const ceilTF = Math.min(state.metrics.hardware.effPeakTflops, (state.metrics.hardware.effBwGBs * ai) / 1000);
        const zone = ai >= state.metrics.hardware.ridgePoint ? "Compute-Bound" : "Memory-Bound";
        state.rooflineHover = { x, ai, ceilTF };
        dom.rooflineTooltip.firstElementChild.textContent =
          `I = ${ai.toFixed(1)} FLOPs/Byte → Ceiling: ${ceilTF.toFixed(1)} TFLOPs/s (${zone})`;
        drawRooflineChart();
      }
    });

    dom.rooflineCanvas.addEventListener("mouseleave", () => {
      state.rooflineHover = null;
      dom.rooflineTooltip.firstElementChild.textContent = "Hover chart to inspect Roofline bounds";
      drawRooflineChart();
    });

    // Pareto interactive hover & click-to-select batch
    dom.paretoCanvas.addEventListener("mousemove", evt => {
      const rect = dom.paretoCanvas.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const y = evt.clientY - rect.top;
      let best = -1, bestDist = 24;
      state.paretoPoints.forEach(p => {
        const d = Math.hypot(p.px - x, p.py - y);
        if (d < bestDist) { bestDist = d; best = p.idx; }
      });
      if (best !== state.paretoHoverIdx) {
        state.paretoHoverIdx = best;
        dom.paretoCanvas.style.cursor = best >= 0 ? "pointer" : "default";
        if (best >= 0) {
          const pt = state.sweep[best];
          dom.paretoTooltip.firstElementChild.textContent = pt.isOOM
            ? `Batch ${pt.batchSize}: OUT OF MEMORY (${pt.vramUtilPct.toFixed(0)}% VRAM)`
            : `Batch ${pt.batchSize}: ${fmtNum(pt.totalTokPerSec)} tok/s total | ${pt.tpotMs.toFixed(1)} ms/tok | $${pt.costPer1MTok.toFixed(2)}/1M tok`;
        } else {
          dom.paretoTooltip.firstElementChild.textContent = "Click any batch point on curve to jump directly to that Batch Size";
        }
        drawParetoChart();
      }
    });

    dom.paretoCanvas.addEventListener("click", () => {
      if (state.paretoHoverIdx >= 0 && state.sweep[state.paretoHoverIdx]) {
        state.activePreset = null;
        state.cfg.batchSize = state.sweep[state.paretoHoverIdx].batchSize;
        syncSidebarFromState();
        recomputeAndRenderStatic();
      }
    });

    // Allocator burst simulation
    dom.burstReqBtn.addEventListener("click", addBurstRequest);
    dom.resetBlocksBtn.addEventListener("click", () => {
      resetAllocatorRequests();
      drawAllocatorCanvas();
    });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (resizeTimer) cancelAnimationFrame(resizeTimer);
      resizeTimer = requestAnimationFrame(() => {
        drawRooflineChart();
        drawParetoChart();
        drawAllocatorCanvas();
      });
    });
  }

  function init() {
    dom.gpuSelect.innerHTML = Engine.GPUS.map(
      g => `<option value="${g.id}">${g.name}</option>`
    ).join("");
    dom.modelSelect.innerHTML = Engine.MODELS.map(
      m => `<option value="${m.id}">${m.name}</option>`
    ).join("");

    initPackets();
    resetAllocatorRequests();
    syncSidebarFromState();
    bindEvents();
    recomputeAndRenderStatic();
    requestAnimationFrame(tick);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();