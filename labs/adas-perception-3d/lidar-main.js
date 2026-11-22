/**
 * ThinkModule Lab — Inside the LiDAR Orchestrator + Live APD Oscilloscope Waveform
 */
import { PART_INFO, evaluateLidarPhysics } from "./lidar-physics.js";
import { LidarHardwareScene } from "./lidar-scene.js";

const $ = id => document.getElementById(id);

const dom = {
  viewport: $("viewport"),
  loadingOverlay: $("loadingOverlay"),
  orbitBtn: $("orbitBtn"),
  resetCamBtn: $("resetCamBtn"),

  detStatus: $("detStatus"),
  formulaLive: $("formulaLive"),
  telDist: $("telDist"),
  telTof: $("telTof"),
  telSnr: $("telSnr"),
  telRefl: $("telRefl"),
  telWave: $("telWave"),
  telAz: $("telAz"),

  partTag: $("partTag"),
  partTitle: $("partTitle"),
  partDesc: $("partDesc"),
  partSpecs: $("partSpecs"),
  scopeMeta: $("scopeMeta"),
  waveformCanvas: $("waveformCanvas"),
  insightBox: $("insightBox"),

  simModeGroup: $("simModeGroup"),
  explodeSlider: $("explodeSlider"),
  explodeVal: $("explodeVal"),
  distSlider: $("distSlider"),
  distVal: $("distVal"),
  targetGroup: $("targetGroup"),
  waveGroup: $("waveGroup"),
  partGroup: $("partGroup")
};

const cfg = {
  simMode: "tof",
  distanceM: 30,
  targetId: "car",
  wavelengthId: "1550",
  autoOrbit: true
};

const scene3d = new LidarHardwareScene(dom.viewport);
let phys = evaluateLidarPhysics(cfg);
let lastTs = 0;
let elapsed = 0;

function refreshPhysicsAndLabels() {
  phys = evaluateLidarPhysics(cfg);

  dom.detStatus.textContent = phys.statusText;
  dom.detStatus.className = `status-chip ${phys.statusSeverity}`;

  dom.formulaLive.textContent = `${phys.distanceM.toFixed(1)} m = (0.300 m/ns × ${phys.tofNs.toFixed(1)} ns) / 2`;
  dom.telDist.textContent = `${phys.distanceM.toFixed(1)} m`;
  dom.telTof.textContent = `${phys.tofNs.toFixed(1)} ns`;

  dom.telSnr.textContent = `${phys.snrDb.toFixed(1)} dB`;
  dom.telSnr.className = "m-val " + (!phys.detected ? "danger" : phys.weak ? "warn" : "accent");

  dom.telRefl.textContent = `${Math.round(phys.target.reflectivity * 100)}%`;
  dom.telWave.textContent = phys.wavelength.label;
  dom.scopeMeta.textContent = `Δt = ${phys.tofNs.toFixed(1)} ns`;

  dom.insightBox.textContent = phys.insight;
  dom.insightBox.className = "insight-box" + (!phys.detected ? " danger" : "");
}

function updatePartInspector(key) {
  const info = PART_INFO[key] || PART_INFO.all;
  dom.partTag.textContent = info.tag;
  dom.partTitle.textContent = info.title;
  dom.partDesc.textContent = info.desc;
  dom.partSpecs.innerHTML = info.specs
    .map(s => `<span class="spec-pill">${s}</span>`)
    .join("");

  dom.partGroup.querySelectorAll(".p-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.part === key);
  });
}

function drawOscilloscope(timeSec) {
  const canvas = dom.waveformCanvas;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(Math.floor(rect.width), 200);
  const h = 108;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 10, padR = 10, padT = 16, padB = 18;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const baseY = padT + plotH;

  // Time axis spans 0 .. 850 ns (covers 0 .. 127 m)
  const maxNs = 850;
  const txX = padL + 14;
  const rxX = padL + 14 + (phys.tofNs / maxNs) * (plotW - 22);

  // Subtle oscilloscope grid
  ctx.strokeStyle = "rgba(38, 60, 96, 0.45)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 3; i++) {
    const x = padL + (plotW * i) / 4;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, baseY); ctx.stroke();
  }

  // 6 dB Detection Threshold dashed amber line
  const threshY = baseY - plotH * 0.28;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(251, 191, 36, 0.58)";
  ctx.beginPath(); ctx.moveTo(padL, threshY); ctx.lineTo(padL + plotW, threshY); ctx.stroke();
  ctx.restore();

  ctx.fillStyle = "rgba(251, 191, 36, 0.78)";
  ctx.font = "600 8.5px ui-monospace, Menlo, monospace";
  ctx.textAlign = "right";
  ctx.fillText("6 dB Threshold", padL + plotW - 2, threshY - 4);

  // Analog APD Signal Curve (TX spike + thermal noise floor + RX echo Gaussian)
  const echoHeight = phys.detected
    ? plotH * Math.min(0.88, 0.30 + phys.echoNorm * 0.62)
    : plotH * Math.max(0.08, phys.echoNorm * 0.34);

  ctx.beginPath();
  for (let px = padL; px <= padL + plotW; px += 2) {
    const dTx = (px - txX) / 4.2;
    const dRx = (px - rxX) / 6.0;
    const txPeak = plotH * 0.88 * Math.exp(-dTx * dTx);
    const rxPeak = echoHeight * Math.exp(-dRx * dRx);
    const noise = (Math.sin(px * 0.85 + timeSec * 14) * 0.5 + Math.cos(px * 1.9 - timeSec * 19) * 0.5) * 3.2;
    const y = Math.min(baseY, baseY - 5 - txPeak - rxPeak + noise);
    if (px === padL) ctx.moveTo(px, y);
    else ctx.lineTo(px, y);
  }
  ctx.strokeStyle = phys.detected ? "#34d399" : "#f87171";
  ctx.lineWidth = 1.7;
  ctx.stroke();

  // Live photon sync sweep cursor in ToF mode
  if (cfg.simMode === "tof") {
    const p = scene3d.tofPhase;
    let curX = txX;
    if (p >= 0.10 && p <= 0.86) {
      const f = (p - 0.10) / 0.76;
      curX = txX + f * (rxX - txX);
    } else if (p > 0.86) {
      curX = rxX;
    }
    ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, baseY); ctx.stroke();
  }

  // Labels for TX & RX peaks
  ctx.font = "700 8.5px ui-monospace, Menlo, monospace";
  ctx.fillStyle = phys.wavelength.cssColor;
  ctx.textAlign = "left";
  ctx.fillText("TX", txX - 4, padT - 4);

  ctx.fillStyle = phys.detected ? "#34d399" : "#f87171";
  ctx.textAlign = "center";
  ctx.fillText(phys.detected ? "RX ECHO" : "NO LOCK", Math.min(padL + plotW - 22, Math.max(txX + 26, rxX)), padT - 4);

  // Bottom ns scale ticks
  ctx.fillStyle = "#627594";
  ctx.font = "8.5px ui-monospace, Menlo, monospace";
  ctx.textAlign = "left";
  ctx.fillText("0 ns", padL, h - 4);
  ctx.textAlign = "center";
  ctx.fillText("400 ns (60 m)", padL + plotW * 0.48, h - 4);
  ctx.textAlign = "right";
  ctx.fillText("800 ns (120 m)", padL + plotW, h - 4);
}

// ============================================================================
// UI EVENT BINDINGS
// ============================================================================

dom.simModeGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  cfg.simMode = btn.dataset.sim;
  scene3d.setSimMode(cfg.simMode);
  dom.simModeGroup.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
});

dom.explodeSlider.addEventListener("input", () => {
  const v = parseInt(dom.explodeSlider.value, 10);
  dom.explodeVal.textContent = `${v}%`;
  scene3d.setExplode(v / 100);
});

dom.distSlider.addEventListener("input", () => {
  cfg.distanceM = parseInt(dom.distSlider.value, 10);
  dom.distVal.textContent = `${cfg.distanceM} m`;
  refreshPhysicsAndLabels();
});

dom.targetGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  cfg.targetId = btn.dataset.target;
  dom.targetGroup.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
  refreshPhysicsAndLabels();
});

dom.waveGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".seg");
  if (!btn) return;
  cfg.wavelengthId = btn.dataset.wave;
  dom.waveGroup.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
  refreshPhysicsAndLabels();
});

dom.partGroup.addEventListener("click", evt => {
  const btn = evt.target.closest(".p-btn");
  if (!btn) return;
  const key = btn.dataset.part;
  updatePartInspector(key);
  scene3d.focusPart(key);

  const syncedPct = Math.round(scene3d.explodeTarget * 100);
  dom.explodeSlider.value = String(syncedPct);
  dom.explodeVal.textContent = `${syncedPct}%`;
});

dom.orbitBtn.addEventListener("click", () => {
  cfg.autoOrbit = !cfg.autoOrbit;
  scene3d.setAutoOrbit(cfg.autoOrbit);
  dom.orbitBtn.classList.toggle("active", cfg.autoOrbit);
});

dom.resetCamBtn.addEventListener("click", () => {
  updatePartInspector("all");
  scene3d.focusPart("all");
});

// 60 FPS Animation Loop
function animate(ts) {
  requestAnimationFrame(animate);
  if (!lastTs) lastTs = ts;
  const dt = Math.min((ts - lastTs) / 1000, 0.05);
  lastTs = ts;
  elapsed += dt;

  scene3d.update(dt, elapsed, phys);
  const deg = ((scene3d.rotorAngle * 180) / Math.PI) % 360;
  dom.telAz.textContent = `${deg.toFixed(1)}°`;
  drawOscilloscope(elapsed);
}

updatePartInspector("all");
refreshPhysicsAndLabels();
requestAnimationFrame(animate);

setTimeout(() => {
  dom.loadingOverlay.classList.add("hidden");
}, 260);