/**
 * ThinkModule — Autonomous Driving & 3D LiDAR Cinema Studio Controller
 * ============================================================================
 * Connects SimulationEngine, Cinema3DRenderer, BEVRadarRenderer, and all
 * interactive HUD / Scenario / Playback controls with zero uncoordinated timers.
 */

import { SimulationEngine, SCENARIOS, INTRO_HOLD_SEC } from "./sim-engine.js";
import { Cinema3DRenderer, BEVRadarRenderer } from "./renderers.js";

const defaultQuality = (navigator.hardwareConcurrency || 4) <= 4 ? "laptop" : "high";

const engine = new SimulationEngine({
  scenarioId: "highway_flow",
  quality: defaultQuality,
});

const dom = {
  mainCanvas: document.getElementById("main-canvas"),
  labelsLayer: document.getElementById("labels-layer"),
  splitDivider: document.getElementById("split-divider"),
  modeBtns: Array.from(document.querySelectorAll(".vis-btn")),
  camBtns: Array.from(document.querySelectorAll(".cam-btn")),
  speedBtns: Array.from(document.querySelectorAll(".spd-btn")),
  selectQuality: document.getElementById("select-quality"),
  liveDot: document.getElementById("live-dot"),
  hudState: document.getElementById("hud-state"),
  hudSpeed: document.getElementById("hud-speed"),
  hudAccel: document.getElementById("hud-accel"),
  hudTracks: document.getElementById("hud-tracks"),
  hudTtc: document.getElementById("hud-ttc"),
  hudPoints: document.getElementById("hud-points"),
  hudTime: document.getElementById("hud-time"),
  narratorPhase: document.getElementById("narrator-phase"),
  narratorTitle: document.getElementById("narrator-title"),
  narratorText: document.getElementById("narrator-text"),
  narratorNote: document.getElementById("narrator-sensor-note"),
  narratorBody: document.getElementById("narrator-body"),
  btnNarratorToggle: document.getElementById("btn-narrator-toggle"),
  bevCanvas: document.getElementById("bev-canvas"),
  bevBody: document.getElementById("bev-body"),
  btnBevToggle: document.getElementById("btn-bev-toggle"),
  startupPill: document.getElementById("startup-pill"),
  scenarioDock: document.getElementById("scenario-dock"),
  btnPlay: document.getElementById("btn-play-pause"),
  btnStep: document.getElementById("btn-step"),
  btnRestart: document.getElementById("btn-restart"),
  slider: document.getElementById("timeline-slider"),
  markers: document.getElementById("timeline-markers"),
  timeCur: document.getElementById("time-current"),
  timeTot: document.getElementById("time-total"),
};

function syncCamButtons(activeMode) {
  dom.camBtns.forEach((b) => b.classList.toggle("active", b.dataset.cam === activeMode));
}

const cinema = new Cinema3DRenderer(dom.mainCanvas, dom.labelsLayer, (newMode) => {
  syncCamButtons(newMode);
});
const bev = new BEVRadarRenderer(dom.bevCanvas);

dom.selectQuality.value = defaultQuality;
cinema.setQuality(defaultQuality);

/* --------------------------------------------------------------------------
   Build 4 Interactive Real-World Scenario Cards
   -------------------------------------------------------------------------- */
const scenarioCards = [];
for (const sc of SCENARIOS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "sc-card";
  btn.dataset.scenario = sc.id;
  btn.innerHTML = `
    <div class="sc-top">
      <span class="sc-num">${sc.num}</span>
      <span class="sc-title">${sc.shortTitle}</span>
    </div>
    <span class="sc-sub">${sc.summary}</span>
  `;
  btn.addEventListener("click", () => selectScenario(sc.id));
  dom.scenarioDock.appendChild(btn);
  scenarioCards.push(btn);
}

function rebuildTimelineMarkers(scDef) {
  dom.slider.max = String(Math.round(scDef.duration * 10));
  dom.timeTot.textContent = `${scDef.duration.toFixed(1)}s`;
  dom.markers.innerHTML = "";
  for (let i = 1; i < scDef.phases.length; i++) {
    const pct = (scDef.phases[i].start / scDef.duration) * 100;
    const tick = document.createElement("span");
    tick.className = "phase-tick";
    tick.style.left = `${pct}%`;
    dom.markers.appendChild(tick);
  }
}

function selectScenario(id) {
  engine.setScenario(id);
  scenarioCards.forEach((c) => c.classList.toggle("active", c.dataset.scenario === id));
  rebuildTimelineMarkers(engine.scenarioDef);
  syncPlayButton();
  dom.startupPill.classList.remove("is-faded");
  updateHUD(engine.getState(), true);
}

// Highlight initial scenario
scenarioCards[0].classList.add("active");
rebuildTimelineMarkers(engine.scenarioDef);

/* --------------------------------------------------------------------------
   Visual Layer, Camera, Quality & Collapsible HUD Controls
   -------------------------------------------------------------------------- */
function selectVisualMode(mode) {
  cinema.setVisualMode(mode);
  dom.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  dom.splitDivider.classList.toggle("hidden", mode !== "split_compare");
}

dom.modeBtns.forEach((btn) => {
  btn.addEventListener("click", () => selectVisualMode(btn.dataset.mode));
});

dom.camBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.cam;
    cinema.setCameraMode(mode);
    syncCamButtons(mode);
  });
});

dom.selectQuality.addEventListener("change", () => {
  const q = dom.selectQuality.value;
  engine.setQuality(q);
  cinema.setQuality(q);
  updateHUD(engine.getState(), true);
});

dom.btnNarratorToggle.addEventListener("click", () => {
  const hidden = dom.narratorBody.classList.toggle("hidden");
  dom.btnNarratorToggle.textContent = hidden ? "+" : "−";
});

dom.btnBevToggle.addEventListener("click", () => {
  const hidden = dom.bevBody.classList.toggle("hidden");
  dom.btnBevToggle.textContent = hidden ? "+" : "−";
  if (!hidden) requestAnimationFrame(() => bev.resize());
});

/* --------------------------------------------------------------------------
   Playback, Speed & Deterministic Timeline Scrubbing
   -------------------------------------------------------------------------- */
function syncPlayButton() {
  const paused = engine.clock.paused;
  dom.btnPlay.textContent = paused ? "Resume" : "Pause";
  dom.btnPlay.classList.toggle("is-paused", paused);
  dom.liveDot.classList.toggle("is-paused", paused);
}

dom.btnPlay.addEventListener("click", () => {
  engine.togglePause();
  syncPlayButton();
  updateHUD(engine.getState(), true);
});

dom.btnStep.addEventListener("click", () => {
  if (!engine.clock.paused) {
    engine.setPaused(true);
    syncPlayButton();
  }
  engine.stepSingleFrame();
  updateHUD(engine.getState(), true);
});

dom.btnRestart.addEventListener("click", () => {
  const wasPaused = engine.clock.paused;
  engine.restart();
  engine.setPaused(wasPaused);
  syncPlayButton();
  updateHUD(engine.getState(), true);
});

function setSpeed(val) {
  engine.setSpeed(val);
  dom.speedBtns.forEach((b) => b.classList.toggle("active", Number(b.dataset.speed) === val));
}

dom.speedBtns.forEach((btn) => {
  btn.addEventListener("click", () => setSpeed(Number(btn.dataset.speed)));
});

// Deterministic Scrubber
let isScrubbing = false;
let preScrubPaused = false;

dom.slider.addEventListener("pointerdown", () => {
  isScrubbing = true;
  preScrubPaused = engine.clock.paused;
  engine.setPaused(true);
  syncPlayButton();
});

dom.slider.addEventListener("input", () => {
  engine.seekToTime(Number(dom.slider.value) / 10);
  updateHUD(engine.getState(), true);
});

const endScrub = () => {
  if (!isScrubbing) return;
  isScrubbing = false;
  engine.setPaused(preScrubPaused);
  syncPlayButton();
};
dom.slider.addEventListener("pointerup", endScrub);
dom.slider.addEventListener("change", endScrub);

/* --------------------------------------------------------------------------
   Keyboard Shortcuts (Space, ., R, 1-4 Scenarios, V View Mode)
   -------------------------------------------------------------------------- */
window.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

  if (e.code === "Space") {
    e.preventDefault();
    engine.togglePause();
    syncPlayButton();
    updateHUD(engine.getState(), true);
  } else if (e.key === ".") {
    e.preventDefault();
    engine.setPaused(true);
    syncPlayButton();
    engine.stepSingleFrame();
    updateHUD(engine.getState(), true);
  } else if (e.key === "r" || e.key === "R") {
    engine.restart();
    syncPlayButton();
    updateHUD(engine.getState(), true);
  } else if (["1", "2", "3", "4"].includes(e.key)) {
    const idx = Number(e.key) - 1;
    if (SCENARIOS[idx]) selectScenario(SCENARIOS[idx].id);
  } else if (e.key === "v" || e.key === "V") {
    const modes = ["cinema_ar", "lidar_cloud", "sensor_fusion", "split_compare"];
    const next = modes[(modes.indexOf(cinema.visualMode) + 1) % modes.length];
    selectVisualMode(next);
  }
});

/* --------------------------------------------------------------------------
   Resize & Tab Visibility Guards
   -------------------------------------------------------------------------- */
const handleResize = () => {
  cinema.resize();
  bev.resize();
};
window.addEventListener("resize", handleResize);
new ResizeObserver(handleResize).observe(document.getElementById("stage-wrap"));

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    engine.notifyVisibilityResume(performance.now());
  }
});

/* --------------------------------------------------------------------------
   Synchronized HUD Update (~16 Hz throttled DOM writes)
   -------------------------------------------------------------------------- */
let lastHudMs = 0;

function updateHUD(state, force = false) {
  const now = performance.now();
  if (!force && now - lastHudMs < 60) return;
  lastHudMs = now;

  const tel = state.telemetry;
  const ph = state.scenario.phase;

  // Startup establishing pill fades cleanly once motion ramps in
  dom.startupPill.classList.toggle("is-faded", state.clock.simTime >= INTRO_HOLD_SEC);

  dom.hudSpeed.innerHTML = `${tel.egoSpeedKmh.toFixed(1)} <small>km/h</small>`;
  const sign = tel.accelMs2 > 0.02 ? "+" : "";
  dom.hudAccel.innerHTML = `${sign}${tel.accelMs2.toFixed(2)} <small>m/s&sup2;</small>`;
  dom.hudTracks.textContent = `${tel.activeTracks} / ${tel.totalTracks}`;
  dom.hudTtc.textContent = Number.isFinite(tel.minTtcS) ? `${tel.minTtcS.toFixed(2)} s` : "∞";
  dom.hudPoints.textContent = tel.points.toLocaleString("en-US");
  dom.hudTime.innerHTML = `${tel.simTimeS.toFixed(2)} <small>s</small>`;

  dom.hudState.textContent = tel.controlState.replace(/_/g, " ");
  dom.hudState.className = "state-badge";
  if (tel.controlState === "AEB_BRAKING") dom.hudState.classList.add("state-critical");
  else if (tel.controlState === "ACC_FOLLOW" || tel.controlState === "CUT_IN_YIELD")
    dom.hudState.classList.add("state-caution");
  else if (tel.controlState === "CRUISING") dom.hudState.classList.add("state-cruise");

  // Live Technical Narrator Card
  dom.narratorPhase.textContent = `PHASE 0${ph.index + 1}/0${ph.total} · ${ph.tag}`;
  dom.narratorTitle.textContent = ph.title;
  dom.narratorText.textContent = ph.text;
  dom.narratorNote.textContent = ph.sensorNote;

  if (!isScrubbing) {
    dom.slider.value = String(Math.round(tel.simTimeS * 10));
  }
  dom.timeCur.textContent = `${tel.simTimeS.toFixed(1)}s`;
}

/* --------------------------------------------------------------------------
   Main Separated Render & Simulation Loop
   -------------------------------------------------------------------------- */
handleResize();
syncPlayButton();
updateHUD(engine.getState(), true);

function animate(nowMs) {
  requestAnimationFrame(animate);

  // Advances deterministic simulation ONLY when not paused
  engine.updateWallClock(nowMs);
  const state = engine.getState();

  // Rendering & 360° Orbit remain silky smooth even when world is paused
  cinema.render(state);
  if (!dom.bevBody.classList.contains("hidden")) {
    bev.render(state);
  }
  updateHUD(state, false);
}

requestAnimationFrame(animate);