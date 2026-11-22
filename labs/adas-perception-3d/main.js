/**
 * ThinkModule — Multi-View Autonomous Driving & LiDAR Controller
 * ============================================================================
 * Connects SimulationEngine (headless physics/sensors) with MultiViewRenderer
 * and all interactive DOM controls, timeline scrubbing, viewport expansion,
 * keyboard shortcuts, and the Engineering Object Inspector.
 */

import { SimulationEngine, SCENARIOS } from "./sim-engine.js";
import { MultiViewRenderer } from "./renderers.js";

// Detect modest hardware defaults gracefully
const defaultQuality = (navigator.hardwareConcurrency || 4) <= 4 ? "low" : "balanced";

const engine = new SimulationEngine({
  scenarioId: "guided_sequence",
  quality: defaultQuality,
  trafficDensity: "moderate",
});

const renderer = new MultiViewRenderer({
  v1: document.getElementById("canvas-view-1"),
  v2: document.getElementById("canvas-view-2"),
  v3: document.getElementById("canvas-view-3"),
  v4: document.getElementById("canvas-view-4"),
});
renderer.setQuality(defaultQuality);

// Shared Visualization Settings (strictly separated from physical WorldState)
const settings = {
  expandedView: null, // null = Quad Overview, 1..4 = Expanded Detail
  selectedTrackId: null,
  debugMode: false,
  showLidar: true,
  showDetection: true,
  showTracking: true,
  v1Rays: true,
  v1Boxes: true,
  v1Path: true,
  lidarColorMode: "distance",
  lidarPointSize: 2.2,
  lidarFrustum: true,
  v3Boxes: true,
  v3Trajectories: true,
  v3Occlusion: true,
  v3Labels: true,
};

// DOM References
const dom = {
  stage: document.getElementById("viewport-stage"),
  vpCards: [1, 2, 3, 4].map((i) => document.getElementById(`vp-${i}`)),
  phaseBadge: document.getElementById("phase-badge"),
  phaseDesc: document.getElementById("phase-desc"),
  telSpeed: document.getElementById("tel-speed"),
  telAccel: document.getElementById("tel-accel"),
  telHz: document.getElementById("tel-hz"),
  telTracks: document.getElementById("tel-tracks"),
  telTtc: document.getElementById("tel-ttc"),
  telState: document.getElementById("tel-state"),
  telTime: document.getElementById("tel-time"),
  hintText: document.getElementById("interaction-hint"),
  btnQuadReset: document.getElementById("btn-quad-reset"),
  v1CamStat: document.getElementById("v1-cam-stat"),
  v2PointsStat: document.getElementById("v2-points-stat"),
  v3TracksStat: document.getElementById("v3-tracks-stat"),
  v4StateBadge: document.getElementById("v4-state-badge"),
  v4LatestEvent: document.getElementById("v4-latest-event"),
  slider: document.getElementById("timeline-slider"),
  markers: document.getElementById("timeline-markers"),
  timeCur: document.getElementById("time-current"),
  timeTot: document.getElementById("time-total"),
  btnPlay: document.getElementById("btn-play-pause"),
  btnStep: document.getElementById("btn-step-frame"),
  btnRestart: document.getElementById("btn-restart"),
  selectScenario: document.getElementById("select-scenario"),
  selectDensity: document.getElementById("select-density"),
  selectQuality: document.getElementById("select-quality"),
  btnDebug: document.getElementById("btn-debug-toggle"),
  toggleLidar: document.getElementById("toggle-lidar"),
  toggleDetection: document.getElementById("toggle-detection"),
  toggleTracking: document.getElementById("toggle-tracking"),
  inspector: document.getElementById("inspector-panel"),
  inspId: document.getElementById("insp-id"),
  inspClass: document.getElementById("insp-class"),
  inspStatus: document.getElementById("insp-status"),
  inspDist: document.getElementById("insp-dist"),
  inspRelVel: document.getElementById("insp-relvel"),
  inspAbsVel: document.getElementById("insp-absvel"),
  inspTtc: document.getElementById("insp-ttc"),
  inspHits: document.getElementById("insp-hits"),
  inspOccl: document.getElementById("insp-occl"),
  inspConf: document.getElementById("insp-conf"),
  eventLog: document.getElementById("event-log-list"),
  keysModal: document.getElementById("keys-modal"),
};

// Initialize Scenario Selector & Quality Selector
dom.selectQuality.value = defaultQuality;
for (const sc of SCENARIOS) {
  const opt = document.createElement("option");
  opt.value = sc.id;
  opt.textContent = sc.name;
  dom.selectScenario.appendChild(opt);
}

function rebuildTimelineMarkers(scenario) {
  dom.slider.max = String(Math.round(scenario.duration * 10));
  dom.timeTot.textContent = `${scenario.duration.toFixed(1)}s`;
  dom.markers.innerHTML = "";
  for (let i = 1; i < scenario.phases.length; i++) {
    const pct = (scenario.phases[i].start / scenario.duration) * 100;
    const tick = document.createElement("span");
    tick.className = "phase-tick";
    tick.style.left = `${pct}%`;
    dom.markers.appendChild(tick);
  }
}

rebuildTimelineMarkers(engine.scenarioDef);

/* --------------------------------------------------------------------------
   Viewport Hybrid Expansion Controller (Never restarts simulation!)
   -------------------------------------------------------------------------- */
function setExpandedView(viewNum) {
  const target = settings.expandedView === viewNum ? null : viewNum;
  settings.expandedView = target;

  dom.stage.classList.toggle("layout-quad", target === null);
  dom.stage.classList.toggle("layout-expanded", target !== null);
  dom.btnQuadReset.classList.toggle("hidden", target === null);

  dom.vpCards.forEach((card, idx) => {
    card.classList.toggle("is-expanded", target === idx + 1);
  });

  if (target === null) {
    dom.hintText.textContent =
      "Select any viewport to inspect its underlying system. Click a tracked vehicle in View 03 for live telemetry.";
  } else {
    const titles = [
      "View 01 · 3D Environment detail active — switch camera modes or orbit freely.",
      "View 02 · 3D LiDAR Point Cloud detail active — drag to orbit, scroll to zoom, adjust color & noise.",
      "View 03 · Object Detection & Tracking detail active — click any vehicle to inspect telemetry.",
      "View 04 · Autonomous Driving Intelligence active — live rule-based IDM / AEB telemetry.",
    ];
    dom.hintText.textContent = titles[target - 1];
  }

  requestAnimationFrame(() => renderer.resizeAll());
}

// Header & Expand Button clicks
document.querySelectorAll(".vp-expand-btn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    setExpandedView(Number(btn.dataset.expand));
  });
});

dom.vpCards.forEach((card, idx) => {
  const viewNum = idx + 1;
  const header = card.querySelector(".vp-header");
  header.addEventListener("click", () => setExpandedView(viewNum));

  // Clicking a minimized sidebar thumbnail expands it directly
  card.addEventListener("click", (e) => {
    if (e.target.closest("[data-no-expand]") || e.target.closest("button")) return;
    if (settings.expandedView !== null && settings.expandedView !== viewNum) {
      setExpandedView(viewNum);
    }
  });
});

dom.btnQuadReset.addEventListener("click", () => {
  if (settings.expandedView !== null) setExpandedView(settings.expandedView);
});

/* --------------------------------------------------------------------------
   Playback, Speed & Deterministic Timeline Scrubbing
   -------------------------------------------------------------------------- */
function syncPlayButton() {
  const paused = engine.clock.paused;
  dom.btnPlay.textContent = paused ? "Resume" : "Pause";
  dom.btnPlay.classList.toggle("is-paused", paused);
}

dom.btnPlay.addEventListener("click", () => {
  engine.togglePause();
  syncPlayButton();
  updateDOM(engine.getState(), true);
});

dom.btnStep.addEventListener("click", () => {
  if (!engine.clock.paused) {
    engine.setPaused(true);
    syncPlayButton();
  }
  engine.stepSingleFrame();
  updateDOM(engine.getState(), true);
});

dom.btnRestart.addEventListener("click", () => {
  const wasPaused = engine.clock.paused;
  engine.restart();
  engine.setPaused(wasPaused);
  syncPlayButton();
  updateDOM(engine.getState(), true);
});

// Speed Multiplier Buttons
const speedBtns = Array.from(document.querySelectorAll(".speed-btn"));
function setSimSpeed(val) {
  engine.setSpeed(val);
  speedBtns.forEach((b) => b.classList.toggle("active", Number(b.dataset.speed) === val));
}

speedBtns.forEach((btn) => {
  btn.addEventListener("click", () => setSimSpeed(Number(btn.dataset.speed)));
});

// Timeline Scrubber
let isScrubbing = false;
let preScrubPaused = false;

dom.slider.addEventListener("pointerdown", () => {
  isScrubbing = true;
  preScrubPaused = engine.clock.paused;
  engine.setPaused(true);
  syncPlayButton();
});

dom.slider.addEventListener("input", () => {
  const tSec = Number(dom.slider.value) / 10;
  engine.seekToTime(tSec);
  updateDOM(engine.getState(), true);
});

const finishScrub = () => {
  if (!isScrubbing) return;
  isScrubbing = false;
  engine.setPaused(preScrubPaused);
  syncPlayButton();
};
dom.slider.addEventListener("pointerup", finishScrub);
dom.slider.addEventListener("change", finishScrub);

/* --------------------------------------------------------------------------
   Scenario, Traffic Density, Quality & Layer Controls
   -------------------------------------------------------------------------- */
dom.selectScenario.addEventListener("change", () => {
  settings.selectedTrackId = null;
  dom.inspector.classList.add("hidden");
  engine.setScenario(dom.selectScenario.value);
  rebuildTimelineMarkers(engine.scenarioDef);
  syncPlayButton();
  updateDOM(engine.getState(), true);
  requestAnimationFrame(() => renderer.resizeAll());
});

dom.selectDensity.addEventListener("change", () => {
  engine.setTrafficDensity(dom.selectDensity.value);
});

dom.selectQuality.addEventListener("change", () => {
  const q = dom.selectQuality.value;
  engine.setQuality(q);
  renderer.setQuality(q);
  updateDOM(engine.getState(), true);
});

dom.btnDebug.addEventListener("click", () => {
  settings.debugMode = !settings.debugMode;
  dom.btnDebug.classList.toggle("active", settings.debugMode);
});

// Synchronized Layer Toggles
dom.toggleLidar.addEventListener("click", () => {
  settings.showLidar = !settings.showLidar;
  dom.toggleLidar.classList.toggle("active", settings.showLidar);
  document.getElementById("chk-v1-rays").checked = settings.showLidar;
});

dom.toggleDetection.addEventListener("click", () => {
  settings.showDetection = !settings.showDetection;
  dom.toggleDetection.classList.toggle("active", settings.showDetection);
  document.getElementById("chk-v1-boxes").checked = settings.showDetection;
  document.getElementById("chk-v3-boxes").checked = settings.showDetection;
});

dom.toggleTracking.addEventListener("click", () => {
  settings.showTracking = !settings.showTracking;
  dom.toggleTracking.classList.toggle("active", settings.showTracking);
  document.getElementById("chk-v3-trajectories").checked = settings.showTracking;
  document.getElementById("chk-v3-labels").checked = settings.showTracking;
});

/* --------------------------------------------------------------------------
   Per-Viewport Controls (Cameras, Point Cloud Settings, Overlays)
   -------------------------------------------------------------------------- */
const camBtns = Array.from(document.querySelectorAll("[data-cam]"));
const camLabels = {
  follow: "Follow Cam",
  bev: "Bird's-Eye",
  sensor: "Sensor POV",
  orbit: "Free Orbit",
  track: "Tracking Cam",
};

function selectCamera(mode) {
  renderer.view1.setCameraMode(mode);
  dom.v1CamStat.textContent = camLabels[mode] || mode;
  camBtns.forEach((b) => b.classList.toggle("active", b.dataset.cam === mode));
}

camBtns.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectCamera(btn.dataset.cam);
  });
});

document.getElementById("btn-reset-cam-1").addEventListener("click", (e) => {
  e.stopPropagation();
  renderer.view1.resetCamera();
  selectCamera(renderer.view1.camMode);
});

document.getElementById("btn-reset-cam-2").addEventListener("click", (e) => {
  e.stopPropagation();
  renderer.view2.resetCamera();
});

// View 01 checkboxes
document.getElementById("chk-v1-rays").addEventListener("change", (e) => {
  settings.v1Rays = e.target.checked;
});
document.getElementById("chk-v1-boxes").addEventListener("change", (e) => {
  settings.v1Boxes = e.target.checked;
});
document.getElementById("chk-v1-path").addEventListener("change", (e) => {
  settings.v1Path = e.target.checked;
});

// View 02 LiDAR controls
document.getElementById("select-lidar-color").addEventListener("change", (e) => {
  settings.lidarColorMode = e.target.value;
});
document.getElementById("select-lidar-density").addEventListener("change", (e) => {
  engine.setLidarDensityScale(Number(e.target.value));
  updateDOM(engine.getState(), true);
});
document.getElementById("range-point-size").addEventListener("input", (e) => {
  settings.lidarPointSize = Number(e.target.value);
});
document.getElementById("chk-lidar-frustum").addEventListener("change", (e) => {
  settings.lidarFrustum = e.target.checked;
});
document.getElementById("chk-lidar-noise").addEventListener("change", (e) => {
  engine.setLidarNoise(e.target.checked);
});

// View 03 Tracking controls
document.getElementById("chk-v3-boxes").addEventListener("change", (e) => {
  settings.v3Boxes = e.target.checked;
});
document.getElementById("chk-v3-trajectories").addEventListener("change", (e) => {
  settings.v3Trajectories = e.target.checked;
});
document.getElementById("chk-v3-occlusion").addEventListener("change", (e) => {
  settings.v3Occlusion = e.target.checked;
});
document.getElementById("chk-v3-labels").addEventListener("change", (e) => {
  settings.v3Labels = e.target.checked;
});

/* --------------------------------------------------------------------------
   View 03 Click-to-Inspect Vehicle Engineering Inspector
   -------------------------------------------------------------------------- */
document.getElementById("canvas-view-3").addEventListener("click", (e) => {
  const pickedId = renderer.view3.pickTrackAt(e.clientX, e.clientY);
  if (pickedId) {
    e.stopPropagation();
    settings.selectedTrackId = pickedId;
    dom.inspector.classList.remove("hidden");
    updateInspector(engine.getState());
    requestAnimationFrame(() => renderer.resizeAll());
  }
});

document.getElementById("btn-close-inspector").addEventListener("click", () => {
  settings.selectedTrackId = null;
  dom.inspector.classList.add("hidden");
  requestAnimationFrame(() => renderer.resizeAll());
});

document.getElementById("btn-insp-track-cam").addEventListener("click", () => {
  selectCamera("track");
});

/* --------------------------------------------------------------------------
   Keyboard Controls & Shortcuts Modal
   -------------------------------------------------------------------------- */
document.getElementById("btn-help-keys").addEventListener("click", () => {
  dom.keysModal.classList.toggle("hidden");
});
document.getElementById("btn-close-keys").addEventListener("click", () => {
  dom.keysModal.classList.add("hidden");
});
dom.keysModal.addEventListener("click", (e) => {
  if (e.target === dom.keysModal) dom.keysModal.classList.add("hidden");
});

window.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  if ((tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") && e.key !== "Escape") {
    return;
  }

  if (e.code === "Space") {
    e.preventDefault();
    engine.togglePause();
    syncPlayButton();
    updateDOM(engine.getState(), true);
  } else if (e.key === ".") {
    e.preventDefault();
    engine.setPaused(true);
    syncPlayButton();
    engine.stepSingleFrame();
    updateDOM(engine.getState(), true);
  } else if (e.key === "r" || e.key === "R") {
    engine.restart();
    syncPlayButton();
    updateDOM(engine.getState(), true);
  } else if (["1", "2", "3", "4"].includes(e.key)) {
    setExpandedView(Number(e.key));
  } else if (e.key === "Escape") {
    if (!dom.keysModal.classList.contains("hidden")) {
      dom.keysModal.classList.add("hidden");
    } else if (settings.expandedView !== null) {
      setExpandedView(settings.expandedView);
    } else if (settings.selectedTrackId) {
      settings.selectedTrackId = null;
      dom.inspector.classList.add("hidden");
      requestAnimationFrame(() => renderer.resizeAll());
    }
  } else if (e.key === "[" || e.key === "]") {
    const speeds = [0.25, 0.5, 1, 2];
    const curIdx = speeds.indexOf(engine.clock.speed);
    const nextIdx = e.key === "[" ? Math.max(0, curIdx - 1) : Math.min(speeds.length - 1, curIdx + 1);
    setSimSpeed(speeds[nextIdx]);
  } else if (e.key === "d" || e.key === "D") {
    settings.debugMode = !settings.debugMode;
    dom.btnDebug.classList.toggle("active", settings.debugMode);
  } else if (e.key === "?") {
    dom.keysModal.classList.toggle("hidden");
  }
});

/* --------------------------------------------------------------------------
   Resize Handling & Browser Tab Visibility Safety
   -------------------------------------------------------------------------- */
const ro = new ResizeObserver(() => renderer.resizeAll());
ro.observe(dom.stage);
window.addEventListener("resize", () => renderer.resizeAll());

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    engine.notifyVisibilityResume(performance.now());
  }
});

/* --------------------------------------------------------------------------
   DOM Telemetry & Inspector Synchronization (Throttled to ~15 Hz)
   -------------------------------------------------------------------------- */
let lastDomUpdateMs = 0;

function updateInspector(state) {
  if (!settings.selectedTrackId) return;
  const trk = state.tracks.find((t) => t.id === settings.selectedTrackId);
  if (!trk) {
    dom.inspStatus.textContent = "OUT OF RANGE";
    dom.inspHits.textContent = "0 pts";
    return;
  }
  dom.inspId.textContent = trk.id;
  dom.inspClass.textContent = trk.label;
  dom.inspStatus.textContent = trk.status;
  dom.inspDist.textContent = `${trk.distance.toFixed(1)} m`;
  const sign = trk.relSpeed >= 0 ? "+" : "";
  dom.inspRelVel.textContent = `${sign}${trk.relSpeed.toFixed(1)} m/s`;
  dom.inspAbsVel.textContent = `${(trk.absSpeed * 3.6).toFixed(1)} km/h`;
  dom.inspTtc.textContent = Number.isFinite(trk.ttc) ? `${trk.ttc.toFixed(2)} s` : "∞";
  dom.inspHits.textContent = `${trk.hitCount} pts`;
  dom.inspOccl.textContent = `${Math.round(trk.occludedRatio * 100)}%`;
  dom.inspConf.textContent = `${Math.round(trk.confidence * 100)}%`;

  // Sync event log
  dom.eventLog.innerHTML = state.events
    .slice(0, 8)
    .map(
      (ev) =>
        `<li class="${ev.category === "WARNING" ? "ev-warn" : ""}"><span class="ev-t">[${ev.time.toFixed(1)}s]</span>${ev.message}</li>`
    )
    .join("");
}

function updateDOM(state, force = false) {
  const now = performance.now();
  if (!force && now - lastDomUpdateMs < 65) return;
  lastDomUpdateMs = now;

  const tel = state.telemetry;
  const ph = state.scenario.phase;

  dom.phaseBadge.textContent = `PHASE 0${ph.index + 1}/0${ph.total} · ${ph.name.toUpperCase()}`;
  dom.phaseDesc.textContent = ph.detail;

  // Guided Hybrid Spotlight in Quad Mode
  dom.vpCards.forEach((card, idx) => {
    card.classList.toggle("is-spotlight", settings.expandedView === null && ph.spotlightView === idx + 1);
  });

  dom.telSpeed.innerHTML = `${tel.egoSpeedKmh.toFixed(1)} <small>km/h</small>`;
  const accSign = tel.accelMs2 > 0.02 ? "+" : "";
  dom.telAccel.innerHTML = `${accSign}${tel.accelMs2.toFixed(2)} <small>m/s&sup2;</small>`;
  dom.telHz.innerHTML = `${tel.scanRateHz.toFixed(1)} <small>Hz</small>`;
  dom.telTracks.textContent = String(tel.activeTracks);
  dom.telTtc.textContent = Number.isFinite(tel.minTtcS) ? `${tel.minTtcS.toFixed(2)} s` : "∞";
  dom.telTime.innerHTML = `${tel.simTimeS.toFixed(2)} <small>s</small>`;

  dom.telState.textContent = tel.controlState.replace(/_/g, " ");
  dom.telState.className = "tel-v state-tag";
  if (tel.controlState === "AEB_BRAKING") dom.telState.classList.add("state-critical");
  else if (tel.controlState === "ACC_FOLLOW" || tel.controlState === "CUT_IN_YIELD")
    dom.telState.classList.add("state-caution");
  else if (tel.controlState === "INIT" || tel.controlState === "CALIBRATING")
    dom.telState.classList.add("state-init");

  // Viewport header badges
  dom.v2PointsStat.textContent = `${state.lidar.activePoints.toLocaleString("en-US")} pts · ${tel.scanRateHz.toFixed(0)} Hz`;
  dom.v3TracksStat.textContent = `${tel.activeTracks} confirmed / ${tel.totalTracked} tracked`;
  dom.v4StateBadge.textContent = `Rule-Based · ${tel.controlState.replace(/_/g, " ")}`;
  if (state.events.length > 0) {
    const latest = state.events[0];
    dom.v4LatestEvent.textContent = `[${latest.time.toFixed(1)}s] ${latest.message}`;
  }

  // Timeline Scrubber sync
  if (!isScrubbing) {
    dom.slider.value = String(Math.round(tel.simTimeS * 10));
  }
  dom.timeCur.textContent = `${tel.simTimeS.toFixed(1)}s`;

  if (settings.selectedTrackId) {
    updateInspector(state);
  }
}

/* --------------------------------------------------------------------------
   Main Separated Render & Simulation Loop
   -------------------------------------------------------------------------- */
renderer.resizeAll();
syncPlayButton();
updateDOM(engine.getState(), true);

function animate(nowMs) {
  requestAnimationFrame(animate);

  // Advances simulation ONLY if not paused (fixed 1/60s ticks, clamped on tab switch)
  engine.updateWallClock(nowMs);

  const state = engine.getState();

  // Render always stays responsive so user can orbit cameras, inspect vehicles, or scrub while paused
  renderer.renderAll(state, settings);
  updateDOM(state, false);
}

requestAnimationFrame(animate);